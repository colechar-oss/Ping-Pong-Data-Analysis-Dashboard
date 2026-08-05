"""
Fit and validate an Elo probability scaling constant with time-series cross-validation.

Run from IDLE:
    1. Set INPUT_FILE and OUTPUT_DIR below.
    2. Press F5.

What it does:
    - Loads a game history CSV with pre-game Elo values
    - Searches for the best Elo scaling constant on the training portion
    - Uses time-series CV to avoid leaking future games into calibration
    - Evaluates the chosen scale on a final holdout block
    - Saves:
        * elo_scale_search.csv
        * elo_holdout_predictions.csv
        * elo_reliability_bins.csv
        * elo_scale_fit.png
        * elo_reliability_plot.png

Required columns in INPUT_FILE:
    - team1_pre_elo
    - team2_pre_elo
    - team1_actual_win

Recommended columns:
    - game_id (for chronological ordering)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = r"C:\Users\colechar\Documents\Python code files\PingPong\ping_pong_output\game_history_with_elo.csv"
OUTPUT_DIR = r"C:\Users\colechar\Documents\Python code files\PingPong\ping_pong_output\elo_scale_fit"

TRAIN_FRACTION = 0.80
N_SPLITS = 5

SCALE_MIN = 1.0
SCALE_MAX = 1000.0
SCALE_STEP = 1.0

BASELINE_SCALE = 50.0
N_BINS = 10

# ============================================================
# SETUP
# ============================================================

outdir = Path(OUTPUT_DIR)
outdir.mkdir(parents=True, exist_ok=True)

history = pd.read_csv(INPUT_FILE)

required = ["team1_pre_elo", "team2_pre_elo", "team1_actual_win"]
missing = [c for c in required if c not in history.columns]
if missing:
    raise ValueError(f"Missing required columns:\n{missing}")

# Sort chronologically if possible.
if "game_id" in history.columns:
    history = history.sort_values("game_id").reset_index(drop=True).copy()
elif "day" in history.columns:
    history = history.sort_values(["day"]).reset_index(drop=True).copy()
else:
    history = history.reset_index(drop=True).copy()

history["elo_diff_pre"] = history["team1_pre_elo"] - history["team2_pre_elo"]
history["team1_actual_win"] = history["team1_actual_win"].astype(int)

split_idx = int(len(history) * TRAIN_FRACTION)
split_idx = max(1, min(split_idx, len(history) - 1))

train_df = history.iloc[:split_idx].copy()
holdout_df = history.iloc[split_idx:].copy()


# ============================================================
# FUNCTIONS
# ============================================================

def elo_probability(elo_diff, scale):
    """
    Convert Elo difference to win probability.

    Positive elo_diff means Team 1 is stronger.
    Larger scale -> flatter curve -> probabilities closer to 50%.
    """
    elo_diff = np.asarray(elo_diff, dtype=float)
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / scale))


def safe_log_loss(y_true, y_prob):
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return log_loss(y_true, y_prob)


def score_predictions(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(safe_log_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def cv_score_scale(train_data, scale, n_splits=5):
    """
    Time-series cross-validation score for one scale.

    For each fold:
      - use only the test fold predictions
      - compute Brier, log loss, accuracy
    """
    if len(train_data) < n_splits + 1:
        raise ValueError("Not enough training games for the requested number of CV splits.")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []

    for _, test_idx in tscv.split(train_data):
        test = train_data.iloc[test_idx].copy()

        probs = elo_probability(test["elo_diff_pre"], scale)
        metrics = score_predictions(test["team1_actual_win"], probs)
        fold_metrics.append(metrics)

    out = pd.DataFrame(fold_metrics)
    return {
        "scale": float(scale),
        "mean_brier_score": float(out["brier_score"].mean()),
        "std_brier_score": float(out["brier_score"].std(ddof=0)),
        "mean_log_loss": float(out["log_loss"].mean()),
        "std_log_loss": float(out["log_loss"].std(ddof=0)),
        "mean_accuracy": float(out["accuracy"].mean()),
        "std_accuracy": float(out["accuracy"].std(ddof=0)),
    }


def reliability_bins(y_true, y_prob, n_bins=10):
    df = pd.DataFrame(
        {
            "team1_actual_win": np.asarray(y_true, dtype=int),
            "pred_prob": np.asarray(y_prob, dtype=float),
        }
    )

    bins = pd.cut(
        df["pred_prob"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
    )

    out = (
        df.assign(bin=bins)
        .groupby("bin", observed=False)
        .agg(
            count=("team1_actual_win", "count"),
            mean_pred=("pred_prob", "mean"),
            actual_rate=("team1_actual_win", "mean"),
        )
        .reset_index()
        .dropna(subset=["mean_pred", "actual_rate"])
    )

    out["abs_gap"] = (out["mean_pred"] - out["actual_rate"]).abs()
    out["weighted_gap"] = out["abs_gap"] * out["count"] / max(len(df), 1)
    return out


# ============================================================
# GRID SEARCH ON TRAINING PORTION ONLY
# ============================================================

print("\nSearching for the best Elo scaling constant on the training portion...\n")

scales = np.arange(SCALE_MIN, SCALE_MAX + SCALE_STEP, SCALE_STEP)

search_rows = []
best_row = None

for scale in scales:
    row = cv_score_scale(train_df, scale, n_splits=N_SPLITS)
    search_rows.append(row)

    if best_row is None or row["mean_brier_score"] < best_row["mean_brier_score"]:
        best_row = row

search_df = pd.DataFrame(search_rows).sort_values("mean_brier_score", ascending=True).reset_index(drop=True)
best_scale = float(best_row["scale"])

search_df.to_csv(str(outdir / "elo_scale_search.csv"), index=False)

print("=" * 70)
print(f"Best scale from CV = {best_scale:.3f}")
print(f"Mean CV Brier      = {best_row['mean_brier_score']:.5f}")
print(f"Mean CV Log Loss   = {best_row['mean_log_loss']:.5f}")
print(f"Mean CV Accuracy   = {best_row['mean_accuracy']:.3f}")
print("=" * 70)


# ============================================================
# HOLDOUT EVALUATION
# ============================================================

holdout_prob_best = elo_probability(holdout_df["elo_diff_pre"], best_scale)
holdout_prob_baseline = elo_probability(holdout_df["elo_diff_pre"], BASELINE_SCALE)

best_metrics = score_predictions(holdout_df["team1_actual_win"], holdout_prob_best)
baseline_metrics = score_predictions(holdout_df["team1_actual_win"], holdout_prob_baseline)

holdout_predictions = holdout_df[[
    c for c in ["game_id", "day", "date", "p1", "p2", "p3", "p4", "score1", "score2", "team1_actual_win", "elo_diff_pre"]
    if c in holdout_df.columns
]].copy()

holdout_predictions["pred_prob_best_scale"] = holdout_prob_best
holdout_predictions["pred_prob_baseline_scale"] = holdout_prob_baseline
holdout_predictions["pred_win_best_scale"] = (holdout_predictions["pred_prob_best_scale"] >= 0.5).astype(int)
holdout_predictions["correct_best_scale"] = (holdout_predictions["pred_win_best_scale"] == holdout_predictions["team1_actual_win"]).astype(int)

holdout_predictions.to_csv(str(outdir / "elo_holdout_predictions.csv"), index=False)

reliability_df = reliability_bins(
    holdout_predictions["team1_actual_win"],
    holdout_predictions["pred_prob_best_scale"],
    n_bins=N_BINS,
)
reliability_df.to_csv(str(outdir / "elo_reliability_bins.csv"), index=False)

print("\nHoldout performance (best scale):")
print(f"  Brier score  = {best_metrics['brier_score']:.5f}")
print(f"  Log loss     = {best_metrics['log_loss']:.5f}")
print(f"  Accuracy     = {best_metrics['accuracy']:.3f}")

print("\nHoldout performance (baseline scale = 50):")
print(f"  Brier score  = {baseline_metrics['brier_score']:.5f}")
print(f"  Log loss     = {baseline_metrics['log_loss']:.5f}")
print(f"  Accuracy     = {baseline_metrics['accuracy']:.3f}")


# ============================================================
# PLOTS
# ============================================================

# 1) CV scale search curve
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    search_df["scale"],
    search_df["mean_brier_score"],
    linewidth=3,
)

ax.scatter(
    best_scale,
    best_row["mean_brier_score"],
    color="red",
    s=100,
    zorder=5,
)

ax.set_xlabel("Elo scaling constant")
ax.set_ylabel("Mean CV Brier score")
ax.set_title("Time-Series CV: Optimizing Elo Scaling Constant")

fig.tight_layout()
scale_plot_path = outdir / "elo_scale_fit.png"
fig.savefig(str(scale_plot_path), dpi=250, bbox_inches="tight")
plt.close(fig)

# 2) Reliability diagram on holdout
fig, ax = plt.subplots(figsize=(7, 7))

ax.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Perfect calibration")

if not reliability_df.empty:
    ax.plot(
        reliability_df["mean_pred"],
        reliability_df["actual_rate"],
        marker="o",
        linewidth=2.5,
        label="Best scale (holdout)",
    )

    for _, r in reliability_df.iterrows():
        ax.annotate(
            str(int(r["count"])),
            (r["mean_pred"], r["actual_rate"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

ax.set_xlabel("Mean predicted win probability")
ax.set_ylabel("Observed win rate")
ax.set_title("Elo Calibration Reliability Plot")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="best")

fig.tight_layout()
reliability_plot_path = outdir / "elo_reliability_plot.png"
fig.savefig(str(reliability_plot_path), dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {scale_plot_path}")
print(f"Saved: {reliability_plot_path}")
print(f"Saved: {outdir / 'elo_scale_search.csv'}")
print(f"Saved: {outdir / 'elo_holdout_predictions.csv'}")
print(f"Saved: {outdir / 'elo_reliability_bins.csv'}")

plt.show()
