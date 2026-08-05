"""Ping pong doubles analysis pipeline for lab log files.

Run from IDLE:
  1. Open this file in IDLE.
  2. Set INPUT_FILE and OUTPUT_DIR in the CONFIG section.
  3. Press F5.

Supported input formats:
  - .txt lab log with one game per line and a blank line between days
  - .csv with columns p1,p2,p3,p4 and score1,score2 (or score)

Outputs:
  - Elo ratings
  - Optional TrueSkill ratings if installed
  - Position advantages
  - Best teammate combinations
  - Win probability estimates
  - Player Elo over time
  - Partnership network graph
  - Overall all-time rankings
  - Corner/position analysis
  - Interactive HTML dashboard
"""

from __future__ import annotations

import json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False
    print("No SHAP")

import argparse
import re
from collections import defaultdict, deque
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sqrt
from scipy.stats import norm

from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG FOR IDLE USERS
# ============================================================
INPUT_FILE = "PingPongData.txt"
OUTPUT_DIR = "ping_pong_output"
DEFAULT_ELO = 1500.0
K_FACTOR = 10.0
MIN_PAIR_GAMES_FOR_DISPLAY = 2

# ============================================================
# OPTIONAL IMPORTS
# ============================================================
try:
    import trueskill  # type: ignore
    TRUESKILL_AVAILABLE = True
except Exception:
    trueskill = None
    TRUESKILL_AVAILABLE = False
    print("No TrueSkill")

try:
    import networkx as nx  # type: ignore
    NETWORKX_AVAILABLE = True
except Exception:
    nx = None
    NETWORKX_AVAILABLE = False
    print("No NetWorkX")

try:
    import plotly.express as px  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False
    print("No Plotly")

# ============================================================
# TEXT PARSING
# ============================================================
GAME_LINE_RE = re.compile(
    r"^\s*(?:\d+\)\s*)?"
    r"(?P<p1>[A-Za-z0-9_\-]+)\s*,\s*(?P<p2>[A-Za-z0-9_\-]+)"
    r"\s+Vs\.?\s+"
    r"(?P<p3>[A-Za-z0-9_\-]+)\s*,\s*(?P<p4>[A-Za-z0-9_\-]+)"
    r"\s*:\s*"
    r"(?P<score1>\d+)\s*[-–]\s*(?P<score2>\d+)\s*$",
    re.IGNORECASE,
)


def parse_txt_log(path: str | Path) -> pd.DataFrame:
    """Parse a human-written text log where blank lines separate days."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: List[dict] = []
    current_day = 1
    game_in_day = 0

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if game_in_day > 0:
                current_day += 1
                game_in_day = 0
            continue

        if line.lower().startswith("day") and "vs" not in line.lower():
            continue

        m = GAME_LINE_RE.match(line)
        if not m:
            raise ValueError(
                f"Could not parse line: {line!r}\n"
                "Expected something like: '1) TT, NS Vs. SY, CH: 21-13'"
            )

        game_in_day += 1
        d = m.groupdict()
        rows.append(
            {
                "day": current_day,
                "game_in_day": game_in_day,
                "p1": d["p1"].strip(),
                "p2": d["p2"].strip(),
                "p3": d["p3"].strip(),
                "p4": d["p4"].strip(),
                "score1": int(d["score1"]),
                "score2": int(d["score2"]),
            }
        )

    if not rows:
        raise ValueError("No games were found in the text file.")

    df = pd.DataFrame(rows)
    df["game_id"] = np.arange(1, len(df) + 1)
    df["date"] = pd.NaT
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {c.lower(): c for c in df.columns}
    aliases = {
        "p1": ["p1", "pos1", "position1", "corner1", "player1"],
        "p2": ["p2", "pos2", "position2", "corner2", "player2"],
        "p3": ["p3", "pos3", "position3", "corner3", "player3"],
        "p4": ["p4", "pos4", "position4", "corner4", "player4"],
        "score1": ["score1", "team1_score", "t1", "left_score", "a_score"],
        "score2": ["score2", "team2_score", "t2", "right_score", "b_score"],
        "score": ["score", "result"],
        "date": ["date", "game_date", "timestamp"],
        "day": ["day"],
        "game_id": ["game_id", "id"],
    }

    rename_map = {}
    for standard, options in aliases.items():
        for opt in options:
            if opt in lower_cols:
                rename_map[lower_cols[opt]] = standard
                break
    return df.rename(columns=rename_map).copy()


def load_games(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".txt":
        return parse_txt_log(path)

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    required = ["p1", "p2", "p3", "p4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required player columns: " + ", ".join(missing))

    if "score" in df.columns:
        scores = df["score"].astype(str).str.extract(r"(?P<score1>\d+)\s*[-:,/]\s*(?P<score2>\d+)")
        if scores.isna().any().any():
            raise ValueError("Could not parse 'score'. Use a format like 21-13.")
        df["score1"] = scores["score1"].astype(int)
        df["score2"] = scores["score2"].astype(int)
    else:
        if "score1" not in df.columns or "score2" not in df.columns:
            raise ValueError("Provide either 'score' or both 'score1' and 'score2'.")
        df["score1"] = pd.to_numeric(df["score1"], errors="raise").astype(int)
        df["score2"] = pd.to_numeric(df["score2"], errors="raise").astype(int)

    if "day" not in df.columns:
        df["day"] = np.nan
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    if "game_id" not in df.columns:
        df["game_id"] = np.arange(1, len(df) + 1)

    for c in required:
        df[c] = df[c].astype(str).str.strip()
    return df.dropna(subset=["p1", "p2", "p3", "p4", "score1", "score2"]).reset_index(drop=True)

# ============================================================
# GAME HELPERS
# ============================================================

def team1_players(row: pd.Series) -> Tuple[str, str]:
    return row["p1"], row["p2"]


def team2_players(row: pd.Series) -> Tuple[str, str]:
    return row["p3"], row["p4"]


def team1_won(row: pd.Series) -> bool:
    return bool(row["score1"] > row["score2"])


def point_diff(row: pd.Series) -> int:
    return int(row["score1"] - row["score2"])


def overtime(row: pd.Series) -> bool:
    return max(int(row["score1"]), int(row["score2"])) > 21


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 1000.0))

def elo_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 500.0))

def trueskill_expected_win_prob(team1_ratings, team2_ratings, beta: float) -> float:
    """
    Approximate TrueSkill Team 1 win probability using mu and sigma.
    team*_ratings should be lists of trueskill.Rating objects.
    """
    mu_diff = sum(r.mu for r in team1_ratings) - sum(r.mu for r in team2_ratings)
    variance = (
        sum(r.sigma ** 2 for r in team1_ratings)
        + sum(r.sigma ** 2 for r in team2_ratings)
        + (len(team1_ratings) + len(team2_ratings)) * beta ** 2
    )
    return float(norm.cdf(mu_diff / sqrt(variance)))

# ============================================================
# ELO AND OPTIONAL TRUESKILL
# ============================================================

def margin_multiplier(score1: int, score2: int) -> float:
    mov = abs(score1 - score2)
    return 1.0 + (mov / 21.0)


def run_elo(df: pd.DataFrame, k_factor: float = K_FACTOR, default_elo: float = DEFAULT_ELO):
    ratings: Dict[str, float] = defaultdict(lambda: default_elo)
    history_rows = []

    for _, row in df.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)
        r1 = float(np.mean([ratings[p] for p in t1]))
        r2 = float(np.mean([ratings[p] for p in t2]))
        exp1 = elo_expected_score(r1, r2)
        actual1 = 1.0 if team1_won(row) else 0.0
        mov_mult = margin_multiplier(int(row["score1"]), int(row["score2"]))
        delta = k_factor * mov_mult * (actual1 - exp1)

        for p in t1:
            ratings[p] += delta / 2.0
        for p in t2:
            ratings[p] -= delta / 2.0

        history_rows.append(
            {
                "game_id": row["game_id"],
                "day": row.get("day", np.nan),
                "date": row.get("date", pd.NaT),
                "p1": row["p1"], "p2": row["p2"], "p3": row["p3"], "p4": row["p4"],
                "score1": row["score1"], "score2": row["score2"],
                "team1_pre_elo": r1,
                "team2_pre_elo": r2,
                "team1_expected_win_prob": exp1,
                "team1_actual_win": actual1,
                "team_delta": delta,
                "point_diff": point_diff(row),
            }
        )

    history = pd.DataFrame(history_rows)
    elo_df = pd.DataFrame([{ "player": p, "elo": r } for p, r in ratings.items()]).sort_values("elo", ascending=False).reset_index(drop=True)
    return history, elo_df


def run_trueskill(df: pd.DataFrame):
    if not TRUESKILL_AVAILABLE:
        return None

    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings: Dict[str, object] = defaultdict(lambda: env.create_rating())

    for _, row in df.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)
        team1 = [ratings[p] for p in t1]
        team2 = [ratings[p] for p in t2]

        if team1_won(row):
            new_t1, new_t2 = env.rate(
                                            [[team1[0], team1[1]], [team2[0], team2[1]]],
                                            ranks=[0, 1]
                                        )
        else:
            new_t1, new_t2 = env.rate([[team1[0], team1[1]], [team2[0], team2[1]]], ranks=[1, 0])

        for p, r in zip(t1, new_t1):
            ratings[p] = r
        for p, r in zip(t2, new_t2):
            ratings[p] = r

    return pd.DataFrame(
        [{
            "player": p,
            "trueskill_mu": r.mu,
            "trueskill_sigma": r.sigma,
            "trueskill_conservative": r.mu - 3 * r.sigma,
        } for p, r in ratings.items()]
    ).sort_values(["trueskill_conservative", "trueskill_mu"], ascending=False).reset_index(drop=True)

# ============================================================
# SUMMARIES
# ============================================================

def build_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    players = pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K"))
    rows = []

    for player in players:
        mask = (df["p1"] == player) | (df["p2"] == player) | (df["p3"] == player) | (df["p4"] == player)
        games = df[mask]
        if games.empty:
            continue

        wins = losses = pf = pa = diff_sum = ot_games = 0
        pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        pos_wins = {1: 0, 2: 0, 3: 0, 4: 0}

        for _, row in games.iterrows():
            pos = [k for k, v in {1: row["p1"], 2: row["p2"], 3: row["p3"], 4: row["p4"]}.items() if v == player][0]
            pos_counts[pos] += 1

            if player in (row["p1"], row["p2"]):
                p_for, p_against = row["score1"], row["score2"]
                win = row["score1"] > row["score2"]
            else:
                p_for, p_against = row["score2"], row["score1"]
                win = row["score2"] > row["score1"]

            pf += int(p_for)
            pa += int(p_against)
            diff_sum += int(p_for - p_against)
            ot_games += int(overtime(row))
            if win:
                wins += 1
                pos_wins[pos] += 1
            else:
                losses += 1

        out = {
            "player": player,
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "win_pct": wins / len(games),
            "points_for": pf,
            "points_against": pa,
            "point_diff": diff_sum,
            "avg_point_diff": diff_sum / len(games),
            "overtime_games": ot_games,
        }
        for pos in range(1, 5):
            out[f"pos{pos}_games"] = pos_counts[pos]
            out[f"pos{pos}_wins"] = pos_wins[pos]
            out[f"pos{pos}_win_pct"] = pos_wins[pos] / pos_counts[pos] if pos_counts[pos] else np.nan
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["wins", "point_diff"], ascending=False).reset_index(drop=True)
    return out_df


def build_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"games": 0, "wins": 0, "point_diff": 0})

    for _, row in df.iterrows():
        t1 = pair_key(row["p1"], row["p2"])
        t2 = pair_key(row["p3"], row["p4"])
        diff = int(row["score1"] - row["score2"])

        stats[t1]["games"] += 1
        stats[t1]["wins"] += int(row["score1"] > row["score2"])
        stats[t1]["point_diff"] += diff

        stats[t2]["games"] += 1
        stats[t2]["wins"] += int(row["score2"] > row["score1"])
        stats[t2]["point_diff"] -= diff

    rows = []
    for (a, b), s in stats.items():
        rows.append(
            {
                "player_a": a,
                "player_b": b,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["games"] - s["wins"],
                "win_pct": s["wins"] / s["games"],
                "point_diff": s["point_diff"],
                "avg_point_diff": s["point_diff"] / s["games"],
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["win_pct", "point_diff", "games"], ascending=[False, False, False]).reset_index(drop=True)
    return out_df


def build_side_summary(df: pd.DataFrame) -> pd.DataFrame:
    games = len(df)
    rows = []
    for side_name, win_col, loss_col in [("1,2 side", "score1", "score2"), ("3,4 side", "score2", "score1")]:
        wins = int((df[win_col] > df[loss_col]).sum())
        diff = int((df[win_col] - df[loss_col]).sum())
        rows.append({
            "side": side_name,
            "games": games,
            "wins": wins,
            "losses": games - wins,
            "win_pct": wins / games if games else np.nan,
            "point_diff": diff,
            "avg_point_diff": diff / games if games else np.nan,
        })
    return pd.DataFrame(rows)


def build_overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    players = pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K"))
    rows = []

    for player in players:
        mask = (df["p1"] == player) | (df["p2"] == player) | (df["p3"] == player) | (df["p4"] == player)
        games = df[mask].copy()
        if games.empty:
            continue

        wins = 0
        point_diff_sum = 0
        for _, row in games.iterrows():
            if player in (row["p1"], row["p2"]):
                win = row["score1"] > row["score2"]
                diff = int(row["score1"] - row["score2"])
            else:
                win = row["score2"] > row["score1"]
                diff = int(row["score2"] - row["score1"])
            wins += int(win)
            point_diff_sum += diff

        rows.append(
            {
                "player": player,
                "games": len(games),
                "wins": wins,
                "losses": len(games) - wins,
                "win_pct": wins / len(games),
                "point_diff": point_diff_sum,
                "avg_point_diff": point_diff_sum / len(games),
            }
        )

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["win_pct", "point_diff", "games"], ascending=[False, False, False]).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_corner_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corner in [1, 2, 3, 4]:
        corner_col = f"p{corner}"
        for player in pd.unique(df[corner_col].values):
            games = df[df[corner_col] == player]
            if games.empty:
                continue

            if corner in (1, 2):
                wins = int((games["score1"] > games["score2"]).sum())
                point_diff_sum = int((games["score1"] - games["score2"]).sum())
            else:
                wins = int((games["score2"] > games["score1"]).sum())
                point_diff_sum = int((games["score2"] - games["score1"]).sum())

            rows.append(
                {
                    "corner": corner,
                    "player": player,
                    "games": len(games),
                    "wins": wins,
                    "losses": len(games) - wins,
                    "win_pct": wins / len(games),
                    "point_diff": point_diff_sum,
                    "avg_point_diff": point_diff_sum / len(games),
                }
            )

    corner_df = pd.DataFrame(rows)
    if not corner_df.empty:
        corner_df = corner_df.sort_values(["corner", "win_pct", "point_diff", "games"], ascending=[True, False, False, False]).reset_index(drop=True)
        corner_df["corner_rank"] = corner_df.groupby("corner").cumcount() + 1
    return corner_df


def build_teammate_matrix(df: pd.DataFrame, metric: str = "win_pct") -> pd.DataFrame:
    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    matrix = pd.DataFrame(index=players, columns=players, dtype=float)
    for p in players:
        matrix.loc[p, p] = np.nan

    pair_stats = defaultdict(lambda: {"games": 0, "wins": 0, "point_diff": 0})
    for _, row in df.iterrows():
        teams = [
            ((row["p1"], row["p2"]), row["score1"], row["score2"]),
            ((row["p3"], row["p4"]), row["score2"], row["score1"]),
        ]
        for (a, b), team_score, opp_score in teams:
            key = tuple(sorted((a, b)))
            pair_stats[key]["games"] += 1
            pair_stats[key]["wins"] += int(team_score > opp_score)
            pair_stats[key]["point_diff"] += int(team_score - opp_score)

    for (a, b), stats in pair_stats.items():
        if metric == "games":
            value = stats["games"]
        elif metric == "wins":
            value = stats["wins"]
        elif metric == "point_diff":
            value = stats["point_diff"]
        else:
            value = stats["wins"] / stats["games"] if stats["games"] else np.nan
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    return matrix


def build_teammate_winpct_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return build_teammate_matrix(df, metric="win_pct")

def build_player_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """Track each player's Elo after every game."""
    ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    rows = []

    for _, row in df.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)

        r1 = float(np.mean([ratings[p] for p in t1]))
        r2 = float(np.mean([ratings[p] for p in t2]))
        exp1 = elo_expected_score(r1, r2)
        actual1 = 1.0 if team1_won(row) else 0.0

        try:
            mov_mult = margin_multiplier(int(row["score1"]), int(row["score2"]))
        except NameError:
            mov_mult = 1.0

        delta = K_FACTOR * mov_mult * (actual1 - exp1)

        for p in t1:
            ratings[p] += delta / 2.0
        for p in t2:
            ratings[p] -= delta / 2.0

        for p in [row["p1"], row["p2"], row["p3"], row["p4"]]:
            rows.append(
                {
                    "game_id": row["game_id"],
                    "day": row.get("day", np.nan),
                    "date": row.get("date", pd.NaT),
                    "player": p,
                    "elo_after_game": ratings[p],
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["player", "game_id"]).reset_index(drop=True)
    return out

def build_matchup_heatmap(pred_df: pd.DataFrame, players: List[str]) -> pd.DataFrame:
    """
    Build a 56x56 heatmap matrix:
    rows = ordered Team 1 pairs
    cols = ordered Team 2 pairs
    cells = predicted Team 1 win probability
    """
    labels = [f"{a} + {b}" for a, b in permutations(players, 2)]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for _, row in pred_df.iterrows():
        matrix.loc[row["team1"], row["team2"]] = row["pred_team1_win_prob"]

    return matrix


def build_corner_heatmap_of_heatmaps(elo_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    Outer grid: corner 1 & 2 combinations, ordered:
        FZ, SY, CH, NS, TT, AC, TD, RS
    Inner grids: corner 3 & 4 combinations from the remaining 6 players,
    using only the upper triangle.

    Cell color: average inner heatmap probability.
    """
    if elo_df.empty:
        return None

    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]
    ratings = elo_df.set_index("player")["elo"].to_dict()

    missing = [p for p in player_order if p not in ratings]
    if missing:
        raise ValueError(f"Missing Elo ratings for: {missing}")

    n = len(player_order)
    fig, axes = plt.subplots(n, n, figsize=(24, 24), constrained_layout=False)

    fig.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.94,
    bottom=0.06,
    wspace=0.04,
    hspace=0.18      # was 0.02
)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0.50, vmax=0.680)

    for i, p1 in enumerate(player_order):
        for j, p2 in enumerate(player_order):
            ax = axes[i, j]

            # Outer grid upper triangle only
            if j <= i:
                ax.axis("off")
                continue

            team1_elo = (ratings[p1] + ratings[p2]) / 2.0
            remaining = [p for p in player_order if p not in (p1, p2)]  # 6 players

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []

            # Inner grid upper triangle only
            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    if c <= r:
                        continue
                    team2_elo = (ratings[p3] + ratings[p4]) / 2.0
                    prob = elo_expected_score(team1_elo, team2_elo)
                    inner[r, c] = prob
                    inner_vals.append(prob)

            avg_prob = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_prob)) if not np.isnan(avg_prob) else (1, 1, 1, 1)

            # Outer cell background and border colored by average inner probability
            ax.set_facecolor(cell_color)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.3)
                spine.set_edgecolor(cell_color)

            ax.imshow(
                np.ma.masked_invalid(inner),
                cmap=cmap,
                norm=norm,
                origin="upper",
                interpolation="nearest",
            )

            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(remaining, fontsize=4, rotation=90)
            ax.set_yticklabels(remaining, fontsize=4)

            ax.tick_params(length=0)
            ax.set_title(f"{p1}-{p2}\nAvg: {avg_prob:.1%}", fontsize=7, pad=2)

    # One shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Team 1 win probability", rotation=90)

    fig.suptitle("Corner-Based Heat Map of Heat Maps", fontsize=18, y=0.995)

    for j, player in enumerate(player_order):
        axes[0, j].set_title(
            player,
            fontsize=16,
            fontweight="bold",
            color="navy",
            pad=18,
        )

    for i, player in enumerate(player_order):
        axes[i, 0].set_ylabel(
            player,
            fontsize=16,
            fontweight="bold",
            color="navy",
            rotation=0,
            labelpad=25,
            va="center",
        )

    fig.text(
    0.5,
    0.97,
    "Corner 2 (Team 1)",
    ha="center",
    fontsize=18,
    fontweight="bold",
    )

    fig.text(
        0.03,
        0.5,
        "Corner 1 (Team 1)",
        rotation=90,
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    path = outdir / "corner_heatmap_of_heatmaps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

def build_corner_heatmap_of_heatmaps_plotly(
    elo_df: pd.DataFrame,
    auto_scale: bool = False,
) -> Optional[str]:
    """
    Interactive Plotly version of the corner-based heatmap-of-heatmaps.

    Outer grid:
        8x8 Corner 1/2 player pairs (upper triangle only)

    Inner grids:
        6x6 Corner 3/4 player pairs for the remaining players
        (upper triangle only, shown in the upper-right orientation)

    Each large cell gets a border/background color based on the
    average win probability of its inner heatmap.
    """
    if elo_df.empty:
        return None

    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]
    ratings = elo_df.set_index("player")["elo"].to_dict()

    missing = [p for p in player_order if p not in ratings]
    if missing:
        raise ValueError(f"Missing Elo ratings for: {missing}")

    if auto_scale:
        all_probs = []
        for i, p1 in enumerate(player_order):
            for j, p2 in enumerate(player_order):
                if j <= i:
                    continue
                team1_elo = (ratings[p1] + ratings[p2]) / 2.0
                remaining = [p for p in player_order if p not in (p1, p2)][::-1]
                for r, p3 in enumerate(remaining):
                    for c, p4 in enumerate(remaining):
                        if c <= r:
                            continue
                        team2_elo = (ratings[p3] + ratings[p4]) / 2.0
                        all_probs.append(elo_expected_score(team1_elo, team2_elo))
        if all_probs:
            zmin = float(np.min(all_probs))
            zmax = float(np.max(all_probs))
        else:
            zmin, zmax = 0.0, 1.0
    else:
        zmin, zmax = 0.0, 1.0

    n = len(player_order)
    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=0.025,
        vertical_spacing=0.03,
    )

    colorscale = "Viridis"
    cmap = plt.cm.viridis
    norm = Normalize(vmin=zmin, vmax=zmax)

    def axis_suffix(idx: int) -> str:
        return "" if idx == 1 else str(idx)

    def rgba_css(color_rgba, alpha: float = 0.18) -> str:
        r, g, b, _ = color_rgba
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    for i, p1 in enumerate(player_order, start=1):
        for j, p2 in enumerate(player_order, start=1):
            if j <= i:
                # Hide the lower triangle of the outer 8x8 grid.
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
                continue

            team1_elo = (ratings[p1] + ratings[p2]) / 2.0
            remaining = [p for p in player_order if p not in (p1, p2)][::-1] 

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []

            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    if c <= r:
                        continue  # upper triangle only

                    team2_elo = (ratings[p3] + ratings[p4]) / 2.0
                    prob = elo_expected_score(team1_elo, team2_elo)
                    inner[r, c] = prob
                    inner_vals.append(prob)

            avg_prob = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_prob)) if not np.isnan(avg_prob) else (1, 1, 1, 1)

            cell_idx = (i - 1) * n + j
            ref = axis_suffix(cell_idx)

            fig.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                line=dict(color=rgba_css(cell_color, 0.95), width=4),
                fillcolor=rgba_css(cell_color, 0.60),
                layer="below",
            )

            heatmap_kwargs = dict(
                z=inner,
                x=remaining,
                y=remaining,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                showscale=False,
                hovertemplate=(
                    f"Team 1: {p1} + {p2}<br>"
                    f"Team 2: %{{y}} + %{{x}}<br>"
                    "Win prob: %{z:.1%}<extra></extra>"
                ),
            )
            if i == 1 and j == 2:
                heatmap_kwargs["showscale"] = True
                heatmap_kwargs["colorbar"] = dict(title="Win %")

            fig.add_trace(
                go.Heatmap(**heatmap_kwargs),
                row=i,
                col=j,
            )


            # ---------- Mini heatmap title ----------
            fig.add_annotation(
                text=(
                     f"<b>{p1} + {p2}: {avg_prob:.1%}</b><br>"
                ),
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                x=0.5,
                y=1.13,          # Slightly above the mini heatmap
                showarrow=False,
                align="center",
                font=dict(
                    size=11,
                    color=rgba_css(cell_color, 1.0),
                ),
            )
            

            fig.update_xaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                tickangle=0,
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                row=i,
                col=j,
            )

            fig.update_yaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                autorange="reversed",
                row=i,
                col=j,
            )

    # Outer-grid labels: column headers across the top
    for j, player in enumerate(player_order, start=1):
        axis_name = "xaxis" if j == 1 else f"xaxis{j}"
        dom = fig.layout[axis_name].domain
        xmid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=xmid,
            y=1.04,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            font=dict(size=18, color="navy", family="Arial Black",),
        )

    # Outer-grid labels: row headers down the left
    for i, player in enumerate(player_order, start=1):
        axis_idx = (i - 1) * n + 1
        axis_name = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
        dom = fig.layout[axis_name].domain
        ymid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=-0.015,
            y=ymid,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            textangle=0,
            xanchor="right",
            font=dict(size=18, color="navy", family="Arial Black",),
        )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text="Corner 2 (Team 1)",
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Corner 1 (Team 1)",
        showarrow=False,
        textangle=-90,
        font=dict(size=18, color="black"),
    )

    fig.update_layout(
        title="Corner-Based Heat Map of Heat Maps",
        height=1800,
        width=1800,
        margin=dict(l=140, r=60, t=140, b=70),
        dragmode="zoom",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

from collections import Counter

def build_corner_heatmap_of_heatmaps_plotly_game_counts(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None

    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]
    players_in_data = set(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    missing = [p for p in player_order if p not in players_in_data]
    if missing:
        raise ValueError(f"Missing players in data: {missing}")

    n = len(player_order)
    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=0.025,
        vertical_spacing=0.03,
    )

    def axis_suffix(idx: int) -> str:
        return "" if idx == 1 else str(idx)

    def rgba_css(color_rgba, alpha: float = 0.18) -> str:
        r, g, b, _ = color_rgba
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    # Count each unique matchup once per appearance in the data
    game_counts = Counter(
        canonical_unique_game_key(row)
        for _, row in df.iterrows()
    )

    max_count = max(game_counts.values()) if game_counts else 1
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=max_count)

    for i, p1 in enumerate(player_order, start=1):
        for j, p2 in enumerate(player_order, start=1):
            if j <= i:
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
                continue

            remaining = [p for p in player_order if p not in (p1, p2)][::-1]

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []

            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    if c <= r:
                        continue

                    outer_team = tuple(sorted((p1, p2)))
                    inner_team = tuple(sorted((p3, p4)))
                    game_key = tuple(sorted((outer_team, inner_team)))

                    count = game_counts.get(game_key, 0)
                    inner[r, c] = count
                    inner_vals.append(count)

            avg_count = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_count)) if not np.isnan(avg_count) else (1, 1, 1, 1)

            cell_idx = (i - 1) * n + j
            ref = axis_suffix(cell_idx)

            fig.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                line=dict(color=rgba_css(cell_color, 0.95), width=4),
                fillcolor=rgba_css(cell_color, 0.60),
                layer="below",
            )

            heatmap_kwargs = dict(
                z=inner,
                x=remaining,
                y=remaining,
                zmin=0,
                zmax=max_count,
                colorscale="Blues",
                showscale=False,
                hovertemplate=(
                    f"Team 1: {p1} + {p2}<br>"
                    f"Team 2: %{{y}} + %{{x}}<br>"
                    "Games played: %{z:.0f}<extra></extra>"
                ),
            )

            if i == 1 and j == 2:
                heatmap_kwargs["showscale"] = True
                heatmap_kwargs["colorbar"] = dict(title="Games")

            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=i, col=j)

            fig.add_annotation(
                text=f"<b>{p1} + {p2}: {avg_count:.1f}</b>",
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                x=0.5,
                y=1.13,
                showarrow=False,
                align="center",
                font=dict(size=11, color=rgba_css(cell_color, 1.0)),
            )

            fig.update_xaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                tickangle=0,
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                row=i,
                col=j,
            )

            fig.update_yaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                autorange="reversed",
                row=i,
                col=j,
            )

    for j, player in enumerate(player_order, start=1):
        axis_name = "xaxis" if j == 1 else f"xaxis{j}"
        dom = fig.layout[axis_name].domain
        xmid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=xmid,
            y=1.04,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    for i, player in enumerate(player_order, start=1):
        axis_idx = (i - 1) * n + 1
        axis_name = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
        dom = fig.layout[axis_name].domain
        ymid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=-0.015,
            y=ymid,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            xanchor="right",
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text="Corner 2 (Team 1)",
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Corner 1 (Team 1)",
        showarrow=False,
        textangle=-90,
        font=dict(size=18, color="black"),
    )

    fig.update_layout(
        title="Corner-Based Heat Map of Games Played",
        height=1800,
        width=1800,
        margin=dict(l=140, r=60, t=140, b=70),
        dragmode="zoom",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_data_completion_donuts(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    Plot three donut charts showing progress through the possible game space.
    """

    if df.empty:
        return None

    TOTAL_GAMES = 1680
    TOTAL_UNIQUE_GAMES = 210
    TOTAL_PAIRS = 28

    # -----------------------------
    # Games played
    # -----------------------------
    games_played = len(df)

    ordered_games = set()

    for _, row in df.iterrows():

        ordered_games.add(
            (
                row["p1"],   # Corner 1
                row["p2"],   # Corner 2
                row["p3"],   # Corner 3
                row["p4"],   # Corner 4
            )
        )

    unique_ordered_games = len(ordered_games)
    # -----------------------------
    # Unique games
    # -----------------------------
    unique_games = set()

    for _, row in df.iterrows():

        t1 = tuple(sorted((row["p1"], row["p2"])))
        t2 = tuple(sorted((row["p3"], row["p4"])))

        unique_games.add(tuple(sorted((t1, t2))))

    unique_games_played = len(unique_games)

    # -----------------------------
    # Unique teammate pairs
    # -----------------------------
    pairs = set()

    for _, row in df.iterrows():

        pairs.add(tuple(sorted((row["p1"], row["p2"]))))
        pairs.add(tuple(sorted((row["p3"], row["p4"]))))

    unique_pairs = len(pairs)

    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    metrics = [
        ("Total Possible Games", unique_ordered_games, TOTAL_GAMES),
        ("Total Possible Unique Games", unique_games_played, TOTAL_UNIQUE_GAMES),
        ("Total Possible Pairs", unique_pairs, TOTAL_PAIRS),
    ]

    for ax, (title, value, total) in zip(axes, metrics):

        pct = value / total

        ax.pie(
            [value, total-value],
            colors=["tab:blue","tab:red"],
            startangle=90,
            wedgeprops=dict(
                width=0.38,
                edgecolor="white"
            ),
        )

        ax.text(
            0,
            0,
            f"{pct:.1%}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_title(
            f"{title}\n{value}/{total}",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(
        f"Total Games Played: {games_played}",
        fontsize=20,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0,0,1,0.90])

    path = outdir / "completion_donuts.png"

    fig.savefig(path, dpi=250, bbox_inches="tight")

    plt.close(fig)

    return path


def longest_run(values: List[int], target: int = 1) -> int:
    best = 0
    cur = 0
    for v in values:
        if v == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def build_pair_synergy_summary(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Synergy = actual win rate - expected win rate for each teammate pair.

    Uses the Elo expectation from each game in history, so the synergy is
    based on game-time strength rather than final Elo.
    """
    df = df.reset_index(drop=True)
    history = history.reset_index(drop=True)

    if len(df) != len(history):
        raise ValueError("df and history must have the same number of rows.")

    stats = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "expected_wins": 0.0,
        "point_diff": 0,
    })

    for i in range(len(df)):
        game = df.iloc[i]
        h = history.iloc[i]

        t1_pair = tuple(sorted((game["p1"], game["p2"])))
        t2_pair = tuple(sorted((game["p3"], game["p4"])))

        t1_win = int(h["team1_actual_win"])
        t1_exp = float(h["team1_expected_win_prob"])
        pdiff = int(game["score1"] - game["score2"])

        # Team 1 pair
        stats[t1_pair]["games"] += 1
        stats[t1_pair]["wins"] += t1_win
        stats[t1_pair]["expected_wins"] += t1_exp
        stats[t1_pair]["point_diff"] += pdiff

        # Team 2 pair
        stats[t2_pair]["games"] += 1
        stats[t2_pair]["wins"] += (1 - t1_win)
        stats[t2_pair]["expected_wins"] += (1.0 - t1_exp)
        stats[t2_pair]["point_diff"] += -pdiff

    rows = []
    for (a, b), s in stats.items():
        actual_win_pct = s["wins"] / s["games"] if s["games"] else np.nan
        expected_win_pct = s["expected_wins"] / s["games"] if s["games"] else np.nan
        rows.append(
            {
                "player_a": a,
                "player_b": b,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["games"] - s["wins"],
                "actual_win_pct": actual_win_pct,
                "expected_win_pct": expected_win_pct,
                "synergy": actual_win_pct - expected_win_pct,
                "point_diff": s["point_diff"],
                "avg_point_diff": s["point_diff"] / s["games"] if s["games"] else np.nan,
            }
        )

    synergy_df = pd.DataFrame(rows)
    if not synergy_df.empty:
        synergy_df = synergy_df.sort_values(
            ["synergy", "actual_win_pct", "games"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return synergy_df



def plot_pair_synergy(synergy_df: pd.DataFrame, outdir: Path, top_n: int = 28) -> Optional[Path]:
    if synergy_df.empty:
        return None

    top = synergy_df.head(top_n).copy()
    top["pair"] = top["player_a"] + " + " + top["player_b"]
    colors = ["tab:green" if x >= 0 else "tab:red" for x in top["synergy"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top["pair"], top["synergy"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Pair Synergy (Actual Win % - Expected Win %)")
    ax.set_xlabel("Synergy")
    ax.set_ylabel("Pair")
    ax.invert_yaxis()
    fig.tight_layout()

    path = outdir / "pair_synergy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path




def build_pair_synergy_from_predictions(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    prob_col: str,
) -> pd.DataFrame:
    """
    Synergy = actual win rate - expected win rate for each teammate pair,
    using predicted probabilities from any model.

    df must contain p1,p2,p3,p4,score1,score2,game_id.
    pred_df must contain game_id and prob_col.
    """
    if df.empty or pred_df.empty:
        return pd.DataFrame()

    base = df[["game_id", "p1", "p2", "p3", "p4", "score1", "score2"]].copy()
    preds = pred_df.copy()

    if "game_id" in preds.columns:
        preds = preds[["game_id", prob_col]].copy()
        merged = base.merge(preds, on="game_id", how="inner").sort_values("game_id").reset_index(drop=True)
    else:
        if len(base) != len(preds):
            raise ValueError("df and pred_df must have the same number of rows if pred_df has no game_id column.")
        merged = base.reset_index(drop=True).copy()
        merged[prob_col] = preds[prob_col].values

    merged["team1_actual_win"] = (merged["score1"] > merged["score2"]).astype(int)

    stats = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "expected_wins": 0.0,
        "point_diff": 0,
    })

    for _, row in merged.iterrows():
        t1_pair = tuple(sorted((row["p1"], row["p2"])))
        t2_pair = tuple(sorted((row["p3"], row["p4"])))

        t1_win = int(row["team1_actual_win"])
        t1_exp = float(row[prob_col])
        pdiff = int(row["score1"] - row["score2"])

        stats[t1_pair]["games"] += 1
        stats[t1_pair]["wins"] += t1_win
        stats[t1_pair]["expected_wins"] += t1_exp
        stats[t1_pair]["point_diff"] += pdiff

        stats[t2_pair]["games"] += 1
        stats[t2_pair]["wins"] += (1 - t1_win)
        stats[t2_pair]["expected_wins"] += (1.0 - t1_exp)
        stats[t2_pair]["point_diff"] += -pdiff

    rows = []
    for (a, b), s in stats.items():
        actual_win_pct = s["wins"] / s["games"] if s["games"] else np.nan
        expected_win_pct = s["expected_wins"] / s["games"] if s["games"] else np.nan
        rows.append(
            {
                "player_a": a,
                "player_b": b,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["games"] - s["wins"],
                "actual_win_pct": actual_win_pct,
                "expected_win_pct": expected_win_pct,
                "synergy": actual_win_pct - expected_win_pct,
                "point_diff": s["point_diff"],
                "avg_point_diff": s["point_diff"] / s["games"] if s["games"] else np.nan,
            }
        )

    synergy_df = pd.DataFrame(rows)
    if not synergy_df.empty:
        synergy_df = synergy_df.sort_values(
            ["synergy", "actual_win_pct", "games"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return synergy_df


def plot_pair_synergy_generic(
    synergy_df: pd.DataFrame,
    outdir: Path,
    filename: str,
    title: str,
    top_n: int = 28,
) -> Optional[Path]:
    if synergy_df.empty:
        return None

    top = synergy_df.head(top_n).copy()
    top["pair"] = top["player_a"] + " + " + top["player_b"]
    colors = ["tab:green" if x >= 0 else "tab:red" for x in top["synergy"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top["pair"], top["synergy"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Synergy")
    ax.set_ylabel("Pair")
    ax.invert_yaxis()
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_probability_histogram(
    pred_df: pd.DataFrame,
    prob_col: str,
    outdir: Path,
    filename: str,
    title: str,
) -> Optional[Path]:
    if pred_df.empty or prob_col not in pred_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_df[prob_col].dropna(), bins=20)
    ax.set_title(title)
    ax.set_xlabel("Predicted Team 1 win probability")
    ax.set_ylabel("Count")
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def build_elo_probability_space_df(elo_df: pd.DataFrame) -> pd.DataFrame:
    players = sorted(elo_df["player"].astype(str).tolist())
    ratings = elo_df.set_index("player")["elo"].to_dict()

    rows = []
    seen = set()

    for quad in combinations(players, 4):
        a, b, c, d = quad
        pairings = [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]

        for team1, team2 in pairings:
            key = tuple(sorted((tuple(sorted(team1)), tuple(sorted(team2)))))
            if key in seen:
                continue
            seen.add(key)

            team1_elo = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in team1]))
            team2_elo = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in team2]))
            prob = elo_expected_score(team1_elo, team2_elo)

            rows.append(
                {
                    "config_label": f"{team1[0]} + {team1[1]} vs {team2[0]} + {team2[1]}",
                    "team1_elo": team1_elo,
                    "team2_elo": team2_elo,
                    "pred_team1_win_prob": prob,
                }
            )

    return pd.DataFrame(rows)


def build_trueskill_probability_space_df(trueskill_df: pd.DataFrame) -> pd.DataFrame:
    if trueskill_df is None or trueskill_df.empty:
        return pd.DataFrame()

    players = sorted(trueskill_df["player"].astype(str).tolist())
    skill = trueskill_df.set_index("player")[["trueskill_mu", "trueskill_sigma"]].to_dict("index")
    env = trueskill.TrueSkill(draw_probability=0.0)

    rows = []
    seen = set()

    for quad in combinations(players, 4):
        a, b, c, d = quad
        pairings = [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]

        for team1, team2 in pairings:
            key = tuple(sorted((tuple(sorted(team1)), tuple(sorted(team2)))))
            if key in seen:
                continue
            seen.add(key)

            t1 = [
                trueskill.Rating(mu=skill[p]["trueskill_mu"], sigma=skill[p]["trueskill_sigma"])
                for p in team1
            ]
            t2 = [
                trueskill.Rating(mu=skill[p]["trueskill_mu"], sigma=skill[p]["trueskill_sigma"])
                for p in team2
            ]
            prob = trueskill_expected_win_prob(t1, t2, env.beta)

            rows.append(
                {
                    "config_label": f"{team1[0]} + {team1[1]} vs {team2[0]} + {team2[1]}",
                    "pred_team1_win_prob": prob,
                }
            )

    return pd.DataFrame(rows)


def build_tree_probability_space_df(
    model,
    feature_cols: list[str],
    state: dict,
) -> pd.DataFrame:
    players = sorted(state["ratings"].keys())
    rows = []

    for p1, p2, p3, p4 in permutations(players, 4):
        feature_row = build_tree_matchup_feature_row(p1, p2, p3, p4, state)
        prob = tree_predict_prob(model, feature_row, feature_cols)
        rows.append(
            {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "config_label": f"{p1}, {p2} vs {p3}, {p4}",
                "pred_team1_win_prob": prob,
            }
        )

    return pd.DataFrame(rows)


def build_tree_configuration_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    counts = defaultdict(int)

    for _, row in df.iterrows():
        key = (row["p1"], row["p2"], row["p3"], row["p4"])
        counts[key] += 1

    rows = []
    for p1, p2, p3, p4 in permutations(players, 4):
        rows.append(
            {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "config_label": f"{p1}, {p2} vs {p3}, {p4}",
                "games_played": int(counts.get((p1, p2, p3, p4), 0)),
            }
        )
    return pd.DataFrame(rows)


def build_probability_strip_heatmap_html(
    space_df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_title: str,
    colorscale: str = "Viridis",
) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        return None
    if space_df.empty or value_col not in space_df.columns:
        return None

    df = space_df.reset_index(drop=True).copy()
    values = df[value_col].astype(float).to_numpy()[None, :]
    labels = df["config_label"].astype(str).tolist()

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=[""],
            zmin=float(np.nanmin(df[value_col])) if np.isfinite(df[value_col]).any() else None,
            zmax=float(np.nanmax(df[value_col])) if np.isfinite(df[value_col]).any() else None,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate="%{x}<br>" + f"{colorbar_title}: %{{z:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_game_highlights(df: pd.DataFrame, history: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    One figure with five panels:
      1) top 3 biggest upsets
      2) top 3 most dominant expected victories
      3) top 3 longest win streaks
      4) top 3 longest loss streaks
      5) top 3 largest absolute Elo jumps
    """
    df = df.reset_index(drop=True)
    history = history.reset_index(drop=True)

    if len(df) != len(history) or df.empty:
        return None

    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    outcomes = {p: [] for p in players}

    upset_rows = []
    dominant_rows = []
    jump_rows = []

    for i in range(len(df)):
        g = df.iloc[i]
        h = history.iloc[i]

        team1_win = int(h["team1_actual_win"])
        team1_expected = float(h["team1_expected_win_prob"])

        # Winner expectation: probability assigned to the actual winner.
        winner_expected = team1_expected if team1_win == 1 else (1.0 - team1_expected)

        label = f'G{int(g["game_id"])}: {g["p1"]}/{g["p2"]} vs {g["p3"]}/{g["p4"]} ({g["score1"]}-{g["score2"]})'

        margin = abs(int(g["score1"]) - int(g["score2"]))
        margin_factor1 = (1.0 + (margin / 21.0)*0.10)
        margin_factor2 = (1.0 + (margin / 21.0))

        upset_rows.append(
            {
                "label": label,
                "upset_score": (1.0 - winner_expected) * margin_factor1,
                "winner_expected": winner_expected,
                "margin": margin,
            }
        )
        dominant_rows.append(
            {
                "label": label,
                "dominance_score": winner_expected * margin_factor2,
                "winner_expected": winner_expected,
                "margin": margin,
            }
        )

        team_delta = float(h["team_delta"])

        # Positive Elo gain for each player on the winning team
        elo_gain_each = abs(team_delta) / 2.0

        match_label = (
            f'G{int(g["game_id"])}: '
            f'{g["p1"]}/{g["p2"]} vs '
            f'{g["p3"]}/{g["p4"]} '
            f'({g["score1"]}-{g["score2"]})'
        )

        # Only keep ONE record per game (winning team)
        if team1_win == 1:
            winning_team = f'{g["p1"]}/{g["p2"]}'
        else:
            winning_team = f'{g["p3"]}/{g["p4"]}'

        jump_rows.append(
            {
                "player": winning_team,
                "match": match_label,
                "elo_gain_each": elo_gain_each,
            }
        )

        # Track player outcomes for streaks.
        if team1_win == 1:
            t1_outcome = 1
            t2_outcome = 0
        else:
            t1_outcome = 0
            t2_outcome = 1

        outcomes[g["p1"]].append(t1_outcome)
        outcomes[g["p2"]].append(t1_outcome)
        outcomes[g["p3"]].append(t2_outcome)
        outcomes[g["p4"]].append(t2_outcome)

    # Build streak table
    streak_rows = []
    for p, seq in outcomes.items():
        streak_rows.append(
            {
                "player": p,
                "longest_win_streak": longest_run(seq, 1),
                "longest_loss_streak": longest_run(seq, 0),
            }
        )
    streak_df = pd.DataFrame(streak_rows)

    upset_df = pd.DataFrame(upset_rows).sort_values("upset_score", ascending=False).head(3)
    dominant_df = pd.DataFrame(dominant_rows).sort_values("dominance_score", ascending=False).head(3)
    win_streak_df = streak_df.sort_values("longest_win_streak", ascending=False).head(3)
    loss_streak_df = streak_df.sort_values("longest_loss_streak", ascending=False).head(3)
    jump_df = (
            pd.DataFrame(jump_rows)
            .sort_values("elo_gain_each", ascending=False)
            .head(3)
        )


    # Furthest Into Overtime
    # -----------------------------
    overtime_df = (
        df.assign(
            winning_score=df[["score1", "score2"]].max(axis=1),
            total_points=df["score1"] + df["score2"],
            label=df.apply(
                lambda r:
                    f'G{int(r["game_id"])}: '
                    f'{r["p1"]}/{r["p2"]} vs '
                    f'{r["p3"]}/{r["p4"]} '
                    f'({r["score1"]}-{r["score2"]})',
                axis=1,
            ),
        )
        .query("winning_score > 21")
        .sort_values(
            ["winning_score", "total_points"],
            ascending=False,
        )
        .head(3)
    )

    
    fig, axes = plt.subplots(6, 1, figsize=(12, 24))
    panels = [
        (upset_df, "Top 3 Biggest Upsets", "upset_score", "tab:red"),
        (dominant_df, "Top 3 Most Dominant Expected Victories", "dominance_score", "tab:blue"),
        (win_streak_df, "Top 3 Longest Win Streaks", "longest_win_streak", "tab:green"),
        (loss_streak_df, "Top 3 Longest Loss Streaks", "longest_loss_streak", "tab:orange"),
        (jump_df, "Top 3 Largest Elo Jumps", "elo_gain_each", "tab:purple"),
        (overtime_df, "Top 3 Furthest Into Overtime","winning_score","tab:cyan"),
    ]

    for ax, (plot_df, title, value_col, color) in zip(axes, panels):
        if plot_df.empty:
            ax.set_axis_off()
            continue

        if "player" in plot_df.columns:
            y = plot_df["player"]
        else:
            y = plot_df["label"]

        ax.barh(y, plot_df[value_col], color=color)
        if title == "Top 3 Largest Elo Jumps":

            for y_pos, (_, row) in enumerate(plot_df.iterrows()):

                ax.text(
                    row["elo_gain_each"] + 0.15,
                    y_pos,
                    f'{row["match"]}\n(+{row["elo_gain_each"]:.2f} Elo each)',
                    fontsize=8,
                    va="center",
                )
        ax.set_title(title)
        ax.invert_yaxis()

    fig.tight_layout()
    path = outdir / "game_highlights.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path



def canonical_unique_game_key(row: pd.Series) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Canonical representation of a unique game, ignoring team order."""
    t1 = tuple(sorted((row["p1"], row["p2"])))
    t2 = tuple(sorted((row["p3"], row["p4"])))
    return tuple(sorted((t1, t2)))


def build_unplayed_unique_games(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
    trueskill_df: pd.DataFrame | None = None,
    model=None,
    feature_cols: list[str] | None = None,
    ml_state: dict | None = None,
) -> pd.DataFrame:
    """
    Build all legal ordered corner-dependent games that have NOT yet appeared in the data.

    An unplayed game is defined as the exact ordered 4-corner state:
      (p1, p2, p3, p4)

    This keeps corner order distinct, so the same four players in a different
    corner arrangement are treated as separate game states.
    """
    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()}

    played = {
        (str(row["p1"]), str(row["p2"]), str(row["p3"]), str(row["p4"]))
        for _, row in df.iterrows()
    }

    ts_lookup = {}
    ts_env = None
    if trueskill_df is not None and not trueskill_df.empty and TRUESKILL_AVAILABLE:
        ts_lookup = trueskill_df.set_index("player")[["trueskill_mu", "trueskill_sigma"]].to_dict("index")
        ts_env = trueskill.TrueSkill(draw_probability=0.0)

    rows = []

    for p1, p2, p3, p4 in permutations(players, 4):
        if (p1, p2, p3, p4) in played:
            continue

        team1_elo = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in (p1, p2)]))
        team2_elo = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in (p3, p4)]))
        elo_prob = float(elo_expected_score(team1_elo, team2_elo))

        ts_prob = np.nan
        if ts_env is not None and ts_lookup:
            team1_ratings = [
                trueskill.Rating(mu=ts_lookup[p1]["trueskill_mu"], sigma=ts_lookup[p1]["trueskill_sigma"]),
                trueskill.Rating(mu=ts_lookup[p2]["trueskill_mu"], sigma=ts_lookup[p2]["trueskill_sigma"]),
            ]
            team2_ratings = [
                trueskill.Rating(mu=ts_lookup[p3]["trueskill_mu"], sigma=ts_lookup[p3]["trueskill_sigma"]),
                trueskill.Rating(mu=ts_lookup[p4]["trueskill_mu"], sigma=ts_lookup[p4]["trueskill_sigma"]),
            ]
            ts_prob = float(trueskill_expected_win_prob(team1_ratings, team2_ratings, ts_env.beta))

        tree_prob = np.nan
        if model is not None and feature_cols is not None and ml_state is not None:
            feature_row = build_tree_matchup_feature_row(p1, p2, p3, p4, ml_state)
            tree_prob = float(tree_predict_prob(model, feature_row, feature_cols))

        probs = [p for p in [elo_prob, ts_prob, tree_prob] if pd.notna(p)]
        combined_prob = float(np.mean(probs)) if probs else np.nan

        rows.append(
            {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "players": [p1, p2, p3, p4],
                "team1": f"{p1} + {p2}",
                "team2": f"{p3} + {p4}",
                "elo_pred_team1_win_prob": elo_prob,
                "trueskill_pred_team1_win_prob": ts_prob,
                "tree_pred_team1_win_prob": tree_prob,
                "pred_team1_win_prob": combined_prob,
                "closeness": abs(combined_prob - 0.5) if pd.notna(combined_prob) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["closeness", "pred_team1_win_prob"],
            ascending=[True, True],
        ).reset_index(drop=True)
    return out

def build_prediction_accuracy(history_df: pd.DataFrame) -> pd.DataFrame:

    """Compute Brier score and rolling averages."""

    out = history_df.copy()

    out["brier"] = (
        out["team1_expected_win_prob"]
        - out["team1_actual_win"]
    ) ** 2

    out["rolling_brier"] = (
        out["brier"]
        .rolling(window=80, min_periods=1)
        .mean()
    )

    out["correct_prediction"] = (
        (
            (out["team1_expected_win_prob"] >= 0.5)
            ==
            (out["team1_actual_win"] == 1)
        )
    ).astype(int)

    out["rolling_accuracy"] = (
        out["correct_prediction"]
        .rolling(window=80, min_periods=1)
        .mean()
    )

    return out

def build_elo_leadership_summary(history_df: pd.DataFrame, default_elo: float = DEFAULT_ELO) -> pd.DataFrame:
    """
    Replays the Elo updates and counts:
      - games led by each player
      - longest consecutive leading streak
      - final Elo at the end of the run

    A player is considered "leading" on a game if they have the highest Elo
    after that game's update. If there is a tie, all tied players are counted
    as leaders for that game.
    """
    if history_df.empty:
        return pd.DataFrame()

    ratings: Dict[str, float] = defaultdict(lambda: default_elo)
    lead_counts = defaultdict(int)
    current_streak = defaultdict(int)
    best_streak = defaultdict(int)
    current_leaders: set[str] = set()

    # Track each player's Elo over time so we can compute variation and peak.
    elo_history = defaultdict(list)

    for _, row in history_df.iterrows():
        t1 = (row["p1"], row["p2"])
        t2 = (row["p3"], row["p4"])
        delta = float(row["team_delta"])

        # Replay the same Elo update used in run_elo()
        for p in t1:
            ratings[p] += delta / 2.0
        for p in t2:
            ratings[p] -= delta / 2.0

        top_elo = max(ratings.values())
        leaders = {p for p, r in ratings.items() if abs(r - top_elo) < 1e-9}

        # Count this game for every current leader
        for p in leaders:
            lead_counts[p] += 1
            current_streak[p] = current_streak[p] + 1 if p in current_leaders else 1
            best_streak[p] = max(best_streak[p], current_streak[p])

        # Reset streaks for non-leaders
        for p in ratings:
            if p not in leaders:
                current_streak[p] = 0

        current_leaders = leaders

        for p, r in ratings.items():
            elo_history[p].append(r)

    rows = []
    total_games = len(history_df)

    for p, r in ratings.items():
        rows.append(
            {
                "player": p,
                "games_led": lead_counts[p],
                "games_led_pct": lead_counts[p] / total_games if total_games else np.nan,
                "longest_leading_streak": best_streak[p],
                "final_elo": r,
                "elo_std": float(np.std(elo_history[p], ddof=0)) if elo_history[p] else np.nan,
                "max_elo": float(np.max(elo_history[p])) if elo_history[p] else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["games_led", "longest_leading_streak", "final_elo"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def plot_elo_leadership(leadership_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if leadership_df.empty:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    top_games = leadership_df.sort_values("games_led", ascending=False).copy()
    top_streak = leadership_df.sort_values("longest_leading_streak", ascending=False).copy()

    axes[0].barh(top_games["player"], top_games["games_led"])
    axes[0].set_title("Games Spent Leading Elo")
    axes[0].set_xlabel("Games as #1 Elo")
    axes[0].invert_yaxis()

    axes[1].barh(top_streak["player"], top_streak["longest_leading_streak"])
    axes[1].set_title("Longest Elo-Leading Streak")
    axes[1].set_xlabel("Consecutive games as #1 Elo")
    axes[1].invert_yaxis()

    fig.tight_layout()
    path = outdir / "elo_leadership.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_elo_leadership_timeline(history_df: pd.DataFrame, default_elo: float = DEFAULT_ELO) -> pd.DataFrame:
    """
    Replay Elo after each game and record the player(s) who are leading at that moment.

    Returns one row per leader per game, so ties will produce multiple rows.
    """
    if history_df.empty:
        return pd.DataFrame()

    ratings: Dict[str, float] = defaultdict(lambda: default_elo)
    rows = []

    for _, row in history_df.iterrows():
        t1 = (row["p1"], row["p2"])
        t2 = (row["p3"], row["p4"])
        delta = float(row["team_delta"])

        for p in t1:
            ratings[p] += delta / 2.0
        for p in t2:
            ratings[p] -= delta / 2.0

        top_elo = max(ratings.values())
        leaders = [p for p, r in ratings.items() if abs(r - top_elo) < 1e-9]

        for leader in leaders:
            rows.append(
                {
                    "game_id": row["game_id"],
                    "leader_player": leader,
                    "leader_elo": ratings[leader],
                }
            )

    return pd.DataFrame(rows)

def build_trueskill_player_history(df: pd.DataFrame) -> pd.DataFrame:
    if not TRUESKILL_AVAILABLE:
        return pd.DataFrame()

    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings: Dict[str, object] = defaultdict(lambda: env.create_rating())
    rows = []

    for _, row in df.sort_values("game_id").iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)

        team1 = [ratings[p] for p in t1]
        team2 = [ratings[p] for p in t2]

        ranks = [0, 1] if team1_won(row) else [1, 0]
        new_t1, new_t2 = env.rate([team1, team2], ranks=ranks)

        for p, r in zip(t1, new_t1):
            ratings[p] = r
        for p, r in zip(t2, new_t2):
            ratings[p] = r

        for p in sorted(set(t1 + t2)):
            r = ratings[p]
            rows.append(
                {
                    "game_id": row["game_id"],
                    "player": p,
                    "trueskill_mu_after": r.mu,
                    "trueskill_sigma_after": r.sigma,
                    "trueskill_conservative_after": r.mu - 3 * r.sigma,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["player", "game_id"]).reset_index(drop=True)
    return out


def build_trueskill_leadership_timeline(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replays TrueSkill after each game and records the overall leader after that game.
    This mirrors the Elo leadership timeline logic.
    """
    if history_df.empty or not TRUESKILL_AVAILABLE:
        return pd.DataFrame()

    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings: Dict[str, object] = defaultdict(lambda: env.create_rating())
    rows = []

    for _, row in history_df.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)
        team1 = [ratings[p] for p in t1]
        team2 = [ratings[p] for p in t2]

        ranks = [0, 1] if team1_won(row) else [1, 0]
        new_t1, new_t2 = env.rate([team1, team2], ranks=ranks)

        for p, r in zip(t1, new_t1):
            ratings[p] = r
        for p, r in zip(t2, new_t2):
            ratings[p] = r

        top_score = max(r.mu - 3 * r.sigma for r in ratings.values())
        leaders = [p for p, r in ratings.items() if abs((r.mu - 3 * r.sigma) - top_score) < 1e-9]

        for leader in leaders:
            leader_rating = ratings[leader]
            rows.append(
                {
                    "game_id": row["game_id"],
                    "leader_player": leader,
                    "leader_ts_mu": leader_rating.mu,
                    "leader_ts_sigma": leader_rating.sigma,
                    "leader_ts_conservative": leader_rating.mu - 3 * leader_rating.sigma,
                }
            )

    return pd.DataFrame(rows)


def plot_trueskill_over_time(
    ts_player_history: pd.DataFrame,
    history_df: pd.DataFrame,
    outdir: Path,
    max_players: int = 8,
) -> Optional[Path]:
    if ts_player_history.empty:
        return None

    players = ts_player_history["player"].value_counts().head(max_players).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))

    line_colors: Dict[str, str] = {}
    for player in players:
        g = ts_player_history[ts_player_history["player"] == player]
        (line,) = ax.plot(
            g["game_id"],
            g["trueskill_conservative_after"],
            marker="",
            label=player,
        )
        line_colors[player] = line.get_color()

    leader_timeline = build_trueskill_leadership_timeline(history_df)
    for _, row in leader_timeline.iterrows():
        leader = row["leader_player"]
        ax.scatter(
            row["game_id"],
            row["leader_ts_conservative"],
            marker="*",
            s=160,
            facecolors="white",
            edgecolors=line_colors.get(leader, "black"),
            linewidths=1.6,
            zorder=20,
        )

    ax.set_title("Player TrueSkill Over Time")
    ax.set_xlabel("Game")
    ax.set_ylabel("TrueSkill conservative rating")
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / "trueskill_over_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
def build_trueskill_leadership_summary(
    history_df: pd.DataFrame,
    ts_player_history: pd.DataFrame,
    trueskill_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summary table for TrueSkill leadership and peak strength.
    """
    if history_df.empty or trueskill_df is None or trueskill_df.empty:
        return pd.DataFrame()

    # Count how often each player is the overall leader after each game.
    leader_timeline = build_trueskill_leadership_timeline(history_df)
    lead_counts = leader_timeline["leader_player"].value_counts().to_dict()

    # Longest leading streak.
    current_streak = defaultdict(int)
    best_streak = defaultdict(int)
    current_leaders = set()

    for game_id, grp in leader_timeline.groupby("game_id", sort=True):
        leaders = set(grp["leader_player"].tolist())
        for p in leaders:
            current_streak[p] = current_streak[p] + 1 if p in current_leaders else 1
            best_streak[p] = max(best_streak[p], current_streak[p])
        for p in set(leader_timeline["leader_player"].unique()):
            if p not in leaders:
                current_streak[p] = 0
        current_leaders = leaders

    # Peak values from per-game TrueSkill history.
    peak_mu = {}
    peak_conservative = {}
    if not ts_player_history.empty:
        peak_mu = ts_player_history.groupby("player")["trueskill_mu_after"].max().to_dict()
        peak_conservative = ts_player_history.groupby("player")["trueskill_conservative_after"].max().to_dict()

    final_df = trueskill_df.set_index("player")
    total_games = len(history_df)

    rows = []
    for p in final_df.index:
        rows.append(
            {
                "player": p,
                "games_led": int(lead_counts.get(p, 0)),
                "games_led_pct": lead_counts.get(p, 0) / total_games if total_games else np.nan,
                "longest_leading_streak": int(best_streak.get(p, 0)),
                "final_mu": float(final_df.loc[p, "trueskill_mu"]),
                "final_sigma": float(final_df.loc[p, "trueskill_sigma"]),
                "final_conservative": float(final_df.loc[p, "trueskill_conservative"]),
                "peak_mu": float(peak_mu.get(p, np.nan)),
                "peak_conservative": float(peak_conservative.get(p, np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["games_led", "longest_leading_streak", "final_mu"], ascending=[False, False, False]).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def build_trueskill_prediction_accuracy(df: pd.DataFrame, default_draw_prob: float = 0.0) -> pd.DataFrame:
    """
    Replay TrueSkill game-by-game and compute:
      - predicted Team 1 win probability
      - Brier score
      - rolling Brier score
      - rolling accuracy

    This mirrors build_prediction_accuracy(...) for Elo.
    """
    if not TRUESKILL_AVAILABLE:
        return pd.DataFrame()

    work = df.sort_values("game_id").reset_index(drop=True).copy()

    env = trueskill.TrueSkill(draw_probability=default_draw_prob)
    ratings: Dict[str, object] = defaultdict(lambda: env.create_rating())

    rows = []

    for _, row in work.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)

        # Pre-game TrueSkill team strength using mu
        team1_ratings = [ratings[p] for p in t1]
        team2_ratings = [ratings[p] for p in t2]
        team1_expected_win_prob = trueskill_expected_win_prob(team1_ratings, team2_ratings, env.beta)
        team1_actual_win = 1.0 if team1_won(row) else 0.0

        rows.append(
            {
                "game_id": row["game_id"],
                "day": row.get("day", np.nan),
                "date": row.get("date", pd.NaT),
                "p1": row["p1"],
                "p2": row["p2"],
                "p3": row["p3"],
                "p4": row["p4"],
                "score1": int(row["score1"]),
                "score2": int(row["score2"]),
                "team1_expected_win_prob": team1_expected_win_prob,
                "team1_actual_win": team1_actual_win,
            }
        )

        ranks = [0, 1] if team1_actual_win == 1.0 else [1, 0]
        team1_ratings = [ratings[p] for p in t1]
        team2_ratings = [ratings[p] for p in t2]
        new_t1, new_t2 = env.rate([team1_ratings, team2_ratings], ranks=ranks)

        for p, r in zip(t1, new_t1):
            ratings[p] = r
        for p, r in zip(t2, new_t2):
            ratings[p] = r

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["brier"] = (out["team1_expected_win_prob"] - out["team1_actual_win"]) ** 2
    out["rolling_brier"] = out["brier"].rolling(window=80, min_periods=1).mean()
    out["correct_prediction"] = (
        (out["team1_expected_win_prob"] >= 0.5) == (out["team1_actual_win"] == 1)
    ).astype(int)
    out["rolling_accuracy"] = out["correct_prediction"].rolling(window=80, min_periods=1).mean()

    return out

def plot_trueskill_prediction_accuracy(trueskill_prediction_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    Plot TrueSkill rolling Brier score and rolling accuracy over time.
    """
    if trueskill_prediction_df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        trueskill_prediction_df["game_id"],
        trueskill_prediction_df["rolling_brier"],
        linewidth=3,
        label="Rolling Brier Score",
    )
    ax1.set_ylabel("Brier Score")
    ax1.set_xlabel("Game")
    ax1.set_ylim(0, 0.35)

    ax2 = ax1.twinx()
    ax2.plot(
        trueskill_prediction_df["game_id"],
        trueskill_prediction_df["rolling_accuracy"],
        "--",
        linewidth=2,
        label="Rolling Accuracy",
    )
    ax2.set_ylabel("Prediction Accuracy")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("TrueSkill Prediction Quality Over Time")
    fig.tight_layout()

    path = outdir / "trueskill_prediction_quality.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def build_trueskill_corner_heatmap_of_heatmaps_plotly(trueskill_df: pd.DataFrame) -> Optional[str]:
    if trueskill_df.empty:
        return None

    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]

    skill = trueskill_df.set_index("player")[["trueskill_mu", "trueskill_sigma"]].to_dict("index")
    env = trueskill.TrueSkill(draw_probability=0.0)

    missing = [p for p in player_order if p not in skill]
    if missing:
        raise ValueError(f"Missing TrueSkill ratings for: {missing}")

    n = len(player_order)
    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.025, vertical_spacing=0.03)

    zmin, zmax = 0, 1
    colorscale = "Viridis"
    cmap = plt.cm.viridis
    norm = Normalize(vmin=zmin, vmax=zmax)

    def axis_suffix(idx: int) -> str:
        return "" if idx == 1 else str(idx)

    def rgba_css(color_rgba, alpha: float = 0.18) -> str:
        r, g, b, _ = color_rgba
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    for i, p1 in enumerate(player_order, start=1):
        for j, p2 in enumerate(player_order, start=1):
            if j <= i:
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
                continue

            remaining = [p for p in player_order if p not in (p1, p2)][::-1]

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []

            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    if c <= r:
                        continue

                    team1_ratings = [
                        trueskill.Rating(mu=skill[p1]["trueskill_mu"], sigma=skill[p1]["trueskill_sigma"]),
                        trueskill.Rating(mu=skill[p2]["trueskill_mu"], sigma=skill[p2]["trueskill_sigma"]),
                    ]
                    team2_ratings = [
                        trueskill.Rating(mu=skill[p3]["trueskill_mu"], sigma=skill[p3]["trueskill_sigma"]),
                        trueskill.Rating(mu=skill[p4]["trueskill_mu"], sigma=skill[p4]["trueskill_sigma"]),
                    ]

                    prob = trueskill_expected_win_prob(team1_ratings, team2_ratings, env.beta)
                    inner[r, c] = prob
                    inner_vals.append(prob)

            avg_prob = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_prob)) if not np.isnan(avg_prob) else (1, 1, 1, 1)

            cell_idx = (i - 1) * n + j
            ref = axis_suffix(cell_idx)

            fig.add_shape(
                type="rect",
                x0=0, x1=1, y0=0, y1=1,
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                line=dict(color=rgba_css(cell_color, 0.95), width=4),
                fillcolor=rgba_css(cell_color, 0.60),
                layer="below",
            )

            heatmap_kwargs = dict(
                z=inner,
                x=remaining,
                y=remaining,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                showscale=False,
                hovertemplate=(
                    f"Team 1: {p1} + {p2}<br>"
                    f"Team 2: %{{y}} + %{{x}}<br>"
                    "Win prob: %{z:.1%}<extra></extra>"
                ),
            )
            if i == 1 and j == 2:
                heatmap_kwargs["showscale"] = True
                heatmap_kwargs["colorbar"] = dict(title="Win %")

            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=i, col=j)

            fig.add_annotation(
                text=f"<b>{p1} + {p2}: {avg_prob:.1%}</b>",
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                x=0.5,
                y=1.13,
                showarrow=False,
                align="center",
                font=dict(size=11, color=rgba_css(cell_color, 1.0)),
            )

            fig.update_xaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                tickangle=0,
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                autorange="reversed",
                row=i,
                col=j,
            )

    for j, player in enumerate(player_order, start=1):
        axis_name = "xaxis" if j == 1 else f"xaxis{j}"
        dom = fig.layout[axis_name].domain
        xmid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=xmid,
            y=1.04,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    for i, player in enumerate(player_order, start=1):
        axis_idx = (i - 1) * n + 1
        axis_name = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
        dom = fig.layout[axis_name].domain
        ymid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=-0.015,
            y=ymid,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            xanchor="right",
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text="Corner 2 (Team 1)",
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Corner 1 (Team 1)",
        showarrow=False,
        textangle=-90,
        font=dict(size=18, color="black"),
    )

    fig.update_layout(
        title="TrueSkill Corner-Based Heat Map of Heat Maps",
        height=1800,
        width=1800,
        margin=dict(l=140, r=60, t=140, b=70),
        dragmode="zoom",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

def format_shap_explanation_probability(
    shap_values_1d: np.ndarray,
    feature_row: dict,
    feature_cols: list[str],
    base_prob: float,
    predicted_prob: float,
    top_k: int = 4,
) -> str:
    """
    Turn one SHAP vector into a short hover-friendly explanation in probability space.
    SHAP values are formatted as percentage points.
    """
    contribs = pd.Series(np.asarray(shap_values_1d, dtype=float), index=feature_cols)

    # Sort by absolute impact, largest first
    contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)

    lines = [
        f"<b>Win prob: {predicted_prob:.1%}</b>",
        f"Base prob: {base_prob:.1%}",
        "",
        "<b>Top SHAP drivers</b>",
    ]
    for feat, val in contribs.head(top_k).items():
        feat_val = feature_row.get(feat, np.nan)
        lines.append(f"{feat}: {val:+.1%} pts (value={feat_val:.3g})")

    return "<br>".join(lines)

def build_tree_corner_heatmap_of_heatmaps_plotly(
    model,
    feature_cols: list[str],
    state: dict,
    shap_explainer=None,
    shap_top_k: int = 4,
) -> Optional[str]:
    """
    Full ordered-space 8x8 corner heatmap-of-heatmaps for the tree model.

    Off-diagonal outer cells are shown (p1 != p2).
    Inner 6x6 heatmaps show all ordered opponent pairs from the remaining
    six players, excluding the diagonal only (p3 != p4).
    """
    if not PLOTLY_AVAILABLE:
        return None
    if shap_explainer is None and SHAP_AVAILABLE:
        shap_explainer = shap.TreeExplainer(model)
    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]
    n = len(player_order)

    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.025, vertical_spacing=0.03)

    zmin, zmax = 0, 1
    colorscale = "Viridis"
    cmap = plt.cm.viridis
    norm = Normalize(vmin=zmin, vmax=zmax)

    def axis_suffix(idx: int) -> str:
        return "" if idx == 1 else str(idx)

    def rgba_css(color_rgba, alpha: float = 0.18) -> str:
        r, g, b, _ = color_rgba
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    for i, p1 in enumerate(player_order, start=1):
        for j, p2 in enumerate(player_order, start=1):
            # Impossible outer square: the same player cannot appear in both corners.
            if p1 == p2:
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
                continue

            remaining = [p for p in player_order if p not in (p1, p2)][::-1]

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []
            inner_custom = np.empty((6, 6), dtype=object)
            inner_custom[:] = ""

            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    # Only impossible cells are skipped: same player cannot occupy both Team 2 corners.
                    if p3 == p4:
                        continue

                    feature_row = build_tree_matchup_feature_row(p1, p2, p3, p4, state)
                    x_one = pd.DataFrame(
                        [[feature_row.get(col, np.nan) for col in feature_cols]],
                        columns=feature_cols,
                    )

                    prob = float(model.predict_proba(x_one)[0, 1])

                    shap_text = ""
                    if SHAP_AVAILABLE and shap_explainer is not None:
                        shap_vals = shap_explainer.shap_values(x_one)

                        # Binary classification can return a list or an array depending on SHAP version/model.
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[1]
                            expected_value = shap_explainer.expected_value[1]
                        else:
                            shap_vals = np.asarray(shap_vals)
                            if shap_vals.ndim == 3:
                                shap_vals = shap_vals[:, :, 1]
                            expected_value = (
                                shap_explainer.expected_value[1]
                                if np.ndim(shap_explainer.expected_value) > 0
                                else shap_explainer.expected_value
                            )

                        shap_text = format_shap_explanation_probability(
                            shap_values_1d=np.asarray(shap_vals)[0],
                            feature_row=feature_row,
                            feature_cols=feature_cols,
                            base_prob=float(expected_value),
                            predicted_prob=prob,
                            top_k=shap_top_k,
                        )

                    inner[r, c] = prob
                    inner_custom[r, c] = shap_text
                    inner_vals.append(prob)

            avg_prob = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_prob)) if not np.isnan(avg_prob) else (1, 1, 1, 1)

            cell_idx = (i - 1) * n + j
            ref = axis_suffix(cell_idx)

            fig.add_shape(
                type="rect",
                x0=0, x1=1, y0=0, y1=1,
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                line=dict(color=rgba_css(cell_color, 0.95), width=4),
                fillcolor=rgba_css(cell_color, 0.60),
                layer="below",
            )

            heatmap_kwargs = dict(
                z=inner,
                x=remaining,
                y=remaining,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                showscale=False,
                customdata=inner_custom,
                hovertemplate=(
                    f"Team 1: {p1} + {p2}<br>"
                    f"Team 2: %{{y}} + %{{x}}<br>"
                    "Win prob: %{z:.1%}<br><br>"
                    "%{customdata}<extra></extra>"
                ),
            )
            if i == 1 and j == 2:
                heatmap_kwargs["showscale"] = True
                heatmap_kwargs["colorbar"] = dict(title="Win %")

            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=i, col=j)

            fig.add_annotation(
                text=f"<b>{p1} + {p2}: {avg_prob:.1%}</b>",
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                x=0.5,
                y=1.13,
                showarrow=False,
                align="center",
                font=dict(size=11, color=rgba_css(cell_color, 1.0)),
            )

            fig.update_xaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                tickangle=0,
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                autorange="reversed",
                row=i,
                col=j,
            )

    for j, player in enumerate(player_order, start=1):
        axis_name = "xaxis" if j == 1 else f"xaxis{j}"
        dom = fig.layout[axis_name].domain
        xmid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=xmid,
            y=1.04,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    for i, player in enumerate(player_order, start=1):
        axis_idx = (i - 1) * n + 1
        axis_name = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
        dom = fig.layout[axis_name].domain
        ymid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=-0.015,
            y=ymid,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            xanchor="right",
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text="Corner 2 (Team 1)",
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Corner 1 (Team 1)",
        showarrow=False,
        textangle=-90,
        font=dict(size=18, color="black"),
    )

    fig.update_layout(
        title="Tree Model Full Ordered Probability Space Heat Map",
        height=1800,
        width=1800,
        margin=dict(l=140, r=60, t=140, b=70),
        dragmode="zoom",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_tree_corner_heatmap_of_heatmaps_plotly_game_counts(df: pd.DataFrame) -> Optional[str]:
    """Full ordered-space 8x8 heatmap-of-heatmaps showing games played per configuration."""
    if not PLOTLY_AVAILABLE:
        return None
    if df.empty:
        return None

    player_order = ["FZ", "SY", "CH", "NS", "AC", "TT", "TD", "RS"]
    players_in_data = set(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    missing = [p for p in player_order if p not in players_in_data]
    if missing:
        raise ValueError(f"Missing players in data: {missing}")

    n = len(player_order)
    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.025, vertical_spacing=0.03)

    def axis_suffix(idx: int) -> str:
        return "" if idx == 1 else str(idx)

    def rgba_css(color_rgba, alpha: float = 0.18) -> str:
        r, g, b, _ = color_rgba
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    counts = defaultdict(int)
    for _, row in df.iterrows():
        counts[(row["p1"], row["p2"], row["p3"], row["p4"])] += 1

    max_count = max(counts.values()) if counts else 1
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=max_count)

    for i, p1 in enumerate(player_order, start=1):
        for j, p2 in enumerate(player_order, start=1):
            if p1 == p2:
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)
                continue

            remaining = [p for p in player_order if p not in (p1, p2)][::-1]

            inner = np.full((6, 6), np.nan, dtype=float)
            inner_vals = []

            for r, p3 in enumerate(remaining):
                for c, p4 in enumerate(remaining):
                    if p3 == p4:
                        continue

                    count = counts.get((p1, p2, p3, p4), 0)
                    inner[r, c] = count
                    inner_vals.append(count)

            avg_count = float(np.nanmean(inner_vals)) if inner_vals else np.nan
            cell_color = cmap(norm(avg_count)) if not np.isnan(avg_count) else (1, 1, 1, 1)

            cell_idx = (i - 1) * n + j
            ref = axis_suffix(cell_idx)

            fig.add_shape(
                type="rect",
                x0=0, x1=1, y0=0, y1=1,
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                line=dict(color=rgba_css(cell_color, 0.95), width=4),
                fillcolor=rgba_css(cell_color, 0.60),
                layer="below",
            )

            heatmap_kwargs = dict(
                z=inner,
                x=remaining,
                y=remaining,
                zmin=0,
                zmax=max_count,
                colorscale="Blues",
                showscale=False,
                hovertemplate=(
                    f"Team 1: {p1} + {p2}<br>"
                    f"Team 2: %{{y}} + %{{x}}<br>"
                    "Games played: %{z:.0f}<extra></extra>"
                ),
            )
            if i == 1 and j == 2:
                heatmap_kwargs["showscale"] = True
                heatmap_kwargs["colorbar"] = dict(title="Games")

            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=i, col=j)

            fig.add_annotation(
                text=f"<b>{p1} + {p2}: {avg_count:.1f}</b>",
                xref=f"x{ref} domain",
                yref=f"y{ref} domain",
                x=0.5,
                y=1.13,
                showarrow=False,
                align="center",
                font=dict(size=11, color=rgba_css(cell_color, 1.0)),
            )

            fig.update_xaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                tickangle=0,
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showticklabels=True,
                tickfont=dict(size=10),
                ticks="outside",
                ticklen=8,
                tickcolor="rgba(0,0,0,0)",
                autorange="reversed",
                row=i,
                col=j,
            )

    for j, player in enumerate(player_order, start=1):
        axis_name = "xaxis" if j == 1 else f"xaxis{j}"
        dom = fig.layout[axis_name].domain
        xmid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=xmid,
            y=1.04,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    for i, player in enumerate(player_order, start=1):
        axis_idx = (i - 1) * n + 1
        axis_name = "yaxis" if axis_idx == 1 else f"yaxis{axis_idx}"
        dom = fig.layout[axis_name].domain
        ymid = (dom[0] + dom[1]) / 2.0
        fig.add_annotation(
            x=-0.015,
            y=ymid,
            xref="paper",
            yref="paper",
            text=player,
            showarrow=False,
            xanchor="right",
            font=dict(size=18, color="navy", family="Arial Black"),
        )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text="Corner 2 (Team 1)",
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Corner 1 (Team 1)",
        showarrow=False,
        textangle=-90,
        font=dict(size=18, color="black"),
    )

    fig.update_layout(
        title="Tree Model Games Played by Ordered Configuration",
        height=1800,
        width=1800,
        margin=dict(l=140, r=60, t=140, b=70),
        dragmode="zoom",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def tree_predict_prob(model, feature_row: dict, feature_cols: list[str]) -> float:
    X = pd.DataFrame([[feature_row.get(c, np.nan) for c in feature_cols]], columns=feature_cols)
    return float(model.predict_proba(X)[0, 1])

def build_tree_matchup_feature_row(
    p1: str,
    p2: str,
    p3: str,
    p4: str,
    state: dict,
    recent_window: int = 5,
    default_elo: float = DEFAULT_ELO,
) -> dict:
    """
    Build one pre-game feature row for a hypothetical matchup.
    Uses the same feature definitions as build_ml_training_table().

    Assumes `state` contains:
      ratings, ts_ratings, player_games, player_wins, player_point_diff,
      player_recent_results, player_recent_point_diffs, player_elo_history,
      player_peak_elo, player_partner_games, player_partner_wins,
      player_ts_mu_history, player_opponent_games, player_opponent_wins,
      pair_games, pair_wins, pair_point_diff, matchup_games, matchup_wins,
      matchup_point_diff, corner_games, corner_wins
    """
    ratings = state.get("ratings", {})
    ts_ratings = state.get("ts_ratings") or {}

    player_games = state.get("player_games", {})
    player_wins = state.get("player_wins", {})
    player_point_diff = state.get("player_point_diff", {})
    player_recent_results = state.get("player_recent_results", {})
    player_recent_point_diffs = state.get("player_recent_point_diffs", {})
    player_elo_history = state.get("player_elo_history", {})
    player_peak_elo = state.get("player_peak_elo", {})

    player_partner_games = state.get("player_partner_games", {})
    player_partner_wins = state.get("player_partner_wins", {})
    player_ts_mu_history = state.get("player_ts_mu_history", {})
    player_opponent_games = state.get("player_opponent_games", {})
    player_opponent_wins = state.get("player_opponent_wins", {})

    pair_games = state.get("pair_games", {})
    pair_wins = state.get("pair_wins", {})
    pair_point_diff = state.get("pair_point_diff", {})

    matchup_games = state.get("matchup_games", {})
    matchup_wins = state.get("matchup_wins", {})
    matchup_point_diff = state.get("matchup_point_diff", {})

    corner_games = state.get("corner_games", {})
    corner_wins = state.get("corner_wins", {})

    default_ts = trueskill.Rating() if TRUESKILL_AVAILABLE else None
    elo_prob_fn = globals().get("elo_expected_score", expected_score)

    def safe_div(num: float, den: float, fallback: float = np.nan) -> float:
        return float(num / den) if den else float(fallback)

    def safe_mean(values, fallback: float = np.nan) -> float:
        values = list(values)
        return float(np.mean(values)) if values else float(fallback)

    def safe_std(values, fallback: float = 1.0) -> float:
        values = list(values)
        if len(values) < 2:
            return float(fallback)
        sd = float(np.std(values, ddof=0))
        return sd if sd > 0 else float(fallback)

    def get_nested(mapping, key1, key2, default=0):
        return mapping.get(key1, {}).get(key2, default)

    def recent_elo_std(player: str) -> float:
        hist = list(player_elo_history.get(player, []))
        if len(hist) < 2:
            return 0.0
        return float(np.std(hist, ddof=0))

    def recent_elo_trend(player: str) -> float:
        hist = list(player_elo_history.get(player, []))
        if not hist:
            return 0.0
        current = float(ratings.get(player, default_elo))
        return float(current - np.mean(hist))

    def recent_ts_momentum(player: str) -> float:
        if not TRUESKILL_AVAILABLE:
            return 0.0
        hist = list(player_ts_mu_history.get(player, []))
        if not hist:
            return 0.0
        current_mu = float(ts_ratings.get(player, default_ts).mu)
        return float(current_mu - np.mean(hist))

    def corner_win_rate(player: str, corner: int) -> float:
        g = get_nested(corner_games, player, corner, 0)
        w = get_nested(corner_wins, player, corner, 0)
        return safe_div(w, g, 0.5)

    def opponent_experience(player: str, opp_a: str, opp_b: str) -> tuple[float, float]:
        games = get_nested(player_opponent_games, player, opp_a, 0) + get_nested(player_opponent_games, player, opp_b, 0)
        wins = get_nested(player_opponent_wins, player, opp_a, 0) + get_nested(player_opponent_wins, player, opp_b, 0)
        return float(games), float(safe_div(wins, games, 0.5))

    def player_specialization(player: str) -> float:
        rates = []
        partners = player_partner_games.get(player, {})
        for partner, games in partners.items():
            if games > 0:
                rates.append(get_nested(player_partner_wins, player, partner, 0) / games)
        if len(rates) < 2:
            return 0.0
        return float(np.std(rates, ddof=0))

    # -----------------------------
    # Team / pair identity
    # -----------------------------
    team1 = (p1, p2)
    team2 = (p3, p4)
    team1_pair = pair_key(p1, p2)
    team2_pair = pair_key(p3, p4)
    matchup_key = (team1_pair, team2_pair)

    # -----------------------------
    # Elo level features
    # -----------------------------
    p1_elo = float(ratings.get(p1, default_elo))
    p2_elo = float(ratings.get(p2, default_elo))
    p3_elo = float(ratings.get(p3, default_elo))
    p4_elo = float(ratings.get(p4, default_elo))

    team1_elo = float(np.mean([p1_elo, p2_elo]))
    team2_elo = float(np.mean([p3_elo, p4_elo]))
    elo_diff = team1_elo - team2_elo
    elo_abs_diff = abs(elo_diff)
    elo_expected_prob = float(elo_prob_fn(team1_elo, team2_elo))

    team1_max_elo = float(max(p1_elo, p2_elo))
    team2_max_elo = float(max(p3_elo, p4_elo))
    team1_min_elo = float(min(p1_elo, p2_elo))
    team2_min_elo = float(min(p3_elo, p4_elo))

    team1_balance = team1_max_elo - team1_min_elo
    team2_balance = team2_max_elo - team2_min_elo

    team1_std = float(np.std([p1_elo, p2_elo], ddof=0))
    team2_std = float(np.std([p3_elo, p4_elo], ddof=0))

    # historical Elo peak / gap
    team1_peak_elo = float(np.mean([player_peak_elo.get(p1, default_elo), player_peak_elo.get(p2, default_elo)]))
    team2_peak_elo = float(np.mean([player_peak_elo.get(p3, default_elo), player_peak_elo.get(p4, default_elo)]))
    team1_peak_gap = team1_peak_elo - team1_elo
    team2_peak_gap = team2_peak_elo - team2_elo

    # historical Elo volatility
    team1_elo_volatility = float(np.mean([recent_elo_std(p1), recent_elo_std(p2)]))
    team2_elo_volatility = float(np.mean([recent_elo_std(p3), recent_elo_std(p4)]))

    # side-based gap
    elo_confidence_gap_side = (team1_elo - team1_elo_volatility) - (team2_elo + team2_elo_volatility)

    # favorite/underdog gap
    if team1_elo >= team2_elo:
        favorite_elo = team1_elo
        favorite_vol = team1_elo_volatility
        underdog_elo = team2_elo
        underdog_vol = team2_elo_volatility
    else:
        favorite_elo = team2_elo
        favorite_vol = team2_elo_volatility
        underdog_elo = team1_elo
        underdog_vol = team1_elo_volatility

    elo_confidence_gap = (favorite_elo - favorite_vol) - (underdog_elo + underdog_vol)

    # -----------------------------
    # TrueSkill features
    # -----------------------------
    if TRUESKILL_AVAILABLE and ts_ratings:
        t1_ts1 = ts_ratings.get(p1, default_ts)
        t1_ts2 = ts_ratings.get(p2, default_ts)
        t2_ts1 = ts_ratings.get(p3, default_ts)
        t2_ts2 = ts_ratings.get(p4, default_ts)

        team1_ts_mu = float(np.mean([t1_ts1.mu, t1_ts2.mu]))
        team2_ts_mu = float(np.mean([t2_ts1.mu, t2_ts2.mu]))
        team1_ts_sigma = float(np.mean([t1_ts1.sigma, t1_ts2.sigma]))
        team2_ts_sigma = float(np.mean([t2_ts1.sigma, t2_ts2.sigma]))

        team1_ts_conservative = team1_ts_mu - 3 * team1_ts_sigma
        team2_ts_conservative = team2_ts_mu - 3 * team2_ts_sigma

        ts_mu_diff = team1_ts_mu - team2_ts_mu
        ts_sigma_diff = team1_ts_sigma - team2_ts_sigma
        ts_cons_diff = team1_ts_conservative - team2_ts_conservative

        team1_ts_balance = abs(t1_ts1.mu - t1_ts2.mu)
        team2_ts_balance = abs(t2_ts1.mu - t2_ts2.mu)

        team1_ts_momentum = float(np.mean([recent_ts_momentum(p1), recent_ts_momentum(p2)]))
        team2_ts_momentum = float(np.mean([recent_ts_momentum(p3), recent_ts_momentum(p4)]))

        ts_confidence_gap_side = (team1_ts_mu - team1_ts_sigma) - (team2_ts_mu + team2_ts_sigma)

        if team1_ts_mu >= team2_ts_mu:
            favorite_ts_mu = team1_ts_mu
            favorite_ts_sigma = team1_ts_sigma
            underdog_ts_mu = team2_ts_mu
            underdog_ts_sigma = team2_ts_sigma
        else:
            favorite_ts_mu = team2_ts_mu
            favorite_ts_sigma = team2_ts_sigma
            underdog_ts_mu = team1_ts_mu
            underdog_ts_sigma = team1_ts_sigma

        ts_confidence_gap = (favorite_ts_mu - favorite_ts_sigma) - (underdog_ts_mu + underdog_ts_sigma)
    else:
        team1_ts_mu = team2_ts_mu = np.nan
        team1_ts_sigma = team2_ts_sigma = np.nan
        team1_ts_conservative = team2_ts_conservative = np.nan
        ts_mu_diff = ts_sigma_diff = ts_cons_diff = np.nan
        team1_ts_balance = team2_ts_balance = np.nan
        team1_ts_momentum = team2_ts_momentum = np.nan
        ts_confidence_gap_side = np.nan
        ts_confidence_gap = np.nan

    # -----------------------------
    # Team specialization / rating disagreement
    # -----------------------------
    team1_specialization = float(np.mean([player_specialization(p1), player_specialization(p2)]))
    team2_specialization = float(np.mean([player_specialization(p3), player_specialization(p4)]))
    specialization_gap = team1_specialization - team2_specialization

    elo_pool = list(ratings.values())
    elo_mean = float(np.mean(elo_pool)) if elo_pool else default_elo
    elo_sd = safe_std(elo_pool, fallback=1.0)

    team1_elo_z = (team1_elo - elo_mean) / elo_sd
    team2_elo_z = (team2_elo - elo_mean) / elo_sd

    if TRUESKILL_AVAILABLE and ts_ratings:
        ts_pool = [r.mu for r in ts_ratings.values()]
        ts_mean = float(np.mean(ts_pool)) if ts_pool else 0.0
        ts_sd = safe_std(ts_pool, fallback=1.0)

        team1_ts_z = (team1_ts_mu - ts_mean) / ts_sd
        team2_ts_z = (team2_ts_mu - ts_mean) / ts_sd

        rating_disagreement = abs((team1_elo_z - team2_elo_z) - (team1_ts_z - team2_ts_z))
    else:
        rating_disagreement = np.nan

    # -----------------------------
    # Experience / recent form
    # -----------------------------
    team1_games_played = float(np.mean([player_games.get(p1, 0), player_games.get(p2, 0)]))
    team2_games_played = float(np.mean([player_games.get(p3, 0), player_games.get(p4, 0)]))

    team1_win_rate = float(np.mean([
        safe_div(player_wins.get(p1, 0), player_games.get(p1, 0), 0.5),
        safe_div(player_wins.get(p2, 0), player_games.get(p2, 0), 0.5),
    ]))
    team2_win_rate = float(np.mean([
        safe_div(player_wins.get(p3, 0), player_games.get(p3, 0), 0.5),
        safe_div(player_wins.get(p4, 0), player_games.get(p4, 0), 0.5),
    ]))

    team1_recent_win_rate = float(np.mean([
        safe_mean(player_recent_results.get(p1, []), 0.5),
        safe_mean(player_recent_results.get(p2, []), 0.5),
    ]))
    team2_recent_win_rate = float(np.mean([
        safe_mean(player_recent_results.get(p3, []), 0.5),
        safe_mean(player_recent_results.get(p4, []), 0.5),
    ]))

    team1_avg_point_diff = float(np.mean([
        safe_div(player_point_diff.get(p1, 0), player_games.get(p1, 0), 0.0),
        safe_div(player_point_diff.get(p2, 0), player_games.get(p2, 0), 0.0),
    ]))
    team2_avg_point_diff = float(np.mean([
        safe_div(player_point_diff.get(p3, 0), player_games.get(p3, 0), 0.0),
        safe_div(player_point_diff.get(p4, 0), player_games.get(p4, 0), 0.0),
    ]))

    team1_recent_point_diff = float(np.mean([
        safe_mean(player_recent_point_diffs.get(p1, []), 0.0),
        safe_mean(player_recent_point_diffs.get(p2, []), 0.0),
    ]))
    team2_recent_point_diff = float(np.mean([
        safe_mean(player_recent_point_diffs.get(p3, []), 0.0),
        safe_mean(player_recent_point_diffs.get(p4, []), 0.0),
    ]))

    team1_recent_elo_trend = float(np.mean([recent_elo_trend(p1), recent_elo_trend(p2)]))
    team2_recent_elo_trend = float(np.mean([recent_elo_trend(p3), recent_elo_trend(p4)]))

    # -----------------------------
    # Chemistry / matchup familiarity
    # -----------------------------
    team1_pair_games = float(pair_games.get(team1_pair, 0))
    team2_pair_games = float(pair_games.get(team2_pair, 0))
    team1_pair_win_rate = safe_div(pair_wins.get(team1_pair, 0), pair_games.get(team1_pair, 0), 0.5)
    team2_pair_win_rate = safe_div(pair_wins.get(team2_pair, 0), pair_games.get(team2_pair, 0), 0.5)
    team1_pair_point_diff = float(pair_point_diff.get(team1_pair, 0))
    team2_pair_point_diff = float(pair_point_diff.get(team2_pair, 0))

    h2h_games = float(matchup_games.get(matchup_key, 0))
    h2h_win_rate = safe_div(matchup_wins.get(matchup_key, 0), matchup_games.get(matchup_key, 0), 0.5)
    h2h_point_diff = float(matchup_point_diff.get(matchup_key, 0))

    t1_opp_g1, t1_opp_wr1 = opponent_experience(p1, p3, p4)
    t1_opp_g2, t1_opp_wr2 = opponent_experience(p2, p3, p4)
    t2_opp_g3, t2_opp_wr3 = opponent_experience(p3, p1, p2)
    t2_opp_g4, t2_opp_wr4 = opponent_experience(p4, p1, p2)

    team1_opponent_experience = float(np.mean([t1_opp_g1, t1_opp_g2]))
    team2_opponent_experience = float(np.mean([t2_opp_g3, t2_opp_g4]))
    team1_opponent_win_rate = float(np.mean([t1_opp_wr1, t1_opp_wr2]))
    team2_opponent_win_rate = float(np.mean([t2_opp_wr3, t2_opp_wr4]))

    # team specialization
    team1_specialization = float(np.mean([player_specialization(p1), player_specialization(p2)]))
    team2_specialization = float(np.mean([player_specialization(p3), player_specialization(p4)]))
    specialization_gap = team1_specialization - team2_specialization

    # -----------------------------
    # Corner / position
    # -----------------------------
    p1_corner_rate = corner_win_rate(p1, 1)
    p2_corner_rate = corner_win_rate(p2, 2)
    p3_corner_rate = corner_win_rate(p3, 3)
    p4_corner_rate = corner_win_rate(p4, 4)

    team1_corner_strength = float(np.mean([p1_corner_rate, p2_corner_rate]))
    team2_corner_strength = float(np.mean([p3_corner_rate, p4_corner_rate]))

    # final feature row
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,

        # Elo
        "team1_elo_pre": team1_elo,
        "team2_elo_pre": team2_elo,
        "elo_diff_pre": elo_diff,
        "elo_abs_diff_pre": elo_abs_diff,
        "elo_expected_prob_pre": elo_expected_prob,
        "team1_max_elo_pre": team1_max_elo,
        "team2_max_elo_pre": team2_max_elo,
        "team1_min_elo_pre": team1_min_elo,
        "team2_min_elo_pre": team2_min_elo,
        "team1_balance_pre": team1_balance,
        "team2_balance_pre": team2_balance,
        "team1_elo_std_pre": team1_std,
        "team2_elo_std_pre": team2_std,
        "team1_peak_elo_pre": team1_peak_elo,
        "team2_peak_elo_pre": team2_peak_elo,
        "team1_peak_gap_pre": team1_peak_gap,
        "team2_peak_gap_pre": team2_peak_gap,

        # Elo stability / confidence
        "team1_elo_volatility_pre": team1_elo_volatility,
        "team2_elo_volatility_pre": team2_elo_volatility,
        "elo_confidence_gap_pre": elo_confidence_gap,
        "elo_confidence_gap_side_pre": elo_confidence_gap_side,

        # TrueSkill
        "team1_ts_mu_pre": team1_ts_mu,
        "team2_ts_mu_pre": team2_ts_mu,
        "team1_ts_sigma_pre": team1_ts_sigma,
        "team2_ts_sigma_pre": team2_ts_sigma,
        "team1_ts_conservative_pre": team1_ts_conservative,
        "team2_ts_conservative_pre": team2_ts_conservative,
        "ts_mu_diff_pre": ts_mu_diff,
        "ts_sigma_diff_pre": ts_sigma_diff,
        "ts_cons_diff_pre": ts_cons_diff,
        "team1_ts_balance_pre": team1_ts_balance,
        "team2_ts_balance_pre": team2_ts_balance,
        "team1_ts_momentum_pre": team1_ts_momentum,
        "team2_ts_momentum_pre": team2_ts_momentum,
        "ts_confidence_gap_pre": ts_confidence_gap_side,
        "ts_confidence_gap_favorite_pre": ts_confidence_gap,

        # Team specialization / disagreement
        "team1_specialization_pre": team1_specialization,
        "team2_specialization_pre": team2_specialization,
        "specialization_gap_pre": specialization_gap,
        "rating_disagreement_pre": rating_disagreement,

        # Experience / form
        "team1_games_played_pre": team1_games_played,
        "team2_games_played_pre": team2_games_played,
        "team1_win_rate_pre": team1_win_rate,
        "team2_win_rate_pre": team2_win_rate,
        "team1_recent_win_rate_pre": team1_recent_win_rate,
        "team2_recent_win_rate_pre": team2_recent_win_rate,
        "team1_avg_point_diff_pre": team1_avg_point_diff,
        "team2_avg_point_diff_pre": team2_avg_point_diff,
        "team1_recent_point_diff_pre": team1_recent_point_diff,
        "team2_recent_point_diff_pre": team2_recent_point_diff,
        "team1_recent_elo_trend_pre": team1_recent_elo_trend,
        "team2_recent_elo_trend_pre": team2_recent_elo_trend,

        # Chemistry
        "team1_pair_games_pre": team1_pair_games,
        "team2_pair_games_pre": team2_pair_games,
        "team1_pair_win_rate_pre": team1_pair_win_rate,
        "team2_pair_win_rate_pre": team2_pair_win_rate,
        "team1_pair_point_diff_pre": team1_pair_point_diff,
        "team2_pair_point_diff_pre": team2_pair_point_diff,
        "h2h_games_pre": h2h_games,
        "h2h_win_rate_pre": h2h_win_rate,
        "h2h_point_diff_pre": h2h_point_diff,
        "team1_opponent_experience_pre": team1_opponent_experience,
        "team2_opponent_experience_pre": team2_opponent_experience,
        "team1_opponent_win_rate_pre": team1_opponent_win_rate,
        "team2_opponent_win_rate_pre": team2_opponent_win_rate,

        # Corner / position
        "p1_corner_win_rate_pre": p1_corner_rate,
        "p2_corner_win_rate_pre": p2_corner_rate,
        "p3_corner_win_rate_pre": p3_corner_rate,
        "p4_corner_win_rate_pre": p4_corner_rate,
        "team1_corner_strength_pre": team1_corner_strength,
        "team2_corner_strength_pre": team2_corner_strength,
        "corner_strength_diff_pre": team1_corner_strength - team2_corner_strength,
    }


# ============================================================
# PREDICTIONS
# ============================================================

def predict_match_prob(team1: Tuple[str, str], team2: Tuple[str, str], elo_df: pd.DataFrame, default_elo: float = DEFAULT_ELO) -> float:
    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()}
    r1 = np.mean([ratings.get(p, default_elo) for p in team1])
    r2 = np.mean([ratings.get(p, default_elo) for p in team2])
    return elo_expected_score(r1, r2)


def add_prediction_columns(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Create all possible current matchups and always place the stronger team in team1."""
    players = list(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    if len(players) < 4:
        return pd.DataFrame(columns=["team1", "team2", "team1_elo", "team2_elo", "elo_diff", "avg_elo", "pred_team1_win_prob"])

    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()}
    rows = []
    matchup_id = 1

    for quad in combinations(players, 4):
        a, b, c, d = quad
        pairings = [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]
        for team_x, team_y in pairings:
            elo_x = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in team_x]))
            elo_y = float(np.mean([ratings.get(p, DEFAULT_ELO) for p in team_y]))
            if elo_y > elo_x:
                team1, team2 = team_y, team_x
                team1_elo, team2_elo = elo_y, elo_x
            else:
                team1, team2 = team_x, team_y
                team1_elo, team2_elo = elo_x, elo_y

            rows.append(
                {
                    "matchup_id": matchup_id,
                    "team1": f"{team1[0]} + {team1[1]}",
                    "team2": f"{team2[0]} + {team2[1]}",
                    "team1_elo": team1_elo,
                    "team2_elo": team2_elo,
                    "elo_diff": team1_elo - team2_elo,
                    "avg_elo": (team1_elo + team2_elo) / 2.0,
                    "pred_team1_win_prob": elo_expected_score(team1_elo, team2_elo),
                }
            )
            matchup_id += 1

    return pd.DataFrame(rows)


# ============================================================
# ML FEATURE ENGINEERING, TRAINING, AND EXPLANATIONS
# ============================================================

def build_ml_training_table(
    df: pd.DataFrame,
    recent_window: int = 5,
    default_elo: float = DEFAULT_ELO,
    return_state: bool = False
) -> pd.DataFrame:
    """
    Convert each historical game into a pre-game feature row.

    All features are computed using only information available BEFORE the game.
    """

    work = df.sort_values("game_id").reset_index(drop=True).copy()

    # Replay state
    ratings = defaultdict(lambda: default_elo)

    # Player state
    player_games = defaultdict(int)
    player_wins = defaultdict(int)
    player_point_diff = defaultdict(int)
    player_recent_results = defaultdict(lambda: deque(maxlen=recent_window))
    player_recent_point_diffs = defaultdict(lambda: deque(maxlen=recent_window))
    player_elo_history = defaultdict(lambda: deque(maxlen=recent_window))
    player_peak_elo = defaultdict(lambda: default_elo)
    player_partner_games = defaultdict(lambda: defaultdict(int))
    player_partner_wins = defaultdict(lambda: defaultdict(int))
    player_ts_mu_history = defaultdict(lambda: deque(maxlen=recent_window))
    player_opponent_games = defaultdict(lambda: defaultdict(int))
    player_opponent_wins = defaultdict(lambda: defaultdict(int))
    
    if TRUESKILL_AVAILABLE:
        ts_env = trueskill.TrueSkill(draw_probability=0.0)
        ts_ratings = defaultdict(lambda: ts_env.create_rating())
    else:
        ts_env = None
        ts_ratings = None

    # Teammate pair state
    pair_games = defaultdict(int)
    pair_wins = defaultdict(int)
    pair_point_diff = defaultdict(int)

    # Exact matchup state: (team1_pair, team2_pair)
    matchup_games = defaultdict(int)
    matchup_wins = defaultdict(int)
    matchup_point_diff = defaultdict(int)

    # Corner state: actual corner number 1, 2, 3, 4
    corner_games = defaultdict(lambda: defaultdict(int))
    corner_wins = defaultdict(lambda: defaultdict(int))

    rows = []

    def safe_div(num: float, den: float, fallback: float = np.nan) -> float:
        return float(num / den) if den else float(fallback)

    def mean_or_default(values, fallback: float) -> float:
        values = list(values)
        return float(np.mean(values)) if values else float(fallback)

    def recent_mean(dq, fallback: float = 0.0) -> float:
        return mean_or_default(dq, fallback)

    def recent_elo_trend(player: str) -> float:
        hist = player_elo_history[player]
        if not hist:
            return 0.0
        return float(ratings[player] - np.mean(hist))

    def corner_win_rate(player: str, corner: int) -> float:
        return safe_div(corner_wins[player][corner], corner_games[player][corner], 0.5)
    
    def recent_elo_std(player: str) -> float:
        hist = player_elo_history[player]
        if len(hist) < 2:
            return 0.0
        return float(np.std(hist, ddof=0))

    def recent_ts_momentum(player: str) -> float:
        hist = player_ts_mu_history[player]
        if not hist:
            return 0.0
        return float(ts_ratings[player].mu - np.mean(hist)) if TRUESKILL_AVAILABLE else 0.0

    def opponent_experience(player: str, opp_a: str, opp_b: str) -> tuple[float, float]:
        games = player_opponent_games[player][opp_a] + player_opponent_games[player][opp_b]
        wins = player_opponent_wins[player][opp_a] + player_opponent_wins[player][opp_b]
        win_rate = safe_div(wins, games, 0.5)
        return float(games), float(win_rate)

    def player_specialization(player: str) -> float:
        """
        Standard deviation of this player's historical pair win rates across teammates.
        Higher = more teammate-dependent / more specialized.
        """
        rates = []
        for partner, games in player_partner_games[player].items():
            if games > 0:
                rates.append(player_partner_wins[player][partner] / games)

        if len(rates) < 2:
            return 0.0
        return float(np.std(rates, ddof=0))


    def safe_std(values, fallback: float = 1.0) -> float:
        values = list(values)
        if len(values) < 2:
            return float(fallback)
        sd = float(np.std(values, ddof=0))
        return sd if sd > 0 else float(fallback)

    for _, row in work.iterrows():
        p1, p2, p3, p4 = row["p1"], row["p2"], row["p3"], row["p4"]

        team1 = (p1, p2)
        team2 = (p3, p4)
        team1_pair = pair_key(p1, p2)
        team2_pair = pair_key(p3, p4)

        # -----------------------------
        # Pre-game Elo features
        # -----------------------------
        p1_elo = ratings[p1]
        p2_elo = ratings[p2]
        p3_elo = ratings[p3]
        p4_elo = ratings[p4]

        team1_elo = float(np.mean([p1_elo, p2_elo]))
        team2_elo = float(np.mean([p3_elo, p4_elo]))

        team1_max_elo = float(max(p1_elo, p2_elo))
        team2_max_elo = float(max(p3_elo, p4_elo))
        team1_min_elo = float(min(p1_elo, p2_elo))
        team2_min_elo = float(min(p3_elo, p4_elo))

        team1_balance = team1_max_elo - team1_min_elo
        team2_balance = team2_max_elo - team2_min_elo

        team1_std = float(np.std([p1_elo, p2_elo]))
        team2_std = float(np.std([p3_elo, p4_elo]))

        elo_diff = team1_elo - team2_elo
        elo_abs_diff = abs(elo_diff)
        elo_expected_prob = elo_expected_score(team1_elo, team2_elo)
        if TRUESKILL_AVAILABLE:
            team1_ts_mu = float(np.mean([ts_ratings[p].mu for p in team1]))
            team2_ts_mu = float(np.mean([ts_ratings[p].mu for p in team2]))
            team1_ts_sigma = float(np.mean([ts_ratings[p].sigma for p in team1]))
            team2_ts_sigma = float(np.mean([ts_ratings[p].sigma for p in team2]))
            team1_ts_conservative = team1_ts_mu - 3 * team1_ts_sigma
            team2_ts_conservative = team2_ts_mu - 3 * team2_ts_sigma
            ts_mu_diff = team1_ts_mu - team2_ts_mu
            ts_sigma_diff = team1_ts_sigma - team2_ts_sigma
            ts_cons_diff = team1_ts_conservative - team2_ts_conservative
        else:
            team1_ts_mu = team2_ts_mu = np.nan
            team1_ts_sigma = team2_ts_sigma = np.nan
            team1_ts_conservative = team2_ts_conservative = np.nan
            ts_mu_diff = ts_sigma_diff = ts_cons_diff = np.nan

        # -----------------------------
        # Rating confidence / overlap
        # -----------------------------
        team1_elo_volatility = float(np.mean([recent_elo_std(p1), recent_elo_std(p2)]))
        team2_elo_volatility = float(np.mean([recent_elo_std(p3), recent_elo_std(p4)]))

        # Side-based version (kept for reference / interpretability)
        elo_confidence_gap_side = (
            (team1_elo - team1_elo_volatility)
            - (team2_elo + team2_elo_volatility)
        )

        # Favorite / underdog version (independent of Team 1 vs Team 2 side)
        if team1_elo >= team2_elo:
            favorite_elo = team1_elo
            favorite_vol = team1_elo_volatility
            underdog_elo = team2_elo
            underdog_vol = team2_elo_volatility
        else:
            favorite_elo = team2_elo
            favorite_vol = team2_elo_volatility
            underdog_elo = team1_elo
            underdog_vol = team1_elo_volatility

        elo_confidence_gap = (
            (favorite_elo - favorite_vol)
            - (underdog_elo + underdog_vol)
        )

        if TRUESKILL_AVAILABLE:
            team1_ts_balance = abs(ts_ratings[p1].mu - ts_ratings[p2].mu)
            team2_ts_balance = abs(ts_ratings[p3].mu - ts_ratings[p4].mu)

            team1_ts_momentum = float(np.mean([recent_ts_momentum(p1), recent_ts_momentum(p2)]))
            team2_ts_momentum = float(np.mean([recent_ts_momentum(p3), recent_ts_momentum(p4)]))

            ts_confidence_gap = (team1_ts_mu - team1_ts_sigma) - (team2_ts_mu + team2_ts_sigma)

            if team1_ts_mu >= team2_ts_mu:
                favorite_ts_mu = team1_ts_mu
                favorite_ts_sigma = team1_ts_sigma
                underdog_ts_mu = team2_ts_mu
                underdog_ts_sigma = team2_ts_sigma
            else:
                favorite_ts_mu = team2_ts_mu
                favorite_ts_sigma = team2_ts_sigma
                underdog_ts_mu = team1_ts_mu
                underdog_ts_sigma = team1_ts_sigma

            ts_confidence_gap_favorite = (
                (favorite_ts_mu - favorite_ts_sigma)
                - (underdog_ts_mu + underdog_ts_sigma)
            )
        else:
            team1_ts_balance = team2_ts_balance = np.nan
            team1_ts_momentum = team2_ts_momentum = np.nan
            ts_confidence_gap = np.nan
            ts_confidence_gap_favorite = np.nan

        # -----------------------------
        # Team specialization / rating disagreement
        # -----------------------------
        team1_specialization = float(np.mean([
            player_specialization(p1),
            player_specialization(p2),
        ]))
        team2_specialization = float(np.mean([
            player_specialization(p3),
            player_specialization(p4),
        ]))
        specialization_gap = team1_specialization - team2_specialization

        # Compare Elo and TrueSkill on a common standardized scale.
        elo_pool = list(ratings.values())
        elo_mean = float(np.mean(elo_pool)) if elo_pool else default_elo
        elo_sd = safe_std(elo_pool, fallback=1.0)

        team1_elo_z = (team1_elo - elo_mean) / elo_sd
        team2_elo_z = (team2_elo - elo_mean) / elo_sd

        if TRUESKILL_AVAILABLE:
            ts_pool = [r.mu for r in ts_ratings.values()]
            ts_mean = float(np.mean(ts_pool)) if ts_pool else 0.0
            ts_sd = safe_std(ts_pool, fallback=1.0)

            team1_ts_z = (team1_ts_mu - ts_mean) / ts_sd
            team2_ts_z = (team2_ts_mu - ts_mean) / ts_sd

            rating_disagreement = abs((team1_elo_z - team2_elo_z) - (team1_ts_z - team2_ts_z))
        else:
            rating_disagreement = np.nan

        team1_peak_elo = float(np.mean([player_peak_elo[p1], player_peak_elo[p2]]))
        team2_peak_elo = float(np.mean([player_peak_elo[p3], player_peak_elo[p4]]))

        team1_peak_gap = team1_peak_elo - team1_elo
        team2_peak_gap = team2_peak_elo - team2_elo

        # -----------------------------
        # Player-level experience / form
        # -----------------------------
        team1_games_played = float(np.mean([player_games[p1], player_games[p2]]))
        team2_games_played = float(np.mean([player_games[p3], player_games[p4]]))

        team1_win_rate = float(np.mean([
            safe_div(player_wins[p1], player_games[p1], 0.5),
            safe_div(player_wins[p2], player_games[p2], 0.5),
        ]))
        team2_win_rate = float(np.mean([
            safe_div(player_wins[p3], player_games[p3], 0.5),
            safe_div(player_wins[p4], player_games[p4], 0.5),
        ]))

        team1_recent_win_rate = float(np.mean([
            recent_mean(player_recent_results[p1], 0.5),
            recent_mean(player_recent_results[p2], 0.5),
        ]))
        team2_recent_win_rate = float(np.mean([
            recent_mean(player_recent_results[p3], 0.5),
            recent_mean(player_recent_results[p4], 0.5),
        ]))

        team1_avg_point_diff = float(np.mean([
            safe_div(player_point_diff[p1], player_games[p1], 0.0),
            safe_div(player_point_diff[p2], player_games[p2], 0.0),
        ]))
        team2_avg_point_diff = float(np.mean([
            safe_div(player_point_diff[p3], player_games[p3], 0.0),
            safe_div(player_point_diff[p4], player_games[p4], 0.0),
        ]))

        team1_recent_point_diff = float(np.mean([
            recent_mean(player_recent_point_diffs[p1], 0.0),
            recent_mean(player_recent_point_diffs[p2], 0.0),
        ]))
        team2_recent_point_diff = float(np.mean([
            recent_mean(player_recent_point_diffs[p3], 0.0),
            recent_mean(player_recent_point_diffs[p4], 0.0),
        ]))

        team1_recent_elo_trend = float(np.mean([recent_elo_trend(p1), recent_elo_trend(p2)]))
        team2_recent_elo_trend = float(np.mean([recent_elo_trend(p3), recent_elo_trend(p4)]))

        # -----------------------------
        # Chemistry / teammate history
        # -----------------------------
        team1_pair_win_rate = safe_div(pair_wins[team1_pair], pair_games[team1_pair], 0.5)
        team2_pair_win_rate = safe_div(pair_wins[team2_pair], pair_games[team2_pair], 0.5)

        team1_pair_point_diff = float(pair_point_diff[team1_pair])
        team2_pair_point_diff = float(pair_point_diff[team2_pair])

        # -----------------------------
        # Head-to-head history
        # -----------------------------
        matchup_key = (team1_pair, team2_pair)
        h2h_games = matchup_games[matchup_key]
        h2h_win_rate = safe_div(matchup_wins[matchup_key], h2h_games, 0.5)
        h2h_point_diff = float(matchup_point_diff[matchup_key])


        # -----------------------------
        # Opponent experience
        # -----------------------------
        t1_opp_g1, t1_opp_wr1 = opponent_experience(p1, p3, p4)
        t1_opp_g2, t1_opp_wr2 = opponent_experience(p2, p3, p4)
        t2_opp_g3, t2_opp_wr3 = opponent_experience(p3, p1, p2)
        t2_opp_g4, t2_opp_wr4 = opponent_experience(p4, p1, p2)

        team1_opponent_experience = float(np.mean([t1_opp_g1, t1_opp_g2]))
        team2_opponent_experience = float(np.mean([t2_opp_g3, t2_opp_g4]))

        team1_opponent_win_rate = float(np.mean([t1_opp_wr1, t1_opp_wr2]))
        team2_opponent_win_rate = float(np.mean([t2_opp_wr3, t2_opp_wr4]))

        # -----------------------------
        # Corner / position features
        # -----------------------------
        p1_corner_rate = corner_win_rate(p1, 1)
        p2_corner_rate = corner_win_rate(p2, 2)
        p3_corner_rate = corner_win_rate(p3, 3)
        p4_corner_rate = corner_win_rate(p4, 4)

        team1_corner_strength = float(np.mean([p1_corner_rate, p2_corner_rate]))
        team2_corner_strength = float(np.mean([p3_corner_rate, p4_corner_rate]))

        rows.append(
            {
                # traceability
                "game_id": row.get("game_id", np.nan),
                "day": row.get("day", np.nan),
                "date": row.get("date", pd.NaT),
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "score1": int(row["score1"]),
                "score2": int(row["score2"]),
                "point_diff": int(row["score1"] - row["score2"]),

                # target
                "team1_actual_win": int(row["score1"] > row["score2"]),

                # Elo features
                "team1_elo_pre": team1_elo,
                "team2_elo_pre": team2_elo,
                "elo_diff_pre": elo_diff,
                "elo_abs_diff_pre": elo_abs_diff,
                "elo_expected_prob_pre": elo_expected_prob,
                "team1_max_elo_pre": team1_max_elo,
                "team2_max_elo_pre": team2_max_elo,
                "team1_min_elo_pre": team1_min_elo,
                "team2_min_elo_pre": team2_min_elo,
                "team1_balance_pre": team1_balance,
                "team2_balance_pre": team2_balance,
                "team1_elo_std_pre": team1_std,
                "team2_elo_std_pre": team2_std,
                "team1_peak_elo_pre": team1_peak_elo,
                "team2_peak_elo_pre": team2_peak_elo,
                "team1_peak_gap_pre": team1_peak_gap,
                "team2_peak_gap_pre": team2_peak_gap,
                "team1_ts_mu_pre": team1_ts_mu,
                "team2_ts_mu_pre": team2_ts_mu,
                "team1_ts_sigma_pre": team1_ts_sigma,
                "team2_ts_sigma_pre": team2_ts_sigma,
                "team1_ts_conservative_pre": team1_ts_conservative,
                "team2_ts_conservative_pre": team2_ts_conservative,
                "ts_mu_diff_pre": ts_mu_diff,
                "ts_sigma_diff_pre": ts_sigma_diff,
                "ts_cons_diff_pre": ts_cons_diff,

                "team1_specialization_pre": team1_specialization,
                "team2_specialization_pre": team2_specialization,
                "specialization_gap_pre": specialization_gap,
                "rating_disagreement_pre": rating_disagreement,

                "team1_elo_volatility_pre": team1_elo_volatility,
                "team2_elo_volatility_pre": team2_elo_volatility,
                
                "elo_confidence_gap_pre": elo_confidence_gap,
                "elo_confidence_gap_side_pre": elo_confidence_gap_side,
                "ts_confidence_gap_favorite_pre": ts_confidence_gap_favorite,

                "team1_ts_balance_pre": team1_ts_balance,
                "team2_ts_balance_pre": team2_ts_balance,
                "team1_ts_momentum_pre": team1_ts_momentum,
                "team2_ts_momentum_pre": team2_ts_momentum,
                

                "team1_opponent_experience_pre": team1_opponent_experience,
                "team2_opponent_experience_pre": team2_opponent_experience,
                "team1_opponent_win_rate_pre": team1_opponent_win_rate,
                "team2_opponent_win_rate_pre": team2_opponent_win_rate,

                # experience / form
                "team1_games_played_pre": team1_games_played,
                "team2_games_played_pre": team2_games_played,
                "team1_win_rate_pre": team1_win_rate,
                "team2_win_rate_pre": team2_win_rate,
                "team1_recent_win_rate_pre": team1_recent_win_rate,
                "team2_recent_win_rate_pre": team2_recent_win_rate,
                "team1_avg_point_diff_pre": team1_avg_point_diff,
                "team2_avg_point_diff_pre": team2_avg_point_diff,
                "team1_recent_point_diff_pre": team1_recent_point_diff,
                "team2_recent_point_diff_pre": team2_recent_point_diff,
                "team1_recent_elo_trend_pre": team1_recent_elo_trend,
                "team2_recent_elo_trend_pre": team2_recent_elo_trend,

                # chemistry
                "team1_pair_games_pre": float(pair_games[team1_pair]),
                "team2_pair_games_pre": float(pair_games[team2_pair]),
                "team1_pair_win_rate_pre": team1_pair_win_rate,
                "team2_pair_win_rate_pre": team2_pair_win_rate,
                "team1_pair_point_diff_pre": team1_pair_point_diff,
                "team2_pair_point_diff_pre": team2_pair_point_diff,

                # exact matchup familiarity
                "h2h_games_pre": float(h2h_games),
                "h2h_win_rate_pre": h2h_win_rate,
                "h2h_point_diff_pre": h2h_point_diff,

                # corner / position
                "p1_corner_win_rate_pre": p1_corner_rate,
                "p2_corner_win_rate_pre": p2_corner_rate,
                "p3_corner_win_rate_pre": p3_corner_rate,
                "p4_corner_win_rate_pre": p4_corner_rate,
                "team1_corner_strength_pre": team1_corner_strength,
                "team2_corner_strength_pre": team2_corner_strength,
                "corner_strength_diff_pre": team1_corner_strength - team2_corner_strength,
            }
        )

        # -----------------------------
        # Update replay state AFTER feature extraction
        # -----------------------------
        team1_wins = int(row["score1"] > row["score2"])
        team2_wins = 1 - team1_wins
        team1_diff = int(row["score1"] - row["score2"])
        team2_diff = -team1_diff

        mov_mult = margin_multiplier(int(row["score1"]), int(row["score2"]))
        delta = K_FACTOR * mov_mult * (team1_wins - elo_expected_prob)

        for p in team1:
            ratings[p] += delta / 2.0
        for p in team2:
            ratings[p] -= delta / 2.0


        # ----------------------------------------
        # Update TrueSkill
        # ----------------------------------------  

        if TRUESKILL_AVAILABLE:
            team1_ts = [ts_ratings[p] for p in team1]
            team2_ts = [ts_ratings[p] for p in team2]
            ranks = [0, 1] if team1_wins else [1, 0]
            new_t1, new_t2 = ts_env.rate([team1_ts, team2_ts], ranks=ranks)

            for p, r in zip(team1, new_t1):
                ts_ratings[p] = r
            for p, r in zip(team2, new_t2):
                ts_ratings[p] = r
        if TRUESKILL_AVAILABLE:
            for p in team1:
                player_ts_mu_history[p].append(ts_ratings[p].mu)
            for p in team2:
                player_ts_mu_history[p].append(ts_ratings[p].mu)
        # ----------------------------------------
        # Update career peak Elo
        # ----------------------------------------
        player_peak_elo[p1] = max(player_peak_elo[p1], ratings[p1])
        player_peak_elo[p2] = max(player_peak_elo[p2], ratings[p2])
        player_peak_elo[p3] = max(player_peak_elo[p3], ratings[p3])
        player_peak_elo[p4] = max(player_peak_elo[p4], ratings[p4])

         # player updates
        player_games[p1] += 1
        player_wins[p1] += team1_wins
        player_point_diff[p1] += team1_diff
        player_recent_results[p1].append(team1_wins)
        player_recent_point_diffs[p1].append(team1_diff)
        player_elo_history[p1].append(ratings[p1])

        player_games[p2] += 1
        player_wins[p2] += team1_wins
        player_point_diff[p2] += team1_diff
        player_recent_results[p2].append(team1_wins)
        player_recent_point_diffs[p2].append(team1_diff)
        player_elo_history[p2].append(ratings[p2])

        player_games[p3] += 1
        player_wins[p3] += team2_wins
        player_point_diff[p3] += team2_diff
        player_recent_results[p3].append(team2_wins)
        player_recent_point_diffs[p3].append(team2_diff)
        player_elo_history[p3].append(ratings[p3])

        player_games[p4] += 1
        player_wins[p4] += team2_wins
        player_point_diff[p4] += team2_diff
        player_recent_results[p4].append(team2_wins)
        player_recent_point_diffs[p4].append(team2_diff)
        player_elo_history[p4].append(ratings[p4])

        # true corner updates
        corner_games[p1][1] += 1
        corner_wins[p1][1] += team1_wins

        corner_games[p2][2] += 1
        corner_wins[p2][2] += team1_wins

        corner_games[p3][3] += 1
        corner_wins[p3][3] += team2_wins

        corner_games[p4][4] += 1
        corner_wins[p4][4] += team2_wins
        # teammate pair updates
        pair_games[team1_pair] += 1
        pair_wins[team1_pair] += team1_wins
        pair_point_diff[team1_pair] += team1_diff

        pair_games[team2_pair] += 1
        pair_wins[team2_pair] += team2_wins
        pair_point_diff[team2_pair] += team2_diff

        # ordered matchup updates
        matchup_games[matchup_key] += 1
        matchup_wins[matchup_key] += team1_wins
        matchup_point_diff[matchup_key] += team1_diff

        for a in team1:
            for b in team2:
                player_opponent_games[a][b] += 1
                player_opponent_wins[a][b] += team1_wins

                player_opponent_games[b][a] += 1
                player_opponent_wins[b][a] += team2_wins
                
        # Update teammate-history tables for specialization
        player_partner_games[p1][p2] += 1
        player_partner_wins[p1][p2] += team1_wins
        player_partner_games[p2][p1] += 1
        player_partner_wins[p2][p1] += team1_wins

        player_partner_games[p3][p4] += 1
        player_partner_wins[p3][p4] += team2_wins
        player_partner_games[p4][p3] += 1
        player_partner_wins[p4][p3] += team2_wins

    ml_df = pd.DataFrame(rows)
    if not ml_df.empty:
        ml_df = ml_df.sort_values("game_id").reset_index(drop=True)

    state = {
        "ratings": dict(ratings),
        "ts_ratings": dict(ts_ratings) if TRUESKILL_AVAILABLE else None,
        "player_games": dict(player_games),
        "player_wins": dict(player_wins),
        "player_point_diff": dict(player_point_diff),
        "player_recent_results": dict(player_recent_results),
        "player_recent_point_diffs": dict(player_recent_point_diffs),
        "player_elo_history": dict(player_elo_history),
        "player_peak_elo": dict(player_peak_elo),
        "player_partner_games": dict(player_partner_games),
        "player_partner_wins": dict(player_partner_wins),
        "player_opponent_games": dict(player_opponent_games),
        "player_opponent_wins": dict(player_opponent_wins),
        "pair_games": dict(pair_games),
        "pair_wins": dict(pair_wins),
        "pair_point_diff": dict(pair_point_diff),
        "matchup_games": dict(matchup_games),
        "matchup_wins": dict(matchup_wins),
        "matchup_point_diff": dict(matchup_point_diff),
        "corner_games": dict(corner_games),
        "corner_wins": dict(corner_wins),
        "player_ts_mu_history": dict(player_ts_mu_history),
    }
    return (ml_df, state) if return_state else ml_df

    return ml_df


from pathlib import Path

def write_feature_dictionary_text(outdir: Path) -> Path:
    """
    Write a text file describing the ML features, grouped by ablation category.
    """
    sections = [
        (
            "Metadata / labels (not model inputs)",
            [
                ("game_id", "Chronological game index."),
                ("day", "Day number from the log, if available."),
                ("date", "Game date, if available."),
                ("p1, p2, p3, p4", "Player IDs / corner assignments for the game."),
                ("score1, score2", "Final score for Team 1 and Team 2."),
                ("team1_actual_win", "Target label: 1 if Team 1 won, else 0."),
                ("point_diff", "Final point differential (score1 - score2)."),
            ],
        ),
        (
            "Elo / rating level",
            [
                ("team1_elo_pre", "Average pre-game Elo of Team 1."),
                ("team2_elo_pre", "Average pre-game Elo of Team 2."),
                ("elo_diff_pre", "Team 1 Elo minus Team 2 Elo."),
                ("elo_abs_diff_pre", "Absolute Elo difference between the teams."),
                ("elo_expected_prob_pre", "Pre-game Team 1 win probability from Elo."),
                ("team1_max_elo_pre", "Higher of the two Team 1 player Elos."),
                ("team2_max_elo_pre", "Higher of the two Team 2 player Elos."),
                ("team1_min_elo_pre", "Lower of the two Team 1 player Elos."),
                ("team2_min_elo_pre", "Lower of the two Team 2 player Elos."),
            ],
        ),
        (
            "Elo stability / confidence",
            [
                ("team1_balance_pre", "Difference between Team 1 player Elos (max - min)."),
                ("team2_balance_pre", "Difference between Team 2 player Elos (max - min)."),
                ("team1_elo_std_pre", "Standard deviation of Team 1 player Elos."),
                ("team2_elo_std_pre", "Standard deviation of Team 2 player Elos."),
                ("team1_peak_elo_pre", "Average of Team 1 players' historical peak Elos."),
                ("team2_peak_elo_pre", "Average of Team 2 players' historical peak Elos."),
                ("team1_peak_gap_pre", "Average distance between Team 1's current Elo and historical peak Elo."),
                ("team2_peak_gap_pre", "Average distance between Team 2's current Elo and historical peak Elo."),
                ("team1_elo_volatility_pre", "Average historical Elo volatility (rolling Elo standard deviation) of Team 1 players."),
                ("team2_elo_volatility_pre", "Average historical Elo volatility (rolling Elo standard deviation) of Team 2 players."),
                ("elo_confidence_gap_pre", "Favorite-versus-underdog Elo confidence gap. Positive values indicate the favorite remains stronger even after accounting for rating volatility."),
                ("elo_confidence_gap_side_pre", "Team-1 minus Team-2 Elo confidence gap. Retained mainly for comparison with earlier models."),
            ],
        ),
        (
            "TrueSkill / uncertainty",
            [
                ("team1_ts_mu_pre", "Average pre-game TrueSkill μ of Team 1."),
                ("team2_ts_mu_pre", "Average pre-game TrueSkill μ of Team 2."),
                ("team1_ts_sigma_pre", "Average pre-game TrueSkill σ of Team 1."),
                ("team2_ts_sigma_pre", "Average pre-game TrueSkill σ of Team 2."),
                ("team1_ts_conservative_pre", "Average conservative TrueSkill estimate (μ − 3σ) for Team 1."),
                ("team2_ts_conservative_pre", "Average conservative TrueSkill estimate (μ − 3σ) for Team 2."),
                ("ts_mu_diff_pre", "Difference in average TrueSkill μ between Team 1 and Team 2."),
                ("ts_sigma_diff_pre", "Difference in average TrueSkill σ between Team 1 and Team 2."),
                ("ts_cons_diff_pre", "Difference in conservative TrueSkill estimates between Team 1 and Team 2."),
                ("team1_ts_balance_pre", "Difference in TrueSkill μ between the two Team 1 players."),
                ("team2_ts_balance_pre", "Difference in TrueSkill μ between the two Team 2 players."),
                ("team1_ts_momentum_pre", "Average recent change in TrueSkill μ for Team 1."),
                ("team2_ts_momentum_pre", "Average recent change in TrueSkill μ for Team 2."),
                ("ts_confidence_gap_pre", "Side-dependent TrueSkill confidence gap (Team 1 vs Team 2)."),
                ("ts_confidence_gap_favorite_pre", "Favorite-versus-underdog TrueSkill confidence gap. Positive values indicate the favorite remains stronger after accounting for uncertainty."),
            ],
        ),
        (
            "Experience / recent form",
            [
                ("team1_games_played_pre", "Average games played by Team 1 players."),
                ("team2_games_played_pre", "Average games played by Team 2 players."),
                ("team1_win_rate_pre", "Average career win rate of Team 1 players."),
                ("team2_win_rate_pre", "Average career win rate of Team 2 players."),
                ("team1_recent_win_rate_pre", "Average recent win rate of Team 1 players."),
                ("team2_recent_win_rate_pre", "Average recent win rate of Team 2 players."),
                ("team1_avg_point_diff_pre", "Average career point differential of Team 1 players."),
                ("team2_avg_point_diff_pre", "Average career point differential of Team 2 players."),
                ("team1_recent_point_diff_pre", "Average recent point differential of Team 1 players."),
                ("team2_recent_point_diff_pre", "Average recent point differential of Team 2 players."),
                ("team1_recent_elo_trend_pre", "Recent change in Elo form for Team 1."),
                ("team2_recent_elo_trend_pre", "Recent change in Elo form for Team 2."),
            ],
        ),
        (
            "Chemistry / matchup familiarity",
            [
                ("team1_pair_games_pre", "Number of prior games played by Team 1 as a pair."),
                ("team2_pair_games_pre", "Number of prior games played by Team 2 as a pair."),
                ("team1_pair_win_rate_pre", "Historical win rate of the Team 1 pair."),
                ("team2_pair_win_rate_pre", "Historical win rate of the Team 2 pair."),
                ("team1_pair_point_diff_pre", "Historical point differential of the Team 1 pair."),
                ("team2_pair_point_diff_pre", "Historical point differential of the Team 2 pair."),
                ("h2h_games_pre", "Prior head-to-head games between these two pairs."),
                ("h2h_win_rate_pre", "Team 1 pair win rate in this exact matchup."),
                ("h2h_point_diff_pre", "Historical point differential in this exact matchup."),
                ("team1_opponent_experience_pre", "Average prior games Team 1 players have played against the current opponents."),
                ("team2_opponent_experience_pre", "Average prior games Team 2 players have played against the current opponents."),
                ("team1_opponent_win_rate_pre", "Average prior win rate of Team 1 players vs the current opponents."),
                ("team2_opponent_win_rate_pre", "Average prior win rate of Team 2 players vs the current opponents."),
                ("team1_specialization_pre", "Average teammate-dependence of Team 1 players, measured as the standard deviation of their historical pair win rates across partners."),
                ("team2_specialization_pre", "Average teammate-dependence of Team 2 players, measured as the standard deviation of their historical pair win rates across partners."),
                ("specialization_gap_pre", "Team 1 specialization minus Team 2 specialization."),
            ],
        ),
        (
            "Corner / position",
            [
                ("p1_corner_win_rate_pre", "Player in corner 1 historical win rate."),
                ("p2_corner_win_rate_pre", "Player in corner 2 historical win rate."),
                ("p3_corner_win_rate_pre", "Player in corner 3 historical win rate."),
                ("p4_corner_win_rate_pre", "Player in corner 4 historical win rate."),
                ("team1_corner_strength_pre", "Average corner strength of Team 1."),
                ("team2_corner_strength_pre", "Average corner strength of Team 2."),
                ("corner_strength_diff_pre", "Team 1 corner strength minus Team 2 corner strength."),
            ],
        ),
        (
            "Rating agreement / disagreement",
            [
                ("rating_disagreement_pre",
                 "Absolute difference between the standardized Elo matchup difference and standardized TrueSkill matchup difference. Large values indicate the two rating systems disagree about the teams."),
            ],
        ),
    ]

    lines = []
    lines.append("Ping Pong ML Feature Dictionary")
    lines.append("=" * 34)
    lines.append("")
    lines.append("All features below are computed using only pre-game information.")
    lines.append("")
    lines.append("")
    lines.append("Feature naming convention:")
    lines.append("  *_pre                = calculated before the game begins")
    lines.append("  *_diff              = Team 1 value minus Team 2 value")
    lines.append("  *_gap               = difference between related metrics")
    lines.append("  *_favorite          = calculated using the favorite/underdog ordering rather than Team 1/Team 2 ordering")
    lines.append("  *_volatility        = historical variation of a player's Elo")
    lines.append("  *_momentum          = recent change in rating")
    lines.append("")

    for section_title, items in sections:
        lines.append(section_title)
        lines.append("-" * len(section_title))
        for name, desc in items:
            lines.append(f"{name}: {desc}")
        lines.append("")

    path = outdir / "ml_feature_dictionary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train_gradient_boosted_model(
    ml_df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: int = 42,
):
    """
    Train a gradient boosted tree with a chronological split.
    Automatically uses all columns ending with '_pre' as features.
    """
    if ml_df.empty:
        raise ValueError("ml_df is empty.")

    df = ml_df.sort_values("game_id").reset_index(drop=True).copy()
    feature_cols = [c for c in df.columns if c.endswith("_pre")]

    if len(df) < 3:
        raise ValueError("Not enough games to train/test split.")

    split_idx = int(len(df) * (1.0 - test_frac))
    split_idx = max(1, min(split_idx, len(df) - 1))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    X_train = train[feature_cols].copy()
    y_train = train["team1_actual_win"].astype(int).copy()
    X_test = test[feature_cols].copy()
    y_test = test["team1_actual_win"].astype(int).copy()

    model = GradientBoostingClassifier(
        random_state=random_state,
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
    )
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "brier_score": float(brier_score_loss(y_test, test_prob)),
        "log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan,
    }
    metrics_df = pd.DataFrame([metrics])

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    predictions_df = test[[
        "game_id", "day", "date", "p1", "p2", "p3", "p4",
        "score1", "score2", "team1_actual_win"
    ]].copy()
    predictions_df["pred_team1_win_prob"] = test_prob
    predictions_df["pred_team1_win"] = test_pred
    predictions_df["correct"] = (predictions_df["pred_team1_win"] == predictions_df["team1_actual_win"]).astype(int)

    return model, feature_cols, metrics_df, importance_df, predictions_df, X_test


def plot_feature_importance(importance_df: pd.DataFrame, outdir: Path, top_n: int = 20) -> Optional[Path]:
    if importance_df.empty:
        return None

    top = importance_df.head(top_n).sort_values("importance", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Gradient Boosted Tree Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    path = outdir / "feature_importance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    outdir: Path,
    top_n: int = 20,
) -> Optional[Path]:
    if not SHAP_AVAILABLE or X_test.empty:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        max_display=min(top_n, X_test.shape[1]),
    )

    path = outdir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path

def build_correct_vs_incorrect_feature_summary(
    X_test: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare feature values on correctly predicted vs incorrectly predicted test games.

    Returns a table with:
      - correct_mean
      - incorrect_mean
      - raw_gap
      - standardized_gap

    standardized_gap = (correct_mean - incorrect_mean) / feature_std
    so different feature scales are comparable.
    """
    if X_test.empty or predictions_df.empty:
        return pd.DataFrame()

    x = X_test.reset_index(drop=True).copy()
    p = predictions_df.reset_index(drop=True).copy()

    if len(x) != len(p):
        raise ValueError("X_test and predictions_df must have the same number of rows.")

    if "correct" not in p.columns:
        raise ValueError("predictions_df must contain a 'correct' column.")

    x["correct"] = p["correct"].astype(int)

    rows = []
    for col in X_test.columns:
        vals = pd.to_numeric(x[col], errors="coerce")
        correct_vals = vals[x["correct"] == 1].dropna()
        incorrect_vals = vals[x["correct"] == 0].dropna()

        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            correct_mean = float(correct_vals.mean()) if len(correct_vals) else np.nan
            incorrect_mean = float(incorrect_vals.mean()) if len(incorrect_vals) else np.nan
        else:
            correct_mean = float(correct_vals.mean())
            incorrect_mean = float(incorrect_vals.mean())

        raw_gap = correct_mean - incorrect_mean
        feature_std = float(vals.std(ddof=0))
        standardized_gap = raw_gap / feature_std if feature_std and not np.isnan(feature_std) else np.nan

        rows.append(
            {
                "feature": col,
                "correct_mean": correct_mean,
                "incorrect_mean": incorrect_mean,
                "raw_gap": raw_gap,
                "feature_std": feature_std,
                "standardized_gap": standardized_gap,
                "abs_standardized_gap": abs(standardized_gap) if not np.isnan(standardized_gap) else np.nan,
                "n_correct": int((x["correct"] == 1).sum()),
                "n_incorrect": int((x["correct"] == 0).sum()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_standardized_gap", ascending=False).reset_index(drop=True)
    return out


def plot_correct_vs_incorrect_features(
    summary_df: pd.DataFrame,
    outdir: Path,
    top_n: int = 15,
) -> Optional[Path]:
    """
    Plot the features whose standardized means differ most between correct and incorrect predictions.

    Positive bars mean the feature tends to be larger when the model is correct.
    Negative bars mean the feature tends to be larger when the model is wrong.
    """
    if summary_df.empty:
        return None

    plot_df = summary_df.dropna(subset=["standardized_gap"]).head(top_n).copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values("standardized_gap", ascending=True).copy()
    colors = ["tab:green" if v >= 0 else "tab:red" for v in plot_df["standardized_gap"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["feature"], plot_df["standardized_gap"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Features Associated With Correct vs Incorrect Predictions")
    ax.set_xlabel("Standardized mean difference\n(correct - incorrect)")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    path = Path(outdir) / "correct_vs_incorrect_features.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def build_top_mispredictions_df(predictions_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return the most confident mistakes from the test set.
    """
    if predictions_df.empty:
        return pd.DataFrame()

    out = predictions_df.copy()
    out["abs_prob_error"] = (out["pred_team1_win_prob"] - out["team1_actual_win"]).abs()

    cols = [
        "game_id", "day", "date", "p1", "p2", "p3", "p4",
        "score1", "score2", "team1_actual_win",
        "pred_team1_win_prob", "pred_team1_win", "correct", "abs_prob_error",
    ]
    cols = [c for c in cols if c in out.columns]

    return out.sort_values("abs_prob_error", ascending=False).head(top_n)[cols].reset_index(drop=True)


def plot_model_performance_report(
    predictions_df: pd.DataFrame,
    outdir: Path,
) -> Optional[Path]:
    """
    Create a compact performance figure for the gradient-boosted model.

    Panels:
      1) Cumulative test accuracy over time
      2) Confusion matrix
      3) Calibration curve by probability bins
      4) Predicted probability histogram
    """
    if predictions_df.empty:
        return None

    df = predictions_df.sort_values("game_id").reset_index(drop=True).copy()
    df["correct"] = (df["pred_team1_win"] == df["team1_actual_win"]).astype(int)
    df["cumulative_accuracy"] = df["correct"].expanding().mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Cumulative accuracy
    ax = axes[0, 0]
    ax.plot(df["game_id"], df["cumulative_accuracy"], linewidth=2)
    ax.set_title("Cumulative Test Accuracy")
    ax.set_xlabel("Game")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    # 2) Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(df["team1_actual_win"], df["pred_team1_win"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Team 1 loss", "Team 1 win"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")

    # 3) Calibration by bins
    ax = axes[1, 0]
    bins = pd.cut(df["pred_team1_win_prob"], bins=np.linspace(0.0, 1.0, 6), include_lowest=True)
    cal = (
        df.assign(prob_bin=bins)
        .groupby("prob_bin", observed=False)
        .agg(
            mean_pred=("pred_team1_win_prob", "mean"),
            actual_rate=("team1_actual_win", "mean"),
            n=("game_id", "count"),
        )
        .reset_index()
        .dropna(subset=["mean_pred", "actual_rate"])
    )
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    if not cal.empty:
        ax.plot(cal["mean_pred"], cal["actual_rate"], marker="o")
        for _, r in cal.iterrows():
            ax.annotate(str(int(r["n"])), (r["mean_pred"], r["actual_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Calibration")
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 4) Probability histogram
    ax = axes[1, 1]
    ax.hist(df["pred_team1_win_prob"], bins=10)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Team 1 win probability")
    ax.set_ylabel("Count")

    fig.tight_layout()
    path = outdir / "model_performance_report.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _score_gboost_feature_set(
    ml_df: pd.DataFrame,
    feature_cols: List[str],
    split_idx: int,
    random_state: int = 42,
) -> dict:
    """
    Train/evaluate one gradient-boosted model on a fixed chronological split.
    Returns metrics only.
    """
    if not feature_cols:
        raise ValueError("feature_cols is empty.")

    df = ml_df.sort_values("game_id").reset_index(drop=True).copy()
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    X_train = train[feature_cols].copy()
    y_train = train["team1_actual_win"].astype(int).copy()
    X_test = test[feature_cols].copy()
    y_test = test["team1_actual_win"].astype(int).copy()

    model = GradientBoostingClassifier(
        random_state=random_state,
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
    )
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    return {
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "brier_score": float(brier_score_loss(y_test, test_prob)),
        "log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan,
    }


def run_ablation_study(
    ml_df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare baseline performance vs. models with one feature group removed.

    Positive accuracy_delta means removing that group improved accuracy.
    Positive brier_improvement means removing that group improved Brier score.
    """
    if ml_df.empty:
        raise ValueError("ml_df is empty.")

    df = ml_df.sort_values("game_id").reset_index(drop=True).copy()
    all_features = [c for c in df.columns if c.endswith("_pre")]

    if len(df) < 3:
        raise ValueError("Not enough games for ablation study.")

    split_idx = int(len(df) * (1.0 - test_frac))
    split_idx = max(1, min(split_idx, len(df) - 1))

    # Feature groups to remove one at a time
    feature_groups = [
        (
            "No Elo group",
            [
                "team1_elo_pre", "team2_elo_pre", "elo_diff_pre", "elo_abs_diff_pre",
                "elo_expected_prob_pre", "team1_max_elo_pre", "team2_max_elo_pre",
                "team1_min_elo_pre", "team2_min_elo_pre", "team1_balance_pre",
                "team2_balance_pre", "team1_elo_std_pre", "team2_elo_std_pre",
            ],
        ),
        (
            "No recent form",
            [
                "team1_recent_win_rate_pre", "team2_recent_win_rate_pre",
                "team1_recent_point_diff_pre", "team2_recent_point_diff_pre",
                "team1_recent_elo_trend_pre", "team2_recent_elo_trend_pre",
            ],
        ),
        (
            "No chemistry",
            [
                "team1_pair_games_pre", "team2_pair_games_pre",
                "team1_pair_win_rate_pre", "team2_pair_win_rate_pre",
                "team1_pair_point_diff_pre", "team2_pair_point_diff_pre",
                "h2h_games_pre", "h2h_win_rate_pre", "h2h_point_diff_pre",
            ],
        ),
        (
            "No experience",
            [
                "team1_games_played_pre", "team2_games_played_pre",
                "team1_win_rate_pre", "team2_win_rate_pre",
                "team1_avg_point_diff_pre", "team2_avg_point_diff_pre",
            ],
        ),
        (
            "No corner features",
            [
                "p1_corner_win_rate_pre", "p2_corner_win_rate_pre",
                "p3_corner_win_rate_pre", "p4_corner_win_rate_pre",
                "team1_corner_strength_pre", "team2_corner_strength_pre",
                "corner_strength_diff_pre",
            ],
        ),
    ]

    baseline_metrics = _score_gboost_feature_set(
        ml_df=df,
        feature_cols=all_features,
        split_idx=split_idx,
        random_state=random_state,
    )

    rows = [
        {
            "ablation": "Baseline (all features)",
            "removed_group": "None",
            "n_features": int(len(all_features)),
            **baseline_metrics,
            "accuracy_delta": 0.0,
            "brier_improvement": 0.0,
            "log_loss_improvement": 0.0,
            "roc_auc_delta": 0.0,
        }
    ]

    for group_name, drop_cols in feature_groups:
        drop_set = set(drop_cols)
        keep_cols = [c for c in all_features if c not in drop_set]

        if not keep_cols:
            continue

        metrics = _score_gboost_feature_set(
            ml_df=df,
            feature_cols=keep_cols,
            split_idx=split_idx,
            random_state=random_state,
        )

        rows.append(
            {
                "ablation": group_name,
                "removed_group": group_name,
                "n_features": int(len(keep_cols)),
                **metrics,
                "accuracy_delta": float(metrics["accuracy"] - baseline_metrics["accuracy"]),
                # positive means the ablated model improved on Brier (lower is better)
                "brier_improvement": float(baseline_metrics["brier_score"] - metrics["brier_score"]),
                "log_loss_improvement": float(baseline_metrics["log_loss"] - metrics["log_loss"]),
                "roc_auc_delta": float(metrics["roc_auc"] - baseline_metrics["roc_auc"]) if not np.isnan(metrics["roc_auc"]) and not np.isnan(baseline_metrics["roc_auc"]) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("accuracy_delta", ascending=False).reset_index(drop=True)
    return out


def plot_ablation_study(ablation_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    Visualize how much each feature group matters relative to the baseline.
    """
    if ablation_df.empty:
        return None

    plot_df = ablation_df[ablation_df["ablation"] != "Baseline (all features)"].copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values("accuracy_delta", ascending=True).copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Accuracy change vs baseline
    colors_acc = ["tab:green" if v >= 0 else "tab:red" for v in plot_df["accuracy_delta"]]
    axes[0].barh(plot_df["ablation"], plot_df["accuracy_delta"], color=colors_acc)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Ablation: Accuracy Change vs Baseline")
    axes[0].set_xlabel("Accuracy delta (removed model - baseline)")
    axes[0].set_ylabel("Feature group")

    # Brier improvement vs baseline
    plot_df_brier = plot_df.sort_values("brier_improvement", ascending=True).copy()
    colors_brier = ["tab:green" if v >= 0 else "tab:red" for v in plot_df_brier["brier_improvement"]]
    axes[1].barh(plot_df_brier["ablation"], plot_df_brier["brier_improvement"], color=colors_brier)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Ablation: Brier Improvement vs Baseline")
    axes[1].set_xlabel("Baseline Brier - removed Brier")
    axes[1].set_ylabel("Feature group")

    fig.tight_layout()
    path = outdir / "ablation_study.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run_time_series_cv(
    ml_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window cross validation for time-ordered sports data.

    Returns:
        cv_fold_df: one row per fold with metrics
        cv_summary_df: mean/std summary across folds
        cv_oof_df: out-of-fold predictions for all test folds
    """
    if ml_df.empty:
        raise ValueError("ml_df is empty.")

    df = ml_df.sort_values("game_id").reset_index(drop=True).copy()
    feature_cols = [c for c in df.columns if c.endswith("_pre")]

    if len(df) < n_splits + 1:
        raise ValueError("Not enough games for the requested number of CV folds.")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rows = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        X_train = train[feature_cols].copy()
        y_train = train["team1_actual_win"].astype(int).copy()
        X_test = test[feature_cols].copy()
        y_test = test["team1_actual_win"].astype(int).copy()

        model = GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        )
        model.fit(X_train, y_train)

        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        fold_rows.append(
            {
                "fold": fold,
                "train_games": int(len(train)),
                "test_games": int(len(test)),
                "accuracy": float(accuracy_score(y_test, test_pred)),
                "brier_score": float(brier_score_loss(y_test, test_prob)),
                "log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-6, 1 - 1e-6))),
                "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else np.nan,
            }
        )

        oof = test[[
            "game_id", "day", "date", "p1", "p2", "p3", "p4",
            "score1", "score2", "team1_actual_win"
        ]].copy()
        oof["fold"] = fold
        oof["pred_team1_win_prob"] = test_prob
        oof["pred_team1_win"] = test_pred
        oof["correct"] = (oof["pred_team1_win"] == oof["team1_actual_win"]).astype(int)
        oof_rows.append(oof)

    cv_fold_df = pd.DataFrame(fold_rows)
    cv_oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()

    if not cv_fold_df.empty:
        cv_summary_df = pd.DataFrame(
            [
                {
                    "metric": "accuracy",
                    "mean": float(cv_fold_df["accuracy"].mean()),
                    "std": float(cv_fold_df["accuracy"].std(ddof=0)),
                },
                {
                    "metric": "brier_score",
                    "mean": float(cv_fold_df["brier_score"].mean()),
                    "std": float(cv_fold_df["brier_score"].std(ddof=0)),
                },
                {
                    "metric": "log_loss",
                    "mean": float(cv_fold_df["log_loss"].mean()),
                    "std": float(cv_fold_df["log_loss"].std(ddof=0)),
                },
                {
                    "metric": "roc_auc",
                    "mean": float(cv_fold_df["roc_auc"].mean()),
                    "std": float(cv_fold_df["roc_auc"].std(ddof=0)),
                },
            ]
        )
    else:
        cv_summary_df = pd.DataFrame()

    return cv_fold_df, cv_summary_df, cv_oof_df


def plot_time_series_cv(cv_fold_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """
    Plot cross-validation performance by fold.
    """
    if cv_fold_df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        cv_fold_df["fold"],
        cv_fold_df["accuracy"],
        marker="o",
        linewidth=2,
        label="Accuracy",
    )
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(
        cv_fold_df["fold"],
        cv_fold_df["brier_score"],
        marker="s",
        color='red',
        linewidth=2,
        label="Brier Score",
    )
    ax2.set_ylabel("Brier Score")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title("Time-Series Cross-Validation Performance by Fold")
    fig.tight_layout()

    path = outdir / "time_series_cv_performance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

# ============================================================
# PLOTS
# ============================================================

def plot_player_elo(elo_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if elo_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(elo_df["player"], elo_df["elo"])
    ax.set_title("Player Elo Ratings")
    ax.set_xlabel("Player")
    ax.set_ylabel("Elo")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = outdir / "player_elo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pair_strength(pair_df: pd.DataFrame, outdir: Path, top_n: int = 10) -> Optional[Path]:
    if pair_df.empty:
        return None
    top = pair_df[pair_df["games"] >= MIN_PAIR_GAMES_FOR_DISPLAY].head(top_n).copy()
    if top.empty:
        top = pair_df.head(min(top_n, len(pair_df))).copy()
    top["pair"] = top["player_a"] + " + " + top["player_b"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["pair"], top["win_pct"])
    ax.set_title("Best Teammate Combinations")
    ax.set_xlabel("Win %")
    ax.set_ylabel("Pair")
    ax.invert_yaxis()
    fig.tight_layout()
    path = outdir / "best_partnerships.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_improvement_curves(
    improvement_df: pd.DataFrame,
    history_df: pd.DataFrame,
    outdir: Path,
    max_players: int = 8,
) -> Optional[Path]:
    if improvement_df.empty or history_df.empty:
        return None

    players = improvement_df["player"].value_counts().head(max_players).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))

    line_colors: Dict[str, str] = {}

    for player in players:
        g = improvement_df[improvement_df["player"] == player]
        (line,) = ax.plot(g["game_id"], g["elo_after_game"], marker="", label=player)
        line_colors[player] = line.get_color()

    leader_timeline = build_elo_leadership_timeline(history_df)

    # Stars mark the overall Elo leader after each game, even if that player
    # did not play in that game.
    for _, row in leader_timeline.iterrows():
        leader = row["leader_player"]
        ax.scatter(
            row["game_id"],
            row["leader_elo"],
            marker="*",
            s=160,
            facecolors="white",
            edgecolors=line_colors.get(leader, "black"),
            linewidths=1.6,
            zorder=20,
        )

    ax.set_title("Player Elo Over Time")
    ax.set_xlabel("Game")
    ax.set_ylabel("Elo rating")
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / "improvement_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_network(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if not NETWORKX_AVAILABLE:
        return None
    G = nx.Graph()
    players = pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K"))
    G.add_nodes_from(players)

    edge_counts = defaultdict(int)
    edge_wins = defaultdict(int)
    for _, row in df.iterrows():
        for team, win in [((row["p1"], row["p2"]), team1_won(row)), ((row["p3"], row["p4"]), not team1_won(row))]:
            e = pair_key(team[0], team[1])
            edge_counts[e] += 1
            edge_wins[e] += int(win)

    for e, n in edge_counts.items():
        G.add_edge(e[0], e[1], weight=n, win_pct=edge_wins[e] / n)

    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=42)
    widths = [G[u][v]["weight"] for u, v in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.65, ax=ax)
    ax.set_title("Partnership Network Graph")
    ax.axis("off")
    fig.tight_layout()
    path = outdir / "partnership_network.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def mask_lower_triangle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = np.tril(np.ones(out.shape, dtype=bool))
    return out.mask(mask)


def plot_teammate_winpct_heatmap(winpct_matrix: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if winpct_matrix.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.ma.masked_invalid(mask_lower_triangle(winpct_matrix).to_numpy(dtype=float))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(winpct_matrix.columns)))
    ax.set_yticks(range(len(winpct_matrix.index)))
    ax.set_xticklabels(winpct_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(winpct_matrix.index)
    ax.set_title("Teammate Win Percentage Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Win %")
    fig.tight_layout()
    path = outdir / "teammate_winpct_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_prediction_accuracy(prediction_df, outdir):

    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.plot(
        prediction_df["game_id"],
        prediction_df["rolling_brier"],
        linewidth=3,
        label="Rolling Brier Score",
    )

    ax1.set_ylabel("Brier Score")
    ax1.set_xlabel("Game")
    ax1.set_ylim(0,0.35)

    ax2 = ax1.twinx()

    ax2.plot(
        prediction_df["game_id"],
        prediction_df["rolling_accuracy"],
        "--",
        linewidth=2,
        label="Rolling Accuracy",
    )

    ax2.set_ylabel("Prediction Accuracy")
    ax2.set_ylim(0,1)
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
    )

    fig.suptitle("Prediction Quality Over Time")

    fig.tight_layout()

    path = outdir/"prediction_quality.png"

    fig.savefig(path,dpi=250,bbox_inches="tight")
    plt.close(fig)

    return path

def plot_combined_accuracy_comparison(
    elo_prediction_df: pd.DataFrame,
    trueskill_prediction_df: pd.DataFrame,
    gboost_predictions_df: pd.DataFrame,
    outdir: Path,
) -> Optional[Path]:
    """
    Combine Elo, TrueSkill, and Gradient Boosted Tree accuracy curves on one plot.

    Elo / TrueSkill:
        uses rolling_accuracy over the full game history.

    Gradient Boosted Tree:
        uses cumulative accuracy over the held-out test set.
    """
    if elo_prediction_df.empty and trueskill_prediction_df.empty and gboost_predictions_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))

    # Elo
    if not elo_prediction_df.empty and "rolling_accuracy" in elo_prediction_df.columns:
        ax.plot(
            elo_prediction_df["game_id"],
            elo_prediction_df["rolling_accuracy"],
            linewidth=2.5,
            label="Elo rolling accuracy",
        )

    # TrueSkill
    if not trueskill_prediction_df.empty and "rolling_accuracy" in trueskill_prediction_df.columns:
        ax.plot(
            trueskill_prediction_df["game_id"],
            trueskill_prediction_df["rolling_accuracy"],
            linewidth=2.5,
            label="TrueSkill rolling accuracy",
        )

    # Gradient Boosted Tree
    if not gboost_predictions_df.empty and "correct" in gboost_predictions_df.columns:
        gdf = gboost_predictions_df.sort_values("game_id").reset_index(drop=True).copy()
        gdf["cumulative_accuracy"] = gdf["correct"].expanding().mean()

        ax.plot(
            gdf["game_id"],
            gdf["cumulative_accuracy"],
            linewidth=2.5,
            label="GBoost cumulative accuracy",
        )

    ax.set_title("Accuracy Comparison Over Time")
    ax.set_xlabel("Game")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / "combined_accuracy_comparison.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


def ordered_game_key_string(p1: str, p2: str, p3: str, p4: str) -> str:
    return f"{p1}|{p2}|{p3}|{p4}"


def unique_game_key_string(p1: str, p2: str, p3: str, p4: str) -> str:
    t1 = ",".join(sorted((str(p1), str(p2))))
    t2 = ",".join(sorted((str(p3), str(p4))))
    return "||".join(sorted((t1, t2)))


def build_corner_game_explorer_df(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
    trueskill_df: Optional[pd.DataFrame] = None,
    model=None,
    feature_cols: Optional[list[str]] = None,
    ml_state: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build the full corner-dependent game explorer table.

    Each row is one ordered state (p1,p2 vs p3,p4).
    Played status is tracked two ways:
      - played_ordered: exact corner order has appeared in the data
      - played_unique: matchup has appeared in any orientation
    """
    if df is None or df.empty:
        return pd.DataFrame()

    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()} if elo_df is not None and not elo_df.empty else {}

    played_ordered_keys = {
        ordered_game_key_string(row["p1"], row["p2"], row["p3"], row["p4"])
        for _, row in df.iterrows()
    }
    played_unique_keys = {
        unique_game_key_string(row["p1"], row["p2"], row["p3"], row["p4"])
        for _, row in df.iterrows()
    }

    ts_lookup = {}
    if trueskill_df is not None and not trueskill_df.empty:
        ts_lookup = trueskill_df.set_index("player")[["trueskill_mu", "trueskill_sigma"]].to_dict("index")

    rows = []
    env = trueskill.TrueSkill(draw_probability=0.0) if TRUESKILL_AVAILABLE else None

    for p1, p2, p3, p4 in permutations(players, 4):
        ordered_key = ordered_game_key_string(p1, p2, p3, p4)
        unique_key = unique_game_key_string(p1, p2, p3, p4)

        t1_elo = float(np.mean([ratings.get(p1, DEFAULT_ELO), ratings.get(p2, DEFAULT_ELO)]))
        t2_elo = float(np.mean([ratings.get(p3, DEFAULT_ELO), ratings.get(p4, DEFAULT_ELO)]))
        elo_prob = float(elo_expected_score(t1_elo, t2_elo))

        ts_prob = np.nan
        if TRUESKILL_AVAILABLE and ts_lookup:
            t1 = [
                trueskill.Rating(mu=ts_lookup[p1]["trueskill_mu"], sigma=ts_lookup[p1]["trueskill_sigma"]),
                trueskill.Rating(mu=ts_lookup[p2]["trueskill_mu"], sigma=ts_lookup[p2]["trueskill_sigma"]),
            ]
            t2 = [
                trueskill.Rating(mu=ts_lookup[p3]["trueskill_mu"], sigma=ts_lookup[p3]["trueskill_sigma"]),
                trueskill.Rating(mu=ts_lookup[p4]["trueskill_mu"], sigma=ts_lookup[p4]["trueskill_sigma"]),
            ]
            ts_prob = float(trueskill_expected_win_prob(t1, t2, env.beta))

        tree_prob = np.nan
        if model is not None and feature_cols is not None and ml_state is not None:
            feature_row = build_tree_matchup_feature_row(p1, p2, p3, p4, ml_state)
            tree_prob = float(tree_predict_prob(model, feature_row, feature_cols))

        sort_prob = tree_prob if np.isfinite(tree_prob) else (ts_prob if np.isfinite(ts_prob) else elo_prob)

        rows.append(
            {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "players": [p1, p2, p3, p4],
                "team1": f"{p1} + {p2}",
                "team2": f"{p3} + {p4}",
                "ordered_key": ordered_key,
                "unique_key": unique_key,
                "played_ordered": ordered_key in played_ordered_keys,
                "played_unique": unique_key in played_unique_keys,
                "elo_pred_team1_win_prob": elo_prob,
                "trueskill_pred_team1_win_prob": ts_prob,
                "tree_pred_team1_win_prob": tree_prob,
                "sort_prob": sort_prob,
                "closeness": abs(sort_prob - 0.5),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["closeness", "sort_prob", "unique_key"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def build_game_explorer_widget_html(unplayed_games_df: pd.DataFrame, players: list[str]) -> str:
    unplayed_games_json = json.dumps(unplayed_games_df.to_dict(orient="records"))
    players_json = json.dumps(players)

    return f"""
    <div style="border:1px solid #ddd; border-radius:12px; padding:14px; margin-bottom:18px;">
      <div style="display:flex; flex-wrap:wrap; gap:18px; margin-bottom:12px; align-items:flex-start;">
        <div>
          <div style="font-weight:bold; margin-bottom:6px;">Game status</div>
          <label style="margin-right:12px;">
            <input type="radio" name="game-status" value="all">
            All
          </label>
          <label style="margin-right:12px;">
            <input type="radio" name="game-status" value="played">
            Played
          </label>
          <label>
            <input type="radio" name="game-status" value="unplayed" checked>
            Unplayed
          </label>
        </div>

        <div>
          <div style="font-weight:bold; margin-bottom:6px;">Representation</div>
          <label style="margin-right:12px;">
            <input type="radio" name="game-representation" value="ordered" checked>
            Ordered (1680)
          </label>
          <label>
            <input type="radio" name="game-representation" value="unique">
            Unique (210)
          </label>
        </div>
      </div>

      <div id="player-checkboxes" style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px;"></div>

      <button id="show-games-btn" style="padding:8px 14px; margin-right:8px;">Show 10 Games</button>
      <button id="clear-games-btn" style="padding:8px 14px;">Clear</button>

      <p style="margin-top:10px; color:#666;">
        Select one or more players. The list will show games that include the selected players.
      </p>

      <div id="games-results" style="margin-top:12px;"></div>
    </div>

    <script>
    const unplayedGames = {unplayed_games_json};
    const allPlayers = {players_json};

    function makeCheckboxes() {{
        const container = document.getElementById("player-checkboxes");
        container.innerHTML = "";
        allPlayers.forEach((player) => {{
            const label = document.createElement("label");
            label.style.display = "inline-flex";
            label.style.alignItems = "center";
            label.style.gap = "6px";
            label.style.border = "1px solid #ccc";
            label.style.borderRadius = "999px";
            label.style.padding = "6px 10px";
            label.style.cursor = "pointer";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = player;
            checkbox.className = "pp-player";

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(player));
            container.appendChild(label);
        }});
    }}

    function selectedPlayers() {{
        return Array.from(document.querySelectorAll(".pp-player:checked")).map(el => el.value);
    }}

    function fmtProb(v) {{
        return Number.isFinite(v) ? `${{(v * 100).toFixed(1)}}%` : "-";
    }}

    function getRadioValue(name, fallback) {{
        const el = document.querySelector(`input[name="${{name}}"]:checked`);
        return el ? el.value : fallback;
    }}

    function canonicalUniqueLabel(g) {{
        const t1 = [g.p1, g.p2].slice().sort().join(" + ");
        const t2 = [g.p3, g.p4].slice().sort().join(" + ");
        return [t1, t2].sort().join(" vs ");
    }}

    function toNumberOrZero(v) {{
        const x = Number(v);
        return Number.isFinite(x) ? x : 0.0;
    }}

    function groupGamesIfNeeded(games, representation) {{
        if (representation !== "unique") {{
            return games.slice();
        }}

        const grouped = new Map();

        games.forEach((g) => {{
            const key = canonicalUniqueLabel(g);
            const row = grouped.get(key);

            if (!row) {{
                grouped.set(key, {{
                    ...g,
                    team1: key.split(" vs ")[0],
                    team2: key.split(" vs ")[1],
                    _count: 1,
                    _elo_sum: toNumberOrZero(g.elo_pred_team1_win_prob),
                    _ts_sum: toNumberOrZero(g.trueskill_pred_team1_win_prob),
                    _tree_sum: toNumberOrZero(g.tree_pred_team1_win_prob),
                    _sort_sum: toNumberOrZero(g.sort_prob),
                    _played_ordered: !!g.played_ordered,
                    _played_unique: !!g.played_unique,
                }});
            }} else {{
                row._count += 1;
                row._elo_sum += toNumberOrZero(g.elo_pred_team1_win_prob);
                row._ts_sum += toNumberOrZero(g.trueskill_pred_team1_win_prob);
                row._tree_sum += toNumberOrZero(g.tree_pred_team1_win_prob);
                row._sort_sum += toNumberOrZero(g.sort_prob);
                row._played_ordered = row._played_ordered || !!g.played_ordered;
                row._played_unique = row._played_unique || !!g.played_unique;
            }}
        }});

        return Array.from(grouped.values()).map((g) => {{
            const count = g._count || 1;
            const sortProb = g._sort_sum / count;
            return {{
                ...g,
                elo_pred_team1_win_prob: g._elo_sum / count,
                trueskill_pred_team1_win_prob: g._ts_sum / count,
                tree_pred_team1_win_prob: g._tree_sum / count,
                sort_prob: sortProb,
                closeness: Math.abs(sortProb - 0.5),
            }};
        }});
    }}

    function renderGames(games, representation) {{
        const out = document.getElementById("games-results");

        if (!games.length) {{
            out.innerHTML = "<p>No matching games found.</p>";
            return;
        }}

        const labelTitle = representation === "unique" ? "Unique Game" : "Ordered Game";

        let html = `
          <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <thead>
              <tr>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;">Team 1</th>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;">Team 2</th>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;"># Selected</th>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;">Elo</th>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;">TrueSkill</th>
                <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px;">Tree</th>
              </tr>
            </thead>
            <tbody>
        `;

        games.forEach((g) => {{
            html += `
              <tr>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{g.team1}}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{g.team2}}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{g.matchCount}}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{fmtProb(g.elo_pred_team1_win_prob)}}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{fmtProb(g.trueskill_pred_team1_win_prob)}}</td>
                <td style="padding:8px; border-bottom:1px solid #eee;">${{fmtProb(g.tree_pred_team1_win_prob)}}</td>
              </tr>
            `;
        }});

        html += `
            </tbody>
          </table>
        `;

        out.innerHTML = `
          <div style="font-size:12px; color:#666; margin-bottom:6px;">
            Showing ${{games.length}} ${{labelTitle}}s
          </div>
          ${{html}}
        `;
    }}

    function showGames() {{
        const selected = selectedPlayers();
        const statusMode = getRadioValue("game-status", "unplayed");
        const representation = getRadioValue("game-representation", "ordered");
        const statusKey = representation === "unique" ? "played_unique" : "played_ordered";

        let filtered = groupGamesIfNeeded(unplayedGames, representation).map((g) => {{
            const matches = selected.filter((p) => g.players.includes(p)).length;
            return {{
                ...g,
                matchCount: matches,
            }};
        }});

        if (statusMode === "played") {{
            filtered = filtered.filter((g) => !!g[statusKey]);
        }} else if (statusMode === "unplayed") {{
            filtered = filtered.filter((g) => !g[statusKey]);
        }}

        if (selected.length === 0) {{
            filtered.sort(
                (a, b) =>
                    Math.abs(a.sort_prob - 0.5) - Math.abs(b.sort_prob - 0.5) ||
                    a.sort_prob - b.sort_prob
            );
            renderGames(filtered.slice(0, 10), representation);
            return;
        }}

        filtered = filtered.filter((g) => g.matchCount > 0);

        const picked = [];
        const covered = new Set();

        while (picked.length < 10 && filtered.length > 0) {{
            let bestIdx = 0;
            let bestScore = -Infinity;

            for (let i = 0; i < filtered.length; i++) {{
                const g = filtered[i];

                const newCoverage = g.players.filter(
                    (p) => selected.includes(p) && !covered.has(p)
                ).length;

                const closeness = Math.abs(g.sort_prob - 0.5);

                const score =
                    newCoverage * 100 +
                    g.matchCount * 10 -
                    closeness * 10;

                if (score > bestScore) {{
                    bestScore = score;
                    bestIdx = i;
                }}
            }}

            const best = filtered.splice(bestIdx, 1)[0];
            picked.push(best);

            best.players.forEach((p) => {{
                if (selected.includes(p)) {{
                    covered.add(p);
                }}
            }});
        }}

        renderGames(picked, representation);
    }}

    function clearSelection() {{
        document.querySelectorAll(".pp-player").forEach((el) => {{
            el.checked = false;
        }});
        document.querySelector('input[name="game-status"][value="unplayed"]').checked = true;
        document.querySelector('input[name="game-representation"][value="ordered"]').checked = true;
        showGames();
    }}

    makeCheckboxes();
    document.getElementById("show-games-btn").addEventListener("click", showGames);
    document.getElementById("clear-games-btn").addEventListener("click", clearSelection);
    document.querySelectorAll('input[name="game-status"]').forEach((el) => {{
        el.addEventListener("change", showGames);
    }});
    document.querySelectorAll('input[name="game-representation"]').forEach((el) => {{
        el.addEventListener("change", showGames);
    }});
    document.querySelectorAll(".pp-player").forEach((el) => {{
        el.addEventListener("change", showGames);
    }});

    showGames();
    </script>
    """

def build_dashboard_html(
    outdir: Path,
    elo_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    corner_df: pd.DataFrame,
    player_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    side_df: pd.DataFrame,
    teammate_matrix: pd.DataFrame,
    teammate_winpct_matrix: pd.DataFrame,
    history_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    matchup_heatmap_df: pd.DataFrame,
    corner_heatmap_path: pd.DataFrame,
    unplayed_games_df: pd.DataFrame,
    corner_heatmap_png: Optional[Path],
    corner_heatmap_prob_html: Optional[str],
    corner_heatmap_count_html: Optional[str],
    completion_path: Optional[Path],
    synergy_path: Optional[Path],
    highlights_path: Optional[Path],
    leadership_path: Optional[Path] = None,
    leadership_df: Optional[Path] = None,
    prediction_plot: Optional[Path] = None,
    network_path: Optional[Path] = None,
    feature_importance_path: Optional[Path] = None,
    shap_summary_path: Optional[Path] = None,
    gboost_metrics_df: Optional[pd.DataFrame] = None,
    gboost_predictions_df: Optional[pd.DataFrame] = None,
    model_performance_path: Optional[Path] = None,
    correct_vs_incorrect_df: Optional[pd.DataFrame] = None,
    correct_vs_incorrect_path: Optional[Path] = None,
    ablation_df: Optional[pd.DataFrame] = None,
    ablation_plot_path: Optional[Path] = None,
    cv_fold_df: Optional[pd.DataFrame] = None,
    cv_summary_df: Optional[pd.DataFrame] = None,
    cv_plot_path: Optional[Path] = None,
    trueskill_df: Optional[pd.DataFrame] = None,
    trueskill_history_df: Optional[pd.DataFrame] = None,
    trueskill_over_time_path: Optional[Path] = None,
    trueskill_corner_heatmap_prob_html: Optional[str] = None,
    trueskill_leadership_df: Optional[pd.DataFrame] = None,
    trueskill_prediction_plot: Optional[Path] = None,
    combined_accuracy_plot: Optional[Path] = None,
    tree_probability_space_html: Optional[str] = None,
    tree_corner_heatmap_prob_html: Optional[str] = None,
    trueskill_synergy_path: Optional[Path] = None,
    trueskill_hist_path: Optional[Path] = None,
    tree_synergy_path: Optional[Path] = None,
    tree_hist_path: Optional[Path] = None,
    elo_prob_hist_path: Optional[Path] = None,
    trueskill_prob_hist_path: Optional[Path] = None,
    tree_prob_hist_path: Optional[Path] = None,
    tree_count_heatmap_html: Optional[str] = None,
    comparison_corner_heatmap_prob_html: Optional[str] = None,
):
    if not PLOTLY_AVAILABLE:
        return None

    sections: List[str] = []

    def add_section_title(title: str):
        sections.append(f"""
        <div style="margin-top:40px; margin-bottom:18px;">
            <h1 style="
                font-size:42px;
                margin:0;
                padding:10px 0;
                border-bottom:3px solid #222;
            ">{title}</h1>
        </div>
        """)

    def add_table(df: pd.DataFrame, title: str):
        if df is None or df.empty:
            sections.append(f"<h2>{title}</h2><p>No data available.</p>")
        else:
            sections.append(f"<h2>{title}</h2>" + df.copy().fillna("-").to_html(index=False, border=0))

    def add_html(title: str, html: str):
        sections.append(f"<h2>{title}</h2>{html}")

    def save_plot_html(filename: str, html_fragment: Optional[str]) -> Optional[Path]:
        if not html_fragment:
            return None

        path = outdir / filename
        full_html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            {html_fragment}
        </body>
        </html>
        """
        path.write_text(full_html, encoding="utf-8")
        return path

    def add_three_way_plot_row(
        title: str,
        left_title: str, left_path: Optional[Path],
        middle_title: str, middle_path: Optional[Path],
        right_title: str, right_path: Optional[Path],
    ):
        def iframe_for(path: Optional[Path]) -> str:
            if not path:
                return "<p>No data available.</p>"
            return f"""
            <iframe
                src="{path.name}"
                style="width:100%; height:1800px; border:none; display:block;"
                scrolling="no">
            </iframe>
            """

        sections.append(f"""
        <h2 style="font-size:28px; margin-top:24px; margin-bottom:14px;">{title}</h2>
        <div style="
            display: flex;
            gap: 18px;
            align-items: flex-start;
            width: 250%;
            margin-bottom: 28px;
            overflow-x: auto;
        ">
          <div style="flex: 1 1 0; min-width: 1800; border: 1px solid #ddd; border-radius: 12px; padding: 10px;">
            <h3 style="margin-top: 0; text-align: center; font-size: 32px;">{left_title}</h3>
            {iframe_for(left_path)}
          </div>
          <div style="flex: 1 1 0; min-width: 1800; border: 1px solid #ddd; border-radius: 12px; padding: 10px;">
            <h3 style="margin-top: 0; text-align: center; font-size: 32px;">{middle_title}</h3>
            {iframe_for(middle_path)}
          </div>
          <div style="flex: 1 1 0; min-width: 1800; border: 1px solid #ddd; border-radius: 12px; padding: 10px;">
            <h3 style="margin-top: 0; text-align: center; font-size: 32px;">{right_title}</h3>
            {iframe_for(right_path)}
          </div>
        </div>
        """)

    # ============================================================
    # TOOLS
    # ============================================================
    add_section_title("Tools")

    sections.append(build_game_explorer_widget_html(unplayed_games_df, list(teammate_matrix.index)))

    # ============================================================
    # GENERAL STATS
    # ============================================================
    add_section_title("General Stats")

    players = sorted(set(teammate_matrix.index).union(teammate_matrix.columns))

    if completion_path and completion_path.exists():
        sections.append(
            f"<h2>Dataset Completion</h2>"
            f"<img src='{completion_path.name}' style='width:100%;max-width:1800px;'>"
        )

    add_table(overall_df.head(10), "Overall Ranking")
    add_table(player_df.head(10), "Top Player Summary")
    add_table(corner_df, "Corner Analysis")
    add_table(side_df, "Position / Side Advantage")
    add_table(pair_df.head(10), "Top Partnerships")
    if not pair_df.empty:
        top_pairs = pair_df[pair_df["games"] >= MIN_PAIR_GAMES_FOR_DISPLAY].head(12).copy()
        if top_pairs.empty:
            top_pairs = pair_df.head(min(12, len(pair_df))).copy()
        top_pairs["pair"] = top_pairs["player_a"] + " + " + top_pairs["player_b"]
        add_html(
            "Best Teammate Combinations",
            px.bar(top_pairs, x="pair", y="win_pct").to_html(full_html=False, include_plotlyjs=False),
        )

    

    # Teammate game-count matrix and heatmap
    teammate_games_matrix = pd.DataFrame(np.nan, index=players, columns=players)
    for _, row in pair_df.iterrows():
        a = row["player_a"]
        b = row["player_b"]
        if a in teammate_games_matrix.index and b in teammate_games_matrix.columns:
            teammate_games_matrix.loc[a, b] = row["games"]
            teammate_games_matrix.loc[b, a] = row["games"]

    tm_games_display = mask_lower_triangle(teammate_games_matrix).fillna("-")
    tm_games_display.index.name = "Player"
    add_table(tm_games_display.reset_index().rename(columns={"index": "Player"}), "Teammate Matrix (Games Together)")
    if not teammate_games_matrix.empty:
        add_html(
            "Teammate Matrix (Games Together)",
            px.imshow(
                mask_lower_triangle(teammate_games_matrix),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Teammate Matrix (Games Together)",
            ).to_html(full_html=False, include_plotlyjs=False),
        )

    if network_path and network_path.exists():
        sections.append(f"<h2>Partnership Network Graph</h2><img src='{network_path.name}' style='max-width:100%;height:auto;'>")
    
    # Teammate win percentage matrix and heatmap
    tm_winpct_display = mask_lower_triangle(teammate_winpct_matrix).fillna("-")
    tm_winpct_display.index.name = "Player"
    add_table(tm_winpct_display.reset_index().rename(columns={"index": "Player"}), "Teammate Matrix (Win %)")
    if not teammate_winpct_matrix.empty:
        add_html(
            "Teammate Matrix (Win %)",
            px.imshow(
                mask_lower_triangle(teammate_winpct_matrix),
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Teammate Matrix (Win %)",
            ).to_html(full_html=False, include_plotlyjs=False),
        )


    sections.append(f"<h2>Game Highlights</h2><img src='{highlights_path.name}' style='max-width:100%;height:auto;'>")

    # ============================================================
    # ELO MODEL
    # ============================================================
    add_section_title("ELO Model")

    add_table(elo_df.sort_values("elo", ascending=False).reset_index(drop=True), "Elo Ratings")

    if not improvement_df.empty:
        fig = px.line(
            improvement_df,
            x="game_id",
            y="elo_after_game",
            color="player",
        )

        color_map = {trace.name: trace.line.color for trace in fig.data}
        leader_df = build_elo_leadership_timeline(history_df)

        if not leader_df.empty:
            for player, grp in leader_df.groupby("leader_player"):
                player_color = color_map.get(player, "#444444")
                fig.add_scatter(
                    x=grp["game_id"],
                    y=grp["leader_elo"],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=15,
                        color="white",
                        line=dict(color=player_color, width=2),
                    ),
                    name=f"{player} leader",
                    showlegend=False,
                    hovertemplate=(
                        f"Leader: {player}<br>"
                        "Game %{x}<br>"
                        "Elo %{y:.1f}<extra></extra>"
                    ),
                )

        add_html(
            "Player Elo Over Time",
            fig.to_html(full_html=False, include_plotlyjs=False),
        )

    add_table(leadership_df.head(10), "Elo Leadership Summary")

    if corner_heatmap_prob_html:
        sections.append("<h2>Elo Corner Matchup Win Probability</h2>")
        sections.append(corner_heatmap_prob_html)

    if corner_heatmap_count_html:
        sections.append("<h2>Elo Corner Matchup Games Played</h2>")
        sections.append(corner_heatmap_count_html)

    if elo_prob_hist_path and elo_prob_hist_path.exists():
        sections.append(
            f"<h2>Elo Probability Distribution</h2>"
            f"<img src='{elo_prob_hist_path.name}' style='max-width:100%;height:auto;'>"
        )

    sections.append(f"<h2>Pair Synergy</h2><img src='{synergy_path.name}' style='max-width:100%;height:auto;'>")

    if prediction_plot and prediction_plot.exists():
        sections.append(
            f"<h2>Prediction Quality Over Time</h2>"
            f"<img src='{prediction_plot.name}' style='max-width:100%;height:auto;'>"
        )

    # ============================================================
    # TRUESKILL MODEL
    # ============================================================
    add_section_title("TrueSkill Model")

    if trueskill_df is not None and not trueskill_df.empty:
        add_table(trueskill_df.head(10), "TrueSkill Ratings")

    if trueskill_over_time_path and trueskill_over_time_path.exists():
        sections.append(
            f"<h2>TrueSkill Over Time</h2>"
            f"<img src='{trueskill_over_time_path.name}' style='max-width:100%;height:auto;'>"
        )

    if trueskill_leadership_df is not None and not trueskill_leadership_df.empty:
        add_table(trueskill_leadership_df.head(10), "TrueSkill Leadership Summary")

    if trueskill_corner_heatmap_prob_html:
        sections.append("<h2>TrueSkill Corner Matchup Win Probability</h2>")
        sections.append(trueskill_corner_heatmap_prob_html)

    if trueskill_prediction_plot and trueskill_prediction_plot.exists():
        sections.append(
            f"<h2>TrueSkill Prediction Quality Over Time</h2>"
            f"<img src='{trueskill_prediction_plot.name}' style='max-width:100%;height:auto;'>"
        )

    if trueskill_synergy_path and trueskill_synergy_path.exists():
        sections.append(
            f"<h2>TrueSkill Pair Synergy</h2>"
            f"<img src='{trueskill_synergy_path.name}' style='max-width:100%;height:auto;'>"
        )


    if trueskill_prob_hist_path and trueskill_prob_hist_path.exists():
        sections.append(
            f"<h2>TrueSkill Full Probability Distribution</h2>"
            f"<img src='{trueskill_prob_hist_path.name}' style='max-width:100%;height:auto;'>"
        )

    # ============================================================
    # MACHINE LEARNING GRADIENT BOOSTED TREE MODEL
    # ============================================================
    add_section_title("Machine Learning Gradient Boosted Tree Model")

    if feature_importance_path and feature_importance_path.exists():
        sections.append(
            f"<h2>Feature Importance</h2>"
            f"<img src='{feature_importance_path.name}' style='max-width:100%;height:auto;'>"
        )

    if shap_summary_path and shap_summary_path.exists():
        sections.append(
            f"<h2>SHAP Summary</h2>"
            f"<img src='{shap_summary_path.name}' style='max-width:100%;height:auto;'>"
        )


    if correct_vs_incorrect_df is not None and not correct_vs_incorrect_df.empty:
        add_table(correct_vs_incorrect_df.head(15), "Correct vs Incorrect Feature Summary")

    if correct_vs_incorrect_path and correct_vs_incorrect_path.exists():
        sections.append(
            f"<h2>Correct vs Incorrect Features</h2>"
            f"<img src='{correct_vs_incorrect_path.name}' style='max-width:100%;height:auto;'>"
        )

    if gboost_metrics_df is not None and not gboost_metrics_df.empty:
        add_table(gboost_metrics_df, "Gradient Boosted Tree Metrics")

    if model_performance_path and model_performance_path.exists():
        sections.append(
            f"<h2>Gradient Boosted Tree Performance</h2>"
            f"<img src='{model_performance_path.name}' style='max-width:100%;height:auto;'>"
        )

    if gboost_predictions_df is not None and not gboost_predictions_df.empty:
        top_misses = build_top_mispredictions_df(gboost_predictions_df, top_n=10)
        add_table(top_misses, "Top Mispredictions")


    if ablation_df is not None and not ablation_df.empty:
        add_table(ablation_df, "Ablation Study")
        sections.append(
            "<p class='note'>"
            "Positive accuracy delta means removing that feature group improved accuracy. "
            "Positive Brier improvement means removing that group improved probability quality."
            "</p>"
        )

    if ablation_plot_path and ablation_plot_path.exists():
        sections.append(
            f"<h2>Ablation Study</h2>"
            f"<img src='{ablation_plot_path.name}' style='max-width:100%;height:auto;'>"
        )

    if cv_summary_df is not None and not cv_summary_df.empty:
        add_table(cv_summary_df, "Time-Series Cross-Validation Summary")

    if cv_fold_df is not None and not cv_fold_df.empty:
        add_table(cv_fold_df, "Time-Series Cross-Validation Folds")

    if cv_plot_path and cv_plot_path.exists():
        sections.append(
            f"<h2>Time-Series Cross-Validation Performance</h2>"
            f"<img src='{cv_plot_path.name}' style='max-width:100%;height:auto;'>"
        )
    if tree_probability_space_html:
        sections.append("<h2>Tree Model Full Probability Space</h2>")
        sections.append(tree_probability_space_html)
    elif tree_corner_heatmap_prob_html:
        sections.append("<h2>Tree Model Full Probability Space</h2>")
        sections.append(tree_corner_heatmap_prob_html)

    if tree_count_heatmap_html:
        sections.append("<h2>Tree Model Games Played by Ordered Configuration</h2>")
        sections.append(tree_count_heatmap_html)

    if tree_synergy_path and tree_synergy_path.exists():
        sections.append(
            f"<h2>Tree Model Pair Synergy</h2>"
            f"<img src='{tree_synergy_path.name}' style='max-width:100%;height:auto;'>"
        )


    if tree_prob_hist_path and tree_prob_hist_path.exists():
        sections.append(
            f"<h2>Tree Model Full Probability Distribution</h2>"
            f"<img src='{tree_prob_hist_path.name}' style='max-width:100%;height:auto;'>"
        )

    # ============================================================
    # MODEL COMPARISONS
    # ============================================================
    add_section_title("Model Comparisons")

    if combined_accuracy_plot and combined_accuracy_plot.exists():
        sections.append(
            f"<h2>Combined Prediction Accuracy Comparison</h2>"
            f"<img src='{combined_accuracy_plot.name}' style='max-width:100%;height:auto;'>"
        )

    # Save Plotly fragments into standalone HTML files for side-by-side rendering.
    elo_prob_html_path = save_plot_html("elo_probability_heatmap.html", comparison_corner_heatmap_prob_html or corner_heatmap_prob_html)
    ts_prob_html_path = save_plot_html("trueskill_probability_heatmap.html", trueskill_corner_heatmap_prob_html)
    tree_prob_html_path = save_plot_html("tree_probability_heatmap.html", tree_probability_space_html or tree_corner_heatmap_prob_html)

    if elo_prob_html_path or ts_prob_html_path or tree_prob_html_path:
        add_three_way_plot_row(
            "Probability Heat Maps",
            "Elo", elo_prob_html_path,
            "TrueSkill", ts_prob_html_path,
            "Tree Model", tree_prob_html_path,
        )

    

    # ============================================================
    # FINAL HTML
    # ============================================================
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Ping Pong Dashboard</title>
        <script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }}
            h1 {{ margin-bottom: 0.2em; }}
            h2 {{ margin-top: 2em; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 0.5em; }}
            th, td {{ padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }}
            .note {{ color: #555; margin-bottom: 1em; }}
        </style>
    </head>
    <body>
        <h1>Lab Ping Pong Dashboard</h1>
        <div class='note'>Interactive summary of ratings, matchups, position effects, and all-time rankings.</div>
        {''.join(sections)}
    </body>
    </html>
    """

    path = outdir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path

# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(
    outdir: Path,
    history: pd.DataFrame,
    elo_df: pd.DataFrame,
    player_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    side_df: pd.DataFrame,
    teammate_matrix: pd.DataFrame,
    overall_df: pd.DataFrame,
    corner_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    synergy_df: pd.DataFrame,
    unplayed_games_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    gboost_metrics_df: pd.DataFrame,
    gboost_importance_df: pd.DataFrame,
    gboost_predictions_df: pd.DataFrame,
    trueskill_df: Optional[pd.DataFrame] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Directory for machine learning outputs
    ml_outdir = outdir / "ML Data"
    ml_outdir.mkdir(parents=True, exist_ok=True)
    history.to_csv(outdir / "game_history_with_elo.csv", index=False)
    elo_df.to_csv(outdir / "final_elo_ratings.csv", index=False)
    player_df.to_csv(outdir / "player_summary.csv", index=False)
    pair_df.to_csv(outdir / "pair_summary.csv", index=False)
    side_df.to_csv(outdir / "side_summary.csv", index=False)
    teammate_matrix.to_csv(outdir / "teammate_matrix_win_pct.csv")
    overall_df.to_csv(outdir / "overall_ranking.csv", index=False)
    corner_df.to_csv(outdir / "corner_summary.csv", index=False)
    pred_df.to_csv(outdir / "match_win_probabilities.csv", index=False)
    improvement_df.to_csv(outdir / "player_improvement_curves.csv", index=False)
    synergy_df.to_csv(outdir / "pair_synergy.csv", index=False)
    ml_df.to_csv(ml_outdir / "ml_training_table.csv", index=False)
    gboost_metrics_df.to_csv(ml_outdir / "gboost_metrics.csv", index=False)
    gboost_importance_df.to_csv(ml_outdir / "gboost_feature_importance.csv", index=False)
    gboost_predictions_df.to_csv(ml_outdir / "gboost_test_predictions.csv", index=False)
    if trueskill_df is not None and not trueskill_df.empty:
        trueskill_df.to_csv(outdir / "trueskill_ratings.csv", index=False)
    if not unplayed_games_df.empty:
        unplayed_games_df.to_csv(outdir / "unplayed_unique_games.csv", index=False)

def print_summary(
    elo_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    corner_df: pd.DataFrame,
    player_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    side_df: pd.DataFrame,
    trueskill_df: Optional[pd.DataFrame],
) -> None:
    print("\n=== Elo Ratings ===")
    print(elo_df.to_string(index=False) if not elo_df.empty else "No Elo results.")

    if trueskill_df is not None:
        print("\n=== TrueSkill Ratings ===")
        print(trueskill_df.to_string(index=False) if not trueskill_df.empty else "No TrueSkill results.")

    print("\n=== Overall Ranking ===")
    if not overall_df.empty:
        cols = [c for c in ["rank", "player", "games", "wins", "losses", "win_pct", "point_diff", "avg_point_diff"] if c in overall_df.columns]
        print(overall_df[cols].to_string(index=False))
    else:
        print("No overall ranking available.")

    print("\n=== Corner Analysis ===")
    if not corner_df.empty:
        cols = [c for c in ["corner", "corner_rank", "player", "games", "wins", "losses", "win_pct", "point_diff"] if c in corner_df.columns]
        print(corner_df[cols].head(16).to_string(index=False))
    else:
        print("No corner analysis available.")

    print("\n=== Player Summary ===")
    if not player_df.empty:
        cols = [c for c in ["player", "games", "wins", "losses", "win_pct", "point_diff", "avg_point_diff"] if c in player_df.columns]
        print(player_df[cols].to_string(index=False))
    else:
        print("No player summary.")

    print("\n=== Top Partnerships ===")
    if not pair_df.empty:
        show = pair_df.head(10).copy()
        show["pair"] = show["player_a"] + " + " + show["player_b"]
        cols = [c for c in ["pair", "games", "wins", "losses", "win_pct", "point_diff"] if c in show.columns]
        print(show[cols].to_string(index=False))
    else:
        print("No pair summary.")

    print("\n=== Side / Position Advantage ===")
    print(side_df.to_string(index=False) if not side_df.empty else "No side summary.")

# ============================================================
# Main
# ============================================================

def run_analysis(input_file: str, output_dir: str) -> Path:
    df = load_games(input_file)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Directory for machine learning outputs
    ml_outdir = outdir / "ML Data"
    ml_outdir.mkdir(parents=True, exist_ok=True)

    history, elo_df = run_elo(df)
    prediction_df = build_prediction_accuracy(history)
    trueskill_df = run_trueskill(df)
    trueskill_history_df = build_trueskill_player_history(df)

    # ML dataset
    ml_df, ml_state = build_ml_training_table(df, recent_window=5, return_state=True)
    feature_dictionary_path = write_feature_dictionary_text(ml_outdir)
    #print(f"Saved feature dictionary to: {feature_dictionary_path}")

    # Train model
    gboost_model, feature_cols, gboost_metrics_df, gboost_importance_df, gboost_predictions_df, X_test = train_gradient_boosted_model(
        ml_df,
        test_frac=0.2,
        random_state=42,
    )

    # Plots
    feature_importance_path = plot_feature_importance(gboost_importance_df, outdir)
    shap_summary_path = plot_shap_summary(gboost_model, X_test, outdir)
    model_performance_path = plot_model_performance_report(gboost_predictions_df, outdir)
    top_mispredictions_df = build_top_mispredictions_df(gboost_predictions_df, top_n=10)

    cv_fold_df, cv_summary_df, cv_oof_df = run_time_series_cv(
    ml_df,
    n_splits=5,
    random_state=42,
    )
    cv_plot_path = plot_time_series_cv(cv_fold_df, outdir)

    cv_fold_df.to_csv(ml_outdir / "time_series_cv_folds.csv", index=False)
    cv_summary_df.to_csv(ml_outdir / "time_series_cv_summary.csv", index=False)
    cv_oof_df.to_csv(ml_outdir / "time_series_cv_oof_predictions.csv", index=False)

    ablation_df = run_ablation_study(ml_df, test_frac=0.2, random_state=42)
    ablation_plot_path = plot_ablation_study(ablation_df, outdir)
    ablation_df.to_csv(ml_outdir / "ablation_study.csv", index=False)

    correct_vs_incorrect_df = build_correct_vs_incorrect_feature_summary(X_test, gboost_predictions_df)
    correct_vs_incorrect_path = plot_correct_vs_incorrect_features(correct_vs_incorrect_df, outdir)
    correct_vs_incorrect_df.to_csv(ml_outdir / "correct_vs_incorrect_feature_summary.csv", index=False)

    player_df = build_player_summary(df)
    pair_df = build_pair_summary(df)
    side_df = build_side_summary(df)
    teammate_matrix = build_teammate_winpct_matrix(df)
    overall_df = build_overall_ranking(df)
    corner_df = build_corner_summary(df)
    pred_df = add_prediction_columns(df, elo_df)
    improvement_df = build_player_improvement(df)
    matchup_heatmap_df = build_matchup_heatmap(pred_df, list(elo_df["player"]))
    corner_heatmap_path = build_corner_heatmap_of_heatmaps(elo_df, outdir)
    corner_heatmap_png = build_corner_heatmap_of_heatmaps(elo_df, outdir)
    corner_heatmap_prob_html = build_corner_heatmap_of_heatmaps_plotly(elo_df, auto_scale=True)
    comparison_corner_heatmap_prob_html = build_corner_heatmap_of_heatmaps_plotly(elo_df, auto_scale=False)
    corner_heatmap_count_html = build_corner_heatmap_of_heatmaps_plotly_game_counts(df)
    completion_path = plot_data_completion_donuts(df, outdir)
    synergy_df = build_pair_synergy_summary(df, history)
    synergy_path = plot_pair_synergy(synergy_df, outdir)
    highlights_path = plot_game_highlights(df, history, outdir)
    unplayed_games_df = build_corner_game_explorer_df(
        df,
        elo_df,
        trueskill_df,
        gboost_model,
        feature_cols,
        ml_state,
    )
    leadership_df = build_elo_leadership_summary(history)
    leadership_path = plot_elo_leadership(leadership_df, outdir)
    model_performance_path = plot_model_performance_report(gboost_predictions_df, outdir)
    trueskill_prediction_df = build_trueskill_prediction_accuracy(df)
    shap_background = ml_df[feature_cols].sample(
        n=min(100, len(ml_df)),
        random_state=42,
    ) if not ml_df.empty else X_test.copy()
    tree_shap_explainer = shap.TreeExplainer(
        gboost_model,
        data=shap_background,
        model_output="probability",
        feature_perturbation="interventional",
    )

    elo_space_df = build_elo_probability_space_df(elo_df)
    trueskill_space_df = build_trueskill_probability_space_df(trueskill_df)
    tree_space_df = build_tree_probability_space_df(gboost_model, feature_cols, ml_state)
    elo_prob_hist_path = plot_probability_histogram(
        elo_space_df,
        "pred_team1_win_prob",
        outdir,
        "elo_probability_histogram.png",
        "Elo Predicted Probability Distribution (210 unique games)",
    )
    trueskill_prob_hist_path = plot_probability_histogram(
        trueskill_space_df,
        "pred_team1_win_prob",
        outdir,
        "trueskill_probability_histogram_full_space.png",
        "TrueSkill Predicted Probability Distribution (210 unique games)",
    )
    tree_prob_hist_path = plot_probability_histogram(
        tree_space_df,
        "pred_team1_win_prob",
        outdir,
        "tree_probability_histogram_full_space.png",
        "Tree Model Predicted Probability Distribution (1680 ordered configs)",
    )
    tree_count_heatmap_html = build_tree_corner_heatmap_of_heatmaps_plotly_game_counts(df)

    

    save_outputs(
        outdir,
        history,
        elo_df,
        player_df,
        pair_df,
        side_df,
        teammate_matrix,
        overall_df,
        corner_df,
        pred_df,
        improvement_df,
        synergy_df,
        unplayed_games_df,
        ml_df,
        gboost_metrics_df,
        gboost_importance_df,
        gboost_predictions_df,
        trueskill_df,
    )
    
    plot_player_elo(elo_df, outdir)
    plot_pair_strength(pair_df, outdir)
    plot_improvement_curves(improvement_df,  history, outdir)
    plot_teammate_winpct_heatmap(teammate_matrix, outdir)
    prediction_plot = plot_prediction_accuracy( prediction_df,outdir,)
    network_path = plot_network(df, outdir)

    trueskill_prediction_plot = plot_trueskill_prediction_accuracy(trueskill_prediction_df, outdir)

    trueskill_synergy_df = build_pair_synergy_from_predictions(
        df,
        trueskill_prediction_df,
        "team1_expected_win_prob",
    )
    trueskill_synergy_path = plot_pair_synergy_generic(
        trueskill_synergy_df,
        outdir,
        "trueskill_pair_synergy.png",
        "TrueSkill Pair Synergy (Actual Win % - Expected Win %)",
    )
    trueskill_hist_path = plot_probability_histogram(
        trueskill_space_df,
        "pred_team1_win_prob",
        outdir,
        "trueskill_probability_histogram_full_space.png",
        "TrueSkill Predicted Probability Distribution (210 unique games)",
    )

    combined_accuracy_plot = plot_combined_accuracy_comparison(
                                                                    prediction_df,
                                                                    trueskill_prediction_df,
                                                                    gboost_predictions_df,
                                                                    outdir,
                                                                )
    tree_probability_space_html = build_tree_corner_heatmap_of_heatmaps_plotly(
                                                                                    gboost_model,
                                                                                    feature_cols,
                                                                                    ml_state,
                                                                                    shap_explainer=tree_shap_explainer,
                                                                                    shap_top_k=4,
                                                                                )
    tree_corner_heatmap_prob_html = tree_probability_space_html
    tree_synergy_df = build_pair_synergy_from_predictions(
        df,
        gboost_predictions_df,
        "pred_team1_win_prob",
    )
    tree_synergy_path = plot_pair_synergy_generic(
        tree_synergy_df,
        outdir,
        "tree_pair_synergy.png",
        "Tree Model Pair Synergy (Actual Win % - Predicted Win %)",
    )
    tree_hist_path = plot_probability_histogram(
        tree_space_df,
        "pred_team1_win_prob",
        outdir,
        "tree_probability_histogram_full_space.png",
        "Tree Model Predicted Probability Distribution (1680 ordered configs)",
    )

    if not trueskill_prediction_df.empty:
        trueskill_prediction_df.to_csv(ml_outdir / "trueskill_prediction_quality.csv", index=False)

    if TRUESKILL_AVAILABLE:
        trueskill_over_time_path = plot_trueskill_over_time(trueskill_history_df, history, outdir)
        trueskill_corner_heatmap_prob_html = build_trueskill_corner_heatmap_of_heatmaps_plotly(trueskill_df)
        trueskill_leadership_df = build_trueskill_leadership_summary(history, trueskill_history_df, trueskill_df)
    else:
        trueskill_over_time_path = None
        trueskill_corner_heatmap_prob_html = None
        trueskill_leadership_df = pd.DataFrame()
    if not trueskill_history_df.empty:
        trueskill_history_df.to_csv(ml_outdir / "trueskill_history.csv", index=False)

    dashboard_path = build_dashboard_html(
        outdir,
        elo_df,
        overall_df,
        corner_df,
        player_df,
        pair_df,
        side_df,
        teammate_matrix,
        teammate_matrix,
        history,
        improvement_df,
        pred_df,
        matchup_heatmap_df,
        corner_heatmap_path,
        unplayed_games_df,
        corner_heatmap_png,
        corner_heatmap_prob_html,
        corner_heatmap_count_html,
        completion_path,
        synergy_path,
        highlights_path,
        leadership_path,
        leadership_df,
        prediction_plot,
        network_path,
        feature_importance_path,
        shap_summary_path,
        gboost_metrics_df,
        gboost_predictions_df,
        model_performance_path,
        correct_vs_incorrect_df,
        correct_vs_incorrect_path,
        ablation_df,
        ablation_plot_path,
        cv_fold_df,
        cv_summary_df,
        cv_plot_path,
        trueskill_df,
        trueskill_history_df,
        trueskill_over_time_path,
        trueskill_corner_heatmap_prob_html,
        trueskill_leadership_df,
        trueskill_prediction_plot,
        combined_accuracy_plot,
        tree_probability_space_html,
        tree_corner_heatmap_prob_html,
        trueskill_synergy_path,
        trueskill_hist_path,
        tree_synergy_path,
        tree_hist_path,
        elo_prob_hist_path,
        trueskill_prob_hist_path,
        tree_prob_hist_path,
        tree_count_heatmap_html,
        comparison_corner_heatmap_prob_html
    )

    print_summary(elo_df, overall_df, corner_df, player_df, pair_df, side_df, trueskill_df)
    if dashboard_path is not None:
        print(f"\nInteractive dashboard saved to: {dashboard_path.resolve()}")
    else:
        print("\nInteractive dashboard skipped because Plotly is not installed.")

    print(f"\nAll outputs saved to: {outdir.resolve()}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze lab ping pong data.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input .txt or .csv file path.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()
