"""Ping pong doubles analysis pipeline for lab log files.

Run from IDLE:
  1. Open this file in IDLE.
  2. Set INPUT_FILE and OUTPUT_DIR in the CONFIG section.
  3. Press F5.

Supported input formats:
  - .txt lab log with one game per line and a blank line between days
  - .csv with columns p1,p2,p3,p4 and score1,score2 (or score)

Text log format example:
  1) TT, NS Vs. SY, CH: 21-13
  2) TT, SY Vs. FZ, NS: 14-21

  1) CH, NS Vs. FZ, SY: 21-18

Blank lines separate days. The script assigns Day 1, Day 2, Day 3, ... automatically.

Outputs:
  - Elo ratings
  - Optional TrueSkill ratings if installed
  - Position advantages
  - Best teammate combinations
  - Win probability estimates
  - Player improvement curves
  - Partnership network graph
  - Weekly/monthly rankings
  - Interactive HTML dashboard
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG FOR IDLE USERS
# ============================================================
INPUT_FILE = "PingPongData.txt"   # change this in IDLE if needed
OUTPUT_DIR = "ping_pong_output"
DEFAULT_ELO = 1500.0
K_FACTOR = 24.0
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

try:
    import networkx as nx  # type: ignore
    NETWORKX_AVAILABLE = True
except Exception:
    nx = None
    NETWORKX_AVAILABLE = False

try:
    import plotly.express as px  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False


# ============================================================
# TEXT PARSING
# ============================================================
GAME_LINE_RE = re.compile(
    r"^\s*(?:\d+\)\s*)?"                                  # optional numbering
    r"(?P<p1>[A-Za-z0-9_\-]+)\s*,\s*(?P<p2>[A-Za-z0-9_\-]+)"  # team 1
    r"\s+Vs\.?\s+"                                          # Vs or Vs.
    r"(?P<p3>[A-Za-z0-9_\-]+)\s*,\s*(?P<p4>[A-Za-z0-9_\-]+)"  # team 2
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
            # blank row = end of day; next nonblank line starts a new day
            if game_in_day > 0:
                current_day += 1
                game_in_day = 0
            continue

        # Skip a likely header line if the file has one.
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
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


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
        exp1 = expected_score(r1, r2)
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
            new_t1, new_t2 = trueskill.rate([[team1[0], team1[1]], [team2[0], team2[1]]], ranks=[0, 1], env=env)
        else:
            new_t1, new_t2 = trueskill.rate([[team1[0], team1[1]], [team2[0], team2[1]]], ranks=[1, 0], env=env)

        for p, r in zip(t1, new_t1):
            ratings[p] = r
        for p, r in zip(t2, new_t2):
            ratings[p] = r

    trueskill_df = pd.DataFrame(
        [{
            "player": p,
            "trueskill_mu": r.mu,
            "trueskill_sigma": r.sigma,
            "trueskill_conservative": r.mu - 3 * r.sigma,
        } for p, r in ratings.items()]
    ).sort_values(["trueskill_conservative", "trueskill_mu"], ascending=False).reset_index(drop=True)

    return trueskill_df


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


def build_player_improvement(df: pd.DataFrame) -> pd.DataFrame:
    ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    rows = []

    for _, row in df.iterrows():
        t1 = team1_players(row)
        t2 = team2_players(row)
        r1 = float(np.mean([ratings[p] for p in t1]))
        r2 = float(np.mean([ratings[p] for p in t2]))
        exp1 = expected_score(r1, r2)
        actual1 = 1.0 if team1_won(row) else 0.0
        delta = K_FACTOR * (actual1 - exp1)

        for p in t1:
            ratings[p] += delta / 2.0
        for p in t2:
            ratings[p] -= delta / 2.0

        for p in [row["p1"], row["p2"], row["p3"], row["p4"]]:
            rows.append({"game_id": row["game_id"], "day": row.get("day", np.nan), "date": row.get("date", pd.NaT), "player": p, "elo_after_game": ratings[p]})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["player", "game_id"]).reset_index(drop=True)
    return out


def build_time_rankings(df: pd.DataFrame):
    if df["day"].isna().all():
        return pd.DataFrame(), pd.DataFrame()

    temp = df.copy()
    temp["day"] = pd.to_numeric(temp["day"], errors="coerce")
    temp = temp[temp["day"].notna()].copy()
    if temp.empty:
        return pd.DataFrame(), pd.DataFrame()

    temp["week"] = ((temp["day"] - 1) // 5 + 1).astype(int)
    temp["month"] = ((temp["day"] - 1) // 20 + 1).astype(int)

    def summarize(grouped: pd.core.groupby.DataFrameGroupBy, label_name: str):
        out = []
        for label, g in grouped:
            players = pd.unique(g[["p1", "p2", "p3", "p4"]].values.ravel("K"))
            for player in players:
                mask = (g["p1"] == player) | (g["p2"] == player) | (g["p3"] == player) | (g["p4"] == player)
                gg = g[mask]
                wins = 0
                for _, row in gg.iterrows():
                    if player in (row["p1"], row["p2"]):
                        wins += int(row["score1"] > row["score2"])
                    else:
                        wins += int(row["score2"] > row["score1"])
                out.append({
                    label_name: label,
                    "player": player,
                    "games": len(gg),
                    "wins": wins,
                    "losses": len(gg) - wins,
                    "win_pct": wins / len(gg) if len(gg) else np.nan,
                    "point_diff": int((gg["score1"] - gg["score2"]).sum()),
                })
        return pd.DataFrame(out)

    weekly_df = summarize(temp.groupby("week"), "week").sort_values(["week", "win_pct", "point_diff"], ascending=[True, False, False]).reset_index(drop=True)
    monthly_df = summarize(temp.groupby("month"), "month").sort_values(["month", "win_pct", "point_diff"], ascending=[True, False, False]).reset_index(drop=True)
    return weekly_df, monthly_df


# ============================================================
# PREDICTIONS
# ============================================================

def predict_match_prob(team1: Tuple[str, str], team2: Tuple[str, str], elo_df: pd.DataFrame, default_elo: float = DEFAULT_ELO) -> float:
    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()}
    r1 = np.mean([ratings.get(p, default_elo) for p in team1])
    r2 = np.mean([ratings.get(p, default_elo) for p in team2])
    return expected_score(r1, r2)


def add_prediction_columns(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_team1_win_prob"] = [predict_match_prob((r["p1"], r["p2"]), (r["p3"], r["p4"]), elo_df) for _, r in out.iterrows()]
    return out


# ============================================================
# PLOTS
# ============================================================

def plot_player_elo(elo_df: pd.DataFrame, outdir: Path):
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


def plot_pair_strength(pair_df: pd.DataFrame, outdir: Path):
    if pair_df.empty:
        return None
    top = pair_df[pair_df["games"] >= MIN_PAIR_GAMES_FOR_DISPLAY].head(12).copy()
    if top.empty:
        top = pair_df.head(min(12, len(pair_df))).copy()
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


def plot_daily_results(df: pd.DataFrame, outdir: Path):
    if df["day"].isna().all():
        return None
    counts = df[df["day"].notna()].groupby("day").size().reset_index(name="games")
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(counts["day"], counts["games"], marker="o")
    ax.set_title("Games Per Day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Games")
    fig.tight_layout()
    path = outdir / "games_per_day.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_improvement_curves(improvement_df: pd.DataFrame, outdir: Path, max_players: int = 8):
    if improvement_df.empty:
        return None
    players = improvement_df["player"].value_counts().head(max_players).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    for player in players:
        g = improvement_df[improvement_df["player"] == player]
        ax.plot(g["game_id"], g["elo_after_game"], marker="o", label=player)
    ax.set_title("Player Improvement Curves")
    ax.set_xlabel("Game")
    ax.set_ylabel("Elo after each game")
    ax.legend(loc="best")
    fig.tight_layout()
    path = outdir / "improvement_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_network(df: pd.DataFrame, outdir: Path):
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


# ============================================================
# INTERACTIVE DASHBOARD
# ============================================================

def build_dashboard_html(outdir: Path, elo_df: pd.DataFrame, player_df: pd.DataFrame, pair_df: pd.DataFrame, side_df: pd.DataFrame, history_df: pd.DataFrame, improvement_df: pd.DataFrame, pred_df: pd.DataFrame, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame, network_path: Optional[Path] = None):
    if not PLOTLY_AVAILABLE:
        return None

    sections: List[str] = []

    def add_table(df: pd.DataFrame, title: str):
        if df.empty:
            sections.append(f"<h2>{title}</h2><p>No data available.</p>")
        else:
            sections.append(f"<h2>{title}</h2>" + df.to_html(index=False, border=0))

    add_table(side_df, "Position / Side Advantage")
    add_table(player_df.head(10), "Top Player Summary")
    add_table(pair_df.head(10), "Top Partnerships")

    if not elo_df.empty:
        sections.append(f"<h2>Elo Ratings</h2>{px.bar(elo_df.head(12), x='player', y='elo').to_html(full_html=False, include_plotlyjs=False)}")
    if not pair_df.empty:
        top_pairs = pair_df[pair_df["games"] >= MIN_PAIR_GAMES_FOR_DISPLAY].head(12).copy()
        if top_pairs.empty:
            top_pairs = pair_df.head(min(12, len(pair_df))).copy()
        top_pairs["pair"] = top_pairs["player_a"] + " + " + top_pairs["player_b"]
        sections.append(f"<h2>Best Teammate Combinations</h2>{px.bar(top_pairs, x='pair', y='win_pct').to_html(full_html=False, include_plotlyjs=False)}")
    if not improvement_df.empty:
        sections.append(f"<h2>Player Improvement Curves</h2>{px.line(improvement_df, x='game_id', y='elo_after_game', color='player').to_html(full_html=False, include_plotlyjs=False)}")
    if not pred_df.empty:
        sections.append(f"<h2>Win Probabilities</h2>{px.histogram(pred_df, x='pred_team1_win_prob', nbins=20).to_html(full_html=False, include_plotlyjs=False)}")

    if network_path and network_path.exists():
        sections.append(f"<h2>Partnership Network Graph</h2><img src='{network_path.name}' style='max-width:100%;height:auto;'>")

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
        <div class='note'>Interactive summary of ratings, partnerships, position effects, and time trends.</div>
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

def save_outputs(outdir: Path, history: pd.DataFrame, elo_df: pd.DataFrame, player_df: pd.DataFrame, pair_df: pd.DataFrame, side_df: pd.DataFrame, pred_df: pd.DataFrame, improvement_df: pd.DataFrame, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame, trueskill_df: Optional[pd.DataFrame] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    history.to_csv(outdir / 'game_history_with_elo.csv', index=False)
    elo_df.to_csv(outdir / 'final_elo_ratings.csv', index=False)
    player_df.to_csv(outdir / 'player_summary.csv', index=False)
    pair_df.to_csv(outdir / 'pair_summary.csv', index=False)
    side_df.to_csv(outdir / 'side_summary.csv', index=False)
    pred_df.to_csv(outdir / 'match_win_probabilities.csv', index=False)
    improvement_df.to_csv(outdir / 'player_improvement_curves.csv', index=False)
    if not weekly_df.empty:
        weekly_df.to_csv(outdir / 'weekly_rankings.csv', index=False)
    if not monthly_df.empty:
        monthly_df.to_csv(outdir / 'monthly_rankings.csv', index=False)
    if trueskill_df is not None and not trueskill_df.empty:
        trueskill_df.to_csv(outdir / 'trueskill_ratings.csv', index=False)


def print_summary(elo_df: pd.DataFrame, player_df: pd.DataFrame, pair_df: pd.DataFrame, side_df: pd.DataFrame, trueskill_df: Optional[pd.DataFrame]) -> None:
    print("\n=== Elo Ratings ===")
    print(elo_df.to_string(index=False) if not elo_df.empty else "No Elo results.")

    if trueskill_df is not None:
        print("\n=== TrueSkill Ratings ===")
        print(trueskill_df.to_string(index=False) if not trueskill_df.empty else "No TrueSkill results.")

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

    print("\n=== Position / Side Advantage ===")
    print(side_df.to_string(index=False) if not side_df.empty else "No side summary.")


# ============================================================
# MAIN
# ============================================================

def run_analysis(input_file: str, output_dir: str) -> Path:
    df = load_games(input_file)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    history, elo_df = run_elo(df)
    trueskill_df = run_trueskill(df)
    player_df = build_player_summary(df)
    pair_df = build_pair_summary(df)
    side_df = build_side_summary(df)
    pred_df = add_prediction_columns(df, elo_df)
    improvement_df = build_player_improvement(df)
    weekly_df, monthly_df = build_time_rankings(df)

    save_outputs(outdir, history, elo_df, player_df, pair_df, side_df, pred_df, improvement_df, weekly_df, monthly_df, trueskill_df)

    plot_player_elo(elo_df, outdir)
    plot_pair_strength(pair_df, outdir)
    plot_daily_results(df, outdir)
    plot_improvement_curves(improvement_df, outdir)
    network_path = plot_network(df, outdir)

    dashboard_path = build_dashboard_html(outdir, elo_df, player_df, pair_df, side_df, history, improvement_df, pred_df, weekly_df, monthly_df, network_path)

    print_summary(elo_df, player_df, pair_df, side_df, trueskill_df)
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

