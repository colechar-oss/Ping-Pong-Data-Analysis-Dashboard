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
  - Player Elo over time
  - Partnership network graph
  - Overall all-time rankings
  - Corner/position analysis
  - Interactive HTML dashboard
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from itertools import combinations

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


def build_overall_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Overall all-time ranking across every game in the log."""
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
    """Corner-based performance table for each player at positions 1-4."""
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
    """
    Build an NxN teammate matrix for players.
    metric can be:
      - "games"    -> games played together
      - "wins"     -> wins together
      - "win_pct"  -> teammate win percentage
      - "point_diff" -> total point differential together
    """
    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    matrix = pd.DataFrame(index=players, columns=players, dtype=float)

    # Start with NaN on diagonal
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
    players = sorted(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    matrix = pd.DataFrame(np.nan, index=players, columns=players, dtype=float)

    pair_stats = defaultdict(lambda: {"games": 0, "wins": 0})

    for _, row in df.iterrows():
        teams = [
            ((row["p1"], row["p2"]), row["score1"], row["score2"]),
            ((row["p3"], row["p4"]), row["score2"], row["score1"]),
        ]

        for (a, b), team_score, opp_score in teams:
            key = tuple(sorted((a, b)))
            pair_stats[key]["games"] += 1
            pair_stats[key]["wins"] += int(team_score > opp_score)

    for (a, b), stats in pair_stats.items():
        win_pct = stats["wins"] / stats["games"]
        matrix.loc[a, b] = win_pct
        matrix.loc[b, a] = win_pct

    return matrix


# ============================================================
# PREDICTIONS
# PREDICTIONS
# ============================================================

def predict_match_prob(team1: Tuple[str, str], team2: Tuple[str, str], elo_df: pd.DataFrame, default_elo: float = DEFAULT_ELO) -> float:
    ratings = {row["player"]: float(row["elo"]) for _, row in elo_df.iterrows()}
    r1 = np.mean([ratings.get(p, default_elo) for p in team1])
    r2 = np.mean([ratings.get(p, default_elo) for p in team2])
    return expected_score(r1, r2)


def add_prediction_columns(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Build a predictive dataset for all possible current matchups.

    Team 1 is always assigned to the higher-Elo team so the predicted
    win probability is always at least 0.50.
    """
    players = list(pd.unique(df[["p1", "p2", "p3", "p4"]].values.ravel("K")))
    if len(players) < 4:
        return pd.DataFrame(columns=[
            "team1", "team2", "team1_elo", "team2_elo",
            "elo_diff", "avg_elo", "pred_team1_win_prob"
        ])

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

            prob = expected_score(team1_elo, team2_elo)

            rows.append(
                {
                    "matchup_id": matchup_id,
                    "team1": f"{team1[0]} + {team1[1]}",
                    "team2": f"{team2[0]} + {team2[1]}",
                    "team1_elo": team1_elo,
                    "team2_elo": team2_elo,
                    "elo_diff": team1_elo - team2_elo,
                    "avg_elo": (team1_elo + team2_elo) / 2.0,
                    "pred_team1_win_prob": prob,
                }
            )
            matchup_id += 1

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
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
    ax.set_title("Player Elo Over Time")
    ax.set_xlabel("Game")
    ax.set_ylabel("Elo rating")
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

def plot_teammate_winpct_heatmap(winpct_matrix: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if winpct_matrix.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.ma.masked_invalid(winpct_matrix.to_numpy(dtype=float))

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


# ============================================================
# INTERACTIVE DASHBOARD
# ============================================================

def build_dashboard_html(outdir: Path, elo_df: pd.DataFrame, overall_df: pd.DataFrame, corner_df: pd.DataFrame, player_df: pd.DataFrame, pair_df: pd.DataFrame, side_df: pd.DataFrame,teammate_matrix: pd.DataFrame, teammate_winpct_matrix: pd.DataFrame, history_df: pd.DataFrame, improvement_df: pd.DataFrame, pred_df: pd.DataFrame, network_path: Optional[Path] = None):
    if not PLOTLY_AVAILABLE:
        return None

    sections: List[str] = []

    def add_table(df: pd.DataFrame, title: str):
        if df.empty:
            sections.append(f"<h2>{title}</h2><p>No data available.</p>")
        else:
            clean_df = df.copy().fillna("-")
            sections.append(f"<h2>{title}</h2>" + clean_df.to_html(index=False, border=0))

    def mask_lower_triangle(df: pd.DataFrame) -> pd.DataFrame:
        """Leave only the upper triangle of a square dataframe."""
        out = df.copy()

        mask = np.tril(np.ones(out.shape, dtype=bool))

        out = out.mask(mask)

        return out
    
    add_table(overall_df.head(10), "Overall Ranking")
    add_table(corner_df, "Corner Analysis")
    add_table(side_df, "Position / Side Advantage")


    players = list(teammate_matrix.index)
    teammate_games_matrix = pd.DataFrame(np.nan, index=players, columns=players)
    np.fill_diagonal(teammate_games_matrix.values, np.nan)

    for _, row in pair_df.iterrows():
        a = row["player_a"]
        b = row["player_b"]
        if a in teammate_games_matrix.index and b in teammate_games_matrix.columns:
            teammate_games_matrix.loc[a, b] = row["games"]
            teammate_games_matrix.loc[b, a] = row["games"]

    tm_games_display = mask_lower_triangle(teammate_games_matrix)
    tm_games_display = tm_games_display.fillna("-")
    tm_games_display.index.name = "Player"
    add_table(
        tm_games_display.reset_index().rename(columns={"index": "Player"}),
        "Teammate Matrix (Games Together)",
    )

    if not teammate_games_matrix.empty:
        fig_games = px.imshow(
            mask_lower_triangle(teammate_games_matrix),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Teammate Matrix (Games Together)",
        )
        sections.append(fig_games.to_html(full_html=False, include_plotlyjs=False))

    
    tm_winpct_display = mask_lower_triangle(teammate_matrix)
    tm_winpct_display.index.name = "Player"
    tm_winpct_display = tm_winpct_display.fillna("-")

    add_table(
        tm_winpct_display.reset_index().rename(columns={"index": "Player"}),
        "Teammate Matrix (Win %)",
    )

    if not teammate_matrix.empty:
        fig = px.imshow(
            mask_lower_triangle(teammate_matrix),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Teammate Matrix (Win %)",
        )
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))
    add_table(player_df.head(10), "Top Player Summary")
    add_table(pair_df.head(10), "Top Partnerships")

    add_table(
    elo_df.sort_values("elo", ascending=False).reset_index(drop=True),
    "Elo Ratings")
    
    if not pair_df.empty:
        top_pairs = pair_df[pair_df["games"] >= MIN_PAIR_GAMES_FOR_DISPLAY].head(12).copy()
        if top_pairs.empty:
            top_pairs = pair_df.head(min(12, len(pair_df))).copy()
        top_pairs["pair"] = top_pairs["player_a"] + " + " + top_pairs["player_b"]
        sections.append(f"<h2>Best Teammate Combinations</h2>{px.bar(top_pairs, x='pair', y='win_pct').to_html(full_html=False, include_plotlyjs=False)}")
    if not improvement_df.empty:
        sections.append(f"<h2>Player Elo Over Time</h2>{px.line(improvement_df, x='game_id', y='elo_after_game', color='player').to_html(full_html=False, include_plotlyjs=False)}")
    if not pred_df.empty and "pred_team1_win_prob" in pred_df.columns:
        sections.append(f"<h2>Predicted Win Probabilities for All Possible Matchups</h2>{px.histogram(pred_df, x='pred_team1_win_prob', nbins=20).to_html(full_html=False, include_plotlyjs=False)}")

    if network_path and network_path.exists():
        sections.append(f"<h2>Partnership Network Graph</h2><img src='{network_path.name}' style='max-width:100%;height:auto;'>")

    tm_display = mask_lower_triangle(teammate_winpct_matrix).fillna("-")
    tm_display.index.name = "Player"

    add_table(
        tm_display.reset_index().rename(columns={"index": "Player"}),
        "Teammate Matrix (Win %)",
    )

    tm_heatmap = mask_lower_triangle(teammate_winpct_matrix)

    if not tm_heatmap.empty:
        fig = px.imshow(
            tm_heatmap,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Teammate Matrix (Win %)",
        )
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))

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
    trueskill_df: Optional[pd.DataFrame] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    history.to_csv(outdir / 'game_history_with_elo.csv', index=False)
    elo_df.to_csv(outdir / 'final_elo_ratings.csv', index=False)
    player_df.to_csv(outdir / 'player_summary.csv', index=False)
    pair_df.to_csv(outdir / 'pair_summary.csv', index=False)
    side_df.to_csv(outdir / 'side_summary.csv', index=False)
    teammate_matrix.to_csv(outdir / "teammate_matrix_win_pct.csv")
    overall_df.to_csv(outdir / 'overall_ranking.csv', index=False)
    corner_df.to_csv(outdir / 'corner_summary.csv', index=False)
    pred_df.to_csv(outdir / 'match_win_probabilities.csv', index=False)
    improvement_df.to_csv(outdir / 'player_improvement_curves.csv', index=False)
    if trueskill_df is not None and not trueskill_df.empty:
        trueskill_df.to_csv(outdir / 'trueskill_ratings.csv', index=False)


def print_summary(elo_df: pd.DataFrame, overall_df: pd.DataFrame, corner_df: pd.DataFrame, player_df: pd.DataFrame, pair_df: pd.DataFrame, side_df: pd.DataFrame, trueskill_df: Optional[pd.DataFrame]) -> None:
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

def run_analysis(input_file: str, output_dir: str) -> Path:
    df = load_games(input_file)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    history, elo_df = run_elo(df)
    trueskill_df = run_trueskill(df)
    player_df = build_player_summary(df)
    pair_df = build_pair_summary(df)
    side_df = build_side_summary(df)

    teammate_matrix = build_teammate_winpct_matrix(df)
    overall_df = build_overall_ranking(df)
    corner_df = build_corner_summary(df)
    pred_df = add_prediction_columns(df, elo_df)
    improvement_df = build_player_improvement(df)

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
        trueskill_df,
    )

    plot_player_elo(elo_df, outdir)
    plot_pair_strength(pair_df, outdir)
    plot_teammate_winpct_heatmap(teammate_matrix, outdir)
    maybe_plot_network(df, outdir)

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
    )

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
