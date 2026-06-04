"""
src/ingestion/player_ingest.py

Derives player profile features from the ball-by-ball DuckDB table.
No external API needed for training — everything computed from Cricsheet data.

Computes per player per season:
  Batting: avg, strike rate, boundary %, death-over SR, powerplay SR
  Bowling: economy, wicket rate, death-over economy, dot ball %
  Venue splits: batting SR and bowling economy per venue

Output: data/raw/player_profiles/
  player_batting.parquet
  player_bowling.parquet
  player_venue_splits.parquet

Usage:
    python -m src.ingestion.player_ingest
"""

import duckdb
import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DB_PATH     = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_DIR     = ROOT / "data" / "raw" / "player_profiles"

# Death overs: 16–20, Powerplay: 1–6, Middle: 7–15
POWERPLAY_OVERS = list(range(0, 6))   # over index 0-5
MIDDLE_OVERS    = list(range(6, 15))
DEATH_OVERS     = list(range(15, 20))


def load_balls() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute("SELECT * FROM balls").df()
    con.close()
    df["over"] = pd.to_numeric(df["ball"], errors="coerce").apply(lambda x: int(x) if pd.notna(x) else pd.NA)
    df["runs_off_bat"]  = pd.to_numeric(df["runs_off_bat"],  errors="coerce").fillna(0)
    df["is_wicket"]     = pd.to_numeric(df["is_wicket"],     errors="coerce").fillna(0)
    df["wides"]         = pd.to_numeric(df["wides"],         errors="coerce").fillna(0)
    df["noballs"]       = pd.to_numeric(df["noballs"],       errors="coerce").fillna(0)
    return df


def build_batting_profiles(balls: pd.DataFrame) -> pd.DataFrame:
    """Career and rolling season batting stats per player."""
    # Legal deliveries faced by striker (exclude wides)
    faced = balls[balls["wides"] == 0].copy()

    def phase_sr(df, over_list, suffix):
        phase = df[df["over"].isin(over_list)]
        return phase.groupby(["striker", "season"]).agg(
            **{f"runs_{suffix}":   ("runs_off_bat", "sum"),
               f"balls_{suffix}":  ("runs_off_bat", "count")}
        ).reset_index()

    overall = faced.groupby(["striker", "season"]).agg(
        runs_total       = ("runs_off_bat", "sum"),
        balls_faced      = ("runs_off_bat", "count"),
        fours            = ("is_four",      "sum") if "is_four" in faced.columns else ("runs_off_bat", lambda x: (x==4).sum()),
        sixes            = ("is_six",       "sum") if "is_six"  in faced.columns else ("runs_off_bat", lambda x: (x==6).sum()),
        dismissals       = ("is_wicket",    "sum"),
        innings_count    = ("match_id",     "nunique"),
    ).reset_index()

    overall["batting_avg"] = (
        overall["runs_total"] / overall["dismissals"].replace(0, 1)
    ).round(2)
    overall["strike_rate"] = (
        100 * overall["runs_total"] / overall["balls_faced"].replace(0, 1)
    ).round(2)
    overall["boundary_pct"] = (
        100 * (overall["fours"] + overall["sixes"]) / overall["balls_faced"].replace(0, 1)
    ).round(2)

    pp   = phase_sr(faced, POWERPLAY_OVERS, "pp")
    mid  = phase_sr(faced, MIDDLE_OVERS,    "mid")
    dth  = phase_sr(faced, DEATH_OVERS,     "death")

    for phase_df in [pp, mid, dth]:
        suffix = [c.split("_")[-1] for c in phase_df.columns if c.startswith("runs_")][0]
        phase_df[f"sr_{suffix}"] = (
            100 * phase_df[f"runs_{suffix}"] / phase_df[f"balls_{suffix}"].replace(0, 1)
        ).round(2)

    profile = overall.merge(pp,  on=["striker","season"], how="left")
    profile = profile.merge(mid, on=["striker","season"], how="left")
    profile = profile.merge(dth, on=["striker","season"], how="left")

    # Rolling 3-season form index (weighted average SR, last 3 seasons)
    profile = profile.sort_values(["striker","season"])
    profile["form_sr_roll3"] = (
        profile.groupby("striker")["strike_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(2)

    return profile.rename(columns={"striker": "player"})


def build_bowling_profiles(balls: pd.DataFrame) -> pd.DataFrame:
    """Career and phase bowling stats per bowler."""
    # Legal deliveries bowled (exclude wides and noballs for economy? 
    # Convention: wides and noballs count toward economy)
    legal = balls[(balls["wides"] == 0) & (balls["noballs"] == 0)].copy()

    def phase_econ(df, over_list, suffix):
        phase = df[df["over"].isin(over_list)]
        agg = phase.groupby(["bowler","season"]).agg(
            **{f"runs_conceded_{suffix}": ("total_runs", "sum"),
               f"balls_{suffix}":         ("total_runs", "count"),
               f"wickets_{suffix}":       ("is_wicket",  "sum")}
        ).reset_index()
        agg[f"economy_{suffix}"] = (
            6 * agg[f"runs_conceded_{suffix}"] / agg[f"balls_{suffix}"].replace(0,1)
        ).round(2)
        return agg

    overall = legal.groupby(["bowler","season"]).agg(
        wickets         = ("is_wicket",    "sum"),
        runs_conceded   = ("total_runs",   "sum"),
        balls_bowled    = ("total_runs",   "count"),
        dot_balls       = ("total_runs",   lambda x: (x==0).sum()),
        matches         = ("match_id",     "nunique"),
    ).reset_index()

    overall["economy"]    = (
        6 * overall["runs_conceded"] / overall["balls_bowled"].replace(0,1)
    ).round(2)
    overall["wicket_rate"] = (
        overall["wickets"] / overall["balls_bowled"].replace(0,1)
    ).round(4)
    overall["dot_pct"] = (
        100 * overall["dot_balls"] / overall["balls_bowled"].replace(0,1)
    ).round(2)

    pp  = phase_econ(legal, POWERPLAY_OVERS, "pp")
    mid = phase_econ(legal, MIDDLE_OVERS,    "mid")
    dth = phase_econ(legal, DEATH_OVERS,     "death")

    profile = overall.merge(pp,  on=["bowler","season"], how="left")
    profile = profile.merge(mid, on=["bowler","season"], how="left")
    profile = profile.merge(dth, on=["bowler","season"], how="left")

    # Rolling 3-season economy form
    profile = profile.sort_values(["bowler","season"])
    profile["form_economy_roll3"] = (
        profile.groupby("bowler")["economy"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(2)

    return profile.rename(columns={"bowler": "player"})


def build_venue_splits(balls: pd.DataFrame) -> pd.DataFrame:
    """Per player per venue batting SR and bowling economy."""
    faced = balls[balls["wides"] == 0].copy()
    legal = balls[(balls["wides"] == 0) & (balls["noballs"] == 0)].copy()

    bat = faced.groupby(["striker","venue"]).agg(
        runs_at_venue  = ("runs_off_bat", "sum"),
        balls_at_venue = ("runs_off_bat", "count"),
    ).reset_index()
    bat["sr_at_venue"] = (
        100 * bat["runs_at_venue"] / bat["balls_at_venue"].replace(0,1)
    ).round(2)
    bat = bat.rename(columns={"striker":"player"})

    bowl = legal.groupby(["bowler","venue"]).agg(
        runs_at_venue_bowl  = ("total_runs", "sum"),
        balls_at_venue_bowl = ("total_runs", "count"),
        wickets_at_venue    = ("is_wicket",  "sum"),
    ).reset_index()
    bowl["economy_at_venue"] = (
        6 * bowl["runs_at_venue_bowl"] / bowl["balls_at_venue_bowl"].replace(0,1)
    ).round(2)
    bowl = bowl.rename(columns={"bowler":"player"})

    splits = bat.merge(bowl, on=["player","venue"], how="outer")
    return splits


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[player] Loading ball-by-ball data ...")
    balls = load_balls()
    print(f"[player] {len(balls):,} deliveries loaded.")

    print("[player] Building batting profiles ...")
    batting = build_batting_profiles(balls)
    batting.to_parquet(OUT_DIR / "player_batting.parquet", index=False)
    print(f"  → {len(batting):,} player-season batting rows")

    print("[player] Building bowling profiles ...")
    bowling = build_bowling_profiles(balls)
    bowling.to_parquet(OUT_DIR / "player_bowling.parquet", index=False)
    print(f"  → {len(bowling):,} player-season bowling rows")

    print("[player] Building venue splits ...")
    splits = build_venue_splits(balls)
    splits.to_parquet(OUT_DIR / "player_venue_splits.parquet", index=False)
    print(f"  → {len(splits):,} player-venue rows")

    print("\n[player] Phase 1d complete — player profiles built.")

    # Quick summary
    print("\n[player] Top 10 batters by career strike rate (min 200 balls):")
    top_bat = (
        batting.groupby("player")
        .agg(balls=("balls_faced","sum"), sr=("strike_rate","mean"))
        .query("balls >= 200")
        .sort_values("sr", ascending=False)
        .head(10)
    )
    print(top_bat.to_string())


if __name__ == "__main__":
    run()
