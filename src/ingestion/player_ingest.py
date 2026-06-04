"""
src/ingestion/player_ingest.py
================================
Derives player batting and bowling profiles entirely from the DuckDB
ball-by-ball table. No external API calls required.

STORAGE:
    data/raw/player_profiles/player_batting.parquet
    data/raw/player_profiles/player_bowling.parquet
    data/raw/player_profiles/player_venue_splits.parquet

WHEN TO RE-RUN:
    - First-time setup: run once after cricsheet_ingest.py.
    - End of each IPL season: re-run after cricsheet_ingest.py updates DuckDB
      to include the completed season's data.
    - Never on match day for training data — fully offline.
    - For INFERENCE TIME current-season form: the Ingestion Agent in Phase 4
      will call fetch_current_form() from this module using DuckDB data
      filtered to the current season only.

IDEMPOTENT: Fully recomputes and overwrites all three parquet files on each run.

COMPUTED PROFILES:

    Batting (per player per season):
        batting_avg, strike_rate, boundary_pct, dismissals
        sr_powerplay (overs 1–6), sr_middle (7–15), sr_death (16–20)
        form_sr_roll3 (rolling 3-season strike rate — form proxy)

    Bowling (per player per season):
        economy, wicket_rate, dot_pct
        economy_powerplay, economy_middle, economy_death
        form_economy_roll3 (rolling 3-season economy — form proxy)

    Venue splits (per player per venue, career):
        sr_at_venue (batting), economy_at_venue (bowling)

USAGE:
    ipl_venv/bin/python -m src.ingestion.player_ingest
"""

import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_DIR = ROOT / "data" / "raw" / "player_profiles"

POWERPLAY = list(range(0, 6))
MIDDLE    = list(range(6, 15))
DEATH     = list(range(15, 20))


def _load_balls() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute("SELECT * FROM balls").df()
    con.close()

    df["over"]         = pd.to_numeric(df.get("over"),         errors="coerce")
    df["runs_off_bat"] = pd.to_numeric(df.get("runs_off_bat"), errors="coerce").fillna(0)
    df["is_wicket"]    = pd.to_numeric(df.get("is_wicket"),    errors="coerce").fillna(0)
    df["total_runs"]   = pd.to_numeric(df.get("total_runs"),   errors="coerce").fillna(0)
    df["wides"]        = pd.to_numeric(df.get("wides"),        errors="coerce").fillna(0)
    df["noballs"]      = pd.to_numeric(df.get("noballs"),      errors="coerce").fillna(0)
    df["is_four"]      = pd.to_numeric(df.get("is_four"),      errors="coerce").fillna(0)
    df["is_six"]       = pd.to_numeric(df.get("is_six"),       errors="coerce").fillna(0)
    return df


def _phase_batting_sr(faced: pd.DataFrame, over_list: list, suffix: str) -> pd.DataFrame:
    """Strike rate in a specific over phase, per player per season."""
    phase = faced[faced["over"].isin(over_list)]
    agg   = phase.groupby(["striker", "season"]).agg(
        **{f"runs_{suffix}":  ("runs_off_bat", "sum"),
           f"balls_{suffix}": ("runs_off_bat", "count")}
    ).reset_index()
    agg[f"sr_{suffix}"] = (
        100 * agg[f"runs_{suffix}"] / agg[f"balls_{suffix}"].replace(0, 1)
    ).round(2)
    return agg[["striker", "season", f"sr_{suffix}"]]


def _phase_bowling_economy(legal: pd.DataFrame, over_list: list, suffix: str) -> pd.DataFrame:
    """Economy in a specific over phase, per bowler per season."""
    phase = legal[legal["over"].isin(over_list)]
    agg   = phase.groupby(["bowler", "season"]).agg(
        **{f"runs_{suffix}":  ("total_runs", "sum"),
           f"balls_{suffix}": ("total_runs", "count")}
    ).reset_index()
    agg[f"economy_{suffix}"] = (
        6 * agg[f"runs_{suffix}"] / agg[f"balls_{suffix}"].replace(0, 1)
    ).round(2)
    return agg[["bowler", "season", f"economy_{suffix}"]]


def build_batting_profiles(balls: pd.DataFrame) -> pd.DataFrame:
    """Career and per-season batting stats. Excludes wide deliveries (not faced)."""
    faced = balls[balls["wides"] == 0].copy()

    overall = faced.groupby(["striker", "season"]).agg(
        runs_total    = ("runs_off_bat", "sum"),
        balls_faced   = ("runs_off_bat", "count"),
        fours         = ("is_four",      "sum"),
        sixes         = ("is_six",       "sum"),
        dismissals    = ("is_wicket",    "sum"),
        innings       = ("match_id",     "nunique"),
    ).reset_index()

    overall["batting_avg"]   = (overall["runs_total"] / overall["dismissals"].replace(0, 1)).round(2)
    overall["strike_rate"]   = (100 * overall["runs_total"] / overall["balls_faced"].replace(0, 1)).round(2)
    overall["boundary_pct"]  = (100 * (overall["fours"] + overall["sixes"]) / overall["balls_faced"].replace(0, 1)).round(2)

    pp  = _phase_batting_sr(faced, POWERPLAY, "powerplay")
    mid = _phase_batting_sr(faced, MIDDLE,    "middle")
    dth = _phase_batting_sr(faced, DEATH,     "death")

    profile = overall.merge(pp,  on=["striker", "season"], how="left")
    profile = profile.merge(mid, on=["striker", "season"], how="left")
    profile = profile.merge(dth, on=["striker", "season"], how="left")

    # Rolling 3-season form index (weighted recent performance proxy)
    profile = profile.sort_values(["striker", "season"])
    profile["form_sr_roll3"] = (
        profile.groupby("striker")["strike_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(2)

    return profile.rename(columns={"striker": "player"})


def build_bowling_profiles(balls: pd.DataFrame) -> pd.DataFrame:
    """Career and per-season bowling stats. Uses legal deliveries for economy."""
    legal = balls[(balls["wides"] == 0) & (balls["noballs"] == 0)].copy()

    overall = legal.groupby(["bowler", "season"]).agg(
        wickets      = ("is_wicket",  "sum"),
        runs_conceded= ("total_runs", "sum"),
        balls_bowled = ("total_runs", "count"),
        dot_balls    = ("total_runs", lambda x: (x == 0).sum()),
        matches      = ("match_id",   "nunique"),
    ).reset_index()

    overall["economy"]     = (6 * overall["runs_conceded"] / overall["balls_bowled"].replace(0, 1)).round(2)
    overall["wicket_rate"] = (overall["wickets"] / overall["balls_bowled"].replace(0, 1)).round(4)
    overall["dot_pct"]     = (100 * overall["dot_balls"] / overall["balls_bowled"].replace(0, 1)).round(2)

    pp  = _phase_bowling_economy(legal, POWERPLAY, "powerplay")
    mid = _phase_bowling_economy(legal, MIDDLE,    "middle")
    dth = _phase_bowling_economy(legal, DEATH,     "death")

    profile = overall.merge(pp,  on=["bowler", "season"], how="left")
    profile = profile.merge(mid, on=["bowler", "season"], how="left")
    profile = profile.merge(dth, on=["bowler", "season"], how="left")

    profile = profile.sort_values(["bowler", "season"])
    profile["form_economy_roll3"] = (
        profile.groupby("bowler")["economy"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(2)

    return profile.rename(columns={"bowler": "player"})


def build_venue_splits(balls: pd.DataFrame) -> pd.DataFrame:
    """Career batting SR and bowling economy per player per venue."""
    faced = balls[balls["wides"] == 0].copy()
    legal = balls[(balls["wides"] == 0) & (balls["noballs"] == 0)].copy()

    bat = faced.groupby(["striker", "venue"]).agg(
        runs_at_venue  = ("runs_off_bat", "sum"),
        balls_at_venue = ("runs_off_bat", "count"),
    ).reset_index()
    bat["sr_at_venue"] = (100 * bat["runs_at_venue"] / bat["balls_at_venue"].replace(0, 1)).round(2)

    bowl = legal.groupby(["bowler", "venue"]).agg(
        runs_bowl_venue   = ("total_runs", "sum"),
        balls_bowl_venue  = ("total_runs", "count"),
        wickets_at_venue  = ("is_wicket",  "sum"),
    ).reset_index()
    bowl["economy_at_venue"] = (6 * bowl["runs_bowl_venue"] / bowl["balls_bowl_venue"].replace(0, 1)).round(2)

    bat  = bat.rename(columns={"striker": "player"})
    bowl = bowl.rename(columns={"bowler": "player"})
    return bat.merge(bowl, on=["player", "venue"], how="outer")


def fetch_current_form(player: str, season: str, role: str = "batting") -> dict:
    """
    Fetch current-season form for a player directly from DuckDB.
    Called by the Ingestion Agent at inference time for live matches.
    role: 'batting' or 'bowling'
    """
    con = duckdb.connect(str(DB_PATH))
    if role == "batting":
        result = con.execute(f"""
            SELECT
                SUM(runs_off_bat)                               AS runs,
                COUNT(*)                                        AS balls,
                100.0 * SUM(runs_off_bat) / NULLIF(COUNT(*),0) AS strike_rate,
                SUM(is_wicket)                                  AS dismissals
            FROM balls
            WHERE striker = '{player}' AND season = '{season}' AND wides = 0
        """).df()
    else:
        result = con.execute(f"""
            SELECT
                SUM(total_runs)                                   AS runs_conceded,
                COUNT(*)                                          AS balls,
                6.0 * SUM(total_runs) / NULLIF(COUNT(*),0)       AS economy,
                SUM(is_wicket)                                    AS wickets
            FROM balls
            WHERE bowler = '{player}' AND season = '{season}'
              AND wides = 0 AND noballs = 0
        """).df()
    con.close()
    return result.iloc[0].to_dict() if not result.empty else {}


def run() -> None:
    logger.info("=== Phase 1d: Player Profile Ingestion ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    balls = _load_balls()
    logger.info(f"Loaded {len(balls):,} deliveries from DuckDB.")

    logger.info("Building batting profiles ...")
    batting = build_batting_profiles(balls)
    batting.to_parquet(OUT_DIR / "player_batting.parquet", index=False)
    logger.success(f"Batting: {len(batting):,} player-season rows")

    logger.info("Building bowling profiles ...")
    bowling = build_bowling_profiles(balls)
    bowling.to_parquet(OUT_DIR / "player_bowling.parquet", index=False)
    logger.success(f"Bowling: {len(bowling):,} player-season rows")

    logger.info("Building venue splits ...")
    splits = build_venue_splits(balls)
    splits.to_parquet(OUT_DIR / "player_venue_splits.parquet", index=False)
    logger.success(f"Venue splits: {len(splits):,} player-venue rows")

    logger.info("\nTop 10 batters by career strike rate (min 200 balls):")
    top_bat = (
        batting.groupby("player")
        .agg(balls=("balls_faced", "sum"), sr=("strike_rate", "mean"))
        .query("balls >= 200")
        .sort_values("sr", ascending=False)
        .head(10)
    )
    print(top_bat.to_string())


if __name__ == "__main__":
    run()