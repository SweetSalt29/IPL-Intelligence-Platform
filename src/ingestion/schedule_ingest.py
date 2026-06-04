"""
src/ingestion/schedule_ingest.py
=================================
Builds the IPL schedule dataset from matches already in DuckDB.
Computes per-team scheduling burden features for every match.

STORAGE:
    data/raw/schedule/team_schedule.parquet
    One row per team per match (every match produces 2 rows — one per team).

WHEN TO RE-RUN:
    - First-time setup: run once after cricsheet_ingest.py.
    - End of each IPL season: re-run after cricsheet_ingest.py updates DuckDB.
    - Never needs to run on match day — this is a static derived dataset.
    - If HOME_GROUNDS mapping is updated (new franchise, relocation):
      re-run to recompute home/away flags.

IDEMPOTENT: Fully recomputes and overwrites output on each run.

COMPUTED FEATURES:
    rest_days           — days since team's previous match this season
    back_to_back        — flag: rest_days < 2
    travel_km           — haversine distance from previous venue
    travel_burden_score — composite: travel_km/1000 + 1/rest_days
    is_home             — 1 if playing at designated home ground
    season_match_num    — match number within the season for this team

USAGE:
    ipl_venv/bin/python -m src.ingestion.schedule_ingest
"""

import duckdb
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from loguru import logger

from config.venues import VENUES, HOME_GROUNDS, resolve_venue

ROOT     = Path(__file__).resolve().parents[2]
DB_PATH  = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_PATH = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometres."""
    R    = 6371.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dl   = radians(lon2 - lon1)
    a    = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dl/2)**2
    return round(2 * R * atan2(sqrt(a), sqrt(1 - a)), 1)


def _venue_coords(venue_raw: str) -> tuple[float, float] | None:
    canonical = resolve_venue(venue_raw)
    info      = VENUES.get(canonical)
    return (info["lat"], info["lon"]) if info else None


def load_matches() -> pd.DataFrame:
    con     = duckdb.connect(str(DB_PATH))
    matches = con.execute(
        "SELECT match_id, season, start_date, venue, team1, team2 FROM matches"
    ).df()
    con.close()
    matches["start_date"] = pd.to_datetime(matches["start_date"])
    return matches.sort_values(["season", "start_date"]).reset_index(drop=True)


def build_team_schedule(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each match into two team-level rows and compute scheduling features.
    """
    rows = []
    for _, m in matches.iterrows():
        for team in [m["team1"], m["team2"]]:
            opponent    = m["team2"] if team == m["team1"] else m["team1"]
            home_ground = HOME_GROUNDS.get(team, "")
            canonical_v = resolve_venue(m["venue"])
            is_home     = int(home_ground != "" and canonical_v == home_ground)
            rows.append({
                "match_id":   m["match_id"],
                "season":     m["season"],
                "start_date": m["start_date"],
                "venue":      m["venue"],
                "team":       team,
                "opponent":   opponent,
                "is_home":    is_home,
            })

    df = pd.DataFrame(rows).sort_values(["team", "season", "start_date"]).reset_index(drop=True)

    # Previous match info per team within the same season
    df["prev_venue"]      = df.groupby(["team", "season"])["venue"].shift(1)
    df["prev_match_date"] = df.groupby(["team", "season"])["start_date"].shift(1)
    df["prev_match_date"] = pd.to_datetime(df["prev_match_date"])

    # Rest days (99 = first match of season, no prior match)
    df["rest_days"] = (
        df["start_date"] - df["prev_match_date"]
    ).dt.days.fillna(99).astype(int)

    # Back-to-back: less than 2 days rest
    df["back_to_back"] = (df["rest_days"] < 2).astype(int)

    # Travel distance via haversine
    def calc_travel(row) -> float:
        if pd.isna(row["prev_venue"]):
            return 0.0
        c1 = _venue_coords(row["venue"])
        c2 = _venue_coords(row["prev_venue"])
        if c1 and c2:
            return haversine_km(c1[0], c1[1], c2[0], c2[1])
        return None

    df["travel_km"] = df.apply(calc_travel, axis=1)

    # Travel burden: composite score — high travel + low rest = high burden
    # rest_days clipped at 1 to avoid divide by zero on back-to-back
    df["travel_burden_score"] = (
        df["travel_km"].fillna(0) / 1000.0 +
        (1.0 / df["rest_days"].replace(99, 7).clip(lower=1))
    ).round(4)

    # Match sequence number within season for this team
    df["season_match_num"] = df.groupby(["team", "season"]).cumcount() + 1

    return df[[
        "match_id", "season", "start_date", "venue", "team", "opponent",
        "is_home", "rest_days", "back_to_back",
        "travel_km", "travel_burden_score", "season_match_num",
    ]]


def run() -> None:
    logger.info("=== Phase 1c: Schedule Ingestion ===")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    matches  = load_matches()
    logger.info(f"Loaded {len(matches)} matches from DuckDB.")

    schedule = build_team_schedule(matches)
    schedule.to_parquet(OUT_PATH, index=False)

    logger.success(f"{len(schedule)} team-match rows → {OUT_PATH}")
    logger.info(f"Average rest days (excl. first match): "
                f"{schedule[schedule['rest_days'] < 99]['rest_days'].mean():.1f}")
    logger.info(f"Back-to-back matches: {schedule['back_to_back'].sum()}")
    logger.info(f"Longest travel legs:")
    top_travel = (
        schedule[schedule["travel_km"] > 0]
        .nlargest(5, "travel_km")
        [["team", "start_date", "venue", "travel_km", "rest_days"]]
    )
    print(top_travel.to_string(index=False))


if __name__ == "__main__":
    run()