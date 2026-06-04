"""
src/ingestion/schedule_ingest.py

Builds the IPL schedule dataset including:
  - Match sequence per team per season
  - Travel distance between consecutive venues (haversine)
  - Rest days between consecutive matches
  - Back-to-back flag (< 48 hrs rest)
  - Home / away flag per team per match

Input:  DuckDB matches table (from cricsheet_ingest.py)
Output: data/raw/schedule/team_schedule.parquet
        One row per team per match. Each team appears twice per match.

Usage:
    python -m src.ingestion.schedule_ingest
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

ROOT     = Path(__file__).resolve().parents[2]
DB_PATH  = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_PATH = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"

# Home ground mapping per franchise
# Accounts for franchise renames and relocations over IPL history
HOME_GROUNDS = {
    "Royal Challengers Bangalore":    "M Chinnaswamy Stadium",
    "Royal Challengers Bengaluru":    "M Chinnaswamy Stadium",
    "Mumbai Indians":                 "Wankhede Stadium",
    "Kolkata Knight Riders":          "Eden Gardens",
    "Chennai Super Kings":            "MA Chidambaram Stadium",
    "Delhi Daredevils":               "Arun Jaitley Stadium",
    "Delhi Capitals":                 "Arun Jaitley Stadium",
    "Sunrisers Hyderabad":            "Rajiv Gandhi International Stadium",
    "Deccan Chargers":                "Rajiv Gandhi International Stadium",
    "Kings XI Punjab":                "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Kings":                   "Punjab Cricket Association IS Bindra Stadium",
    "Rajasthan Royals":               "Sawai Mansingh Stadium",
    "Gujarat Titans":                 "Narendra Modi Stadium",
    "Lucknow Super Giants":           "Ekana Cricket Stadium",
    "Kochi Tuskers Kerala":           "Greenfield International Stadium",
    "Pune Warriors":                  "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiant":         "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiants":        "Maharashtra Cricket Association Stadium",
    "Gujarat Lions":                  "Narendra Modi Stadium",
}

# City coordinates for each venue (used in travel calc)
VENUE_CITY = {
    "M Chinnaswamy Stadium":                        (12.9791, 77.5497),
    "Wankhede Stadium":                             (18.9389, 72.8258),
    "Eden Gardens":                                 (22.5645, 88.3433),
    "MA Chidambaram Stadium":                       (13.0628, 80.2791),
    "Arun Jaitley Stadium":                         (28.6364, 77.2195),
    "Feroz Shah Kotla":                             (28.6364, 77.2195),
    "Rajiv Gandhi International Stadium":           (17.4042, 78.5498),
    "Punjab Cricket Association IS Bindra Stadium": (30.6842, 76.7154),
    "Punjab Cricket Association Stadium, Mohali":   (30.6842, 76.7154),
    "Sawai Mansingh Stadium":                       (26.8972, 75.8024),
    "Narendra Modi Stadium":                        (23.0900, 72.0847),
    "Sardar Patel Stadium":                         (23.0900, 72.0847),
    "Brabourne Stadium":                            (18.9322, 72.8264),
    "DY Patil Stadium":                             (19.0435, 72.9987),
    "Dr DY Patil Sports Academy":                   (19.0435, 72.9987),
    "Maharashtra Cricket Association Stadium":      (18.6298, 73.8015),
    "JSCA International Stadium Complex":           (23.3441, 85.3096),
    "Himachal Pradesh Cricket Association Stadium": (32.2190, 76.3234),
    "Barsapara Cricket Stadium":                    (26.1433, 91.7898),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": (17.7231, 83.2183),
    "Barabati Stadium":                             (20.4686, 85.8792),
    "Greenfield International Stadium":             (8.5553,  76.9063),
    "Holkar Cricket Stadium":                       (22.7215, 75.8578),
    "Ekana Cricket Stadium":                        (26.8575, 80.9346),
    "Newlands":                                     (-33.9258, 18.4232),
    "St George's Park":                             (-33.9608, 25.6022),
    "Kingsmead":                                    (-29.8579, 31.0292),
    "SuperSport Park":                              (-25.7547, 28.2267),
    "New Wanderers Stadium":                        (-26.1446, 28.0566),
    "Dubai International Cricket Stadium":          (25.0359, 55.2466),
    "Sheikh Zayed Stadium":                         (24.3886, 54.5195),
    "Sharjah Cricket Stadium":                      (25.3396, 55.3839),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance between two coordinates in km."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi       = radians(lat2 - lat1)
    dlambda    = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


def load_matches() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    matches = con.execute(
        "SELECT match_id, season, start_date, venue, team1, team2 FROM matches"
    ).df()
    con.close()
    matches["start_date"] = pd.to_datetime(matches["start_date"])
    return matches.sort_values(["season", "start_date"]).reset_index(drop=True)


def build_team_schedule(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each match into two rows (one per team).
    Compute per-team sequential features.
    """
    rows = []
    for _, m in matches.iterrows():
        for team in [m["team1"], m["team2"]]:
            opponent = m["team2"] if team == m["team1"] else m["team1"]
            home_ground = HOME_GROUNDS.get(team)
            is_home = (home_ground is not None and
                       m["venue"].startswith(home_ground[:20]))
            rows.append({
                "match_id":   m["match_id"],
                "season":     m["season"],
                "start_date": m["start_date"],
                "venue":      m["venue"],
                "team":       team,
                "opponent":   opponent,
                "is_home":    int(is_home),
            })

    df = pd.DataFrame(rows).sort_values(["team", "season", "start_date"])
    df = df.reset_index(drop=True)

    # Previous match info per team
    df["prev_venue"]      = df.groupby(["team", "season"])["venue"].shift(1)
    df["prev_match_date"] = df.groupby(["team", "season"])["start_date"].shift(1)

    # Rest days — ensure datetime before subtraction
    df["start_date"]       = pd.to_datetime(df["start_date"])
    df["prev_match_date"]  = pd.to_datetime(df["prev_match_date"])
    df["rest_days"] = (
        df["start_date"] - df["prev_match_date"]
    ).dt.days.fillna(99).astype(int)

    # Back-to-back flag: < 2 days rest
    df["back_to_back"] = (df["rest_days"] < 2).astype(int)

    # Travel distance (haversine, km)
    def get_travel(row) -> float | None:
        if pd.isna(row["prev_venue"]):
            return 0.0
        c1 = VENUE_CITY.get(row["venue"])
        c2 = VENUE_CITY.get(row["prev_venue"])
        if c1 and c2:
            return round(haversine_km(c1[0], c1[1], c2[0], c2[1]), 1)
        return None

    df["travel_km"] = df.apply(get_travel, axis=1)

    # Fatigue proxy: high travel + low rest
    df["travel_burden_score"] = (
        df["travel_km"].fillna(0) / 1000.0 +           # normalised distance
        (1 / df["rest_days"].clip(lower=1))             # inverse rest
    ).round(4)

    # Match number in season for this team
    df["season_match_num"] = df.groupby(["team", "season"]).cumcount() + 1

    return df[[
        "match_id", "season", "start_date", "venue", "team", "opponent",
        "is_home", "rest_days", "back_to_back",
        "travel_km", "travel_burden_score", "season_match_num",
        "prev_venue",
    ]]


def run() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[schedule] Loading matches from DuckDB ...")
    matches = load_matches()
    print(f"[schedule] {len(matches)} matches loaded.")

    schedule = build_team_schedule(matches)
    schedule.to_parquet(OUT_PATH, index=False)

    print(f"[schedule] {len(schedule)} team-match rows written to {OUT_PATH}")
    print("\n[schedule] Sample — high travel burden matches:")
    print(
        schedule[schedule["travel_km"] > 1500]
        [["team", "start_date", "venue", "travel_km", "rest_days", "travel_burden_score"]]
        .sort_values("travel_km", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
