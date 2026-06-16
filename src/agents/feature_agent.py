"""
src/agents/feature_agent.py
=============================
Feature Engineering Agent — reads raw data + ext_scores from state,
applies the shared feature schema, and writes the 44-feature vector to state.

This agent is the bridge between raw data and the ML model.
It must apply the EXACT same transformations as the offline ETL (Phase 2).
Both import from src/features/schema_loader.py to ensure consistency.
"""

import pickle
import pandas as pd
from pathlib import Path
from loguru import logger

from src.agents.state import MatchState
from src.features.schema_loader import get_feature_columns, get_fillna_map

ROOT        = Path(__file__).resolve().parents[2]
SCHED_PATH  = ROOT / "data" / "raw" / "schedule"       / "team_schedule.parquet"
SCALER_PATH = ROOT / "data" / "processed"              / "scaler.pkl"

# Load scaler once at import time
_SCALER = None
if SCALER_PATH.exists():
    with open(SCALER_PATH, "rb") as f:
        _SCALER = pickle.load(f)


def _load_venue_stats(venue: str) -> dict:
    """Load historical venue stats from schedule/DuckDB. Quick lookup."""
    try:
        import duckdb
        db  = ROOT / "data" / "processed" / "ipl.duckdb"
        con = duckdb.connect(str(db))
        row = con.execute(f"""
            SELECT
                AVG(CASE WHEN innings=1 THEN total_score END)   AS avg_first_innings,
                AVG(chasing_team_won)                           AS chasing_win_rate
            FROM (
                SELECT m.match_id, m.chasing_team_won,
                       SUM(b.total_runs) AS total_score, b.innings
                FROM matches m
                JOIN balls b ON m.match_id = b.match_id
                WHERE m.venue = '{venue.replace("'","''")}' AND b.innings = 1
                GROUP BY m.match_id, m.chasing_team_won, b.innings
            )
        """).fetchone()
        con.close()
        return {
            "venue_avg_first_innings_score": float(row[0]) if row[0] else 165.0,
            "venue_chasing_win_rate":        float(row[1]) if row[1] else 0.5,
        }
    except Exception:
        return {
            "venue_avg_first_innings_score": 165.0,
            "venue_chasing_win_rate":        0.5,
        }


def _load_team_form(team: str, season: str, role: str) -> dict:
    """Load last-5 win rate and average score for a team from DuckDB."""
    try:
        import duckdb
        db  = ROOT / "data" / "processed" / "ipl.duckdb"
        con = duckdb.connect(str(db))
        # Recent form: last 5 matches for this team
        rows = con.execute(f"""
            SELECT chasing_team_won, team1, team2
            FROM matches
            WHERE (team1 = '{team}' OR team2 = '{team}')
              AND season <= '{season}'
            ORDER BY start_date DESC
            LIMIT 5
        """).df()
        con.close()
        if rows.empty:
            return {}
        wins = sum(
            1 for _, r in rows.iterrows()
            if (r["team2"] == team and r["chasing_team_won"] == 1) or
               (r["team1"] == team and r["chasing_team_won"] == 0)
        )
        return {f"{role}_win_rate_last5": round(wins / len(rows), 3)}
    except Exception:
        return {}


def feature_agent(state: MatchState) -> MatchState:
    """
    LangGraph node — assembles the 44-feature vector aligned to the shared schema.
    Reads: raw_weather, raw_schedule_team1/2, raw_player_batting/bowling, ext_scores
    Writes: feature_vector, availability_flags
    """
    logger.info("[FeatureAgent] Building feature vector ...")

    fillna      = get_fillna_map()
    ext         = state.get("ext_scores", {})
    weather     = state.get("raw_weather", {})
    sched1      = state.get("raw_schedule_team1", {})
    sched2      = state.get("raw_schedule_team2", {})
    players     = state.get("raw_player_batting", {})
    venue       = state.get("venue", "")
    season      = state.get("season", "")
    team1       = state.get("team1", "")
    team2       = state.get("team2", "")
    override    = state.get("human_override", {})

    venue_stats = _load_venue_stats(venue)
    form1       = _load_team_form(team1, season, "team1")
    form2       = _load_team_form(team2, season, "team2")

    p1 = players.get("team1", {})
    p2 = players.get("team2", {})

    raw_vector = {
        # Match context
        "toss_winner_is_team1":        override.get("toss_winner_is_team1",   0),
        "toss_decision_bat":           override.get("toss_decision_bat",       0),
        "is_day_match":                int(not state.get("is_night_match", True)),
        "season_stage":                override.get("season_stage",            0),
        "match_number_in_season":      sched1.get("season_match_num",          1),

        # Venue
        "venue_encoded":               0,   # label encoding not meaningful at inference — use 0
        "team1_home_flag":             sched1.get("is_home",                   0),
        "team2_home_flag":             sched2.get("is_home",                   0),
        "venue_avg_first_innings_score": venue_stats.get("venue_avg_first_innings_score", 165.0),
        "venue_chasing_win_rate":      venue_stats.get("venue_chasing_win_rate",  0.5),
        "venue_avg_powerplay_score":   fillna["venue_avg_powerplay_score"],

        # Environmental (from weather + ext_scores)
        "dew_risk_flag":               ext.get("dew_risk_flag",               0),
        "temp_night_avg":              weather.get("temp_night_avg")          or fillna["temp_night_avg"],
        "humidity_night_avg":          weather.get("humidity_night_avg")      or fillna["humidity_night_avg"],
        "dewpoint_night_avg":          weather.get("dewpoint_night_avg")      or fillna["dewpoint_night_avg"],
        "windspeed_night_avg":         ext.get("windspeed_night_avg")         or fillna["windspeed_night_avg"],
        "pressure_night_avg":          ext.get("pressure_night_avg")          or fillna["pressure_night_avg"],
        "precipitation_mm":            weather.get("precipitation_mm")        or 0.0,

        # Fatigue
        "team1_rest_days":             min(ext.get("team1_rest_days",          7), 14),
        "team2_rest_days":             min(ext.get("team2_rest_days",          7), 14),
        "team1_back_to_back":          ext.get("team1_back_to_back",           0),
        "team2_back_to_back":          ext.get("team2_back_to_back",           0),
        "team1_travel_km":             ext.get("team1_travel_km",              0.0),
        "team2_travel_km":             ext.get("team2_travel_km",              0.0),
        "team1_travel_burden":         ext.get("team1_travel_burden",          0.0),
        "team2_travel_burden":         ext.get("team2_travel_burden",          0.0),
        "team1_season_match_num":      sched1.get("season_match_num",          1),
        "team2_season_match_num":      sched2.get("season_match_num",          1),

        # Team form
        "team1_win_rate_last5":        form1.get("team1_win_rate_last5",       0.5),
        "team2_win_rate_last5":        form2.get("team2_win_rate_last5",       0.5),
        "team1_avg_score_last5":       fillna["team1_avg_score_last5"],
        "team2_avg_score_last5":       fillna["team2_avg_score_last5"],
        "team1_avg_conceded_last5":    fillna["team1_avg_conceded_last5"],
        "team2_avg_conceded_last5":    fillna["team2_avg_conceded_last5"],
        "team1_chasing_win_rate":      fillna["team1_chasing_win_rate"],
        "team2_chasing_win_rate":      fillna["team2_chasing_win_rate"],

        # Player strength
        "team1_top4_avg_sr":           p1.get("top4_avg_sr")                  or fillna["team1_top4_avg_sr"],
        "team2_top4_avg_sr":           p2.get("top4_avg_sr")                  or fillna["team2_top4_avg_sr"],
        "team1_bowling_avg_economy":   p1.get("top4_avg_economy")             or fillna["team1_bowling_avg_economy"],
        "team2_bowling_avg_economy":   p2.get("top4_avg_economy")             or fillna["team2_bowling_avg_economy"],
        "team1_bowling_avg_economy_death": p1.get("top4_avg_economy_death")   or fillna["team1_bowling_avg_economy_death"],
        "team2_bowling_avg_economy_death": p2.get("top4_avg_economy_death")   or fillna["team2_bowling_avg_economy_death"],

        # Pitch
        "pitch_deterioration_index":   ext.get("pitch_deterioration_index",   0.5),
        "venue_spin_factor":           fillna["venue_spin_factor"],
    }

    # Apply human override (last write wins)
    raw_vector.update({k: v for k, v in override.items() if k in raw_vector})

    # Track availability
    availability = {
        k: "provided" if raw_vector[k] != fillna.get(k) else "fillna"
        for k in get_feature_columns()
        if k in raw_vector
    }

    # Apply scaler if available
    if _SCALER:
        try:
            from src.features.schema_loader import get_normalize_features
            import pandas as pd, numpy as np
            norm_cols = get_normalize_features()
            vec_df    = pd.DataFrame([raw_vector])
            for c in norm_cols:
                if c not in vec_df.columns:
                    vec_df[c] = fillna.get(c, 0)
            vec_df[norm_cols] = _SCALER.transform(vec_df[norm_cols])
            raw_vector = vec_df.iloc[0].to_dict()
        except Exception as e:
            state.get("warnings", []).append(f"Scaler apply failed: {e}")

    logger.info(f"  Feature vector: {len(raw_vector)} features | "
                f"fillna applied: {sum(1 for v in availability.values() if v == 'fillna')}")

    return {
        **state,
        "feature_vector":    raw_vector,
        "availability_flags": availability,
    }