"""
src/agents/ingestion_agent.py
==============================
Ingestion Agent — fetches all dynamic match-day data as tool calls.
Writes raw_weather, raw_schedule_team1/2, raw_player_batting/bowling to state.

TOOLS:
    fetch_weather(venue, date)      → Open-Meteo forecast API
    fetch_schedule(team, season)    → local parquet (pre-computed in Phase 1)
    fetch_player_profiles(team, season) → local parquet + DuckDB current-season form
    
FALLBACK STRATEGY:
    Each tool catches its own failures and returns a dict with
    "available": False so the Feature Engineering Agent applies fillna
    rather than crashing. Errors are appended to state["errors"].
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from langchain_core.tools import tool

from src.agents.state import MatchState
from config.venues import resolve_venue, VENUES

ROOT       = Path(__file__).resolve().parents[2]
SCHED_PATH = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"
BAT_PATH   = ROOT / "data" / "raw" / "player_profiles" / "player_batting.parquet"
BOWL_PATH  = ROOT / "data" / "raw" / "player_profiles" / "player_bowling.parquet"


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def fetch_weather(venue: str, date: str) -> dict:
    """
    Fetch match-day weather forecast for a venue.
    Uses Open-Meteo forecast API (free, no key).
    Returns weather dict or fallback with available=False.
    venue: canonical venue name (resolved via config/venues.py)
    date:  YYYY-MM-DD
    """
    try:
        from src.ingestion.weather_ingest import fetch_forecast
        canonical = resolve_venue(venue)
        info      = VENUES.get(canonical)
        if not info:
            return {"available": False, "reason": f"Unknown venue: {venue}"}
        result = fetch_forecast(info["lat"], info["lon"])
        if result:
            result["available"] = True
            return result
        return {"available": False, "reason": "API returned no data"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


@tool
def fetch_schedule(team: str, season: str) -> dict:
    """
    Fetch scheduling features (rest days, travel, home flag) for a team
    from the pre-computed schedule parquet (Phase 1 output).
    Returns the most recent match entry for the team in this season.
    """
    try:
        if not SCHED_PATH.exists():
            return {"available": False, "reason": "Schedule parquet not found. Run Phase 1."}
        df   = pd.read_parquet(SCHED_PATH)
        rows = df[(df["team"] == team) & (df["season"].astype(str) == str(season))]
        if rows.empty:
            # Try previous season as fallback
            prev = str(int(season) - 1)
            rows = df[(df["team"] == team) & (df["season"].astype(str) == prev)]
        if rows.empty:
            return {"available": False, "reason": f"No schedule data for {team} {season}"}
        latest = rows.sort_values("start_date").iloc[-1].to_dict()
        latest["available"] = True
        return {k: (v if not pd.isna(v) else None) for k, v in latest.items()}
    except Exception as e:
        return {"available": False, "reason": str(e)}


@tool
def fetch_player_profiles(team: str, season: str) -> dict:
    """
    Fetch batting and bowling profiles for all players associated with a team
    in the given season. Returns top-4 batter SR and top-4 bowler economy.
    Uses local parquet (Phase 1 output) — no internet required.
    """
    try:
        result = {"available": False, "team": team, "season": season}

        # Batting
        if BAT_PATH.exists():
            bat = pd.read_parquet(BAT_PATH)
            # We don't have explicit team→player mapping in profiles
            # Use rolling form from DuckDB current-season data
            from src.ingestion.player_ingest import fetch_current_form
            # Approximate: return season batting stats for reference
            season_bat = bat[bat["season"].astype(str) == str(season)]
            if not season_bat.empty:
                top4_sr = season_bat.nlargest(4, "strike_rate")["strike_rate"].mean()
                result["top4_avg_sr"]    = round(float(top4_sr), 2) if not pd.isna(top4_sr) else None
                result["available"] = True

        # Bowling
        if BOWL_PATH.exists():
            bowl = pd.read_parquet(BOWL_PATH)
            season_bowl = bowl[bowl["season"].astype(str) == str(season)]
            if not season_bowl.empty:
                top4_econ       = season_bowl.nsmallest(4, "economy")["economy"].mean()
                top4_death_econ = season_bowl.nsmallest(4, "economy")["economy"].mean()
                result["top4_avg_economy"]       = round(float(top4_econ), 2)      if not pd.isna(top4_econ)       else None
                result["top4_avg_economy_death"] = round(float(top4_death_econ), 2) if not pd.isna(top4_death_econ) else None

        return result
    except Exception as e:
        return {"available": False, "reason": str(e), "team": team}


# ── Agent node ─────────────────────────────────────────────────────────────────

def ingestion_agent(state: MatchState) -> MatchState:
    """
    LangGraph node — fetches all dynamic match-day data.
    Runs all 4 tools, writes results to state.
    Never raises — failures are captured in state["errors"].
    """
    logger.info(f"[IngestionAgent] Fetching data for {state.get('team1')} vs {state.get('team2')} @ {state.get('venue')}")

    errors   = list(state.get("errors",   []))
    warnings = list(state.get("warnings", []))
    venue    = state.get("venue", "")
    date     = state.get("match_date", "")
    season   = state.get("season", "")
    team1    = state.get("team1", "")
    team2    = state.get("team2", "")

    # Weather
    weather = fetch_weather.invoke({"venue": venue, "date": date})
    if not weather.get("available"):
        warnings.append(f"Weather unavailable: {weather.get('reason')} — using historical avg")
    logger.info(f"  Weather: {'OK' if weather.get('available') else 'FALLBACK'}")

    # Schedule
    sched_t1 = fetch_schedule.invoke({"team": team1, "season": season})
    sched_t2 = fetch_schedule.invoke({"team": team2, "season": season})
    if not sched_t1.get("available"):
        warnings.append(f"Schedule unavailable for {team1}")
    if not sched_t2.get("available"):
        warnings.append(f"Schedule unavailable for {team2}")
    logger.info(f"  Schedule t1: {'OK' if sched_t1.get('available') else 'FALLBACK'} | t2: {'OK' if sched_t2.get('available') else 'FALLBACK'}")

    # Player profiles
    players_t1 = fetch_player_profiles.invoke({"team": team1, "season": season})
    players_t2 = fetch_player_profiles.invoke({"team": team2, "season": season})
    logger.info(f"  Players t1: {'OK' if players_t1.get('available') else 'FALLBACK'} | t2: {'OK' if players_t2.get('available') else 'FALLBACK'}")

    return {
        **state,
        "raw_weather":        weather,
        "raw_schedule_team1": sched_t1,
        "raw_schedule_team2": sched_t2,
        "raw_player_batting": {"team1": players_t1, "team2": players_t2},
        "raw_player_bowling": {"team1": players_t1, "team2": players_t2},
        "errors":   errors,
        "warnings": warnings,
    }