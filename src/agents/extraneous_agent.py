"""
src/agents/extraneous_agent.py
================================
Extraneous Factor Agent — domain reasoning over raw environmental and
scheduling data. Computes derived risk scores and writes ext_scores to state.

This is NOT just field mapping. It applies cricket-domain logic:
  - Dew risk: dewpoint + humidity threshold → binary flag + severity score
  - Fatigue index: travel burden + rest days + back-to-back combined
  - Pitch Deterioration Index: venue historical pattern proxy
  - Heat stress: temperature + humidity → performance impact score

These computed scores are what the Feature Engineering Agent reads.
Separating this from feature engineering means the domain logic lives
in one place and can be updated without touching the ML feature pipeline.
"""

import math
from loguru import logger
from src.agents.state import MatchState


def _dew_risk_score(weather: dict) -> dict:
    """
    Dew risk assessment for night matches.
    Score 0.0–1.0: 0 = no risk, 1.0 = extreme dew expected.
    Favours chasing team (ball harder to grip for spinners after over 14).
    """
    if not weather.get("available"):
        return {"dew_risk_flag": 0, "dew_risk_score": 0.0, "dew_onset_over": None}

    dewpoint = weather.get("dewpoint_night_avg") or 0.0
    humidity = weather.get("humidity_night_avg") or 0.0

    # Binary flag: dewpoint > 15°C AND humidity > 70%
    flag  = int(dewpoint > 15 and humidity > 70)

    # Continuous score: how severe
    dew_score = 0.0
    if dewpoint > 15 and humidity > 70:
        dew_score = min(1.0, ((dewpoint - 15) / 10) * ((humidity - 70) / 30))

    # Estimated over of dew onset (typically between overs 12–18 in India)
    # Higher dew score → earlier onset
    onset_over = None
    if flag:
        onset_over = max(12, int(18 - dew_score * 6))

    return {
        "dew_risk_flag":  flag,
        "dew_risk_score": round(dew_score, 3),
        "dew_onset_over": onset_over,
    }


def _fatigue_index(schedule: dict, label: str) -> dict:
    """
    Composite fatigue index for one team.
    Combines rest days, travel distance, and back-to-back flag.
    Score 0.0–1.0: 0 = fully rested, 1.0 = maximally fatigued.
    """
    if not schedule.get("available"):
        return {
            f"{label}_fatigue_index":    0.3,   # neutral default
            f"{label}_rest_days":        7,
            f"{label}_travel_km":        0.0,
            f"{label}_back_to_back":     0,
            f"{label}_travel_burden":    0.0,
        }

    rest_days  = int(schedule.get("rest_days",  7))
    travel_km  = float(schedule.get("travel_km", 0) or 0)
    btb        = int(schedule.get("back_to_back", 0) or 0)

    # Clamp rest_days: 99 = first match of season = well rested
    if rest_days >= 99:
        rest_days = 7

    # Fatigue components (each 0–1)
    rest_fatigue    = max(0.0, 1.0 - (rest_days / 5.0))      # 0 days rest = 1.0
    travel_fatigue  = min(1.0, travel_km / 2500.0)            # 2500km = max
    btb_fatigue     = 0.4 if btb else 0.0

    fatigue_index = round(
        0.4 * rest_fatigue + 0.4 * travel_fatigue + 0.2 * btb_fatigue, 3
    )

    return {
        f"{label}_fatigue_index":  fatigue_index,
        f"{label}_rest_days":      min(rest_days, 14),
        f"{label}_travel_km":      travel_km,
        f"{label}_back_to_back":   btb,
        f"{label}_travel_burden":  round(travel_km / 1000 + (1 / max(rest_days, 1)), 4),
    }


def _heat_stress(weather: dict) -> float:
    """
    Heat-humidity composite. High values reduce bowling pace and batting focus.
    Score 0.0–1.0.
    """
    if not weather.get("available"):
        return 0.3
    temp     = weather.get("temp_night_avg") or 27.0
    humidity = weather.get("humidity_night_avg") or 65.0
    # Heat index proxy (simplified)
    heat = (temp - 25) / 15 + (humidity - 60) / 40
    return round(min(1.0, max(0.0, heat / 2)), 3)


def _pitch_deterioration_index(venue: str) -> float:
    """
    Venue-based pitch deterioration proxy.
    In production this would use per-match pitch reports.
    Current implementation uses venue-type heuristics.
    """
    # High-spin / deteriorating venues
    HIGH_SPIN = ["MA Chidambaram Stadium", "Eden Gardens", "Sawai Mansingh Stadium"]
    HIGH_PACE = ["Himachal Pradesh Cricket Association Stadium",
                 "Punjab Cricket Association IS Bindra Stadium"]

    if any(v in venue for v in HIGH_SPIN):
        return 0.75   # pitch deteriorates significantly, favours spinners in inn2
    if any(v in venue for v in HIGH_PACE):
        return 0.35   # pace-friendly, less deterioration
    return 0.50       # neutral default


def extraneous_agent(state: MatchState) -> MatchState:
    """
    LangGraph node — computes all extraneous factor scores.
    Reads raw data from state, writes ext_scores dict to state.
    """
    venue   = state.get("venue", "")
    weather = state.get("raw_weather", {})
    sched1  = state.get("raw_schedule_team1", {})
    sched2  = state.get("raw_schedule_team2", {})

    logger.info(f"[ExtraneousAgent] Computing scores for {venue}")

    dew     = _dew_risk_score(weather)
    fat1    = _fatigue_index(sched1, "team1")
    fat2    = _fatigue_index(sched2, "team2")
    heat    = _heat_stress(weather)
    pdi     = _pitch_deterioration_index(venue)

    ext_scores = {
        # Dew
        **dew,
        # Fatigue
        **fat1,
        **fat2,
        # Environment
        "heat_stress_index":        heat,
        # Pitch
        "pitch_deterioration_index": pdi,
        # Wind (directly from weather)
        "windspeed_night_avg":       weather.get("windspeed_night_avg") or 10.0,
        "pressure_night_avg":        weather.get("pressure_night_avg")  or 1010.0,
    }

    logger.info(f"  dew_risk={dew['dew_risk_flag']} score={dew['dew_risk_score']} "
                f"| t1_fatigue={fat1['team1_fatigue_index']} "
                f"| t2_fatigue={fat2['team2_fatigue_index']} "
                f"| PDI={pdi}")

    return {**state, "ext_scores": ext_scores}