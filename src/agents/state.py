"""
src/agents/state.py
====================
LangGraph shared state — the single object all agents read from and write to.
No agent passes data directly to another agent. Everything goes through state.

FIELDS:
    Input (set by match trigger / human override):
        match_id, team1, team2, venue, match_date, season
        is_night_match, human_override

    Populated by Ingestion Agent:
        raw_weather, raw_schedule_team1, raw_schedule_team2
        raw_player_batting, raw_player_bowling

    Populated by Extraneous Factor Agent:
        ext_scores (dew_risk, fatigue, PDI, travel burden etc.)

    Populated by Feature Engineering Agent:
        feature_vector (44-feature dict aligned to schema)
        availability_flags (which features used fillna)

    Populated by Prediction Agent (via ML model tool):
        win_prob_chasing, win_prob_batting, confidence
        top_drivers, model_version

    Populated by Narrative Agent:
        narrative (plain-English tactical brief)

    System fields:
        errors (list of non-fatal errors — degraded mode)
        warnings (list of warnings)
        rerun_triggered (live monitor flag — not used in Phase 4)
"""

from typing import TypedDict, Optional, Any


class MatchState(TypedDict, total=False):

    # ── Match identity ─────────────────────────────────────────────
    match_id:       str
    team1:          str       # team batting first
    team2:          str       # team batting second (chasing)
    pov_team:       str       # team perspective for narrative
    venue:          str
    match_date:     str       # YYYY-MM-DD
    season:         str
    is_night_match: bool
    human_override: dict      # coach/analyst injected context

    # ── Raw data (written by Ingestion Agent) ──────────────────────
    raw_weather:        dict
    raw_schedule_team1: dict
    raw_schedule_team2: dict
    raw_player_batting: dict  # {team1: [...], team2: [...]}
    raw_player_bowling: dict

    # ── Extraneous scores (written by Extraneous Factor Agent) ─────
    ext_scores: dict   # dew_risk_score, fatigue_t1, fatigue_t2, pdi, etc.

    # ── Feature vector (written by Feature Engineering Agent) ──────
    feature_vector:    dict   # 44 features aligned to schema
    availability_flags: dict  # feature_name → "provided" | "fillna"

    # ── Prediction (written by Prediction Agent) ───────────────────
    win_prob_chasing:  float
    win_prob_batting:  float
    confidence:        str    # high | medium | low
    top_drivers:       list
    model_version:     str

    # ── Narrative (written by Narrative Agent) ─────────────────────
    narrative:         str

    # ── System ────────────────────────────────────────────────────
    errors:            list[str]
    warnings:          list[str]
    rerun_triggered:   bool