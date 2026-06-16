"""
src/agents/narrative_agent.py
================================
Narrative Agent — uses Groq (llama-3.3-70b-versatile, free tier) via
langchain-groq to generate a plain-English tactical brief for team management.

MODEL:
    llama-3.3-70b-versatile on Groq — best quality on the free tier.
    Fast inference (~500 tok/s on Groq LPU), no credit card required.
    Get a free API key at: https://console.groq.com

ENV:
    GROQ_API_KEY — set in .env (required for LLM narrative)

FALLBACK:
    If GROQ_API_KEY is missing or the call fails, produces a rule-based
    template narrative. The system never goes silent.

WHEN TO RE-RUN:
    This agent runs on every prediction request. No offline setup needed.
"""

import os
from loguru import logger
from dotenv import load_dotenv

from src.agents.state import MatchState
from src.features.schema_loader import get_feature_description

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_prompt(state: MatchState) -> str:
    team1       = state.get("team1", "Team 1")
    team2       = state.get("team2", "Team 2")
    venue       = state.get("venue", "the venue")
    win_prob    = state.get("win_prob_chasing", 0.5)
    win_bat     = state.get("win_prob_batting", 0.5)
    confidence  = state.get("confidence", "medium")
    top_drivers = state.get("top_drivers", [])
    ext         = state.get("ext_scores", {})
    override    = state.get("human_override", {})
    warnings    = state.get("warnings", [])

    driver_lines = "\n".join(
        f"  - {get_feature_description(d['feature'])}: {d['value']:.3f} (importance: {d['importance']:.4f})"
        for d in top_drivers[:5]
    ) or "  - No driver data available"

    toss_rec = "Bowl first" if win_prob > 0.52 else "Bat first"
    dew_note = (
        f"Dew expected from ~over {ext.get('dew_onset_over', 14)}. "
        "Spinners lose grip. Plan death bowling accordingly."
        if ext.get("dew_risk_flag") else "No significant dew risk."
    )
    override_note  = f"\nCoach override context: {override}" if override else ""
    warnings_note  = f"\nData warnings: {'; '.join(warnings)}" if warnings else ""

    features = state.get("features", {})
    spin_factor = features.get("venue_spin_factor", 0.5) if isinstance(features, dict) else 0.5
    chasing_win_rate = features.get("venue_chasing_win_rate", 0.5) if isinstance(features, dict) else 0.5

    pov_team = state.get("pov_team")
    if not pov_team or pov_team == "None":
        pov_team = team1
    opponent = team2 if pov_team == team1 else team1

    return f"""You are the Head of Analytics for {pov_team}.
Your job is to analyze the conditions and provide a Toss Recommendation and Squad Roster Tweaks for YOUR team ({pov_team}) against your opponent ({opponent}).

MATCH: {team1} vs {team2} at {venue}

PREDICTION:
  {team2} (chasing) win probability: {win_prob:.1%}
  Confidence: {confidence}

CONDITIONS & FACTORS:
  Dew: {dew_note}
  Pitch Deterioration Index: {ext.get('pitch_deterioration_index', 0.5):.2f}
  Venue Spin Factor: {spin_factor:.2f}
  Venue Chasing Win Rate: {chasing_win_rate:.2%}
{override_note}{warnings_note}

You must output a strictly valid JSON object with EXACTLY these three keys:
{{
  "toss_recommendation": "BOWL FIRST" or "BAT FIRST",
  "toss_rationale": "1-2 sentences explaining why, citing dew or pitch deterioration.",
  "squad_tweaks": "1-2 sentences suggesting changes to {pov_team}'s playing XI, citing spin factor, fatigue, or matchup advantages."
}}
Output ONLY the raw JSON. Do not include markdown formatting like ```json."""


def _fallback_narrative(state: MatchState) -> str:
    team1       = state.get("team1", "Team 1")
    team2       = state.get("team2", "Team 2")
    win_prob    = state.get("win_prob_chasing", 0.5)
    confidence  = state.get("confidence", "medium")
    ext         = state.get("ext_scores", {})
    top_drivers = state.get("top_drivers", [])

    toss_rec = "Bowl first" if win_prob > 0.52 else "Bat first"
    dew_str  = (
        f"Dew risk is high (onset ~over {ext.get('dew_onset_over', 14)}). "
        "Chasing conditions favoured."
        if ext.get("dew_risk_flag") else "No significant dew risk."
    )
    top_feat = top_drivers[0]["feature"] if top_drivers else "match context"
    fat1     = ext.get("team1_fatigue_index", 0.3)
    fat2     = ext.get("team2_fatigue_index", 0.3)

    import json
    return json.dumps({
        "toss_recommendation": toss_rec.upper(),
        "toss_rationale": f"{team2} win probability is {win_prob:.1%}. {dew_str}",
        "squad_tweaks": f"Fatigue metrics ({team1}: {fat1:.2f}, {team2}: {fat2:.2f}) suggest rotating fast bowlers."
    })


def narrative_agent(state: MatchState) -> MatchState:
    """
    LangGraph node — generates tactical brief using llama-3.3-70b-versatile on Groq.
    Falls back to rule-based brief if GROQ_API_KEY is missing or call fails.
    """
    logger.info("[NarrativeAgent] Generating tactical brief via Groq ...")

    api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        logger.warning("  GROQ_API_KEY not set — using fallback narrative.")
        logger.warning("  Get a free key at https://console.groq.com")
        return {**state, "narrative": _fallback_narrative(state)}

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm      = ChatGroq(model=GROQ_MODEL, api_key=api_key, max_tokens=400, temperature=0.3)
        prompt   = _build_prompt(state)
        response = llm.invoke([HumanMessage(content=prompt)])
        narrative = response.content.strip()
        if narrative.startswith("```json"):
            narrative = narrative[7:-3].strip()
        elif narrative.startswith("```"):
            narrative = narrative[3:-3].strip()
        logger.info(f"  Narrative generated via Groq {GROQ_MODEL} ({len(narrative)} chars)")

    except Exception as e:
        logger.warning(f"  Groq call failed: {e} — using fallback narrative.")
        narrative = _fallback_narrative(state)

    return {**state, "narrative": narrative}