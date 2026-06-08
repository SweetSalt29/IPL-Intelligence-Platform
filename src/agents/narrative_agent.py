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

    return f"""You are the intelligence system for an IPL franchise pit-wall.
Produce a concise tactical brief (150–200 words) for team management.
Be direct. No hedging. Use cricket terminology. Speak like a senior analyst.

MATCH: {team1} (batting first) vs {team2} (chasing) at {venue}

PREDICTION:
  {team2} (chasing) win probability: {win_prob:.1%}
  {team1} (batting first) win probability: {win_bat:.1%}
  Confidence: {confidence}

TOP PREDICTION DRIVERS:
{driver_lines}

EXTRANEOUS FACTORS:
  Dew: {dew_note}
  Fatigue (Team 1): {ext.get('team1_fatigue_index', 'N/A')}
  Fatigue (Team 2): {ext.get('team2_fatigue_index', 'N/A')}
  Pitch Deterioration Index: {ext.get('pitch_deterioration_index', 0.5):.2f}
  Heat stress: {ext.get('heat_stress_index', 0.3):.2f}
{override_note}{warnings_note}

TOSS RECOMMENDATION: {toss_rec}

Write the tactical brief now. Start with the toss recommendation.
End with 1–2 specific bowling or batting order suggestions."""


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

    return (
        f"TOSS RECOMMENDATION: {toss_rec}.\n\n"
        f"{team2} holds a {win_prob:.1%} win probability (chasing). "
        f"Confidence: {confidence}. "
        f"Primary driver: {get_feature_description(top_feat)}. "
        f"{dew_str} "
        f"Fatigue — {team1}: {fat1:.2f}, {team2}: {fat2:.2f}. "
        f"{'Watch ' + team1 + ' bowling in death overs — fatigue risk elevated.' if fat1 > 0.6 else ''}"
        f"\n\n[Fallback narrative — set GROQ_API_KEY in .env for LLM brief.]"
    )


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
        narrative = response.content
        logger.info(f"  Narrative generated via Groq {GROQ_MODEL} ({len(narrative)} chars)")

    except Exception as e:
        logger.warning(f"  Groq call failed: {e} — using fallback narrative.")
        narrative = _fallback_narrative(state)

    return {**state, "narrative": narrative}