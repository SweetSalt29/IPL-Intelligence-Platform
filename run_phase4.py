"""
run_phase4.py
=============
Phase 4 — Run the LangGraph multi-agent prediction system.

    ipl_venv/bin/python run_phase4.py

Runs a sample pre-match prediction for MI vs CSK at Wankhede.
Edit the match details at the bottom to predict any match.

REQUIREMENTS:
    - Phase 1, 2, 3 must have run (DuckDB + model artifact must exist)
    - ANTHROPIC_API_KEY in .env for LLM narrative (optional — fallback exists)
    - Model serving endpoint optionally running on :8000 (fallback to direct load)
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger
from src.agents.graph import run_prediction


def main():
    # ── Example match ──────────────────────────────────────────────
    # Edit these fields to predict any match
    result = run_prediction(
        team1          = "Mumbai Indians",
        team2          = "Chennai Super Kings",
        venue          = "Wankhede Stadium",
        match_date     = "2025-04-15",
        season         = "2025",
        is_night_match = True,
        match_id       = "MI_vs_CSK_WAN_20250415",

        # Optional: coach/analyst override — inject known facts not in model
        human_override = {
            "toss_winner_is_team1": 0,      # CSK won toss
            "toss_decision_bat":    0,       # CSK elected to field (bowl first)
            "season_stage":         0,       # league phase
        },
    )

    # ── Print final output ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  TACTICAL BRIEF")
    print("="*60)
    print(result.get("narrative", "No narrative generated."))
    print("\n" + "="*60)
    print("  PREDICTION SUMMARY")
    print("="*60)
    print(f"  Chasing ({result.get('team2')}) win prob : {result.get('win_prob_chasing', 0):.1%}")
    print(f"  Batting ({result.get('team1')}) win prob : {result.get('win_prob_batting', 0):.1%}")
    print(f"  Confidence                     : {result.get('confidence', 'N/A')}")
    print(f"  Model                          : {result.get('model_version', 'N/A')}")
    print(f"\n  Top prediction drivers:")
    for d in result.get("top_drivers", [])[:5]:
        print(f"    {d['feature']:<42} {d['importance']:.4f}")
    if result.get("warnings"):
        print(f"\n  Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    • {w}")
    if result.get("errors"):
        print(f"\n  Errors:")
        for e in result["errors"]:
            print(f"    ✗ {e}")
    print("="*60)


if __name__ == "__main__":
    main()