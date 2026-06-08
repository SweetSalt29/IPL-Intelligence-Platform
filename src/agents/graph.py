"""
src/agents/graph.py
====================
LangGraph supervisor graph — wires all agents into the StateGraph
and defines the execution flow.

GRAPH TOPOLOGY:
    START
      → validation_gate          (check state has minimum required fields)
      → ingestion_agent          (fetch weather, schedule, player data)
      → extraneous_agent         (compute dew, fatigue, PDI scores)
      → feature_agent            (build 44-feature vector from state)
      → prediction_agent         (call ML model tool)
      → narrative_agent          (generate tactical brief via Claude)
    END

All nodes run sequentially. Parallel execution is possible (ingestion +
extraneous can run in parallel) but kept sequential here for debuggability.
Add parallelism in v2 once the graph is stable.

HUMAN OVERRIDE:
    Inject into initial state under "human_override" key before invoking.
    The Feature Agent picks it up and applies last-write-wins.
"""

from langgraph.graph import StateGraph, START, END
from loguru import logger

from src.agents.state           import MatchState
from src.agents.ingestion_agent  import ingestion_agent
from src.agents.extraneous_agent import extraneous_agent
from src.agents.feature_agent    import feature_agent
from src.agents.prediction_agent import prediction_agent
from src.agents.narrative_agent  import narrative_agent


# ── Validation gate ────────────────────────────────────────────────────────────

def validation_gate(state: MatchState) -> MatchState:
    """
    First node — validates minimum required fields in state.
    Appends warnings for missing optional fields.
    Fails fast with clear error if critical fields are absent.
    """
    errors   = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    required = ["team1", "team2", "venue", "match_date", "season"]
    for field in required:
        if not state.get(field):
            errors.append(f"CRITICAL: missing required field '{field}'")

    optional = ["human_override", "is_night_match"]
    for field in optional:
        if field not in state:
            warnings.append(f"Optional field '{field}' not provided — using default")

    if errors:
        logger.error(f"[ValidationGate] Critical errors: {errors}")
    else:
        logger.info(f"[ValidationGate] OK — {state['team1']} vs {state['team2']}")

    return {
        **state,
        "errors":   errors,
        "warnings": warnings,
        "human_override":  state.get("human_override",  {}),
        "is_night_match":  state.get("is_night_match",  True),
        "rerun_triggered": state.get("rerun_triggered", False),
    }


def should_continue(state: MatchState) -> str:
    """
    Conditional edge after validation gate.
    If critical errors exist (missing required fields), route to END.
    Otherwise continue to ingestion.
    """
    errors = state.get("errors", [])
    critical = [e for e in errors if e.startswith("CRITICAL")]
    return "end" if critical else "continue"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(MatchState)

    # Add nodes
    graph.add_node("validation_gate",   validation_gate)
    graph.add_node("ingestion_agent",   ingestion_agent)
    graph.add_node("extraneous_agent",  extraneous_agent)
    graph.add_node("feature_agent",     feature_agent)
    graph.add_node("prediction_agent",  prediction_agent)
    graph.add_node("narrative_agent",   narrative_agent)

    # Edges
    graph.add_edge(START, "validation_gate")

    # Conditional: abort on critical validation errors
    graph.add_conditional_edges(
        "validation_gate",
        should_continue,
        {"continue": "ingestion_agent", "end": END},
    )

    graph.add_edge("ingestion_agent",  "extraneous_agent")
    graph.add_edge("extraneous_agent", "feature_agent")
    graph.add_edge("feature_agent",    "prediction_agent")
    graph.add_edge("prediction_agent", "narrative_agent")
    graph.add_edge("narrative_agent",  END)

    return graph.compile()


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_prediction(
    team1:         str,
    team2:         str,
    venue:         str,
    match_date:    str,
    season:        str,
    is_night_match: bool = True,
    human_override: dict = None,
    match_id:      str  = None,
) -> MatchState:
    """
    Entry point for running a pre-match prediction.
    Returns the final state with all fields populated.

    Args:
        team1:          Team batting first
        team2:          Team batting second (chasing)
        venue:          Venue name (resolved against config/venues.py)
        match_date:     YYYY-MM-DD
        season:         e.g. '2025'
        is_night_match: True for D/N and night matches
        human_override: Optional dict of coach-injected context
        match_id:       Optional match identifier string
    """
    app = build_graph()

    initial_state: MatchState = {
        "match_id":       match_id or f"{team1}_vs_{team2}_{match_date}",
        "team1":          team1,
        "team2":          team2,
        "venue":          venue,
        "match_date":     match_date,
        "season":         season,
        "is_night_match": is_night_match,
        "human_override": human_override or {},
        "errors":         [],
        "warnings":       [],
    }

    logger.info(f"\n{'='*55}")
    logger.info(f"  IPL Prediction: {team1} vs {team2}")
    logger.info(f"  {venue} | {match_date} | {'Night' if is_night_match else 'Day'}")
    logger.info(f"{'='*55}")

    final_state = app.invoke(initial_state)

    logger.info(f"\n{'='*55}")
    logger.info(f"  RESULT")
    logger.info(f"  {team2} (chasing) win prob: {final_state.get('win_prob_chasing', 0):.1%}")
    logger.info(f"  {team1} (batting)  win prob: {final_state.get('win_prob_batting', 0):.1%}")
    logger.info(f"  Confidence: {final_state.get('confidence', 'N/A')}")
    if final_state.get("warnings"):
        logger.warning(f"  Warnings: {len(final_state['warnings'])}")
    if final_state.get("errors"):
        logger.error(f"  Errors: {final_state['errors']}")
    logger.info(f"{'='*55}\n")

    return final_state