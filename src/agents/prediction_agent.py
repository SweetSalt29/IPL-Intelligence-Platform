"""
src/agents/prediction_agent.py
================================
Prediction Agent — assembles the feature vector from state and calls the
ML model as a LangChain tool. Writes win_prob, confidence, and top_drivers
back to state. This agent is a pure tool-caller — no reasoning of its own.

The ML model tool calls the FastAPI /predict endpoint if the server is
running, or falls back to loading the model artifact directly from disk.
This makes it work in both production (server running) and development
(direct artifact load).
"""

import os
import pickle
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from langchain_core.tools import tool

from src.agents.state import MatchState
from src.features.schema_loader import get_feature_columns, get_fillna_map

ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "src" / "models" / "artifacts"
MODEL_API   = os.getenv("MODEL_API_URL", "http://localhost:8000")


# ── ML Model Tool ──────────────────────────────────────────────────────────────

@tool
def predict_win_probability(features: dict) -> dict:
    """
    Call the trained XGBoost model to get win probability.
    Tries the FastAPI endpoint first; falls back to direct artifact load.
    Returns: {win_prob_chasing, win_prob_batting, confidence, top_drivers, model_version}
    """
    # Try API first
    try:
        resp = requests.post(
            f"{MODEL_API}/predict",
            json={"match_id": "inference", "features": features},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "win_prob_chasing": data["win_prob_chasing"],
                "win_prob_batting": data["win_prob_batting"],
                "confidence":       data["confidence"],
                "top_drivers":      data["top_drivers"],
                "model_version":    data["model_version"],
                "source":           "api",
            }
    except Exception:
        pass  # API not running — use direct load

    # Fallback: load artifact directly
    artifacts = sorted(MODELS_DIR.glob("model_v*.pkl"),
                       key=lambda p: int(p.stem.split("_v")[1]))
    if not artifacts:
        return {"error": "No model artifact found. Run run_phase3.py first."}

    with open(artifacts[-1], "rb") as f:
        model = pickle.load(f)

    feature_cols = get_feature_columns()
    fillna       = get_fillna_map()
    row          = {c: features.get(c, fillna[c]) for c in feature_cols}
    X            = pd.DataFrame([row])[feature_cols].fillna(0)

    probs         = model.predict_proba(X)[0]
    win_prob      = float(probs[1])

    # Feature importances from base model
    base = model.estimator if hasattr(model, "estimator") else model
    importances = sorted(
        zip(feature_cols, base.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    top_drivers = [
        {"feature": f, "importance": round(float(i), 4), "value": round(float(row.get(f, 0)), 4)}
        for f, i in importances[:5]
    ]

    return {
        "win_prob_chasing": round(win_prob, 4),
        "win_prob_batting": round(1 - win_prob, 4),
        "confidence":       "high",
        "top_drivers":      top_drivers,
        "model_version":    artifacts[-1].name,
        "source":           "direct",
    }


# ── Agent node ─────────────────────────────────────────────────────────────────

def prediction_agent(state: MatchState) -> MatchState:
    """
    LangGraph node — calls ML model tool, writes prediction to state.
    """
    logger.info("[PredictionAgent] Calling ML model tool ...")

    feature_vector = state.get("feature_vector", {})
    if not feature_vector:
        error = "Feature vector empty — Feature Agent may have failed."
        logger.error(f"  {error}")
        return {
            **state,
            "win_prob_chasing": 0.5,
            "win_prob_batting": 0.5,
            "confidence":       "low",
            "top_drivers":      [],
            "model_version":    "none",
            "errors":           state.get("errors", []) + [error],
        }

    result = predict_win_probability.invoke({"features": feature_vector})

    if "error" in result:
        logger.error(f"  Prediction failed: {result['error']}")
        return {
            **state,
            "win_prob_chasing": 0.5,
            "win_prob_batting": 0.5,
            "confidence":       "low",
            "top_drivers":      [],
            "model_version":    "none",
            "errors":           state.get("errors", []) + [result["error"]],
        }

    logger.info(
        f"  win_prob_chasing={result['win_prob_chasing']:.3f} | "
        f"confidence={result['confidence']} | "
        f"source={result.get('source')} | "
        f"top_driver={result['top_drivers'][0]['feature'] if result['top_drivers'] else 'N/A'}"
    )

    return {
        **state,
        "win_prob_chasing": result["win_prob_chasing"],
        "win_prob_batting": result["win_prob_batting"],
        "confidence":       result["confidence"],
        "top_drivers":      result["top_drivers"],
        "model_version":    result["model_version"],
    }