"""
src/models/serve.py
====================
FastAPI serving endpoint for the trained XGBoost model.
Exposes predict_win_probability() as a REST tool callable by the
LangGraph Prediction Agent in Phase 4.

STORAGE:
    src/models/artifacts/model_v{N}.pkl  — loaded at startup
    src/models/artifacts/model_meta.json — feature list, version info
    data/processed/scaler.pkl            — MinMaxScaler for normalisation

WHEN TO RE-RUN:
    - After Phase 3 training produces a new model artifact: restart server.
    - In production: Docker container restart picks up new artifact automatically
      if artifacts/ is mounted as a volume.

ENDPOINTS:
    POST /predict       — primary prediction endpoint (used by LangGraph agent)
    GET  /health        — liveness check
    GET  /model-info    — current model version, features, last-trained metrics
    POST /drift-check   — compares inference feature vector to training distribution

USAGE:
    ipl_venv/bin/python -m src.models.serve
    # or via runner:
    ipl_venv/bin/python run_phase3.py --serve
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.features.schema_loader import (
    get_feature_columns, get_fillna_map, validate_feature_vector
)

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "models" / "artifacts"
SCALER_PATH= ROOT / "data" / "processed" / "scaler.pkl"


# ── Load artifacts at startup ──────────────────────────────────────────────────

def _load_latest_model():
    """Load the highest-versioned model artifact."""
    artifacts = sorted(MODELS_DIR.glob("model_v*.pkl"),
                       key=lambda p: int(p.stem.split("_v")[1]))
    if not artifacts:
        raise FileNotFoundError(
            f"No model artifacts found in {MODELS_DIR}. Run run_phase3.py first."
        )
    path  = artifacts[-1]
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {path.name}")
    return model, path.name


def _load_meta() -> dict:
    meta_path = MODELS_DIR / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _load_scaler():
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    logger.warning("Scaler not found — normalisation skipped.")
    return None


MODEL, MODEL_NAME = _load_latest_model()
META              = _load_meta()
SCALER            = _load_scaler()
FEATURE_COLS      = get_feature_columns()
FILLNA_MAP        = get_fillna_map()

# Training distribution stats for drift detection (from meta)
TRAIN_FEATURE_MEANS = {
    fi["feature"]: fi["importance"]
    for fi in META.get("top_features", [])
}


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IPL Win Probability API",
    description="Serves calibrated XGBoost win probability for IPL matches.",
    version=META.get("version", "1"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Feature vector for one match.
    All fields optional — missing fields get fillna from schema.
    match_id and match_context are metadata, not model inputs.
    """
    match_id: Optional[str] = None
    features: dict = Field(
        ...,
        description="Dict of feature_name → value. Missing keys get schema fillna."
    )

class PredictResponse(BaseModel):
    match_id:            Optional[str]
    win_prob_chasing:    float   # probability chasing team wins (team batting 2nd)
    win_prob_batting:    float   # probability team batting 1st wins (1 - above)
    confidence:          str     # high | medium | low — based on feature availability
    model_version:       str
    top_drivers:         list[dict]   # top 5 features driving this prediction
    availability_flags:  dict         # which features were missing → fillna applied
    warnings:            list[str]

class DriftCheckRequest(BaseModel):
    features: dict

class DriftCheckResponse(BaseModel):
    drift_detected: bool
    missing_features: list[str]
    extra_features:   list[str]
    warnings:         list[str]

class SimulateMatchRequest(BaseModel):
    match_id: Optional[str] = None
    pov_team: Optional[str] = None
    team1: str
    team2: str
    venue: str
    match_date: str
    season: str
    is_night_match: bool = True
    human_override: dict = {}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_feature_vector(raw: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply fillna, enforce column order, return feature DataFrame + availability flags.
    """
    filled   = {}
    avail    = {}
    for col in FEATURE_COLS:
        if col in raw and raw[col] is not None:
            filled[col] = raw[col]
            avail[col]  = "provided"
        else:
            filled[col] = FILLNA_MAP[col]
            avail[col]  = "fillna"

    df = pd.DataFrame([filled])[FEATURE_COLS]
    return df, avail


def _apply_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the training scaler. Falls back gracefully if scaler missing."""
    if SCALER is None:
        return df
    try:
        from src.features.schema_loader import get_normalize_features
        norm_cols = [c for c in get_normalize_features() if c in df.columns]
        df[norm_cols] = SCALER.transform(df[norm_cols])
    except Exception as e:
        logger.warning(f"Scaler apply failed: {e}")
    return df


def _confidence(avail: dict) -> str:
    """Derive confidence level from feature availability."""
    fillna_count = sum(1 for v in avail.values() if v == "fillna")
    ratio        = fillna_count / len(avail)
    if ratio < 0.10:
        return "high"
    elif ratio < 0.30:
        return "medium"
    return "low"


def _top_drivers(model, df: pd.DataFrame, n: int = 5) -> list[dict]:
    """
    Feature × importance × value for top N features.
    Used by Narrative Agent to explain the prediction.
    """
    base_model = model.estimator if hasattr(model, "estimator") else model
    importances = base_model.feature_importances_
    pairs = sorted(
        zip(FEATURE_COLS, importances, df.iloc[0].values),
        key=lambda x: x[1], reverse=True
    )[:n]
    return [
        {"feature": f, "importance": round(float(i), 4), "value": round(float(v), 4)}
        for f, i, v in pairs
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/model-info")
def model_info():
    return {
        "model_version":  META.get("version"),
        "trained_at":     META.get("trained_at"),
        "n_features":     META.get("n_features"),
        "train_seasons":  META.get("train_seasons"),
        "val_metrics":    next((m for m in META.get("metrics", []) if m["split"] == "val"), {}),
        "top_features":   META.get("top_features", [])[:10],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Primary endpoint — called by LangGraph Prediction Agent.
    Accepts a feature dict, returns calibrated win probability + explanation.
    """
    # Validate
    is_valid, issues = validate_feature_vector(req.features)
    warnings = issues  # non-fatal — fillna applied

    # Build + scale feature vector
    X, avail = _build_feature_vector(req.features)
    X        = _apply_scaler(X)

    # Predict
    try:
        probs = MODEL.predict_proba(X)[0]
        win_prob_chasing = float(probs[1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    confidence = _confidence(avail)
    if confidence == "low":
        warnings.append("Low confidence — >30% of features used fillna defaults.")

    return PredictResponse(
        match_id=req.match_id,
        win_prob_chasing=round(win_prob_chasing, 4),
        win_prob_batting=round(1 - win_prob_chasing, 4),
        confidence=confidence,
        model_version=MODEL_NAME,
        top_drivers=_top_drivers(MODEL, X),
        availability_flags={k: v for k, v in avail.items() if v == "fillna"},
        warnings=warnings,
    )


@app.post("/drift-check", response_model=DriftCheckResponse)
def drift_check(req: DriftCheckRequest):
    """
    Data Validation Gate calls this before predict.
    Checks for missing/extra features vs schema.
    """
    is_valid, issues = validate_feature_vector(req.features)
    expected = set(FEATURE_COLS)
    provided = set(req.features.keys())
    missing  = sorted(expected - provided)
    extra    = sorted(provided - expected)

    drift_detected = len(missing) > int(len(FEATURE_COLS) * 0.3)  # >30% missing = drift

    return DriftCheckResponse(
        drift_detected=drift_detected,
        missing_features=missing,
        extra_features=extra,
        warnings=issues,
    )


@app.post("/simulate-match")
def simulate_match(req: SimulateMatchRequest):
    """
    Run the full 6-agent LangGraph pipeline for a match.
    Called by the frontend Dashboard.
    """
    try:
        from src.agents.graph import run_prediction
        result = run_prediction(
            team1=req.team1,
            team2=req.team2,
            venue=req.venue,
            match_date=req.match_date,
            season=req.season,
            is_night_match=req.is_night_match,
            human_override=req.human_override,
            match_id=req.match_id,
            pov_team=req.pov_team,
        )
        return result
    except Exception as e:
        logger.error(f"Agent pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schedule/2026")
def get_schedule_2026():
    """Returns the schedule for the 2026 season."""
    path = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"
    if not path.exists():
        return {"matches": []}
    import pandas as pd
    df = pd.read_parquet(path)
    # Ensure season is string and filter
    df['season'] = df['season'].astype(str)
    df_2026 = df[df['season'] == '2026'].drop_duplicates('match_id').sort_values('start_date')
    
    matches = []
    for _, row in df_2026.iterrows():
        matches.append({
            "match_id": str(row["match_id"]),
            "team1": row["team"],
            "team2": row["opponent"],
            "venue": row["venue"],
            "date": str(row["start_date"].date()) if hasattr(row["start_date"], "date") else str(row["start_date"])
        })
    return {"matches": matches}

@app.get("/match-result/{match_id}")
def get_match_result(match_id: str):
    """Returns the actual historical result of a match from DuckDB."""
    db_path = ROOT / "data" / "processed" / "ipl.duckdb"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="DuckDB not found")
    
    import duckdb
    with duckdb.connect(str(db_path)) as con:
        res = con.execute("SELECT team1, team2, team1_score, team2_score, chasing_team, chasing_team_won FROM matches WHERE match_id = ?", [match_id]).fetchone()
        
        if not res:
            raise HTTPException(status_code=404, detail="Match not found in database")
            
        team1, team2, team1_score, team2_score, chasing_team, chasing_team_won = res
        
        batting_team = team1 if team2 == chasing_team else team2
        winner = chasing_team if chasing_team_won else batting_team
        
        return {
            "match_id": match_id,
            "batting_team": batting_team,
            "chasing_team": chasing_team,
            "team1_score": team1_score,
            "team2_score": team2_score,
            "winner": winner,
            "chasing_team_won": bool(chasing_team_won)
        }

# Mount dashboard static files at root
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")


# ── Dev server entry ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, uvicorn
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    uvicorn.run("src.models.serve:app", host="0.0.0.0", port=8001, reload=False)