"""
src/models/train.py
====================
Trains the XGBoost win probability model on the Phase 2 feature matrix.

STORAGE:
    src/models/artifacts/model_v{N}.pkl     — trained XGBoost model
    src/models/artifacts/model_meta.json    — feature list, version, eval metrics
    data/processed/scaler.pkl               — MinMaxScaler (fitted in Phase 2 ETL)

WHEN TO RE-RUN:
    - First-time setup: run once after run_phase2.py completes.
    - End of each IPL season: re-run after Phase 2 ETL has absorbed new season data.
    - If feature_schema.yaml changes: must re-run Phase 2 then re-train.
    - Hyperparameter tuning: adjust PARAMS dict, re-run, compare val AUC.

MODEL DESIGN:
    Single XGBoost binary classifier.
    Label: chasing_team_won (1 = team batting second won).
    Evaluation: AUC-ROC on val set + calibration check (Brier score).
    Feature importances logged for Narrative Agent consumption.

USAGE:
    ipl_venv/bin/python -m src.models.train
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    accuracy_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

from src.features.schema_loader import get_feature_columns

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "src" / "models" / "artifacts"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Tuned conservatively for IPL dataset size (~1000–5000 matches).
# Increase n_estimators and reduce learning_rate when real data is available.
PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  1.0,   # adjust if class imbalance > 60/40
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")
    logger.info(f"Loaded — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label y from a split DataFrame."""
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["chasing_team_won"].astype(int)
    return X, y


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> XGBClassifier:
    """Train XGBoost with early stopping on val logloss."""
    model = XGBClassifier(**PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    best = getattr(model, "best_iteration", model.n_estimators)
    logger.info(f"Iterations used: {best}")
    return model


def calibrate_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """
    Platt scaling calibration on val set.
    Raw XGBoost probabilities are often overconfident — calibration
    ensures win_prob=0.7 actually means 70% of the time.
    """
    calibrated = CalibratedClassifierCV(model, cv=5, method="sigmoid")
    # Fit on val set only (model already trained on train set)
    calibrated.fit(X_val, y_val)
    return calibrated


def evaluate(
    model, X: pd.DataFrame, y: pd.Series, split_name: str
) -> dict:
    """Return evaluation metrics dict for a given split."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "split":       split_name,
        "n":           len(y),
        "auc_roc":     round(roc_auc_score(y, probs), 4),
        "brier_score": round(brier_score_loss(y, probs), 4),
        "accuracy":    round(accuracy_score(y, preds), 4),
    }
    logger.info(
        f"{split_name:5s} | AUC: {metrics['auc_roc']:.4f} | "
        f"Brier: {metrics['brier_score']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f}"
    )
    return metrics


def get_feature_importances(
    model: XGBClassifier, feature_cols: list[str]
) -> list[dict]:
    """Return feature importances sorted descending — used by Narrative Agent."""
    raw = model.feature_importances_
    if hasattr(model, 'estimator'):
        # Calibrated wrapper
        raw = model.estimator.feature_importances_
    importances = sorted(
        [{"feature": f, "importance": round(float(v), 6)}
         for f, v in zip(feature_cols, raw)],
        key=lambda x: x["importance"], reverse=True
    )
    return importances


def get_next_version() -> str:
    """Auto-increment model version based on existing artifacts."""
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return "1"
    versions = [int(f.stem.split("_v")[1]) for f in existing]
    return str(max(versions) + 1)


def run() -> None:
    logger.info("=== Phase 3: Model Training ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits()
    X_train, y_train = get_xy(train_df)
    X_val,   y_val   = get_xy(val_df)
    X_test,  y_test  = get_xy(test_df)
    feature_cols     = list(X_train.columns)

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label balance (train) — chasing won: {y_train.mean():.2%}")

    # ── Train ──────────────────────────────────────────────────────
    logger.info("Training XGBoost ...")
    model = train_model(X_train, y_train, X_val, y_val)

    # ── Calibrate ──────────────────────────────────────────────────
    logger.info("Calibrating probabilities (Platt scaling on val set) ...")
    calibrated = calibrate_model(model, X_val, y_val)

    # ── Evaluate ───────────────────────────────────────────────────
    logger.info("\nEvaluation:")
    metrics = [
        evaluate(calibrated, X_train, y_train, "train"),
        evaluate(calibrated, X_val,   y_val,   "val"),
        evaluate(calibrated, X_test,  y_test,  "test"),
    ]

    # ── Feature importances ────────────────────────────────────────
    importances = get_feature_importances(model, feature_cols)
    logger.info("\nTop 10 features by importance:")
    for fi in importances[:10]:
        logger.info(f"  {fi['feature']:<45} {fi['importance']:.4f}")

    # ── Save model artifact ────────────────────────────────────────
    version    = get_next_version()
    model_path = MODELS_DIR / f"model_v{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    logger.success(f"Model saved → {model_path}")

    # ── Save metadata ──────────────────────────────────────────────
    meta = {
        "version":       version,
        "trained_at":    datetime.utcnow().isoformat(),
        "features":      feature_cols,
        "n_features":    len(feature_cols),
        "hyperparams":   {k: v for k, v in PARAMS.items() if k != "use_label_encoder"},
        "metrics":       metrics,
        "top_features":  importances[:15],
        "label":         "chasing_team_won",
        "train_seasons": sorted(train_df["season"].astype(str).unique().tolist()),
        "val_season":    sorted(val_df["season"].astype(str).unique().tolist()),
        "test_season":   sorted(test_df["season"].astype(str).unique().tolist()),
    }
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.success(f"Metadata saved → {meta_path}")

    # ── Sanity: single prediction ──────────────────────────────────
    sample_row = X_val.iloc[[0]].copy()
    prob       = calibrated.predict_proba(sample_row)[0][1]
    logger.info(f"\nSanity check — sample val prediction: {prob:.3f} win prob (chasing team)")

    logger.success(f"\nPhase 3 complete. Model v{version} ready.")
    logger.success("Next: ipl_venv/bin/python run_phase3.py  →  then Phase 3b: serve.py")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.chdir(Path(__file__).resolve().parents[2])
    run()