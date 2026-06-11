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
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# FrozenEstimator replaces deprecated cv='prefit' in sklearn >=1.6
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

from src.features.schema_loader import get_feature_columns

ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "src" / "models" / "artifacts"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# early_stopping_rounds stops training when val logloss plateaus.
# max_depth=3 + min_child_weight=10 + strong regularisation = less overfit
# on ~4k match dataset. Re-tune after each season adds more data.
PARAMS = {
    "n_estimators":          500,
    "max_depth":             3,
    "learning_rate":         0.02,
    "subsample":             0.7,
    "colsample_bytree":      0.7,
    "min_child_weight":      10,
    "gamma":                 0.2,
    "reg_alpha":             0.5,
    "reg_lambda":            2.0,
    "scale_pos_weight":      1.0,
    "eval_metric":           "logloss",
    "early_stopping_rounds": 30,
    "random_state":          42,
    "n_jobs":                -1,
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")
    logger.info(f"Loaded — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    best = getattr(model, "best_iteration", PARAMS["n_estimators"])
    logger.info(f"Best iteration: {best} / {PARAMS['n_estimators']}")
    return model


def calibrate_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """
    Platt scaling calibration on val set.
    Uses FrozenEstimator (sklearn >=1.6) which supersedes cv='prefit'.
    FrozenEstimator wraps a fitted model and prevents re-fitting during
    the calibration step — probabilities are adjusted via sigmoid only.
    Falls back to cv='prefit' for older sklearn versions.
    """
    if HAS_FROZEN:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(model), method="sigmoid"
        )
    else:
        # sklearn <1.6 fallback — suppress deprecation warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")

    calibrated.fit(X_val, y_val)
    return calibrated


def evaluate(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
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


def get_feature_importances(model, feature_cols: list[str]) -> list[dict]:
    base = model.estimator if hasattr(model, "estimator") else model
    # FrozenEstimator wraps the base — unwrap one more level if needed
    if hasattr(base, "estimator"):
        base = base.estimator
    raw = base.feature_importances_
    return sorted(
        [{"feature": f, "importance": round(float(v), 6)} for f, v in zip(feature_cols, raw)],
        key=lambda x: x["importance"], reverse=True,
    )


def get_next_version() -> str:
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return "1"
    return str(max(int(f.stem.split("_v")[1]) for f in existing) + 1)


def purge_old_artifacts() -> None:
    """
    Remove model artifacts that were trained on mock data.
    Keeps only the latest version — avoids serve.py accidentally
    loading an old mock-trained model if v5 is the first real one.
    Called automatically on each training run.
    """
    artifacts = sorted(
        MODELS_DIR.glob("model_v*.pkl"),
        key=lambda p: int(p.stem.split("_v")[1])
    )
    if len(artifacts) <= 1:
        return
    # Keep only the latest — delete all older versions
    for old in artifacts[:-1]:
        old.unlink()
        logger.info(f"Purged old artifact: {old.name}")


def diagnose_fillna(df: pd.DataFrame, split_name: str) -> None:
    """
    Log what percentage of each feature is fillna (== schema default).
    High fillna % means the ETL didn't compute real values — flags
    features that are contributing noise rather than signal.
    """
    from src.features.schema_loader import get_fillna_map
    fillna_map   = get_fillna_map()
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    high_fillna  = []

    for col in feature_cols:
        default = fillna_map.get(col)
        if default is not None:
            pct = (df[col] == default).mean()
            if pct > 0.4:
                high_fillna.append((col, pct))

    if high_fillna:
        logger.warning(f"\n[{split_name}] Features with >40% fillna defaults (noise risk):")
        for col, pct in sorted(high_fillna, key=lambda x: -x[1]):
            logger.warning(f"  {col:<45} {pct:.1%} fillna")
    else:
        logger.success(f"[{split_name}] All features <40% fillna — good coverage.")


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

    # Feature quality check before training
    diagnose_fillna(train_df, "train")

    logger.info("Training XGBoost (early stopping on val logloss) ...")
    model = train_model(X_train, y_train, X_val, y_val)

    logger.info("Calibrating (Platt scaling via FrozenEstimator) ...")
    calibrated = calibrate_model(model, X_val, y_val)

    logger.info("\nEvaluation:")
    metrics = [
        evaluate(calibrated, X_train, y_train, "train"),
        evaluate(calibrated, X_val,   y_val,   "val"),
        evaluate(calibrated, X_test,  y_test,  "test"),
    ]

    importances = get_feature_importances(calibrated, feature_cols)
    logger.info("\nTop 10 features by importance:")
    for fi in importances[:10]:
        logger.info(f"  {fi['feature']:<45} {fi['importance']:.4f}")

    # Save new artifact, then purge old ones
    version    = get_next_version()
    model_path = MODELS_DIR / f"model_v{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated, f)
    logger.success(f"Model saved → {model_path}")

    purge_old_artifacts()

    meta = {
        "version":       version,
        "trained_at":    datetime.utcnow().isoformat(),
        "features":      feature_cols,
        "n_features":    len(feature_cols),
        "hyperparams":   PARAMS,
        "metrics":       metrics,
        "top_features":  importances[:15],
        "label":         "chasing_team_won",
        "train_seasons": sorted(train_df["season"].astype(str).unique().tolist()),
        "val_season":    sorted(val_df["season"].astype(str).unique().tolist()),
        "test_season":   sorted(test_df["season"].astype(str).unique().tolist()),
        "sklearn_frozen": HAS_FROZEN,
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.success(f"Metadata saved → {MODELS_DIR}/model_meta.json")

    prob = calibrated.predict_proba(X_val.iloc[[0]])[0][1]
    logger.info(f"\nSanity check — sample val prediction: {prob:.3f}")
    logger.success(f"\nPhase 3 complete. Model v{version} ready.")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.chdir(Path(__file__).resolve().parents[2])
    run()