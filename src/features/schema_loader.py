"""
src/features/schema_loader.py
==============================
Loads and validates the shared feature schema (config/feature_schema.yaml).
Both the offline ETL and the live inference pipeline import from here.
This is the single source of truth for feature names, dtypes, and fillna values.

WHEN TO USE:
    - ETL (training): import get_feature_columns() to ensure correct column ordering
    - Inference agents: import get_fillna_map() to apply correct fallbacks
    - Narrative Agent: import get_feature_description() for plain-English explanations
"""

import yaml
from pathlib import Path
from functools import lru_cache

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "feature_schema.yaml"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, "r") as f:
        return yaml.safe_load(f)


def get_all_features() -> dict[str, dict]:
    """Return flat dict of {feature_name: feature_config} — excludes label and feature_groups."""
    schema = load_schema()
    features = {}
    for section, entries in schema.items():
        if section in ("label", "feature_groups"):
            continue
        features.update(entries)
    return features


def get_feature_columns() -> list[str]:
    """Ordered list of all feature column names. Order is fixed — never change without retraining."""
    return list(get_all_features().keys())


def get_fillna_map() -> dict[str, float | int]:
    """Returns {feature_name: fillna_value} for all features."""
    return {name: cfg["fillna"] for name, cfg in get_all_features().items()}


def get_normalize_features() -> list[str]:
    """Features that should be min-max normalised before model input."""
    return [name for name, cfg in get_all_features().items() if cfg.get("normalize")]


def get_feature_description(feature_name: str) -> str:
    """Plain-English description for a feature. Used by Narrative Agent."""
    features = get_all_features()
    return features.get(feature_name, {}).get("description", feature_name)


def get_feature_group(group: str) -> list[str]:
    """Return feature names in a named group (e.g. 'environmental', 'fatigue')."""
    schema = load_schema()
    return schema.get("feature_groups", {}).get(group, [])


def validate_feature_vector(vector: dict) -> tuple[bool, list[str]]:
    """
    Validate an inference-time feature vector against the schema.
    Returns (is_valid, list_of_issues).
    Issues are warnings — missing features get fillna applied, not rejected.
    """
    expected = set(get_feature_columns())
    provided = set(vector.keys())
    missing  = expected - provided
    extra    = provided - expected
    issues   = []
    if missing:
        issues.append(f"Missing features (fillna applied): {sorted(missing)}")
    if extra:
        issues.append(f"Extra features (ignored): {sorted(extra)}")
    return len(issues) == 0, issues


if __name__ == "__main__":
    cols = get_feature_columns()
    print(f"Total features: {len(cols)}")
    print(f"Normalised:     {len(get_normalize_features())}")
    print(f"Feature groups: {list(load_schema().get('feature_groups', {}).keys())}")
    print(f"\nAll features:")
    for c in cols:
        cfg = get_all_features()[c]
        print(f"  {c:<45} dtype={cfg['dtype']:<10} fillna={cfg['fillna']}")