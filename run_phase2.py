"""
run_phase2.py
=============
Master runner for Phase 2 — Feature Engineering + ETL.
Run from the project root after Phase 1 completes:

    ipl_venv/bin/python run_phase2.py

OUTPUT:
    data/processed/feature_matrix.parquet  — full dataset
    data/processed/train.parquet           — seasons 2008–2022
    data/processed/val.parquet             — season 2023
    data/processed/test.parquet            — season 2024+
    data/processed/scaler.pkl              — fitted MinMaxScaler (used at inference)

NEXT STEP: Phase 3 — Model Training (run_phase3.py)
"""

import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    from src.features.schema_loader import get_feature_columns, load_schema
    cols = get_feature_columns()
    logger.info(f"Schema loaded: {len(cols)} features across "
                f"{len(load_schema()) - 2} groups")

    from src.features.etl import run
    run()


if __name__ == "__main__":
    main()