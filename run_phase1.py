"""
run_phase1.py
=============
Master runner for Phase 1 — Data Acquisition.
Run from the project root with the venv active:

    ipl_venv/bin/python run_phase1.py
    ipl_venv/bin/python run_phase1.py --force   # re-download Cricsheet zip

ORDER OF EXECUTION:
    1a. cricsheet_ingest  — downloads IPL CSVs, builds DuckDB (balls + matches tables)
    1b. weather_ingest    — fetches Open-Meteo historical weather per match, caches as parquet
    1c. schedule_ingest   — computes travel/rest/home-away per team per match
    1d. player_ingest     — derives batting/bowling profiles from DuckDB

Each step is idempotent — safe to re-run, skips already-done work.
Cricsheet data must complete before the other three steps can run.
"""

import sys
import time
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FORCE = "--force" in sys.argv


def section(title: str) -> None:
    logger.info("")
    logger.info("=" * 55)
    logger.info(f"  {title}")
    logger.info("=" * 55)


def main() -> None:
    t0 = time.time()

    section("PHASE 1a — Cricsheet → DuckDB")
    from src.ingestion.cricsheet_ingest import run as cricsheet
    cricsheet(force=FORCE)

    section("PHASE 1b — Weather (Open-Meteo Historical)")
    from src.ingestion.weather_ingest import run as weather
    weather()

    section("PHASE 1c — Schedule / Travel / Rest Days")
    from src.ingestion.schedule_ingest import run as schedule
    schedule()

    section("PHASE 1d — Player Profiles from Ball-by-Ball")
    from src.ingestion.player_ingest import run as players
    players()

    # ── Summary ────────────────────────────────────────────────
    section(f"PHASE 1 COMPLETE ({time.time() - t0:.0f}s)")

    import duckdb, pandas as pd
    con = duckdb.connect(str(ROOT / "data" / "processed" / "ipl.duckdb"))
    n_balls   = con.execute("SELECT COUNT(*) FROM balls").fetchone()[0]
    n_matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    n_seasons = con.execute("SELECT COUNT(DISTINCT season) FROM matches").fetchone()[0]
    con.close()

    w_path = ROOT / "data" / "raw" / "weather" / "weather_by_match.parquet"
    s_path = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"

    logger.info("Data inventory:")
    logger.info(f"  Deliveries (balls)      : {n_balls:,}")
    logger.info(f"  Matches                 : {n_matches:,}")
    logger.info(f"  Seasons (2008–present)  : {n_seasons}")
    if w_path.exists():
        logger.info(f"  Matches with weather    : {len(pd.read_parquet(w_path)):,}")
    if s_path.exists():
        logger.info(f"  Team-match schedule rows: {len(pd.read_parquet(s_path)):,}")

    logger.info("")
    logger.success("Next step: Phase 2 — Shared Feature Schema + ETL")


if __name__ == "__main__":
    main()