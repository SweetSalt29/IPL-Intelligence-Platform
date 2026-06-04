#!/usr/bin/env python3
"""
run_phase1.py

Master runner for Phase 1 data acquisition.
Run from the project root:
    python run_phase1.py

Steps:
  1a. Cricsheet  — download IPL ball-by-ball CSVs → DuckDB
  1b. Weather    — fetch Open-Meteo historical weather per match
  1c. Schedule   — compute travel/rest/home-away per team per match
  1d. Players    — derive batting/bowling profiles from ball-by-ball

Each step is idempotent: safe to re-run, skips already-done work.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main() -> None:
    t0 = time.time()

    # ── 1a: Cricsheet ──────────────────────────────────────────────
    separator("PHASE 1a — Cricsheet Ball-by-Ball → DuckDB")
    from src.ingestion.cricsheet_ingest import run as run_cricsheet
    run_cricsheet(force_download="--force" in sys.argv)

    # ── 1b: Weather ────────────────────────────────────────────────
    separator("PHASE 1b — Open-Meteo Historical Weather")
    from src.ingestion.weather_ingest import run as run_weather
    run_weather()

    # ── 1c: Schedule ───────────────────────────────────────────────
    separator("PHASE 1c — Schedule / Travel / Rest Days")
    from src.ingestion.schedule_ingest import run as run_schedule
    run_schedule()

    # ── 1d: Player profiles ────────────────────────────────────────
    separator("PHASE 1d — Player Profiles from Ball-by-Ball")
    from src.ingestion.player_ingest import run as run_players
    run_players()

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    separator(f"PHASE 1 COMPLETE — {elapsed:.1f}s")

    from pathlib import Path
    import duckdb

    db = ROOT / "data" / "processed" / "ipl.duckdb"
    con = duckdb.connect(str(db))
    balls   = con.execute("SELECT COUNT(*) FROM balls").fetchone()[0]
    matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    seasons = con.execute(
        "SELECT COUNT(DISTINCT season) FROM matches"
    ).fetchone()[0]
    con.close()

    weather_path = ROOT / "data" / "raw" / "weather" / "weather_by_match.parquet"
    schedule_path = ROOT / "data" / "raw" / "schedule" / "team_schedule.parquet"

    print("Data inventory:")
    print(f"  Ball-by-ball deliveries : {balls:,}")
    print(f"  Matches                 : {matches:,}")
    print(f"  Seasons                 : {seasons}")
    if weather_path.exists():
        import pandas as pd
        w = pd.read_parquet(weather_path)
        print(f"  Matches with weather    : {len(w):,}")
    if schedule_path.exists():
        import pandas as pd
        s = pd.read_parquet(schedule_path)
        print(f"  Team-match schedule rows: {len(s):,}")

    print("\nNext step: Phase 2 — Shared Feature Schema + ETL")


if __name__ == "__main__":
    main()
