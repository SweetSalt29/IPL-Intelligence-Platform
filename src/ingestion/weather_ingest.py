"""
src/ingestion/weather_ingest.py
================================
Fetches historical hourly weather for every IPL match from Open-Meteo
and caches results locally as Parquet. Uses ERA5 reanalysis data — free,
no API key required, covers 1940–present at hourly resolution.

STORAGE:
    data/raw/weather/weather_by_match.parquet
    One row per match. Appended incrementally — safe to interrupt and resume.

WHEN TO RE-RUN:
    - First-time setup: run once after cricsheet_ingest.py to backfill
      weather for all 1,200+ historical matches. Takes ~5 minutes at
      Open-Meteo's free rate (~0.15s per request).
    - End of each IPL season: re-run to fetch weather for new season's matches.
      Only fetches matches not already in the parquet cache — fast.
    - Never run on match day for historical data (it's already cached).
    - For match-day FORECAST data (inference), use weather_ingest.fetch_forecast()
      which calls the Open-Meteo Forecast API separately.

IDEMPOTENT: Already-fetched match IDs are skipped on re-run.

USAGE:
    ipl_venv/bin/python -m src.ingestion.weather_ingest
"""

import time
import requests
import duckdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from config.venues import VENUES, resolve_venue

ROOT     = Path(__file__).resolve().parents[2]
DB_PATH  = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_PATH = ROOT / "data" / "raw" / "weather" / "weather_by_match.parquet"

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "surface_pressure",
    "windspeed_10m",
    "winddirection_10m",
    "cloudcover",
]

# Night match window: 18:00–23:00 local (primary IPL window)
# Day match window:   13:00–18:00 local
NIGHT_HOURS = list(range(18, 24))
DAY_HOURS   = list(range(13, 19))


def _get_pending_matches() -> pd.DataFrame:
    """Return matches from DuckDB that don't yet have weather cached."""
    con     = duckdb.connect(str(DB_PATH))
    matches = con.execute("SELECT match_id, start_date, venue FROM matches ORDER BY start_date").df()
    con.close()

    if OUT_PATH.exists():
        cached   = pd.read_parquet(OUT_PATH)
        done_ids = set(cached["match_id"].astype(str))
        matches  = matches[~matches["match_id"].astype(str).isin(done_ids)]
        logger.info(f"{len(done_ids)} matches already cached. {len(matches)} remaining.")
    else:
        logger.info(f"No cache found. Fetching weather for all {len(matches)} matches.")

    return matches


def _venue_coords(venue_raw: str) -> tuple[float, float] | None:
    """Resolve a venue string to (lat, lon). Returns None if unknown."""
    canonical = resolve_venue(venue_raw)
    info      = VENUES.get(canonical)
    return (info["lat"], info["lon"]) if info else None


def _fetch_historical(lat: float, lon: float, date: str) -> dict | None:
    """
    Call Open-Meteo Archive API for one match date.
    Returns aggregated weather dict for night and day windows, or None on failure.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date,
        "end_date":   date,
        "hourly":     ",".join(WEATHER_VARS),
        "timezone":   "auto",
    }
    try:
        resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"API error ({lat},{lon}) {date}: {e}")
        return None

    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        return None

    wdf         = pd.DataFrame(hourly)
    wdf["hour"] = pd.to_datetime(wdf["time"]).dt.hour
    night       = wdf[wdf["hour"].isin(NIGHT_HOURS)]
    day         = wdf[wdf["hour"].isin(DAY_HOURS)]

    def avg(df, col):
        return float(df[col].mean()) if col in df.columns and len(df) > 0 else None

    dew_temp     = avg(night, "dewpoint_2m")
    humidity     = avg(night, "relative_humidity_2m")
    dew_risk     = int((dew_temp or 0) > 15 and (humidity or 0) > 70) if dew_temp and humidity else None

    return {
        # Night window (primary — most IPL matches)
        "temp_night_avg":      avg(night, "temperature_2m"),
        "humidity_night_avg":  humidity,
        "dewpoint_night_avg":  dew_temp,
        "pressure_night_avg":  avg(night, "surface_pressure"),
        "windspeed_night_avg": avg(night, "windspeed_10m"),
        "winddir_night_avg":   avg(night, "winddirection_10m"),
        "cloudcover_night_avg":avg(night, "cloudcover"),
        # Day window
        "temp_day_avg":        avg(day, "temperature_2m"),
        "humidity_day_avg":    avg(day, "relative_humidity_2m"),
        # Full-day totals
        "precipitation_mm":    float(wdf["precipitation"].sum()) if "precipitation" in wdf.columns else None,
        # Derived extraneous flag
        "dew_risk_flag":       dew_risk,
    }


def fetch_forecast(lat: float, lon: float) -> dict | None:
    """
    Fetch today's forecast from Open-Meteo Forecast API.
    Used by the Ingestion Agent at inference time on match day.
    Returns same schema as _fetch_historical for schema consistency.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(WEATHER_VARS),
        "timezone":  "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Forecast API error: {e}")
        return None

    hourly      = data.get("hourly", {})
    wdf         = pd.DataFrame(hourly)
    wdf["hour"] = pd.to_datetime(wdf["time"]).dt.hour
    night       = wdf[wdf["hour"].isin(NIGHT_HOURS)]

    def avg(df, col):
        return float(df[col].mean()) if col in df.columns and len(df) > 0 else None

    dew_temp = avg(night, "dewpoint_2m")
    humidity = avg(night, "relative_humidity_2m")

    return {
        "temp_night_avg":       avg(night, "temperature_2m"),
        "humidity_night_avg":   humidity,
        "dewpoint_night_avg":   dew_temp,
        "pressure_night_avg":   avg(night, "surface_pressure"),
        "windspeed_night_avg":  avg(night, "windspeed_10m"),
        "winddir_night_avg":    avg(night, "winddirection_10m"),
        "cloudcover_night_avg": avg(night, "cloudcover"),
        "temp_day_avg":         avg(wdf[wdf["hour"].isin(DAY_HOURS)], "temperature_2m"),
        "humidity_day_avg":     avg(wdf[wdf["hour"].isin(DAY_HOURS)], "relative_humidity_2m"),
        "precipitation_mm":     float(wdf["precipitation"].sum()) if "precipitation" in wdf.columns else None,
        "dew_risk_flag":        int((dew_temp or 0) > 15 and (humidity or 0) > 70) if dew_temp and humidity else None,
    }


def run(batch_size: int = 50) -> None:
    logger.info("=== Phase 1b: Weather Ingestion ===")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pending       = _get_pending_matches()
    if pending.empty:
        logger.success("All matches already have weather cached.")
        return

    # Load existing cache
    results       = pd.read_parquet(OUT_PATH).to_dict("records") if OUT_PATH.exists() else []
    unknown       = set()

    for i, row in enumerate(tqdm(pending.itertuples(), total=len(pending), desc="weather")):
        coords = _venue_coords(row.venue)
        if coords is None:
            unknown.add(row.venue)
            continue

        date_str = str(row.start_date)[:10]
        weather  = _fetch_historical(coords[0], coords[1], date_str)

        record   = {"match_id": row.match_id, "start_date": date_str, "venue": row.venue,
                    "lat": coords[0], "lon": coords[1]}
        if weather:
            record.update(weather)
        results.append(record)

        # Checkpoint every batch_size matches
        if (i + 1) % batch_size == 0:
            pd.DataFrame(results).to_parquet(OUT_PATH, index=False)
            logger.info(f"Checkpoint saved — {i+1}/{len(pending)} matches processed.")

        time.sleep(0.15)  # Open-Meteo fair-use: free tier ~10k req/day

    pd.DataFrame(results).to_parquet(OUT_PATH, index=False)
    logger.success(f"Weather cache complete: {len(results)} matches → {OUT_PATH}")

    if unknown:
        logger.warning(f"Unknown venues (no coordinates — add to config/venues.py): {unknown}")


if __name__ == "__main__":
    run()