"""
src/ingestion/weather_ingest.py

Fetches historical hourly weather for every IPL match from Open-Meteo.
Uses ERA5 reanalysis — free, no API key, covers 1940–present.

Fetches variables relevant to our extraneous factor model:
  - temperature_2m          → heat stress
  - relative_humidity_2m    → dew risk proxy
  - dewpoint_2m             → dew point (direct)
  - precipitation            → rain risk
  - surface_pressure        → air density / swing
  - windspeed_10m           → wind effect on shots
  - winddirection_10m       → crosswind flag
  - cloudcover              → visibility / ball sighting

Output: data/raw/weather/weather_by_match.parquet
        one row per match with pre-game and in-match weather averages.

Usage:
    python -m src.ingestion.weather_ingest
"""

import time
import duckdb
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

ROOT        = Path(__file__).resolve().parents[2]
DB_PATH     = ROOT / "data" / "processed" / "ipl.duckdb"
OUT_PATH    = ROOT / "data" / "raw" / "weather" / "weather_by_match.parquet"

# Open-Meteo Historical Weather API
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

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

# All IPL venues with coordinates
# (venue_name → lat, lon) — kept flat here for fast lookup
VENUE_COORDS = {
    "M Chinnaswamy Stadium":                       (12.9791, 77.5497),
    "Wankhede Stadium":                            (18.9389, 72.8258),
    "Eden Gardens":                                (22.5645, 88.3433),
    "MA Chidambaram Stadium":                      (13.0628, 80.2791),
    "Arun Jaitley Stadium":                        (28.6364, 77.2195),
    "Feroz Shah Kotla":                            (28.6364, 77.2195),  # alias
    "Rajiv Gandhi International Stadium":          (17.4042, 78.5498),
    "Punjab Cricket Association IS Bindra Stadium":(30.6842, 76.7154),
    "Punjab Cricket Association Stadium, Mohali":  (30.6842, 76.7154),
    "Sawai Mansingh Stadium":                      (26.8972, 75.8024),
    "Narendra Modi Stadium":                       (23.0900, 72.0847),
    "Sardar Patel Stadium":                        (23.0900, 72.0847),
    "Brabourne Stadium":                           (18.9322, 72.8264),
    "DY Patil Stadium":                            (19.0435, 72.9987),
    "Dr DY Patil Sports Academy":                  (19.0435, 72.9987),
    "Maharashtra Cricket Association Stadium":     (18.6298, 73.8015),
    "JSCA International Stadium Complex":          (23.3441, 85.3096),
    "Himachal Pradesh Cricket Association Stadium":(32.2190, 76.3234),
    "Barsapara Cricket Stadium":                   (26.1433, 91.7898),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": (17.7231, 83.2183),
    "Barabati Stadium":                            (20.4686, 85.8792),
    "Greenfield International Stadium":            (8.5553,  76.9063),
    "Holkar Cricket Stadium":                      (22.7215, 75.8578),
    "Ekana Cricket Stadium":                       (26.8575, 80.9346),
    # South Africa (2009)
    "Newlands":                                    (-33.9258, 18.4232),
    "St George's Park":                            (-33.9608, 25.6022),
    "Kingsmead":                                   (-29.8579, 31.0292),
    "SuperSport Park":                             (-25.7547, 28.2267),
    "New Wanderers Stadium":                       (-26.1446, 28.0566),
    "De Beers Diamond Oval":                       (-28.7377, 24.7478),
    "Buffalo Park":                                (-32.9594, 27.9034),
    # UAE (2014, 2020, 2021)
    "Dubai International Cricket Stadium":         (25.0359, 55.2466),
    "Sheikh Zayed Stadium":                        (24.3886, 54.5195),
    "Sharjah Cricket Stadium":                     (25.3396, 55.3839),
}

# Day match start ~14:00 local, night match ~19:30 local (IPL standard)
MATCH_HOURS_DAY   = list(range(13, 19))  # 1pm-7pm
MATCH_HOURS_NIGHT = list(range(18, 24))  # 6pm-midnight


def get_matches_needing_weather() -> pd.DataFrame:
    """Load match list from DuckDB. Return only matches missing weather data."""
    con = duckdb.connect(str(DB_PATH))
    matches = con.execute(
        "SELECT match_id, start_date, venue FROM matches ORDER BY start_date"
    ).df()
    con.close()

    # Check what we already have
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done_ids = set(existing["match_id"].astype(str))
        matches  = matches[~matches["match_id"].astype(str).isin(done_ids)]
        print(f"[weather] {len(done_ids)} matches already have weather. "
              f"{len(matches)} remaining.")
    else:
        print(f"[weather] Starting fresh. {len(matches)} matches to fetch.")
    return matches


def resolve_coords(venue: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a venue string. None if unknown."""
    if venue in VENUE_COORDS:
        return VENUE_COORDS[venue]
    # Fuzzy: try contains match
    for known, coords in VENUE_COORDS.items():
        if known.lower() in venue.lower() or venue.lower() in known.lower():
            return coords
    return None


def fetch_weather_for_match(
    lat: float, lon: float, date: str
) -> dict | None:
    """
    Call Open-Meteo historical API for one match date.
    Returns aggregated weather dict or None on failure.
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
        print(f"    [weather] API error for ({lat},{lon}) {date}: {e}")
        return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return None

    # Build hourly DataFrame
    wdf = pd.DataFrame(hourly)
    wdf["hour"] = pd.to_datetime(wdf["time"]).dt.hour

    # Night match aggregations (6pm–midnight) — primary IPL window
    night = wdf[wdf["hour"].isin(MATCH_HOURS_NIGHT)]
    day   = wdf[wdf["hour"].isin(MATCH_HOURS_DAY)]

    def safe_mean(series):
        return float(series.mean()) if len(series) > 0 else None

    result = {
        # Night window (primary)
        "temp_night_mean":         safe_mean(night["temperature_2m"])       if "temperature_2m"       in night else None,
        "humidity_night_mean":     safe_mean(night["relative_humidity_2m"]) if "relative_humidity_2m" in night else None,
        "dewpoint_night_mean":     safe_mean(night["dewpoint_2m"])           if "dewpoint_2m"           in night else None,
        "precipitation_total":     float(wdf["precipitation"].sum())         if "precipitation"         in wdf   else None,
        "pressure_night_mean":     safe_mean(night["surface_pressure"])      if "surface_pressure"      in night else None,
        "windspeed_night_mean":    safe_mean(night["windspeed_10m"])          if "windspeed_10m"         in night else None,
        "winddirection_night_mean":safe_mean(night["winddirection_10m"])      if "winddirection_10m"     in night else None,
        "cloudcover_night_mean":   safe_mean(night["cloudcover"])             if "cloudcover"            in night else None,
        # Day window
        "temp_day_mean":           safe_mean(day["temperature_2m"])          if "temperature_2m"        in day   else None,
        "humidity_day_mean":       safe_mean(day["relative_humidity_2m"])    if "relative_humidity_2m"  in day   else None,
        # Dew risk: high if dewpoint_night_mean > 15°C and humidity > 70%
        "dew_risk_flag":           int(
            (safe_mean(night["dewpoint_2m"])           or 0) > 15 and
            (safe_mean(night["relative_humidity_2m"])  or 0) > 70
        ) if "dewpoint_2m" in night and "relative_humidity_2m" in night else None,
    }
    return result


def run(batch_size: int = 50) -> None:
    """Fetch weather for all matches, save incrementally."""
    matches = get_matches_needing_weather()
    if matches.empty:
        print("[weather] All matches already have weather data.")
        return

    # Load existing results to append to
    results = []
    if OUT_PATH.exists():
        results = pd.read_parquet(OUT_PATH).to_dict("records")

    unknown_venues = set()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm(matches.itertuples(), total=len(matches), desc="weather")):
        coords = resolve_coords(row.venue)
        if coords is None:
            unknown_venues.add(row.venue)
            continue

        date_str = str(row.start_date)[:10]  # YYYY-MM-DD
        weather  = fetch_weather_for_match(coords[0], coords[1], date_str)

        record = {
            "match_id":   row.match_id,
            "start_date": date_str,
            "venue":      row.venue,
            "lat":        coords[0],
            "lon":        coords[1],
        }
        if weather:
            record.update(weather)
        results.append(record)

        # Save every batch_size matches
        if (i + 1) % batch_size == 0:
            pd.DataFrame(results).to_parquet(OUT_PATH, index=False)
            print(f"  [weather] Checkpoint saved — {i+1} matches processed.")

        # Respect Open-Meteo fair use: ~10k req/day free
        # ~0.1s sleep keeps us well under limit
        time.sleep(0.1)

    # Final save
    pd.DataFrame(results).to_parquet(OUT_PATH, index=False)
    print(f"\n[weather] Done. {len(results)} matches with weather data.")
    if unknown_venues:
        print(f"[weather] Unknown venues (no coords): {unknown_venues}")
    print(f"[weather] Output: {OUT_PATH}")


if __name__ == "__main__":
    run()
