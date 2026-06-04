"""
src/ingestion/cricsheet_ingest.py

Downloads IPL ball-by-ball CSVs from Cricsheet, parses them into a
unified DuckDB table. Run once, then update at end of each season.

Usage:
    python -m src.ingestion.cricsheet_ingest
"""

import sys
import zipfile
import requests
import pandas as pd
import duckdb
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path.cwd()   # Uses current project folder

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "cricsheet"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = PROCESSED_DIR / "ipl.duckdb"

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"

# Ensure folders exist immediately
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Column dtypes expected from Cricsheet CSV2 format ─────────────────
BALL_DTYPES = {
    "match_id":             "str",
    "season":               "str",
    "start_date":           "str",
    "venue":                "str",
    "innings":              "int8",
    "ball":                 "float32",
    "batting_team":         "str",
    "bowling_team":         "str",
    "striker":              "str",
    "non_striker":          "str",
    "bowler":               "str",
    "runs_off_bat":         "int8",
    "extras":               "int8",
    "wides":                "float32",
    "noballs":              "float32",
    "byes":                 "float32",
    "legbyes":              "float32",
    "penalty":              "float32",
    "wicket_type":          "str",
    "player_dismissed":     "str",
    "other_wicket_type":    "str",
    "other_player_dismissed": "str",
}


def download_cricsheet(force: bool = False) -> Path:
    """Download IPL zip from Cricsheet. Skip if already present."""
    zip_path = RAW_DIR / "ipl_csv2.zip"
    if zip_path.exists() and not force:
        print(f"[cricsheet] Zip already present at {zip_path}. Skipping download.")
        return zip_path

    print(f"[cricsheet] Downloading from {CRICSHEET_URL} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(CRICSHEET_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="cricsheet zip"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[cricsheet] Saved to {zip_path}")
    return zip_path


def extract_csvs(zip_path: Path) -> Path:
    """Extract all CSVs from the zip into RAW_DIR/csv/."""
    csv_dir = RAW_DIR / "csv"
    if csv_dir.exists() and any(csv_dir.glob("*.csv")):
        print(f"[cricsheet] CSVs already extracted at {csv_dir}. Skipping.")
        return csv_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        print(f"[cricsheet] Extracting {len(members)} CSV files ...")
        for member in tqdm(members, desc="extracting"):
            zf.extract(member, csv_dir)
    return csv_dir


def load_all_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Cricsheet CSV2 format: each match produces two files.
      *_info.csv  — match metadata (1 row per match)
      *.csv       — ball-by-ball (N rows per match)

    We parse the ball-by-ball files and attach metadata from info files.
    """
    ball_files = sorted(csv_dir.rglob("[!*_info]*.csv"))
    # filter: exclude info files
    ball_files = [f for f in ball_files if not f.name.endswith("_info.csv")]

    dfs = []
    errors = []
    print(f"[cricsheet] Parsing {len(ball_files)} ball-by-ball files ...")
    for bf in tqdm(ball_files, desc="parsing"):
        try:
            df = pd.read_csv(bf, dtype=str, low_memory=False)
            df["source_file"] = bf.name
            dfs.append(df)
        except Exception as e:
            errors.append((bf.name, str(e)))

    if errors:
        print(f"[cricsheet] WARNING: {len(errors)} files failed to parse:")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[cricsheet] Total deliveries loaded: {len(combined):,}")
    return combined


def clean_balls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise types, derive computed columns, handle known Cricsheet quirks.
    """
    # Numeric casts — coerce errors to NaN
    int_cols   = ["innings", "runs_off_bat", "extras"]
    float_cols = ["ball", "wides", "noballs", "byes", "legbyes", "penalty"]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int8")
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # Date
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["year"]  = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month

    # Over + ball-in-over from ball column (e.g. 3.2 → over=3, ball_in_over=2)
    df["over"]         = df["ball"].dropna().astype(int)
    df["ball_in_over"] = (
        (df["ball"].fillna(0) * 10).astype(int) % 10
    )

    # Is wicket ball?
    df["is_wicket"] = df["wicket_type"].notna().astype("int8")

    # Total runs on delivery
    df["total_runs"] = (
        pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0) +
        pd.to_numeric(df["extras"],       errors="coerce").fillna(0)
    ).astype("int16")

    # Is boundary
    df["is_four"] = (df["runs_off_bat"] == "4").astype("int8") \
        if df["runs_off_bat"].dtype == object \
        else (pd.to_numeric(df["runs_off_bat"], errors="coerce") == 4).astype("int8")
    df["is_six"]  = (df["runs_off_bat"] == "6").astype("int8") \
        if df["runs_off_bat"].dtype == object \
        else (pd.to_numeric(df["runs_off_bat"], errors="coerce") == 6).astype("int8")

    # Season normalise: Cricsheet uses "2009/10" format for some seasons
    df["season"] = df["season"].str.replace(r"/\d+", "", regex=True)

    return df


def derive_match_summary(balls: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball into match-level rows.
    Each match → 1 row with team scores, winner flag, etc.
    This is our primary training label source.
    """
    # Per innings totals
    inn = (
        balls.groupby(["match_id", "innings", "batting_team", "bowling_team"])
        .agg(
            runs      = ("total_runs", "sum"),
            wickets   = ("is_wicket", "sum"),
            balls_faced = ("ball", "count"),
            fours     = ("is_four", "sum"),
            sixes     = ("is_six", "sum"),
        )
        .reset_index()
    )

    # Split innings 1 and 2
    inn1 = inn[inn["innings"] == 1].copy().add_suffix("_inn1").rename(
        columns={"match_id_inn1": "match_id", "innings_inn1": "innings"}
    )
    inn2 = inn[inn["innings"] == 2].copy().add_suffix("_inn2").rename(
        columns={"match_id_inn2": "match_id", "innings_inn2": "innings"}
    )

    matches = balls.drop_duplicates("match_id")[
        ["match_id", "season", "start_date", "venue", "year", "month"]
    ].copy()

    matches = matches.merge(inn1, on="match_id", how="left")
    matches = matches.merge(inn2, on="match_id", how="left")

    # Winner: team2 wins if they exceeded team1 score (chasing)
    matches["team1"] = matches["batting_team_inn1"]
    matches["team2"] = matches["batting_team_inn2"]
    matches["team1_score"] = matches["runs_inn1"]
    matches["team2_score"] = matches["runs_inn2"]

    # 1 = team2 (chasing team) won, 0 = team1 (batting first) won
    matches["chasing_team_won"] = (
        matches["team2_score"] > matches["team1_score"]
    ).astype("int8")

    return matches


def write_to_duckdb(balls: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Write both tables into DuckDB."""

    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    con.execute("DROP TABLE IF EXISTS balls")
    con.execute("DROP TABLE IF EXISTS matches")

    con.register("balls_df", balls)
    con.register("matches_df", matches)

    con.execute("CREATE TABLE balls AS SELECT * FROM balls_df")
    con.execute("CREATE TABLE matches AS SELECT * FROM matches_df")

    n_balls = con.execute("SELECT COUNT(*) FROM balls").fetchone()[0]
    n_matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    print(f"[duckdb] Written: {n_balls:,} deliveries | {n_matches:,} matches")
    print(f"[duckdb] Database saved at: {DB_PATH}")

    con.close()

def run(force_download: bool = False) -> None:
    zip_path = download_cricsheet(force=force_download)
    csv_dir  = extract_csvs(zip_path)
    balls    = load_all_csvs(csv_dir)
    balls    = clean_balls(balls)
    matches  = derive_match_summary(balls)
    write_to_duckdb(balls, matches)
    print("\n[cricsheet] Phase 1a complete — ball-by-ball data in DuckDB.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force_download=force)
