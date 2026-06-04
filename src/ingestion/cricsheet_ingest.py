"""
src/ingestion/cricsheet_ingest.py
==================================
Downloads IPL ball-by-ball CSVs from Cricsheet and loads them into DuckDB.

STORAGE:
    data/raw/cricsheet/ipl_csv2.zip   — raw zip (gitignored after first run)
    data/raw/cricsheet/csv/           — extracted per-match CSVs (gitignored)
    data/processed/ipl.duckdb         — tables: `balls`, `matches`

WHEN TO RE-RUN:
    - First-time setup: run once to build the full historical database (2008–present).
    - End of each IPL season: re-run with --force to pull the updated zip from
      Cricsheet which will include the completed season's matches.
    - Never needs to run during a live match or on match day.

IDEMPOTENT: Safe to re-run. --force flag re-downloads and overwrites.

USAGE:
    ipl_venv/bin/python -m src.ingestion.cricsheet_ingest
    ipl_venv/bin/python -m src.ingestion.cricsheet_ingest --force
"""

import io
import sys
import zipfile
import requests
import pandas as pd
import duckdb
from pathlib import Path
from tqdm import tqdm
from loguru import logger

ROOT          = Path(__file__).resolve().parents[2]
RAW_DIR       = ROOT / "data" / "raw" / "cricsheet"
DB_PATH       = ROOT / "data" / "processed" / "ipl.duckdb"
CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"


def download_cricsheet(force: bool = False) -> Path:
    """
    Download IPL zip from Cricsheet.
    Skips download if zip already exists unless force=True.
    """
    zip_path = RAW_DIR / "ipl_csv2.zip"
    if zip_path.exists() and not force:
        logger.info(f"Zip already present at {zip_path}. Skipping download. Use --force to re-download.")
        return zip_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {CRICSHEET_URL} ...")
    resp  = requests.get(CRICSHEET_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="cricsheet") as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    logger.success(f"Downloaded → {zip_path}")
    return zip_path


def extract_csvs(zip_path: Path, force: bool = False) -> Path:
    """Extract all match CSVs from zip into data/raw/cricsheet/csv/."""
    csv_dir = RAW_DIR / "csv"
    if csv_dir.exists() and any(csv_dir.glob("*.csv")) and not force:
        n = len(list(csv_dir.glob("*.csv")))
        logger.info(f"CSVs already extracted ({n} files). Skipping.")
        return csv_dir

    csv_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        logger.info(f"Extracting {len(members)} CSV files ...")
        for member in tqdm(members, desc="extracting"):
            zf.extract(member, csv_dir)

    logger.success(f"Extracted to {csv_dir}")
    return csv_dir


def load_all_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Parse all ball-by-ball CSVs from csv_dir.
    Cricsheet CSV2 format: each match has one ball-by-ball file.
    Info files (*_info.csv) are skipped — metadata is embedded in ball files.
    """
    ball_files = [f for f in sorted(csv_dir.rglob("*.csv")) if not f.name.endswith("_info.csv")]
    logger.info(f"Parsing {len(ball_files)} ball-by-ball files ...")

    dfs, errors = [], []
    for bf in tqdm(ball_files, desc="parsing CSVs"):
        try:
            df = pd.read_csv(bf, dtype=str, low_memory=False)
            df["source_file"] = bf.name
            dfs.append(df)
        except Exception as e:
            errors.append((bf.name, str(e)))

    if errors:
        logger.warning(f"{len(errors)} files failed to parse:")
        for name, err in errors[:5]:
            logger.warning(f"  {name}: {err}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.success(f"Loaded {len(combined):,} deliveries from {len(dfs)} matches")
    return combined


def clean_balls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise types and derive computed columns.
    Handles known Cricsheet quirks:
      - Season format: '2009/10' normalised to '2009'
      - Numeric columns stored as strings
      - Missing wicket fields stored as NaN
    """
    # Numerics
    for c in ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "penalty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["innings"] = pd.to_numeric(df.get("innings"), errors="coerce")
    df["ball"]    = pd.to_numeric(df.get("ball"),    errors="coerce")

    # Date and time decomposition
    df["start_date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
    df["year"]       = df["start_date"].dt.year
    df["month"]      = df["start_date"].dt.month

    # Over and ball-in-over from ball column (e.g. 3.2 → over=3, ball_in_over=2)
    df["over"]         = df["ball"].apply(lambda x: int(x) if pd.notna(x) else None)
    df["ball_in_over"] = df["ball"].apply(lambda x: round((x % 1) * 10) if pd.notna(x) else None)

    # Derived flags
    df["is_wicket"]  = df["wicket_type"].notna().astype("int8")
    df["total_runs"] = (df["runs_off_bat"] + df["extras"]).astype("int16")
    df["is_four"]    = (df["runs_off_bat"] == 4).astype("int8")
    df["is_six"]     = (df["runs_off_bat"] == 6).astype("int8")
    df["is_dot"]     = ((df["runs_off_bat"] == 0) & (df["wides"] == 0) & (df["noballs"] == 0)).astype("int8")

    # Normalise season: '2009/10' → '2009'
    df["season"] = df["season"].astype(str).str.replace(r"/\d+", "", regex=True).str.strip()

    return df


def derive_match_summary(balls: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate deliveries into one row per match.
    Computes team scores per innings and derives the binary label:
      chasing_team_won = 1 if team batting second exceeded team batting first's score.
    This is the primary training label.
    """
    # Per innings totals
    inn = (
        balls.groupby(["match_id", "innings", "batting_team", "bowling_team"])
        .agg(
            runs        = ("total_runs",   "sum"),
            wickets     = ("is_wicket",    "sum"),
            balls_bowled= ("ball",         "count"),
            fours       = ("is_four",      "sum"),
            sixes       = ("is_six",       "sum"),
        )
        .reset_index()
    )

    inn1 = inn[inn["innings"] == 1].copy()
    inn2 = inn[inn["innings"] == 2].copy()

    # Match-level metadata (one row per match)
    match_meta = balls.drop_duplicates("match_id")[
        ["match_id", "season", "start_date", "venue", "year", "month"]
    ].copy()

    # Merge innings
    matches = match_meta.merge(
        inn1[["match_id","batting_team","bowling_team","runs","wickets","balls_bowled","fours","sixes"]]
            .rename(columns=lambda c: f"{c}_inn1" if c != "match_id" else c),
        on="match_id", how="left"
    ).merge(
        inn2[["match_id","batting_team","runs","wickets"]]
            .rename(columns=lambda c: f"{c}_inn2" if c != "match_id" else c),
        on="match_id", how="left"
    )

    matches = matches.rename(columns={
        "batting_team_inn1": "team1",
        "bowling_team_inn1": "team2",
        "batting_team_inn2": "chasing_team",
        "runs_inn1":         "team1_score",
        "runs_inn2":         "team2_score",
    })

    # Binary label: 1 = chasing team won
    matches["chasing_team_won"] = (
        matches["team2_score"] > matches["team1_score"]
    ).astype("int8")

    return matches


def write_to_duckdb(balls: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Write balls and matches DataFrames to DuckDB, replacing existing tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS balls")
    con.execute("DROP TABLE IF EXISTS matches")
    con.execute("CREATE TABLE balls    AS SELECT * FROM balls")
    con.execute("CREATE TABLE matches  AS SELECT * FROM matches")

    n_balls   = con.execute("SELECT COUNT(*) FROM balls").fetchone()[0]
    n_matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    logger.success(f"DuckDB written: {n_balls:,} deliveries | {n_matches:,} matches")

    logger.info("Matches per season:")
    season_dist = con.execute(
        "SELECT season, COUNT(*) AS matches FROM matches GROUP BY season ORDER BY season"
    ).df()
    print(season_dist.to_string(index=False))
    con.close()


def run(force: bool = False) -> None:
    logger.info("=== Phase 1a: Cricsheet Ingestion ===")
    zip_path = download_cricsheet(force=force)
    csv_dir  = extract_csvs(zip_path, force=force)
    balls    = load_all_csvs(csv_dir)
    balls    = clean_balls(balls)
    matches  = derive_match_summary(balls)
    write_to_duckdb(balls, matches)
    logger.success("Phase 1a complete — data/processed/ipl.duckdb ready.")


if __name__ == "__main__":
    run(force="--force" in sys.argv)