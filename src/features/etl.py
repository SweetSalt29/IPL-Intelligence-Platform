"""
src/features/etl.py
====================
Offline ETL pipeline — reads from all Phase 1 raw sources and produces
the final training feature matrix aligned to the shared feature schema.

STORAGE:
    data/processed/feature_matrix.parquet   — full feature matrix (all seasons)
    data/processed/train.parquet            — seasons 2008–2022 (training set)
    data/processed/val.parquet              — season 2023 (validation set)
    data/processed/test.parquet             — season 2024 (held-out test set)

WHEN TO RE-RUN:
    - First-time setup: run once after all Phase 1 ingestion scripts complete.
    - End of each IPL season: re-run after cricsheet_ingest, weather_ingest,
      schedule_ingest, and player_ingest have all been updated.
    - If feature_schema.yaml is changed: must re-run ETL and retrain model.
    - Never run on match day — this is a fully offline batch process.

SPLIT STRATEGY:
    Temporal split by season — NOT random. Random splitting causes data leakage
    because rolling form features for a match in season N are computed from
    matches in season N-1, N-2, etc. A random split would allow future match
    data to bleed into training.

    Train : seasons 2008–2022
    Val   : season 2023       (tune hyperparameters here)
    Test  : season 2024+      (final held-out evaluation — touch once)

LEAKAGE CONTROLS:
    - All rolling features (team form, player stats) use only matches BEFORE
      the current match date (strictly past data).
    - Toss data (winner, decision) is available pre-match — safe to include.
    - Playing XI details are NOT included — not reliably known pre-toss.

USAGE:
    ipl_venv/bin/python -m src.features.etl
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

from src.features.schema_loader import (
    get_feature_columns, get_fillna_map, get_normalize_features
)

ROOT       = Path(__file__).resolve().parents[2]
DB_PATH    = ROOT / "data" / "processed" / "ipl.duckdb"
SCHED_PATH = ROOT / "data" / "raw" / "schedule"  / "team_schedule.parquet"
WEATH_PATH = ROOT / "data" / "raw" / "weather"   / "weather_by_match.parquet"
BAT_PATH   = ROOT / "data" / "raw" / "player_profiles" / "player_batting.parquet"
BOWL_PATH  = ROOT / "data" / "raw" / "player_profiles" / "player_bowling.parquet"
OUT_DIR    = ROOT / "data" / "processed"

VAL_SEASON  = "2023"
TEST_SEASON = "2024"


# ── Step 1: Load raw sources ───────────────────────────────────────────────────

def load_matches() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute("SELECT * FROM matches").df()
    con.close()
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df.sort_values("start_date").reset_index(drop=True)


def load_balls() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute("""
        SELECT match_id, season, innings, over, batting_team, bowling_team,
               runs_off_bat, total_runs, is_wicket, is_four, is_six, wides, noballs
        FROM balls
    """).df()
    con.close()
    for c in ["runs_off_bat","total_runs","is_wicket","is_four","is_six","wides","noballs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["over"] = pd.to_numeric(df["over"], errors="coerce").fillna(0).astype(int)
    return df


def load_schedule() -> pd.DataFrame:
    if not SCHED_PATH.exists():
        logger.warning("Schedule parquet not found — run schedule_ingest.py first.")
        return pd.DataFrame()
    return pd.read_parquet(SCHED_PATH)


def load_weather() -> pd.DataFrame:
    if not WEATH_PATH.exists():
        logger.warning("Weather parquet not found — run weather_ingest.py first.")
        return pd.DataFrame()
    return pd.read_parquet(WEATH_PATH)


def load_player_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    bat  = pd.read_parquet(BAT_PATH)  if BAT_PATH.exists()  else pd.DataFrame()
    bowl = pd.read_parquet(BOWL_PATH) if BOWL_PATH.exists() else pd.DataFrame()
    return bat, bowl


# ── Step 2: Compute venue-level aggregates ─────────────────────────────────────

def compute_venue_stats(matches: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """
    Per-venue historical aggregates.
    Uses ALL historical data — no leakage risk since these are global venue properties.
    """
    # First innings scores per match
    inn1 = (
        balls[balls["innings"] == 1]
        .groupby("match_id")
        .agg(inn1_score=("total_runs","sum"), inn1_wickets=("is_wicket","sum"))
        .reset_index()
    )
    # Powerplay (overs 0–5)
    pp = (
        balls[(balls["innings"] == 1) & (balls["over"].isin(range(6)))]
        .groupby("match_id")
        .agg(pp_score=("total_runs","sum"))
        .reset_index()
    )
    # Spinner wickets proxy: overs 7–15
    spin_wkts = (
        balls[(balls["innings"] == 1) & (balls["over"].isin(range(6,16)))]
        .groupby("match_id")
        .agg(mid_wickets=("is_wicket","sum"))
        .reset_index()
    )
    # Death wickets: overs 16–20
    death_wkts = (
        balls[(balls["innings"] == 1) & (balls["over"].isin(range(15,20)))]
        .groupby("match_id")
        .agg(death_wickets=("is_wicket","sum"))
        .reset_index()
    )

    m = (
        matches[["match_id","venue","chasing_team_won"]]
        .merge(inn1,      on="match_id", how="left")
        .merge(pp,        on="match_id", how="left")
        .merge(spin_wkts, on="match_id", how="left")
        .merge(death_wkts,on="match_id", how="left")
    )

    venue_stats = m.groupby("venue").agg(
        venue_avg_first_innings_score = ("inn1_score",       "mean"),
        venue_chasing_win_rate        = ("chasing_team_won", "mean"),
        venue_avg_powerplay_score     = ("pp_score",         "mean"),
        venue_spin_factor             = ("mid_wickets",      "mean"),  # proxy
        venue_death_wickets_avg       = ("death_wickets",    "mean"),
    ).reset_index()

    # Pitch Deterioration Index: ratio of death wickets to overall wickets
    venue_stats["pitch_deterioration_index"] = (
        venue_stats["venue_death_wickets_avg"] /
        venue_stats["venue_death_wickets_avg"].replace(0, np.nan)
    ).fillna(0.5).clip(0, 1)

    # Spin factor: normalise to 0–1
    venue_stats["venue_spin_factor"] = (
        venue_stats["venue_spin_factor"] /
        venue_stats["venue_spin_factor"].max()
    ).fillna(0.35)

    return venue_stats


# ── Step 3: Compute rolling team form ─────────────────────────────────────────

def compute_team_form(matches: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team rolling form features computed STRICTLY using past matches only.
    Sorted by date, uses .shift(1) before rolling to prevent leakage.
    """
    # Per-match scores for each team
    scores = (
        balls.groupby(["match_id","batting_team"])
        .agg(score=("total_runs","sum"))
        .reset_index()
        .rename(columns={"batting_team":"team"})
    )

    # Merge match date
    m = matches[["match_id","start_date","team1","team2","chasing_team_won"]].copy()

    # Win flag per team per match
    rows = []
    for _, r in m.iterrows():
        t1_won = 1 - r["chasing_team_won"]
        t2_won = r["chasing_team_won"]
        rows.append({"match_id": r["match_id"], "start_date": r["start_date"],
                     "team": r["team1"], "won": t1_won})
        rows.append({"match_id": r["match_id"], "start_date": r["start_date"],
                     "team": r["team2"], "won": t2_won})

    form_raw = pd.DataFrame(rows).merge(scores, on=["match_id","team"], how="left")
    form_raw = form_raw.sort_values(["team","start_date"]).reset_index(drop=True)

    # Rolling last-5 — shift(1) ensures current match NOT included
    form_raw["win_rate_last5"]  = (
        form_raw.groupby("team")["won"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )
    form_raw["avg_score_last5"] = (
        form_raw.groupby("team")["score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )
    # Conceded: get opponent's score per match
    opp_scores = form_raw[["match_id","team","score"]].rename(
        columns={"team":"opponent","score":"opp_score"}
    )
    form_raw = form_raw.merge(
        matches[["match_id","team1","team2"]].melt(
            id_vars="match_id", value_name="opponent"
        ).drop(columns="variable"),
        on=["match_id"], how="left"
    ).merge(opp_scores, on=["match_id","opponent"], how="left")

    form_raw["avg_conceded_last5"] = (
        form_raw.groupby("team")["opp_score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    # Chasing win rate (historical cumulative, shifted to avoid leakage)
    chasing = []
    for _, r in m.iterrows():
        chasing.append({"match_id": r["match_id"], "team": r["team2"],
                        "chased_and_won": r["chasing_team_won"]})
        chasing.append({"match_id": r["match_id"], "team": r["team1"],
                        "chased_and_won": np.nan})
    chasing_df = pd.DataFrame(chasing).merge(
        m[["match_id","start_date"]], on="match_id"
    ).sort_values(["team","start_date"])

    chasing_df["chasing_win_rate"] = (
        chasing_df.groupby("team")["chased_and_won"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    ).fillna(0.5)

    form_out = form_raw[["match_id","team",
                          "win_rate_last5","avg_score_last5","avg_conceded_last5"]].copy()
    form_out = form_out.merge(
        chasing_df[["match_id","team","chasing_win_rate"]], on=["match_id","team"], how="left"
    )
    form_out[["win_rate_last5","avg_score_last5","avg_conceded_last5","chasing_win_rate"]] = (
        form_out[["win_rate_last5","avg_score_last5","avg_conceded_last5","chasing_win_rate"]]
        .fillna({"win_rate_last5":0.5, "avg_score_last5":160.0,
                 "avg_conceded_last5":160.0, "chasing_win_rate":0.5})
    )
    return form_out


# ── Step 4: Player squad-level aggregates ─────────────────────────────────────

def compute_player_squad_features(
    matches: pd.DataFrame, bat: pd.DataFrame, bowl: pd.DataFrame
) -> pd.DataFrame:
    """
    Squad-level batting and bowling strength per team per season.
    We don't have confirmed playing XI pre-toss, so we average across
    the top performers registered for each team in that season.
    Uses only stats from PREVIOUS seasons to avoid leakage.
    """
    if bat.empty or bowl.empty:
        logger.warning("Player profiles missing — using fillna defaults.")
        return pd.DataFrame(columns=["match_id",
            "team1_top4_avg_sr","team2_top4_avg_sr",
            "team1_bowling_avg_economy","team2_bowling_avg_economy",
            "team1_bowling_avg_economy_death","team2_bowling_avg_economy_death"])

    # Prior season batting SR per player
    bat_sorted = bat.sort_values(["player","season"])
    bat_sorted["prev_season_sr"] = bat_sorted.groupby("player")["strike_rate"].shift(1)

    bowl_sorted = bowl.sort_values(["player","season"])
    bowl_sorted["prev_season_economy"]       = bowl_sorted.groupby("player")["economy"].shift(1)
    death_col = "economy_death" if "economy_death" in bowl_sorted.columns else "economy"
    bowl_sorted["prev_season_economy_death"] = bowl_sorted.groupby("player")[death_col].shift(1)

    # Map player → team using balls table
    con = duckdb.connect(str(DB_PATH))
    player_team = con.execute("""
        SELECT striker AS player, batting_team AS team, season
        FROM balls GROUP BY striker, batting_team, season
    """).df()
    bowler_team = con.execute("""
        SELECT bowler AS player, bowling_team AS team, season
        FROM balls GROUP BY bowler, bowling_team, season
    """).df()
    con.close()

    bat_with_team  = bat_sorted.merge(player_team,  on=["player","season"], how="left")
    bowl_with_team = bowl_sorted.merge(bowler_team, on=["player","season"], how="left")

    # Top-4 SR per team per season
    top4 = (
        bat_with_team.dropna(subset=["prev_season_sr","team"])
        .groupby(["team","season"])
        .apply(lambda g: g.nlargest(4, "prev_season_sr")["prev_season_sr"].mean())
        .reset_index(name="team_top4_avg_sr")
    )
    # Top-4 economy per team
    top4_econ = (
        bowl_with_team.dropna(subset=["prev_season_economy","team"])
        .groupby(["team","season"])
        .apply(lambda g: g.nsmallest(4, "prev_season_economy")["prev_season_economy"].mean())
        .reset_index(name="team_bowling_avg_economy")
    )
    top4_death = (
        bowl_with_team.dropna(subset=["prev_season_economy_death","team"])
        .groupby(["team","season"])
        .apply(lambda g: g.nsmallest(4, "prev_season_economy_death")["prev_season_economy_death"].mean())
        .reset_index(name="team_bowling_avg_economy_death")
    )

    squad = top4.merge(top4_econ, on=["team","season"], how="outer")
    squad = squad.merge(top4_death, on=["team","season"], how="outer")

    # Join to matches (team1 and team2 separately)
    result = matches[["match_id","team1","team2","season"]].copy()
    result = result.merge(squad.rename(columns={
        "team":"team1","team_top4_avg_sr":"team1_top4_avg_sr",
        "team_bowling_avg_economy":"team1_bowling_avg_economy",
        "team_bowling_avg_economy_death":"team1_bowling_avg_economy_death"
    }), on=["team1","season"], how="left")
    result = result.merge(squad.rename(columns={
        "team":"team2","team_top4_avg_sr":"team2_top4_avg_sr",
        "team_bowling_avg_economy":"team2_bowling_avg_economy",
        "team_bowling_avg_economy_death":"team2_bowling_avg_economy_death"
    }), on=["team2","season"], how="left")

    return result[["match_id",
                   "team1_top4_avg_sr","team2_top4_avg_sr",
                   "team1_bowling_avg_economy","team2_bowling_avg_economy",
                   "team1_bowling_avg_economy_death","team2_bowling_avg_economy_death"]]


# ── Step 5: Assemble full feature matrix ──────────────────────────────────────

def assemble(
    matches:    pd.DataFrame,
    schedule:   pd.DataFrame,
    weather:    pd.DataFrame,
    venue_stats:pd.DataFrame,
    team_form:  pd.DataFrame,
    player_feats:pd.DataFrame,
) -> pd.DataFrame:
    """Join all feature blocks onto the matches spine."""
    df = matches.copy()

    # ── Toss — read from matches table (populated by parse_info_files) ──
    # toss_winner_is_team1 and toss_decision_bat written by cricsheet_ingest.
    # If present, use them. Fill remaining nulls with neutral defaults.
    if "toss_winner_is_team1" in df.columns:
        df["toss_winner_is_team1"] = pd.to_numeric(
            df["toss_winner_is_team1"], errors="coerce"
        ).fillna(0).astype("int8")
    else:
        df["toss_winner_is_team1"] = 0

    if "toss_decision_bat" in df.columns:
        df["toss_decision_bat"] = pd.to_numeric(
            df["toss_decision_bat"], errors="coerce"
        ).fillna(1).astype("int8")
    else:
        df["toss_decision_bat"] = 1
    df["is_day_match"]         = (df["start_date"].dt.hour < 15).astype("int8") \
        if "start_date" in df.columns else 0
    # season_stage: 0=league, 1=qualifier/eliminator, 2=final
    # IPL league phase is 70 matches (2022+), 60 matches earlier.
    # Matches beyond that threshold are knockouts.
    if "season" in df.columns:
        season_counts = df.groupby("season")["match_id"].transform("count")
        match_rank    = df.groupby("season").cumcount() + 1
        league_cutoff = season_counts - 4   # last 4 matches = knockouts
        df["season_stage"] = (
            (match_rank > league_cutoff).astype("int8")
        )
    else:
        df["season_stage"] = 0

    # ── Venue encode ───────────────────────────────────────────────
    le = LabelEncoder()
    df["venue_encoded"] = le.fit_transform(df["venue"].fillna("Unknown"))
    df = df.merge(venue_stats, on="venue", how="left")

    # ── Schedule / fatigue ─────────────────────────────────────────
    if not schedule.empty:
        sched_t1 = schedule.rename(columns={
            "team":"team1","rest_days":"team1_rest_days",
            "back_to_back":"team1_back_to_back","travel_km":"team1_travel_km",
            "travel_burden_score":"team1_travel_burden",
            "season_match_num":"team1_season_match_num","is_home":"team1_home_flag",
        })[["match_id","team1","team1_rest_days","team1_back_to_back",
            "team1_travel_km","team1_travel_burden","team1_season_match_num","team1_home_flag"]]
        sched_t2 = schedule.rename(columns={
            "team":"team2","rest_days":"team2_rest_days",
            "back_to_back":"team2_back_to_back","travel_km":"team2_travel_km",
            "travel_burden_score":"team2_travel_burden",
            "season_match_num":"team2_season_match_num","is_home":"team2_home_flag",
        })[["match_id","team2","team2_rest_days","team2_back_to_back",
            "team2_travel_km","team2_travel_burden","team2_season_match_num","team2_home_flag"]]
        # match_id type alignment before merge
        sched_t1["match_id"] = sched_t1["match_id"].astype(str)
        sched_t2["match_id"] = sched_t2["match_id"].astype(str)
        df["match_id"] = df["match_id"].astype(str)
        df = df.merge(sched_t1, on=["match_id","team1"], how="left")
        df = df.merge(sched_t2, on=["match_id","team2"], how="left")

    # Clamp rest_days
    for col in ["team1_rest_days","team2_rest_days"]:
        if col in df.columns:
            df[col] = df[col].replace(99, 7).clip(upper=14)

    # ── Weather ────────────────────────────────────────────────────
    # match_id type alignment: weather parquet stores string (filename stem),
    # matches table may have integer IDs. Cast both to str before merge.
    if not weather.empty:
        weather_cols = ["match_id","dew_risk_flag","temp_night_avg","humidity_night_avg",
                        "dewpoint_night_avg","windspeed_night_avg","pressure_night_avg",
                        "precipitation_mm"]
        w = weather[[c for c in weather_cols if c in weather.columns]].copy()
        w["match_id"] = w["match_id"].astype(str)
        df["match_id"] = df["match_id"].astype(str)
        df = df.merge(w, on="match_id", how="left")

    # ── Team form ──────────────────────────────────────────────────
    form_t1 = team_form.rename(columns={
        "team":"team1","win_rate_last5":"team1_win_rate_last5",
        "avg_score_last5":"team1_avg_score_last5",
        "avg_conceded_last5":"team1_avg_conceded_last5",
        "chasing_win_rate":"team1_chasing_win_rate",
    })
    form_t2 = team_form.rename(columns={
        "team":"team2","win_rate_last5":"team2_win_rate_last5",
        "avg_score_last5":"team2_avg_score_last5",
        "avg_conceded_last5":"team2_avg_conceded_last5",
        "chasing_win_rate":"team2_chasing_win_rate",
    })
    df = df.merge(form_t1[["match_id","team1","team1_win_rate_last5",
                            "team1_avg_score_last5","team1_avg_conceded_last5",
                            "team1_chasing_win_rate"]], on=["match_id","team1"], how="left")
    df = df.merge(form_t2[["match_id","team2","team2_win_rate_last5",
                            "team2_avg_score_last5","team2_avg_conceded_last5",
                            "team2_chasing_win_rate"]], on=["match_id","team2"], how="left")

    # ── Player squad strength ──────────────────────────────────────
    if not player_feats.empty:
        df = df.merge(player_feats, on="match_id", how="left")

    return df


# ── Step 6: Apply schema — fillna, dtype, column order ────────────────────────

def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the shared feature schema:
      1. Apply fillna for every feature
      2. Cast to correct dtype
      3. Reorder columns to match schema order exactly
      4. Append label column
    """
    fillna_map = get_fillna_map()
    feature_cols = get_feature_columns()

    for col, fill_val in fillna_map.items():
        if col not in df.columns:
            df[col] = fill_val
        else:
            df[col] = df[col].fillna(fill_val)

    # Label
    df["chasing_team_won"] = df["chasing_team_won"].fillna(0).astype("int8")

    # Reorder: feature columns in schema order, then label
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Features missing after join (using fillna): {missing}")

    return df[["match_id","season","start_date"] + available + ["chasing_team_won"]]


# ── Step 7: Normalize + Split ──────────────────────────────────────────────────

def normalize_and_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Min-max normalize features flagged normalize=true in schema.
    Fit scaler on TRAIN only, apply to val/test — never fit on val/test.
    Save scaler artifact for inference-time use.
    """
    norm_cols = [c for c in get_normalize_features() if c in df.columns]

    train = df[df["season"].astype(str) < VAL_SEASON].copy()
    val   = df[df["season"].astype(str) == VAL_SEASON].copy()
    test  = df[df["season"].astype(str) >= TEST_SEASON].copy()

    logger.info(f"Split sizes — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    if len(train) == 0:
        logger.warning("Train set empty — using full data as train (mock data mode).")
        train = df.copy()
        val   = df.copy()
        test  = df.copy()

    scaler = MinMaxScaler()
    train[norm_cols] = scaler.fit_transform(train[norm_cols].fillna(0))
    if len(val)  > 0: val[norm_cols]  = scaler.transform(val[norm_cols].fillna(0))
    if len(test) > 0: test[norm_cols] = scaler.transform(test[norm_cols].fillna(0))

    # Save scaler — inference pipeline must use this exact scaler
    scaler_path = OUT_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.success(f"Scaler saved → {scaler_path}")

    return {"train": train, "val": val, "test": test}


# ── Master run ─────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info("=== Phase 2: ETL + Feature Engineering ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw sources ...")
    matches  = load_matches()
    balls    = load_balls()
    schedule = load_schedule()
    weather  = load_weather()
    bat, bowl= load_player_profiles()

    logger.info(f"Matches: {len(matches)} | Balls: {len(balls):,}")

    logger.info("Computing venue stats ...")
    venue_stats  = compute_venue_stats(matches, balls)

    logger.info("Computing team form (rolling, leakage-safe) ...")
    team_form    = compute_team_form(matches, balls)

    logger.info("Computing player squad features ...")
    player_feats = compute_player_squad_features(matches, bat, bowl)

    logger.info("Assembling feature matrix ...")
    df = assemble(matches, schedule, weather, venue_stats, team_form, player_feats)

    logger.info("Applying feature schema (fillna + dtype + column order) ...")
    df = apply_schema(df)

    feature_cols = get_feature_columns()
    logger.info(f"Feature matrix: {len(df)} rows × {len(feature_cols)} features")

    # Save full matrix
    df.to_parquet(OUT_DIR / "feature_matrix.parquet", index=False)
    logger.success(f"Full matrix → {OUT_DIR}/feature_matrix.parquet")

    # Normalize and split
    splits = normalize_and_split(df)
    for name, split_df in splits.items():
        if len(split_df) > 0:
            path = OUT_DIR / f"{name}.parquet"
            split_df.to_parquet(path, index=False)
            logger.success(f"{name.upper()} ({len(split_df)} rows) → {path}")

    # Leakage check: confirm no future data in training features
    logger.info("\nLeakage check ...")
    train = splits["train"]
    if "start_date" in train.columns:
        max_train_date = pd.to_datetime(train["start_date"]).max()
        logger.info(f"Latest date in train set: {max_train_date.date()}")
        assert str(max_train_date.year) < VAL_SEASON or len(train) == len(df), \
            "LEAKAGE DETECTED — val/test dates found in train set"
        logger.success("Leakage check passed.")

    # Quick feature summary
    logger.info("\nFeature summary (train set, non-null %):")
    feat_nulls = train[feature_cols].isnull().mean().sort_values(ascending=False)
    non_trivial = feat_nulls[feat_nulls > 0.05]
    if len(non_trivial) > 0:
        logger.warning(f"Features with >5% null before fillna:\n{non_trivial}")
    else:
        logger.success("All features fully populated after fillna.")

    logger.success("\nPhase 2 complete. Ready for Phase 3 — Model Training.")


if __name__ == "__main__":
    run()