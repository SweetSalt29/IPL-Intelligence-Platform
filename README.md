# IPL Win Prediction System

A multi-agent LangGraph system for pre-match IPL win probability prediction, incorporating ball-by-ball data, weather, player profiles, and scheduling factors.

---

## Architecture

Two pipelines:

* **Offline** — historical data ingestion, feature engineering, model training
* **Live inference** — LangGraph multi-agent system that fetches match-day data, builds features, and calls the trained model as a tool

Full architecture diagram: `IPL_Architecture.png`

---

## Project Structure

```
IPL Project/
├── config/
│   ├── feature_schema.yaml   # Shared feature contract — training + inference
│   └── venues.py             # Venue registry, coordinates, home grounds
├── data/
│   ├── processed/            # DuckDB, parquet splits, scaler (gitignored)
│   └── raw/                  # Cricsheet, weather, schedule, player profiles (gitignored)
├── src/
│   ├── ingestion/            # Phase 1 — data acquisition scripts
│   ├── features/             # Phase 2 — ETL + schema loader
│   ├── models/               # Phase 3 — training + serving
│   └── agents/               # Phase 4 — LangGraph agents
├── run_phase1.py             # Phase 1 master runner
├── run_phase2.py             # Phase 2 master runner
├── setup.py
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd IPL\ Project
python3 -m venv ipl_venv
```

### 2. Install dependencies

```bash
ipl_venv/bin/pip install -r requirements.txt
```

### 3. Install project as editable package

**Required** — makes `src` and `config` importable from anywhere.

```bash
ipl_venv/bin/pip install -e .
```

### 4. Configure environment

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env (required for Phase 4 only)
```

---

## Running the Pipeline

### Phase 1 — Data Acquisition

```bash
ipl_venv/bin/python run_phase1.py
```

Downloads Cricsheet IPL data, fetches historical weather from Open-Meteo, computes schedule and player profiles. Run once on setup, then at end of each IPL season.

**Re-download Cricsheet:**

```bash
ipl_venv/bin/python run_phase1.py --force
```

### Phase 2 — Feature Engineering + ETL

```bash
ipl_venv/bin/python run_phase2.py
```

Builds the 44-feature training matrix, applies the shared schema, temporal splits (train/val/test), saves `scaler.pkl`. Run after Phase 1.

### Phase 3 — Model Training *(coming soon)*

```bash
ipl_venv/bin/python run_phase3.py
```

### Phase 4 — LangGraph Agent System *(coming soon)*

```bash
ipl_venv/bin/python run_phase4.py
```

---

## Data Sources

| Source                 | Stored locally            | Fetched via internet      |
| ---------------------- | ------------------------- | ------------------------- |
| Cricsheet ball-by-ball | ✅ After first download   | First run only            |
| Open-Meteo weather     | ✅ Cached as parquet      | First run + end of season |
| IPL schedule           | ✅ Derived from Cricsheet | Never                     |
| Player profiles        | ✅ Derived from Cricsheet | Never                     |

---

## When to Re-run

| Script                  | When                                              |
| ----------------------- | ------------------------------------------------- |
| `cricsheet_ingest.py` | End of each IPL season                            |
| `weather_ingest.py`   | After Cricsheet update (fetches new matches only) |
| `schedule_ingest.py`  | After Cricsheet update                            |
| `player_ingest.py`    | After Cricsheet update                            |
| `run_phase2.py`       | After any Phase 1 update or schema change         |

---

## Feature Schema

All 44 features are defined in `config/feature_schema.yaml` with source, dtype, fillna fallback, and description. This is the contract between training and inference — any change here requires re-running Phase 2 and retraining the model.

Feature groups: `match_context`, `venue`, `environmental`, `fatigue`, `team_form`, `player_strength`, `pitch`
