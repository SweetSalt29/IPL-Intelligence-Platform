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
├── run_phase3.py             # Phase 3 master runner
├── run_phase4.py             # Phase 4 master runner
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
# Add your GROQ_API_KEY to .env (required for Phase 4 narrative agent)
```

Get a free Groq API key (no credit card) at [https://console.groq.com](https://console.groq.com)

---

## Running the Pipeline

### Phase 1 — Data Acquisition

```bash
ipl_venv/bin/python run_phase1.py
```

Downloads Cricsheet IPL ball-by-ball data (2008–present), fetches historical weather from Open-Meteo, computes schedule (travel, rest days) and player profiles. Run once on setup, then at end of each IPL season.

**Re-download Cricsheet:**

```bash
ipl_venv/bin/python run_phase1.py --force
```

### Phase 2 — Feature Engineering + ETL

```bash
ipl_venv/bin/python run_phase2.py
```

Builds the 44-feature training matrix, applies the shared schema, temporal splits (train/val/test by season), saves `scaler.pkl`. Run after Phase 1.

### Phase 3 — Model Training + Serving

```bash
ipl_venv/bin/python run_phase3.py           # train only
ipl_venv/bin/python run_phase3.py --serve   # train + start API at localhost:8000
ipl_venv/bin/python run_phase3.py --serve --port=8080  # custom port
```

Trains XGBoost classifier with Platt scaling calibration. Saves versioned model artifact to `src/models/artifacts/`. Serves via FastAPI with `/predict`, `/model-info`, and `/drift-check` endpoints.

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Phase 4 — LangGraph Agent System

```bash
ipl_venv/bin/python run_phase4.py
```

Runs the full 6-agent LangGraph pipeline for a pre-match prediction. Edit the match details at the bottom of `run_phase4.py` to predict any match. The narrative agent uses `llama-3.3-70b-versatile` on Groq — falls back to a rule-based brief if `GROQ_API_KEY` is not set.

---

## Agent Pipeline

```
START
  → Validation Gate       check required fields, abort on critical errors
  → Ingestion Agent       fetch weather (Open-Meteo), schedule, player profiles
  → Extraneous Agent      compute dew risk, fatigue index, PDI, heat stress
  → Feature Agent         assemble 44-feature vector aligned to schema
  → Prediction Agent      call XGBoost model tool → win probability
  → Narrative Agent       Groq llama-3.3-70b-versatile → tactical brief
END
```

Human override context (toss result, fitness updates, captain intent) can be injected into the initial state before the graph runs.

---

## Data Sources

| Source                 | Stored locally            | Fetched via internet            |
| ---------------------- | ------------------------- | ------------------------------- |
| Cricsheet ball-by-ball | ✅ After first download   | First run + end of season       |
| Open-Meteo weather     | ✅ Cached as parquet      | First run + end of season       |
| IPL schedule           | ✅ Derived from Cricsheet | Never                           |
| Player profiles        | ✅ Derived from Cricsheet | Never                           |
| Match-day forecast     | ❌ Live on match day      | Ingestion Agent at inference    |

---

## When to Re-run

| Script                 | When                                              |
| ---------------------- | ------------------------------------------------- |
| `cricsheet_ingest.py`  | End of each IPL season                            |
| `weather_ingest.py`    | After Cricsheet update (fetches new matches only) |
| `schedule_ingest.py`   | After Cricsheet update                            |
| `player_ingest.py`     | After Cricsheet update                            |
| `run_phase2.py`        | After any Phase 1 update or schema change         |
| `run_phase3.py`        | After Phase 2 or if hyperparameters are tuned     |

---

## Feature Schema

All 44 features are defined in `config/feature_schema.yaml` with source, dtype, fillna fallback, normalize flag, and plain-English description. This is the contract between training and inference — any change here requires re-running Phase 2 and retraining the model.

Feature groups: `match_context`, `venue`, `environmental`, `fatigue`, `team_form`, `player_strength`, `pitch`

---

## Expected Model Performance

| Dataset          | Expected AUC (real data) |
| ---------------- | ------------------------ |
| Train (2008–22)  | 0.68–0.75                |
| Val (2023)       | 0.62–0.70                |
| Test (2024+)     | 0.60–0.68                |

Pre-match T20 prediction is inherently uncertain — the ceiling for this type of model is ~0.70 AUC. Mock data will show inflated metrics (AUC ~1.0 on 278 rows). Run Phase 1 with real Cricsheet data for meaningful evaluation.

---

## Roadmap

- [x] Phase 1 — Data ingestion
- [x] Phase 2 — Feature engineering + ETL
- [x] Phase 3 — Model training + serving
- [x] Phase 4 — LangGraph multi-agent system
- [ ] Streamlit dashboard (pit-wall UI)
- [ ] Output persistence + post-match feedback loop
- [ ] Retraining trigger (end-of-season automation)