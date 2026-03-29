# Project Summary

Last updated: 2026-03-29

## 1. Project Overview

This repository contains a menstrual health prediction platform with two primary runtime services:

- FastAPI backend for inference and explainability
- React + Vite frontend for user input and result display

The core predictive outputs are:

- period_start risk (period probability and binary prediction)
- lh_surge risk (ovulation probability and binary prediction)

The backend also provides:

- SHAP-based top feature contributions
- rule-based clinical explanation
- optional agentic RAG explanation path (LLM + ChromaDB retrieval)

## 2. Current Architecture

### Backend (production API)

- Location: backend/
- Entry point: backend/main.py
- Framework: FastAPI
- Main endpoints:
  - GET /
  - GET /health
  - GET /features
  - POST /predict

Startup behavior:

- Loads period model from models/period_model.pkl
- Loads ovulation model from models/ovulation_model.pkl when available
- Loads expected feature order from models/trained_features.json

Inference behavior:

1. Accepts a list of daily records in the request body.
2. Builds engineered temporal features.
3. Aligns columns to trained_features.json.
4. Uses the most recent day (last row) for prediction.
5. Returns probabilities, booleans, risk levels, and explanations.

### Frontend (web app)

- Location: frontend/
- Stack: React 18 + Vite 5
- Scripts:
  - npm run dev
  - npm run build
  - npm run preview

The frontend README is still the default Vite template and should be replaced with project-specific documentation.

## 3. Machine Learning and Explainability

### Feature engineering

- Location: backend/ml/features.py
- Includes rolling/lag/trend style transformations and domain-derived features.

### Modeling and evaluation

- Core ML modules:
  - backend/ml/models.py
  - backend/ml/pipeline.py
  - backend/ml/evaluate.py
  - backend/ml/validator.py

### Explainability

- SHAP utility: backend/ml/shap_utils.py
- Inference response includes top feature impacts when include_shap=true.

### Saved artifacts

- models/trained_features.json
- pipeline_outputs/results_period_start.json
- pipeline_outputs/results_lh_surge.json

## 4. Agentic RAG Layer

- Agent entry: backend/agents/explainer.py
- Knowledge base: backend/rag/knowledge_base.py

Behavior at inference time:

- RAG is optional (include_rag flag in request)
- Failure in RAG path does not block prediction; API still returns prediction output

## 5. Data and Research Assets

Main data/research directories present in repository:

- mcphases/
- data/raw/
- EDA/
- notebooks/
- prediction_pipeline/

There is also a separate experimental app stack under menstrual_ai_agentic_rag/, which appears to be related but not the primary FastAPI runtime path.

## 6. Testing Status

- API smoke/regression script: tests/test_api.py
- This file validates health, features, prediction response shape, threshold checks, and latency snapshots.

Current test style is script-based rather than pytest unit/integration structure.

## 7. Containerization and Deployment

### Docker

- Root Dockerfile builds and runs the backend with uvicorn backend.main:app.
- backend/Dockerfile currently exists but is empty.

### Docker Compose

- docker-compose.yml defines two services:
  - backend (port 8000)
  - frontend (port 80)
- Backend mounts models/ and pipeline_outputs/ into /app.
- Frontend depends on backend healthcheck.

## 8. Dependency Snapshot

### Python

From requirements.txt:

- API: fastapi, uvicorn[standard], python-multipart
- ML: lightgbm, scikit-learn, imbalanced-learn, pandas, numpy, shap, joblib, matplotlib
- RAG/LLM: chromadb, sentence-transformers, openai, huggingface_hub
- Config: python-dotenv

### Frontend

From frontend/package.json:

- react, react-dom
- @vitejs/plugin-react, vite

## 9. Known Gaps and Cleanup Opportunities

1. Root README.md is currently empty.
2. frontend/README.md is still the default template.
3. backend/Dockerfile is empty and may cause confusion if someone expects backend-specific container config there.
4. Repository has duplicate-looking training scripts at root (train_and_sav.py and train_and_save.py) that should be clarified or consolidated.
5. Automated tests can be improved by adding pytest-based suites and CI execution.

## 10. Recommended Next Steps

1. Replace root README.md with complete setup, API, and run instructions.
2. Replace frontend/README.md with app-specific frontend usage notes.
3. Decide whether backend/Dockerfile is needed; remove or implement it.
4. Consolidate training script entry points and document the canonical one.
5. Add pytest tests and CI pipeline to enforce API contract and model-loading checks.

## 11. Deep-Dive Technical Appendix (Claude-Ready)

This section is intentionally detailed so another assistant/tool (for example Claude) can answer project questions with minimal extra context.

### Repository identity

- Local workspace root: `D:\MajorProject`
- Git repository remote:
  - `origin (fetch): https://github.com/iiqra-khan/cycleai.git`
  - `origin (push):  https://github.com/iiqra-khan/cycleai.git`
- Practical repository name to use in prompts: **cycleai**

### Primary product objective

Predict near-term menstrual cycle events from hormone + symptom signals and provide explainability at three levels:

1. Model probabilities + binary predictions
2. Feature-level explanation via SHAP
3. Optional clinical narrative via agentic RAG over a local clinical evidence base

### Runtime services and entry points

1. Backend API (FastAPI)
   - Module entry: `backend.main:app`
   - Dev command from repo root:
     - `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
   - If running from `backend/` folder, module path becomes `main:app`

2. Frontend app (React + Vite)
   - Location: `frontend/`
   - Dev command:
     - `npm run dev`
   - API base in code:
     - `VITE_API_URL` env var, fallback `http://127.0.0.1:8000`

3. Containerized runtime
   - Root `Dockerfile` starts backend with:
     - `uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`
   - `docker-compose.yml` orchestrates backend + frontend

### Backend API contract (authoritative)

Base URL (local): `http://127.0.0.1:8000`

Available endpoints:

1. `GET /`
   - Returns service message, running status, docs path

2. `GET /health`
   - Returns booleans for model/feature load state

3. `GET /features`
   - Returns list and count of `trained_features`

4. `POST /predict`
   - Main inference endpoint
   - Request schema: `PredictionRequest`
   - Response schema: `PredictionResponse`

#### `/predict` request input expected by API

Top-level body:

- `days`: array of `DailyInput` (recommended >= 7 days for rolling features)
- `include_rag`: boolean, default `true`
- `include_shap`: boolean, default `true`

`DailyInput` expected fields:

- Required hormone numeric fields:
  - `lh_imputed`
  - `estrogen_imputed`
  - `pdg_imputed`

- Symptom numeric fields (default 0.0 if omitted):
  - `cramps_imputed`
  - `sorebreasts_imputed`
  - `bloating_imputed`
  - `moodswing_imputed`
  - `fatigue_imputed`
  - `headaches_imputed`
  - `foodcravings_imputed`
  - `indigestion_imputed`
  - `exerciselevel_imputed`
  - `stress_imputed`
  - `sleepissue_imputed`
  - `appetite_imputed`

- Boolean flags:
  - `high_estrogen_flag`
  - `estrogen_capped_flag`
  - `is_weekend`

- Temporal identifiers:
  - `id` (participant id)
  - `day_in_study` (day index)

Minimal single-day example payload:

```json
{
  "days": [
    {
      "lh_imputed": 4.6,
      "estrogen_imputed": 138.4,
      "pdg_imputed": 3.6,
      "cramps_imputed": 1,
      "sorebreasts_imputed": 1,
      "bloating_imputed": 1,
      "moodswing_imputed": 1,
      "fatigue_imputed": 1,
      "headaches_imputed": 1,
      "foodcravings_imputed": 1,
      "indigestion_imputed": 1,
      "exerciselevel_imputed": 3,
      "stress_imputed": 3,
      "sleepissue_imputed": 1,
      "appetite_imputed": 3,
      "high_estrogen_flag": false,
      "estrogen_capped_flag": false,
      "is_weekend": false,
      "id": 1,
      "day_in_study": 1
    }
  ],
  "include_rag": true,
  "include_shap": true
}
```

#### `/predict` response output shape

- `period_probability` (float)
- `period_prediction` (bool)
- `period_risk` `{ level, color, description }`
- `ovulation_probability` (float)
- `ovulation_prediction` (bool)
- `ovulation_risk` `{ level, color, description }`
- `top_features` (list of feature importance objects from SHAP path)
- `clinical_explanation` (string)
- `rag_explanation` (string or null)
- `model_used` (string class name of period model)
- `features_used` (int)

Important behavior notes:

1. Backend engineers features, aligns to `trained_features.json`, and predicts on the most recent row only.
2. If ovulation model is absent, ovulation values default to low/false-like output.
3. If RAG fails, prediction still returns; `rag_explanation` includes fallback text.
4. Agent/tool step logs currently print to backend console, not to API response fields.

### Models used

#### Inference-time loaded artifacts (startup)

From `models/`:

- `period_model.pkl` (required for service to predict period)
- `ovulation_model.pkl` (optional at runtime)
- `trained_features.json` (authoritative feature order for inference)

#### Training candidates (defined in code)

In `backend/ml/models.py`, the training pipeline evaluates:

1. Logistic Regression (with scaler, class_weight balanced)
2. Random Forest (balanced_subsample)
3. LightGBM (with computed `scale_pos_weight`)

Best model is selected by AUC-ROC per target in pipeline execution.

### Explainability and RAG internals

1. SHAP
   - Utility: `backend/ml/shap_utils.py`
   - Produces top positive/negative drivers and plots in `pipeline_outputs/`

2. Rule-based explanation
   - Function: `generate_clinical_explanation(...)`
   - Fast local text explanation without LLM calls

3. Agentic RAG
   - Entry: `backend/agents/explainer.py` (`AgenticRAG`)
   - Retrieval source: ChromaDB vector store built from local docs via `backend/rag/knowledge_base.py`
   - Tool-calling loop uses two tools:
     - `retrieve_evidence`
     - `finalize_explanation`
   - LLM client uses `openai` SDK with OpenRouter-compatible base URL

### Output files and artifact folders

1. Model artifacts
   - `models/period_model.pkl`
   - `models/ovulation_model.pkl`
   - `models/trained_features.json`

2. Pipeline result JSON
   - `pipeline_outputs/results_period_start.json`
   - `pipeline_outputs/results_lh_surge.json`

3. Explainability visual artifacts
   - `pipeline_outputs/shap_bar_*.png`
   - `pipeline_outputs/shap_beeswarm_*.png`
   - `pipeline_outputs/shap_waterfall_*.png`

4. Additional mirrored outputs also exist in:
   - root-level `results_period_start.json`
   - root-level `results_lh_surge.json`
   - `prediction_pipeline/pipeline_outputs/`

### High-signal folder structure map

```text
MajorProject/
  backend/
    main.py                 # FastAPI app + API schemas/routes
    agents/explainer.py     # Agentic RAG + tool-calling orchestration
    ml/
      features.py           # feature engineering
      models.py             # model definitions
      pipeline.py           # training/eval orchestration
      evaluate.py           # metrics and CV helpers
      shap_utils.py         # SHAP utilities
      validator.py          # prediction input/model feature alignment checks
    rag/knowledge_base.py   # retrieval index/vector-store build

  frontend/
    src/App.jsx             # main UI, sends /predict request
    src/main.jsx            # React bootstrap

  models/                   # serialized models + trained feature list
  pipeline_outputs/         # metrics json + SHAP image outputs
  tests/test_api.py         # API smoke/contract check script
  docker-compose.yml        # full stack local orchestration
  Dockerfile                # backend container image
```

### Environment and secrets assumptions

Backend reads environment through `python-dotenv` and expects at least:

- `OPENAI_API_KEY` for RAG LLM calls
- Optional `OPENAI_BASE_URL` (defaults to OpenRouter endpoint in code)
- Optional `PORT` in container runtime

### Known operational gotchas

1. Running `python -m uvicorn main:app ...` from repo root fails because `main.py` is under `backend/`.
2. Correct module path from repo root is `backend.main:app`.
3. RAG initialization can be slow on first request due to embedding/vector-store setup.
4. Frontend currently receives only final response payload, not streaming agent step logs.

### Quick prompt template for Claude (copy-ready)

Use this when asking Claude for project help:

```text
You are helping with the cycleai repository (Menstrual Cycle Prediction API + React UI).

Backend:
- FastAPI entry: backend.main:app
- Main endpoint: POST /predict
- Request keys: days (array of DailyInput), include_rag, include_shap
- DailyInput required hormones: lh_imputed, estrogen_imputed, pdg_imputed
- Optional/default symptoms: cramps_imputed, sorebreasts_imputed, bloating_imputed, moodswing_imputed,
  fatigue_imputed, headaches_imputed, foodcravings_imputed, indigestion_imputed,
  exerciselevel_imputed, stress_imputed, sleepissue_imputed, appetite_imputed
- Flags: high_estrogen_flag, estrogen_capped_flag, is_weekend
- IDs: id, day_in_study
- Response includes period/ovulation probabilities + risk objects + top_features + clinical_explanation + rag_explanation

Models:
- Runtime artifacts in models/: period_model.pkl, ovulation_model.pkl, trained_features.json
- Training candidates: Logistic Regression, Random Forest, LightGBM

Explainability:
- SHAP via backend/ml/shap_utils.py
- Agentic RAG via backend/agents/explainer.py and ChromaDB retrieval

Please answer my next question with concrete file-level guidance and command-level steps.
```
