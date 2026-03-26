# Project Summary

## What This Project Is

This repository contains **CycleAI**, a menstrual health prediction system with:

- A **FastAPI backend** for period and ovulation prediction
- A **React + Vite frontend** for entering daily hormone and symptom data
- A **machine learning pipeline** for training, evaluation, and SHAP explainability
- An optional **Agentic RAG explanation layer** using ChromaDB and an OpenAI-compatible API

The app predicts:

- `period_start`
- `lh_surge`

It returns:

- prediction probabilities
- binary predictions
- risk levels
- SHAP feature drivers
- a rule-based clinical explanation
- an optional RAG-based evidence explanation

## Main Runtime Flow

1. The frontend sends daily hormone and symptom inputs to the backend `/predict` endpoint.
2. The backend engineers temporal and physiological features.
3. The backend aligns those features to the trained feature list stored in `models/trained_features.json`.
4. The backend loads and uses serialized models from `models/`.
5. The backend returns prediction results and explanations.

## Most Important Files

### Backend

- `backend/main.py`
  Main FastAPI application. Defines `/`, `/health`, `/features`, and `/predict`.

- `backend/ml/features.py`
  Core feature engineering logic. Creates rolling means, deltas, lag features, LH momentum, symptom burden, and `e2_pdg_ratio`.

- `backend/ml/config.py`
  Defines targets and base input features.

- `backend/ml/models.py`
  Defines the training models:
  - Logistic Regression
  - Random Forest
  - LightGBM

- `backend/ml/pipeline.py`
  Main training and evaluation pipeline.

- `backend/ml/evaluate.py`
  Evaluation metrics and time-series cross-validation.

- `backend/ml/validator.py`
  Validates feature names and ordering before prediction.

- `backend/agents/explainer.py`
  Agentic RAG layer using tool-calling with an OpenAI-compatible client.

- `backend/rag/knowledge_base.py`
  Hardcoded clinical knowledge loaded into ChromaDB.

### Frontend

- `frontend/src/App.jsx`
  Main user interface. Collects input, calls the backend, and renders results.

- `frontend/package.json`
  Frontend dependencies and scripts.

### Models and Outputs

- `models/period_model.pkl`
  Required for period prediction.

- `models/ovulation_model.pkl`
  Required for ovulation prediction.

- `models/trained_features.json`
  Required feature order for inference.

- `pipeline_outputs/results_period_start.json`
  Saved training/evaluation results for period prediction.

- `pipeline_outputs/results_lh_surge.json`
  Saved training/evaluation results for ovulation prediction.

### Deployment and Setup

- `.env`
  Environment configuration for backend and frontend.

- `requirements.txt`
  Python dependencies.

- `Dockerfile`
  Backend container definition.

- `docker-compose.yml`
  Multi-service setup for backend and frontend.

### Testing

- `tests/test_api.py`
  Manual test script for health, features, predictions, response shape, and timing.

## Critical Project Folders

- `backend/`
  Core application logic

- `frontend/`
  User interface

- `models/`
  Serialized trained models required at runtime

- `pipeline_outputs/`
  Saved evaluation metrics and SHAP plots

## Supporting / Research Folders

These appear to be useful for experimentation and research, but are not the main runtime path:

- `EDA/`
- `notebooks/`
- `prediction_pipeline/`
- `mcphases/`
- `menstrual_ai_agentic_rag/`
- `mlops/`

## Inputs Expected by the API

The prediction endpoint expects daily records containing at least:

- `lh_imputed`
- `estrogen_imputed`
- `pdg_imputed`
- symptom scores such as `cramps_imputed`, `bloating_imputed`, `fatigue_imputed`, etc.
- flags like `high_estrogen_flag`, `estrogen_capped_flag`, `is_weekend`
- identifiers like `id` and `day_in_study`

The backend comments indicate that **7 or more days of recent history** are preferred so rolling features behave properly.

## Model Performance Snapshot

Based on saved outputs in `pipeline_outputs/`:

- Best period model: `LightGBM`
- Best ovulation model: `LightGBM`

Approximate saved metrics:

- Period prediction AUC-ROC: `0.9991`
- Ovulation prediction AUC-ROC: `0.9610`

Important top drivers seen in saved outputs:

- Period prediction: cramps, PDG, PDG delta, PDG rolling features
- Ovulation prediction: LH, LH rolling features, LH delta, estrogen-related trends

## Key Dependencies

Backend:

- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `imbalanced-learn`
- `shap`
- `chromadb`
- `sentence-transformers`
- `openai`
- `python-dotenv`
- `joblib`

Frontend:

- `react`
- `react-dom`
- `vite`

## Current Risks and Important Notes

### 1. Secret Exposure

The `.env` file currently contains a live API key in plain text.

Recommended action:

- rotate the key immediately
- replace it with a fresh key
- avoid committing secrets to source control

### 2. Frontend / Backend Data Mismatch

The frontend currently submits a single day by default, while the backend documentation says predictions work best with at least 7 days of history.

This may reduce prediction quality in normal frontend usage.

### 3. Documentation Gap

- `README.md` is empty
- `frontend/README.md` is still the default Vite template

This means the repo currently lacks clear setup and usage documentation.

### 4. Docker Compose May Need Fixing

The `frontend` service block in `docker-compose.yml` appears to have indentation issues and may not run correctly without adjustment.

### 5. RAG Depends on External API Access

The Agentic RAG explanation path requires:

- a valid API key
- network access
- ChromaDB embedding setup

If any of those fail, prediction still works, but the RAG explanation may not.

## What Is Actually Essential to Keep

If you want the minimum critical set for understanding or preserving this project, focus on:

- `backend/`
- `frontend/`
- `models/`
- `pipeline_outputs/`
- `.env` after fixing secrets
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `tests/test_api.py`

## Recommended Next Improvements

- Add a real root `README.md`
- Remove and rotate exposed secrets from `.env`
- Fix `docker-compose.yml`
- Update the frontend to support multi-day history input
- Add clear startup instructions for local and Docker usage
- Add automated tests beyond the current manual API test script

