# =============================================================================
# main.py — FastAPI application
# Run from D:\MajorProject with:
#   uvicorn backend.main:app --reload
# =============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ml.features import engineer_features
from backend.ml.validator import validate_prediction_input, reorder_features_for_model
from backend.ml.shap_utils import get_shap_context
from backend.agents.explainer import AgenticRAG, generate_clinical_explanation

load_dotenv()

# =============================================================================
# App setup
# =============================================================================

app = FastAPI(
    title="Menstrual Cycle Prediction API",
    description="Explainable AI clinical decision support for menstrual health",
    version="1.0.0"
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Load models on startup (once, not on every request)
# =============================================================================

BASE_DIR    = Path(__file__).resolve().parent.parent  # D:\MajorProject
MODELS_DIR  = BASE_DIR / "models"

period_model    = None
ovulation_model = None
trained_features = None

@app.on_event("startup")
def load_models():
    global period_model, ovulation_model, trained_features

    period_path    = MODELS_DIR / "period_model.pkl"
    ovulation_path = MODELS_DIR / "ovulation_model.pkl"
    features_path  = MODELS_DIR / "trained_features.json"

    if not period_path.exists():
        print(f"[WARNING] period_model.pkl not found at {period_path}")
        print("[WARNING] Run notebooks/train.ipynb first to train and save models.")
        return

    period_model    = joblib.load(period_path)
    ovulation_model = joblib.load(ovulation_path) if ovulation_path.exists() else None

    with open(features_path) as f:
        trained_features = json.load(f)

    print(f"[STARTUP] Models loaded successfully.")
    print(f"  Period model:    {type(period_model).__name__}")
    print(f"  Ovulation model: {type(ovulation_model).__name__ if ovulation_model else 'Not found'}")
    print(f"  Features:        {len(trained_features)}")


# =============================================================================
# Request / Response schemas
# =============================================================================

class DailyInput(BaseModel):
    """
    One day of hormone + symptom data for a single participant.
    All fields match your cleaned CSV columns.
    """
    # Required hormone fields
    lh_imputed:        float
    estrogen_imputed:  float
    pdg_imputed:       float

    # Symptom fields (default 0 if not provided)
    cramps_imputed:        float = 0.0
    sorebreasts_imputed:   float = 0.0
    bloating_imputed:      float = 0.0
    moodswing_imputed:     float = 0.0
    fatigue_imputed:       float = 0.0
    headaches_imputed:     float = 0.0
    foodcravings_imputed:  float = 0.0
    indigestion_imputed:   float = 0.0
    exerciselevel_imputed: float = 0.0
    stress_imputed:        float = 0.0
    sleepissue_imputed:    float = 0.0
    appetite_imputed:      float = 0.0

    # Flags
    high_estrogen_flag:   bool = False
    estrogen_capped_flag: bool = False
    is_weekend:           bool = False

    # Required for feature engineering
    id:           int   = 1
    day_in_study: int   = 1


class PredictionRequest(BaseModel):
    """
    Single prediction request — one participant's recent history.
    Pass at least 7 days of history for rolling features to work properly.
    """
    days:             list[DailyInput]
    include_rag:      bool = True    # set False for faster response without RAG
    include_shap:     bool = True


class RiskLevel(BaseModel):
    level:       str    # LOW / MODERATE / HIGH
    color:       str    # green / yellow / red
    description: str


class PredictionResponse(BaseModel):
    # Period prediction
    period_probability:  float
    period_prediction:   bool
    period_risk:         RiskLevel

    # Ovulation prediction
    ovulation_probability: float
    ovulation_prediction:  bool
    ovulation_risk:        RiskLevel

    # Explainability
    top_features:        list[dict]   # [{feature, importance}]
    clinical_explanation: str
    rag_explanation:     str | None   # None if include_rag=False

    # Metadata
    model_used:  str
    features_used: int


# =============================================================================
# Helper: build risk level object
# =============================================================================

def get_risk(proba: float) -> RiskLevel:
    if proba >= 0.7:
        return RiskLevel(
            level="HIGH",
            color="red",
            description="High probability — likely within 1-2 days"
        )
    elif proba >= 0.4:
        return RiskLevel(
            level="MODERATE",
            color="yellow",
            description="Moderate probability — possible within 3-5 days"
        )
    else:
        return RiskLevel(
            level="LOW",
            color="green",
            description="Low probability — unlikely in the near term"
        )


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {
        "message": "Menstrual Cycle Prediction API",
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "period_model":   period_model is not None,
        "ovulation_model": ovulation_model is not None,
        "features_loaded": trained_features is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.

    Send hormone + symptom data for recent days.
    Returns period + ovulation predictions with SHAP + clinical explanation.
    """
    if period_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run notebooks/train.ipynb first."
        )

    # ── Build DataFrame from request ──────────────────────────────────────────
    rows = [day.dict() for day in request.days]
    df   = pd.DataFrame(rows)

    # ── Feature engineering ───────────────────────────────────────────────────
    try:
        df_eng = engineer_features(df)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature engineering failed: {str(e)}"
        )

    # ── Align to trained features ─────────────────────────────────────────────
    for col in trained_features:
        if col not in df_eng.columns:
            df_eng[col] = 0

    # Use the last row (most recent day) for prediction
    last_row = df_eng.tail(1).reindex(columns=trained_features, fill_value=0)

    # ── Period prediction ─────────────────────────────────────────────────────
    try:
        period_proba = float(period_model.predict_proba(last_row)[0, 1])
        period_pred  = bool(period_model.predict(last_row)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Period prediction failed: {str(e)}"
        )

    # ── Ovulation prediction ──────────────────────────────────────────────────
    ovulation_proba = 0.0
    ovulation_pred  = False
    if ovulation_model is not None:
        try:
            ovulation_proba = float(ovulation_model.predict_proba(last_row)[0, 1])
            ovulation_pred  = bool(ovulation_model.predict(last_row)[0])
        except Exception as e:
            print(f"[WARNING] Ovulation prediction failed: {e}")

    # ── SHAP explainability ───────────────────────────────────────────────────
    top_features_list = []
    shap_ctx          = {}

    if request.include_shap:
        try:
            shap_ctx = get_shap_context(
                period_model, last_row,
                trained_features, sample_idx=0
            )
            top_features_list = shap_ctx.get("top_positive_features", [])
        except Exception as e:
            print(f"[WARNING] SHAP failed: {e}")

    # ── Clinical explanation (fast, no LLM) ───────────────────────────────────
    fi_df = pd.DataFrame(top_features_list).rename(
        columns={"shap_value": "importance"}
    ) if top_features_list else None

    clinical_explanation = generate_clinical_explanation(
        period_proba, fi_df, "period_start"
    )

    # ── RAG explanation (LLM-powered) ────────────────────────────────────────
    rag_explanation = None
    if request.include_rag:
        try:
            rag = AgenticRAG()
            rag_context = {
                "target":        "period_start",
                "prediction":    period_proba,
                "top_features":  [f["feature"] for f in top_features_list[:5]],
                "model":         period_model,
                "feature_names": trained_features,
            }
            rag_explanation = rag.retrieve_and_explain(
                rag_context, X_test=last_row
            )
        except Exception as e:
            print(f"[WARNING] RAG failed: {e}")
            rag_explanation = f"RAG unavailable: {str(e)}"

    return PredictionResponse(
        period_probability   = round(period_proba, 4),
        period_prediction    = period_pred,
        period_risk          = get_risk(period_proba),

        ovulation_probability = round(ovulation_proba, 4),
        ovulation_prediction  = ovulation_pred,
        ovulation_risk        = get_risk(ovulation_proba),

        top_features         = top_features_list,
        clinical_explanation = clinical_explanation,
        rag_explanation      = rag_explanation,

        model_used    = type(period_model).__name__,
        features_used = len(trained_features),
    )


@app.get("/features")
def get_features():
    """Returns the list of features the model expects."""
    if trained_features is None:
        raise HTTPException(status_code=503, detail="Features not loaded.")
    return {"features": trained_features, "count": len(trained_features)}