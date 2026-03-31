# =============================================================================
# main.py — FastAPI application
# Run from D:\MajorProject with:
#   uvicorn backend.main:app --reload
# =============================================================================

import os
import json
import time
import uuid
import asyncio
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Callable
import base64
import hashlib
import hmac
from fastapi import Depends, FastAPI, HTTPException, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
import jwt
from jwt.exceptions import InvalidTokenError
from dotenv import load_dotenv

from backend.ml.features import engineer_features
from backend.ml.validator import validate_prediction_input, reorder_features_for_model
from backend.ml.shap_utils import get_shap_context
from backend.ml.config import LEAKY_FEATURES
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
USERS_DB_PATH = BASE_DIR / "backend" / "db" / "users.json"

period_model    = None
ovulation_model = None
trained_features = None
prediction_jobs: dict[str, dict[str, Any]] = {}
feature_contract_ok = True
feature_contract_errors: list[str] = []

RUNTIME_FORBIDDEN_FEATURES = {
    "phase",
    "is_bleeding",
    "is_bleeding_lag1",
    "is_bleeding_lag2",
    "is_luteal",
    "is_follicular",
    "is_ovulatory",
    "is_menstrual",
}

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
users_db: dict[str, dict[str, Any]] = {}
PBKDF2_ITERATIONS = 390_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_old_jobs(max_age_seconds: int = 900):
    now = time.time()
    stale_ids = []
    for job_id, job in prediction_jobs.items():
        if now - job.get("created_at_ts", now) > max_age_seconds:
            stale_ids.append(job_id)

    for job_id in stale_ids:
        prediction_jobs.pop(job_id, None)


def _make_event(event_type: str, message: str, payload: dict | None = None) -> dict:
    return {
        "type": event_type,
        "message": message,
        "timestamp": _utc_now_iso(),
        "payload": payload or {},
    }


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    key_b64 = base64.b64encode(key).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${key_b64}"


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        scheme, iter_s, salt_b64, key_b64 = hashed_password.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False

        iterations = int(iter_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected_key = base64.b64decode(key_b64.encode("ascii"))
        actual_key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual_key, expected_key)
    except Exception:
        return False


def _create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        ) from exc


def _load_users_db_from_disk() -> dict[str, dict[str, Any]]:
    if not USERS_DB_PATH.exists():
        return {}

    try:
        with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
        return {}
    except Exception as exc:
        print(f"[WARNING] Failed to load users db: {exc}")
        return {}


def _save_users_db_to_disk() -> None:
    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users_db, f, indent=2)


def _get_user_or_401(username: str) -> dict[str, Any]:
    user = users_db.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, Any]:
    payload = _decode_access_token(token)
    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )
    return _get_user_or_401(username)


def _get_current_user_from_token_string(token: str) -> dict[str, Any]:
    payload = _decode_access_token(token)
    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )
    return _get_user_or_401(username)

@app.on_event("startup")
def load_models():
    global period_model, ovulation_model, trained_features
    global feature_contract_ok, feature_contract_errors
    global users_db

    users_db = _load_users_db_from_disk()

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

    leaked = sorted(set(trained_features).intersection(RUNTIME_FORBIDDEN_FEATURES))
    if leaked:
        feature_contract_ok = False
        feature_contract_errors = [
            "Model artifacts include deployment-forbidden features: "
            + ", ".join(leaked),
            "Retrain using draft5 flow and replace models/period_model.pkl, models/ovulation_model.pkl, and models/trained_features.json",
        ]
        print("[ERROR] Feature contract check failed.")
        for err in feature_contract_errors:
            print(f"  - {err}")
    else:
        feature_contract_ok = True
        feature_contract_errors = []

    print(f"[STARTUP] Models loaded successfully.")
    print(f"  Period model:    {type(period_model).__name__}")
    print(f"  Ovulation model: {type(ovulation_model).__name__ if ovulation_model else 'Not found'}")
    print(f"  Features:        {len(trained_features)}")

    bootstrap_username = os.getenv("AUTH_BOOTSTRAP_USERNAME")
    bootstrap_password = os.getenv("AUTH_BOOTSTRAP_PASSWORD")
    if bootstrap_username and bootstrap_password and bootstrap_username not in users_db:
        users_db[bootstrap_username] = {
            "username": bootstrap_username,
            "password_hash": _hash_password(bootstrap_password),
            "created_at": _utc_now_iso(),
        }
        _save_users_db_to_disk()
        print(f"[STARTUP] Bootstrapped auth user: {bootstrap_username}")


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


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    username: str
    created_at: str


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
        "feature_contract_ok": feature_contract_ok,
        "feature_contract_errors": feature_contract_errors,
    }


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: AuthRequest):
    username = payload.username.strip().lower()
    if username in users_db:
        raise HTTPException(status_code=409, detail="Username already exists")

    users_db[username] = {
        "username": username,
        "password_hash": _hash_password(payload.password),
        "created_at": _utc_now_iso(),
    }
    _save_users_db_to_disk()

    token = _create_access_token(username)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: AuthRequest):
    username = payload.username.strip().lower()
    user = users_db.get(username)
    if not user or not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = _create_access_token(username)
    return TokenResponse(access_token=token)


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: dict[str, Any] = Depends(get_current_user)):
    return UserResponse(username=current_user["username"], created_at=current_user["created_at"])


def _run_prediction_core(
    request: PredictionRequest,
    emit: Callable[[dict], None] | None = None,
) -> PredictionResponse:
    if period_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run notebooks/train.ipynb first."
        )

    if trained_features is None:
        raise HTTPException(
            status_code=503,
            detail="Feature list not loaded. Check models/trained_features.json"
        )

    if not feature_contract_ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model feature contract is inconsistent with runtime API fields. "
                + " | ".join(feature_contract_errors)
            ),
        )

    feature_names = trained_features

    if emit:
        emit(_make_event("request_received", "Prediction request received."))

    # ── Build DataFrame from request ──────────────────────────────────────────
    rows = [day.dict() for day in request.days]
    df   = pd.DataFrame(rows)

    if emit:
        emit(_make_event(
            "feature_engineering_started",
            "Feature engineering started.",
            {"input_rows": len(df)}
        ))

    # ── Feature engineering ───────────────────────────────────────────────────
    try:
        df_eng = engineer_features(df)
        if emit:
            emit(_make_event(
                "feature_engineering_done",
                "Feature engineering completed.",
                {"engineered_rows": len(df_eng), "engineered_cols": len(df_eng.columns)}
            ))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature engineering failed: {str(e)}"
        )

    # ── Align to trained features ─────────────────────────────────────────────
    missing_cols = []
    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0
            missing_cols.append(col)

    if emit:
        emit(_make_event(
            "feature_alignment_done",
            "Aligned runtime features to trained feature order.",
            {"missing_filled": len(missing_cols), "features_used": len(feature_names)}
        ))

    # Use the last row (most recent day) for prediction
    last_row = df_eng.tail(1).reindex(columns=feature_names, fill_value=0)

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

    if emit:
        emit(_make_event(
            "model_inference_done",
            "Model inference complete.",
            {
                "period_probability": round(period_proba, 4),
                "ovulation_probability": round(ovulation_proba, 4),
            }
        ))

    # ── SHAP explainability ───────────────────────────────────────────────────
    top_features_list = []
    top_negative_list = []

    if request.include_shap:
        try:
            shap_ctx = get_shap_context(
                period_model, last_row,
                feature_names, sample_idx=0
            )
            top_features_list = shap_ctx.get("top_positive_features", [])
            top_negative_list = shap_ctx.get("top_negative_features", [])
            if emit:
                emit(_make_event(
                    "shap_done",
                    "SHAP explainability generated.",
                    {
                        "top_positive_count": len(top_features_list),
                        "top_negative_count": len(top_negative_list),
                    }
                ))
        except Exception as e:
            print(f"[WARNING] SHAP failed: {e}")
            if emit:
                emit(_make_event(
                    "shap_warning",
                    "SHAP failed; continuing without SHAP details.",
                    {"error": str(e)}
                ))

    # ── Clinical explanation (fast, no LLM) ───────────────────────────────────
    fi_df = pd.DataFrame(top_features_list).rename(
        columns={"shap_value": "importance"}
    ) if top_features_list else pd.DataFrame(columns=["feature", "importance"])
    neg_df = pd.DataFrame(top_negative_list) if top_negative_list else pd.DataFrame(columns=["feature", "shap_value"])

    clinical_explanation = generate_clinical_explanation(
        period_proba, fi_df, "period_start", negative_features=neg_df
    )

    # ── RAG explanation (LLM-powered) ────────────────────────────────────────
    rag_explanation = None
    if request.include_rag:
        if emit:
            emit(_make_event("rag_started", "Agentic RAG started."))
        try:
            rag = AgenticRAG(trace_callback=emit)
            rag_context = {
                "target":        "period_start",
                "prediction":    period_proba,
                "top_features":  [f["feature"] for f in top_features_list[:5]],
                "model":         period_model,
                "feature_names": feature_names,
            }
            rag_explanation = rag.retrieve_and_explain(
                rag_context, X_test=last_row
            )
            if emit:
                emit(_make_event("rag_done", "Agentic RAG explanation generated."))
        except Exception as e:
            print(f"[WARNING] RAG failed: {e}")
            rag_explanation = f"RAG unavailable: {str(e)}"
            if emit:
                emit(_make_event(
                    "rag_warning",
                    "RAG failed; returning fallback explanation.",
                    {"error": str(e)}
                ))

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
        features_used = len(feature_names),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    """
    Main prediction endpoint.

    Send hormone + symptom data for recent days.
    Returns period + ovulation predictions with SHAP + clinical explanation.
    """
    _ = current_user
    return _run_prediction_core(request)


@app.post("/predict/jobs")
async def create_prediction_job(
    request: PredictionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Creates an async prediction job and returns a streamable job id."""
    _cleanup_old_jobs()
    _ = current_user

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    prediction_jobs[job_id] = {
        "queue": queue,
        "status": "running",
        "created_at_ts": time.time(),
    }

    async def run_job():
        loop = asyncio.get_running_loop()

        def emit(event: dict):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        try:
            response = await asyncio.to_thread(_run_prediction_core, request, emit)
            payload = response.dict() if hasattr(response, "dict") else response.model_dump()

            emit(_make_event(
                "final_result",
                "Prediction completed successfully.",
                payload
            ))
            emit(_make_event("completed", "Prediction stream completed."))
            prediction_jobs[job_id]["status"] = "completed"
        except Exception as e:
            detail = str(e)
            if isinstance(e, HTTPException):
                detail = str(getattr(e, "detail", detail))

            emit(_make_event("error", "Prediction job failed.", {"error": detail}))
            emit(_make_event("completed", "Prediction stream completed with error."))
            prediction_jobs[job_id]["status"] = "failed"

    asyncio.create_task(run_job())

    return {
        "job_id": job_id,
        "stream_url": f"/predict/stream/{job_id}",
    }


@app.get("/predict/stream/{job_id}")
async def stream_prediction_job(
    job_id: str,
    request: Request,
    token: str | None = Query(default=None),
):
    """Streams live prediction trace events via Server-Sent Events."""
    bearer_token = token
    if not bearer_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

    if not bearer_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    _get_current_user_from_token_string(bearer_token)

    _cleanup_old_jobs()
    job = prediction_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id")

    queue: asyncio.Queue = job["queue"]

    async def event_generator():
        event_id = 0
        while True:
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
                event_id += 1
                payload = json.dumps({"event_id": event_id, **event})
                yield f"id: {event_id}\ndata: {payload}\n\n"

                if event.get("type") == "completed":
                    break
            except asyncio.TimeoutError:
                # Keep-alive comment to avoid idle connection drops.
                yield ": keep-alive\n\n"

        job["created_at_ts"] = time.time() - 7200

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/features")
def get_features(current_user: dict[str, Any] = Depends(get_current_user)):
    """Returns the list of features the model expects."""
    _ = current_user
    if trained_features is None:
        raise HTTPException(status_code=503, detail="Features not loaded.")
    return {"features": trained_features, "count": len(trained_features)}