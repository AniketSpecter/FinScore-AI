"""
backend/main.py
────────────────
FastAPI application for FinScore AI.

Endpoints:
  GET  /health        — Liveness check
  GET  /model-info    — Active model metadata
  GET  /metrics       — Aggregate prediction statistics
  POST /predict       — Run risk scoring on a loan application
  POST /train         — Trigger model retraining in background
"""

import os
import sys
import json
import logging
import datetime
import subprocess
import threading
import math
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import shap
import uvicorn

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import models as db_models
from .database import engine, get_db, SessionLocal
from . import schemas

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger("finscore_backend")

# ─── App ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global database_ready
    log.info("FinScore AI API starting up …")
    try:
        db_models.Base.metadata.create_all(bind=engine)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ready = True
        log.info("Database connection ready.")
    except Exception as e:
        database_ready = False
        log.error(f"Database initialisation failed: {e}")
    load_artefacts()
    yield


app = FastAPI(
    title="FinScore AI — Risk Scoring API",
    description=(
        "AI-powered financial risk scoring and loan approval intelligence. "
        "Predicts probability of default, generates a 0-100 credit safety score, "
        "and provides SHAP-based explanations."
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# This prototype has no browser authentication. Credentials must remain disabled
# when wildcard origins are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── Paths (overridable via env vars) ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.getenv("MODEL_PATH",       os.path.join(BASE_DIR, "..", "models", "best_model.joblib"))
FEATURE_PATH     = os.getenv("FEATURE_COLS_PATH",os.path.join(BASE_DIR, "..", "models", "feature_cols.joblib"))
SCALER_PATH      = os.getenv("SCALER_PATH",      os.path.join(BASE_DIR, "..", "models", "scaler.joblib"))
METADATA_PATH    = os.path.join(BASE_DIR, "..", "models", "model_metadata.json")
ML_PIPELINE_DIR  = os.path.join(BASE_DIR, "..", "ml_pipeline")

# ─── Load artefacts at startup ────────────────────────────────────────────────
best_model   = None
feature_cols = None
scaler       = None
explainer    = None
model_meta   = {}
artefact_error = None
database_ready = False
training_lock = threading.Lock()
training_status = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "detail": "No training job has run in this process.",
}

def load_artefacts():
    global best_model, feature_cols, scaler, explainer, model_meta, artefact_error

    best_model = None
    feature_cols = None
    scaler = None
    explainer = None
    model_meta = {}
    artefact_error = None

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_features = joblib.load(FEATURE_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        if not loaded_features:
            raise ValueError("Feature column artefact is empty.")
        scaler_features = getattr(loaded_scaler, "n_features_in_", len(loaded_features))
        if scaler_features != len(loaded_features):
            raise ValueError("Scaler and feature column artefacts are incompatible.")
    except Exception as e:
        artefact_error = str(e)
        log.error(f"Could not load required model artefacts: {e}")
        return False

    best_model = loaded_model
    feature_cols = loaded_features
    scaler = loaded_scaler
    log.info("Model, feature columns, and scaler loaded.")

    # Build SHAP explainer
    try:
        explainer = shap.TreeExplainer(loaded_model)
        log.info("SHAP TreeExplainer initialised.")
    except Exception:
        try:
            bg = np.zeros((1, len(loaded_features)))
            explainer = shap.LinearExplainer(loaded_model, bg)
            log.info("SHAP LinearExplainer initialised.")
        except Exception as e2:
            log.warning(f"Could not build SHAP explainer: {e2}")

    # Load metadata
    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            model_meta = json.load(f)
    except Exception as e:
        log.warning(f"Could not load model metadata: {e}")
        model_meta = {"model_name": "unknown", "metrics": {}}
    return True


# ─── Risk engine ─────────────────────────────────────────────────────────────

def compute_risk_score(probability: float) -> float:
    """Convert default probability [0,1] → credit safety score [0,100]."""
    return round(max(0.0, min(100.0, (1.0 - probability) * 100)), 2)


def classify_risk(score: float):
    if score >= 75:
        return "Low Risk", "Approve"
    elif score >= 50:
        return "Medium Risk", "Review with Conditions"
    else:
        return "High Risk", "Reject"


def calculate_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Calculate a standard amortized monthly payment in the input currency."""
    monthly_rate = annual_rate / 1200.0
    if monthly_rate == 0:
        return round(principal / months, 2)
    growth = math.pow(1.0 + monthly_rate, months)
    return round(principal * monthly_rate * growth / (growth - 1.0), 2)


def assess_affordability(
    monthly_payment: float,
    monthly_income: Optional[float],
    existing_obligations: float,
):
    if monthly_income is None:
        return None, "Not assessed"

    ratio = round((monthly_payment + existing_obligations) / monthly_income * 100.0, 2)
    if ratio <= 35:
        status_label = "Comfortable"
    elif ratio <= 50:
        status_label = "Manageable"
    else:
        status_label = "High burden"
    return ratio, status_label


def build_input_warnings(input_data: dict) -> List[str]:
    """Flag valid but less-represented values without pretending they are errors."""
    warnings_list = []
    if input_data["credit_amount"] > 7_179:
        warnings_list.append(
            "Loan amount is above about 90% of the training examples; interpret the score cautiously."
        )
    if input_data["duration"] > 36:
        warnings_list.append(
            "Loan duration is longer than about 90% of the training examples."
        )
    return warnings_list


def build_recommended_actions(input_data: dict, affordability_ratio: Optional[float]) -> List[str]:
    """Return transparent, actionable suggestions independent of SHAP sign quirks."""
    actions = []
    if affordability_ratio is not None and affordability_ratio > 50:
        actions.append("Reduce the requested amount or existing obligations to lower monthly burden below 50%.")
    if input_data["credit_amount"] > 7_179:
        actions.append("Compare a smaller loan amount; the request is larger than most training examples.")
    if input_data["duration"] > 36:
        actions.append("Compare a shorter term while checking that the monthly payment remains affordable.")
    if input_data["checking_status"] in {"<0", "no checking"}:
        actions.append("A stable, non-negative checking-account position may strengthen the profile.")
    if input_data["savings_status"] in {"<100", "no known savings"}:
        actions.append("Building an emergency savings buffer may reduce financial vulnerability.")
    if input_data["employment"] in {"unemployed", "<1"}:
        actions.append("More employment stability may improve a future assessment.")
    if input_data["other_payment_plans"] != "none" or input_data["existing_credits"] >= 3:
        actions.append("Review or consolidate existing credit obligations before taking another loan.")
    if input_data["installment_commitment"] == 4:
        actions.append("The highest installment-burden band is selected; consider a lower payment burden.")
    return actions[:4]


def interpret_score(score: float) -> str:
    if score >= 75:
        return "Lower model-estimated default risk (below 25%); eligible for approval review."
    if score >= 50:
        return "Moderate model-estimated default risk (25% to 50%); conditions or manual review are appropriate."
    return "Higher model-estimated default risk (50% or more); do not auto-approve without detailed review."


# ─── Inference helper ─────────────────────────────────────────────────────────

def preprocess_for_inference(input_dict: dict) -> pd.DataFrame:
    """Apply the same preprocessing as training to a single applicant dict."""
    from ml_pipeline.preprocessing import transform_input_for_inference

    return transform_input_for_inference(
        input_dict,
        feature_cols=feature_cols,
        scaler_path=SCALER_PATH,
    )


def extract_shap_row(raw_values, expected_features: int) -> np.ndarray:
    """Normalise SHAP's model/version-dependent output to one class-1 row."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]

    values = np.asarray(raw_values)
    if values.ndim == 1:
        row = values
    elif values.ndim == 2 and values.shape[1] == expected_features:
        row = values[0]
    elif values.ndim == 2 and values.shape[0] == expected_features:
        row = values[:, -1]
    elif values.ndim == 3 and values.shape[1] == expected_features:
        row = values[0, :, -1]
    elif values.ndim == 3 and values.shape[2] == expected_features:
        row = values[-1, 0, :]
    else:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    row = np.asarray(row, dtype=float).reshape(-1)
    if row.size != expected_features:
        raise ValueError(
            f"SHAP returned {row.size} values for {expected_features} features."
        )
    return row


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="API health check")
def health_check():
    global database_ready
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ready = True
    except Exception:
        database_ready = False

    ready = all(
        [database_ready, best_model is not None, feature_cols is not None, scaler is not None]
    )
    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "database_ready": database_ready,
        "model_loaded": best_model is not None,
        "scaler_loaded": scaler is not None,
        "explainer_ready": explainer is not None,
        "artefact_error": artefact_error,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["System"], summary="Service readiness check")
def readiness_check():
    health = health_check()
    return JSONResponse(status_code=200 if health["ready"] else 503, content=health)


@app.get("/model-info", response_model=schemas.ModelInfo, tags=["System"], summary="Active model info")
def get_model_info(db: Session = Depends(get_db)):
    active_ver = (
        db.query(db_models.ModelVersion)
        .filter(db_models.ModelVersion.is_active == 1)
        .first()
    )
    if active_ver:
        return schemas.ModelInfo(
            model_name=active_ver.version_tag,
            model_version=active_ver.version_tag,
            active=True,
            deployed_at=str(active_ver.deployed_at),
            metrics=active_ver.metrics or {},
        )
    # Fallback to local metadata file
    return schemas.ModelInfo(
        model_name=model_meta.get("model_name", "local-file"),
        model_version=model_meta.get("model_version", "local-v1"),
        active=best_model is not None,
        deployed_at=model_meta.get("trained_at", "N/A"),
        metrics=model_meta.get("metrics", {}),
    )


@app.get("/metrics", tags=["Analytics"], summary="Aggregate prediction statistics")
def get_aggregate_metrics(db: Session = Depends(get_db)):
    total, avg_score, avg_probability = db.query(
        func.count(db_models.Prediction.id),
        func.avg(db_models.Prediction.risk_score),
        func.avg(db_models.Prediction.probability_default),
    ).one()
    if not total:
        return {"total_predictions": 0}

    risk_distribution = dict(
        db.query(db_models.Prediction.risk_category, func.count(db_models.Prediction.id))
        .group_by(db_models.Prediction.risk_category)
        .all()
    )
    recommendation_distribution = dict(
        db.query(db_models.Prediction.recommendation, func.count(db_models.Prediction.id))
        .group_by(db_models.Prediction.recommendation)
        .all()
    )

    return {
        "total_predictions": int(total),
        "avg_risk_score": round(float(avg_score), 2),
        "avg_default_prob": round(float(avg_probability), 4),
        "risk_distribution": risk_distribution,
        "recommendation_dist": recommendation_distribution,
    }


@app.get("/history", tags=["Analytics"], summary="Recent prediction history")
def get_history(
    limit: int = Query(50, ge=1, le=100, description="Number of newest records to return"),
    db: Session = Depends(get_db),
):
    preds = (
        db.query(db_models.Prediction)
        .order_by(db_models.Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": p.id,
            "timestamp": str(p.timestamp),
            "risk_score": p.risk_score,
            "risk_category": p.risk_category,
            "recommendation": p.recommendation,
            "probability_default": p.probability_default,
            "model_version": p.model_version,
        }
        for p in preds
    ]


@app.post("/predict", response_model=schemas.PredictionResponse, tags=["Prediction"], summary="Score a loan application")
def predict(application: schemas.LoanApplicationInput, db: Session = Depends(get_db)):
    if best_model is None or feature_cols is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model artefacts are not loaded.")

    input_dict = application.model_dump()
    log.info(f"Prediction request: credit_amount={input_dict.get('credit_amount')} duration={input_dict.get('duration')}")

    # Preprocess
    try:
        model_input = dict(input_dict)
        for field in ("annual_interest_rate", "monthly_income", "existing_monthly_obligations"):
            model_input.pop(field, None)
        df_final = preprocess_for_inference(model_input)
    except Exception as e:
        log.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Feature preprocessing failed.") from e

    # Predict
    try:
        probability = float(best_model.predict_proba(df_final)[0, 1])
    except Exception as e:
        log.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.") from e

    score = compute_risk_score(probability)
    risk_category, recommendation = classify_risk(score)
    monthly_payment = calculate_monthly_payment(
        input_dict["credit_amount"],
        input_dict["annual_interest_rate"],
        input_dict["duration"],
    )
    affordability_ratio, affordability_status = assess_affordability(
        monthly_payment,
        input_dict.get("monthly_income"),
        input_dict["existing_monthly_obligations"],
    )
    if affordability_status == "High burden" and recommendation == "Approve":
        recommendation = "Review with Conditions"
    input_warnings = build_input_warnings(input_dict)
    recommended_actions = build_recommended_actions(input_dict, affordability_ratio)

    # SHAP
    shap_vals: Optional[Dict[str, float]] = None
    positive_factors: List[str] = []
    negative_factors: List[str] = []

    if explainer is not None:
        try:
            raw = explainer.shap_values(df_final)
            sv = extract_shap_row(raw, len(feature_cols))
            shap_vals = {col: float(val) for col, val in zip(feature_cols, sv)}

            # Categorise factors by SHAP sign
            # Positive SHAP = increases default probability = BAD (risk factor)
            # Negative SHAP = decreases default probability = GOOD (protective factor)
            positive_shap = sorted([(k, v) for k, v in shap_vals.items() if v > 0], key=lambda x: x[1], reverse=True)
            negative_shap = sorted([(k, v) for k, v in shap_vals.items() if v < 0], key=lambda x: x[1])
            negative_factors = [k for k, v in positive_shap[:3]]  # top risk-increasing features
            positive_factors = [k for k, v in negative_shap[:3]]  # top risk-reducing features
        except Exception as e:
            log.warning(f"SHAP computation failed: {e}")

    # Persist
    pred_record = db_models.Prediction(
        input_data=input_dict,
        risk_score=score,
        probability_default=probability,
        recommendation=recommendation,
        risk_category=risk_category,
        model_version=model_meta.get("model_version", model_meta.get("model_name", "local-v1")),
    )
    db.add(pred_record)
    # Audit log
    db.add(db_models.AuditLog(action="prediction_made", details=f"score={score} cat={risk_category}"))
    try:
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        log.error(f"Could not persist prediction: {e}")
        raise HTTPException(status_code=503, detail="Prediction storage is unavailable.") from e

    log.info(f"Prediction complete: score={score} category={risk_category} recommendation={recommendation}")

    return schemas.PredictionResponse(
        prediction_id=pred_record.id,
        model_version=pred_record.model_version,
        probability_default=probability,
        risk_score=score,
        risk_category=risk_category,
        recommendation=recommendation,
        estimated_monthly_payment=monthly_payment,
        affordability_ratio=affordability_ratio,
        affordability_status=affordability_status,
        score_interpretation=interpret_score(score),
        input_warnings=input_warnings,
        recommended_actions=recommended_actions,
        shap_values=shap_vals,
        positive_factors=positive_factors,
        negative_factors=negative_factors,
    )


@app.post(
    "/train",
    tags=["MLOps"],
    summary="Trigger model retraining",
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Kick off a background retraining job and return immediately."""
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A training job is already running.")

    try:
        db.add(db_models.AuditLog(action="training_triggered", details="Manual retraining requested via API"))
        db.commit()
    except Exception:
        db.rollback()
        training_lock.release()
        raise

    training_status.update({
        "state": "running",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished_at": None,
        "detail": "Training all candidate models.",
    })

    def run_training():
        try:
            result = subprocess.run(
                [sys.executable, "train.py"],
                cwd=ML_PIPELINE_DIR,
                capture_output=True,
                text=True,
                timeout=60 * 60,
            )
            if result.returncode == 0 and load_artefacts():
                log.info("Retraining completed successfully.")
                training_status.update({
                    "state": "succeeded",
                    "detail": "Training completed and artefacts were reloaded.",
                })
            else:
                detail = (result.stderr or result.stdout or "Unknown training failure.")[-2000:]
                log.error(f"Retraining failed:\n{detail}")
                training_status.update({"state": "failed", "detail": detail})
        except Exception as e:
            log.exception("Retraining job crashed.")
            training_status.update({"state": "failed", "detail": str(e)})
        finally:
            training_status["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            training_lock.release()

            audit_db = SessionLocal()
            try:
                audit_db.add(db_models.AuditLog(
                    action="training_finished",
                    details=f"state={training_status['state']}",
                ))
                audit_db.commit()
            except Exception as e:
                audit_db.rollback()
                log.warning(f"Could not write training audit record: {e}")
            finally:
                audit_db.close()

    background_tasks.add_task(run_training)
    return {"message": "Training job started in background.", "status": "accepted"}


@app.get("/train/status", tags=["MLOps"], summary="Get retraining status")
def get_training_status():
    return dict(training_status)


# ─── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=3022, reload=True)
