"""
ml_pipeline/train.py
──────────────────────
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM on the
German Credit dataset. Tracks all experiments with MLflow. Selects and
saves the best model by ROC-AUC score.

Run from the ml_pipeline/ directory:
    python train.py
"""

import os
import sys
import warnings
import logging
import json
import datetime
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay, brier_score_loss
)
import shap

from preprocessing import get_train_test_data, MODEL_EXCLUDED_COLUMNS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "german_credit.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "mlflow", "artifacts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, model_name: str, out_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(out_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_prob, model_name: str, out_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC Curve — {model_name}")
    ax.legend()
    path = os.path.join(out_dir, f"roc_curve_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def save_shap_summary(model, X_test: pd.DataFrame, model_name: str, out_dir: str):
    """Generate and save a SHAP bar summary plot. Chooses explainer by model type."""
    try:
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.ensemble import RandomForestClassifier as RFC
        if isinstance(model, LR):
            # LinearExplainer needs a background dataset
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
            sv = shap_values
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, X_test, show=False, plot_type="bar")
        path = os.path.join(out_dir, f"shap_summary_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        return path
    except Exception as e:
        log.warning(f"SHAP summary failed for {model_name}: {e}")
        return None


# ─── Train one model ─────────────────────────────────────────────────────────

def train_and_log(name: str, model, X_train, y_train, X_test, y_test, feature_cols, run_artifacts_dir):
    log.info(f"Training {name} ...")
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Artifacts
        cm_path  = save_confusion_matrix(y_test, y_pred, name, run_artifacts_dir)
        roc_path = save_roc_curve(y_test, y_prob, name, run_artifacts_dir)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        # Log model using the appropriate MLflow flavor
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif isinstance(model, LGBMClassifier):
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        log.info(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return model, metrics


# ─── Main training loop ──────────────────────────────────────────────────────

def train_models():
    log.info("=" * 60)

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Training dataset not found at {DATA_PATH}. Run data/fetch_dataset.py first."
        )

    raw_df = pd.read_csv(DATA_PATH)
    profiled_columns = [
        "credit_amount", "duration", "age", "installment_commitment",
        "residence_since", "existing_credits", "num_dependents",
    ]
    input_profile = {
        col: {
            "min": float(raw_df[col].min()),
            "p50": float(raw_df[col].median()),
            "p90": float(raw_df[col].quantile(0.90)),
            "max": float(raw_df[col].max()),
        }
        for col in profiled_columns
    }
    log.info("FinScore AI — Model Training Pipeline")
    log.info("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, feature_cols = get_train_test_data(
        DATA_PATH, scaler_save_path=SCALER_PATH
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}  Features: {len(feature_cols)}")

    # Save feature columns
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    log.info(f"Feature columns saved → {MODELS_DIR}/feature_cols.joblib")

    # MLflow setup
    default_tracking_uri = f"sqlite:///{os.path.join(BASE_DIR, '..', 'mlflow', 'mlflow.db')}"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri))
    mlflow.set_experiment("FinScore_AI_Risk_Models")

    run_artifacts_dir = os.path.join(ARTIFACTS_DIR, "plots")
    os.makedirs(run_artifacts_dir, exist_ok=True)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=42, verbose=-1
        ),
    }

    results = {}
    for name, model in candidates.items():
        try:
            trained_model, metrics = train_and_log(
                name, model, X_train, y_train, X_test, y_test, feature_cols, run_artifacts_dir
            )
            results[name] = (trained_model, metrics)
        except Exception as e:
            log.error(f"Failed to train {name}: {e}")

    # ─── Select a discriminative and well-calibrated model ────────────────
    if not results:
        raise RuntimeError("No candidate model trained successfully.")

    # The score shown to users is derived from predicted probability, so a tiny
    # AUC advantage must not outweigh materially worse probability calibration.
    # Keep models within 0.01 of the best AUC, then choose the lowest Brier score.
    best_auc = max(result[1]["roc_auc"] for result in results.values())
    eligible = [
        name for name, (_, metrics) in results.items()
        if metrics["roc_auc"] >= best_auc - 0.01
    ]
    best_name = min(eligible, key=lambda name: results[name][1]["brier_score"])
    best_model, best_metrics = results[best_name]

    log.info("\n" + "=" * 60)
    log.info(f"Best Model  : {best_name}")
    log.info(f"ROC-AUC     : {best_metrics['roc_auc']}")
    log.info(f"Brier Score : {best_metrics['brier_score']}")
    log.info(f"F1 Score    : {best_metrics['f1']}")
    log.info("=" * 60)

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    log.info(f"Best model saved → {best_model_path}")

    # Save metadata
    trained_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata = {
        "model_name": best_name,
        "model_version": f"{best_name.lower().replace(' ', '-')}-{trained_at[:10]}",
        "trained_at": trained_at,
        "selection_metric": "lowest Brier score among models within 0.01 of best ROC-AUC",
        "dataset": "OpenML credit-g v1",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "excluded_sensitive_fields": sorted(MODEL_EXCLUDED_COLUMNS),
        "input_profile": input_profile,
        "metrics": best_metrics,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate SHAP summary for best model
    log.info("Generating SHAP summary for best model …")
    shap_path = save_shap_summary(best_model, X_test, best_name, run_artifacts_dir)
    if shap_path:
        log.info(f"SHAP summary -> {shap_path}")

    # Print comparison table
    log.info("\nModel Comparison:")
    log.info(f"{'Model':<25} {'ACC':>6} {'PREC':>6} {'REC':>6} {'F1':>6} {'AUC':>6}")
    log.info("-" * 60)
    for n, (_, m) in results.items():
        marker = " << BEST" if n == best_name else ""
        log.info(
            f"{n:<25} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f}{marker}"
        )

    return metadata


if __name__ == "__main__":
    train_models()
