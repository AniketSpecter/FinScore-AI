# 💳 FinScore AI

> **AI-Powered Financial Risk Scoring & Loan Approval Intelligence Platform**
> 
> *MCA AI/ML Internship Portfolio Project — Production-Inspired Prototype*

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=https%3A%2F%2Fgithub.com%2FAniketSpecter%2FFinScore-AI&branch=main&mainModule=frontend%2Fstreamlit_app.py)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?logo=xgboost)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)

---

## One-Click Windows Start

Double-click `finscoreAI_Runner.bat` to prepare missing dependencies/data/models and start the API, dashboard, and MLflow. Run `finscoreAI_Runner.bat --check` for a setup and test preflight without starting services.

See [FinscoreAI.md](FinscoreAI.md) for the complete architecture, API, setup, operations, troubleshooting, responsible-use notes, and enhancement roadmap.

### Streamlit Community Cloud

The hosted entrypoint is `frontend/streamlit_app.py`. It starts the FastAPI scoring service on an internal loopback port, uses the committed model artefacts, and stores prediction history only in temporary cloud storage. The local runner remains the full environment with MLflow and persistent SQLite.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Development Setup](#local-development-setup)
- [API Documentation](#api-documentation)
- [ML Pipeline](#ml-pipeline)
- [Explainable AI](#explainable-ai)
- [MLflow Tracking](#mlflow-tracking)
- [Running Tests](#running-tests)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)

---

## Overview

FinScore AI is a complete, end-to-end machine learning platform that replicates real-world financial risk assessment workflows used by banks and fintech companies. It:

1. **Predicts** the probability a loan applicant will default.
2. **Generates** a credit safety score from **0 to 100** (higher = safer).
3. **Categorises** applicants into Low, Medium, or High Risk tiers.
4. **Recommends** loan decisions: Approve / Review with Conditions / Reject.
5. **Calculates** amortized monthly payment and optional income-based affordability.
6. **Explains** every prediction using SHAP values and plain-language actions.
7. **Tracks** all experiments with MLflow for full reproducibility.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User (Browser)                        │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Streamlit Frontend    │  host :2022
              │  4-page dashboard       │
              │  Risk Scoring           │
              │  Analytics Charts       │
              │  Prediction History     │
              │  System Info            │
              └────────────┬────────────┘
                           │ HTTP REST JSON
              ┌────────────▼────────────┐
              │   FastAPI Backend       │  host :3022
              │  POST /predict          │
              │  GET  /metrics          │
              │  GET  /history          │
              │  GET  /health           │
              │  GET  /model-info       │
              │  POST /train            │
              └──┬─────────────┬────────┘
                 │             │
    ┌────────────▼──┐   ┌──────▼──────────┐
    │  PostgreSQL   │   │  ML Artefacts   │
    │ host :5022    │   │  best_model     │
    │  predictions  │   │  feature_cols   │
    │  audit_logs   │   │  scaler         │
    │  model_ver    │   │  SHAP explainer │
    └───────────────┘   └─────────────────┘
                 │
    ┌────────────▼────────────┐
    │   MLflow Server         │  host :4022
    │  Experiment tracking    │
    │  Model registry         │
    │  Artifact store         │
    └─────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **4 ML Models** | Logistic Regression, Random Forest, XGBoost, LightGBM |
| **Probability-Aware Selection** | Lowest Brier error among models within 0.01 of best held-out ROC-AUC |
| **6 Engineered Features** | EMI Burden, Credit Age Score, Payment Reliability, Employment Stability, Savings-to-Loan Ratio, Loan-to-Duration |
| **Affordability** | Standard amortized payment, payment burden, and >50% guardrail |
| **Explainable AI** | SHAP values per prediction + global summary plots |
| **Risk Engine** | Score 0-100, 3-tier categorisation, 3-tier recommendation |
| **REST API** | FastAPI with Pydantic validation, CORS, structured logging |
| **Dashboard** | Streamlit 4-page app: Scoring, Analytics, History, System |
| **Visualisations** | Gauge, SHAP bar, Correlation Heatmap, Donut charts, Timeline |
| **Database** | PostgreSQL + SQLAlchemy ORM (predictions, audit logs) |
| **MLOps** | MLflow experiment tracking + model registry |
| **Docker** | 4-service docker-compose.yml |
| **Tests** | pytest unit + integration tests |

---

## Project Structure

```
finscore-ai/
├── backend/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, all endpoints
│   ├── database.py       # SQLAlchemy engine + session
│   ├── models.py         # ORM models (Predictions, AuditLogs, etc.)
│   └── schemas.py        # Pydantic request/response schemas
│
├── frontend/
│   └── app.py            # Streamlit 4-page dashboard
│
├── ml_pipeline/
│   ├── preprocessing.py  # Feature engineering + transform pipeline
│   └── train.py          # Training + MLflow logging + model selection
│
├── data/
│   ├── german_credit.csv # Downloaded dataset (1000 rows, 21 cols)
│   └── fetch_dataset.py  # OpenML download script
│
├── models/
│   ├── best_model.joblib     # Trained best model
│   ├── feature_cols.joblib   # Aligned column names
│   ├── scaler.joblib         # Fitted StandardScaler
│   └── model_metadata.json  # Best model name + metrics
│
├── mlflow/
│   ├── mlflow.db         # SQLite tracking store (local dev)
│   └── artifacts/        # Logged plots, SHAP summaries
│
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
│
├── notebooks/            # EDA notebook (Jupyter)
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── docs/
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start (Docker)

> Requires Docker Desktop running on Windows.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd finscore-ai

# 2. Train models first (needs Python + requirements installed locally)
cd ml_pipeline
python train.py
cd ..

# 3. Start all services
docker-compose up --build

# 4. Access services
# Dashboard:  http://127.0.0.1:2022
# API Docs:   http://127.0.0.1:3022/docs
# MLflow:     http://127.0.0.1:4022
```

---

## Local Development Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python data/fetch_dataset.py

# 4. Train models
cd ml_pipeline && python train.py && cd ..

# 5. Start backend
uvicorn backend.main:app --reload --port 3022

# 6. Start frontend (new terminal)
streamlit run frontend/app.py
```

---

## API Documentation

Full Swagger UI available at `http://127.0.0.1:3022/docs`

### `GET /health`
Returns API liveness status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "explainer_ready": true,
  "timestamp": "2026-07-16T00:00:00Z"
}
```

---

### `POST /predict`
Scores a loan application and returns risk prediction with SHAP explanations.

**Request Body:** (18 model fields plus optional affordability inputs — see Swagger)
```json
{
  "checking_status": "0<=X<200",
  "duration": 24,
  "credit_history": "existing paid",
  "purpose": "radio/tv",
  "credit_amount": 5000.0,
  "savings_status": "<100",
  "employment": "1<=X<4",
  "installment_commitment": 2,
  "age": 30,
  "annual_interest_rate": 10.0,
  "monthly_income": 1500.0,
  "existing_monthly_obligations": 100.0
}
```

**Response:**
```json
{
  "probability_default": 0.32,
  "risk_score": 68.0,
  "risk_category": "Medium Risk",
  "recommendation": "Review with Conditions",
  "estimated_monthly_payment": 230.72,
  "affordability_ratio": 22.05,
  "shap_values": { "credit_amount": 0.12, "checking_status_>=200": -0.08, ... },
  "positive_factors": ["credit_age_score", "payment_reliability_score"],
  "negative_factors": ["credit_amount", "checking_status"]
}
```

---

### `GET /metrics`
Returns aggregate prediction statistics.

```json
{
  "total_predictions": 42,
  "avg_risk_score": 65.3,
  "avg_default_prob": 0.28,
  "risk_distribution": { "Low Risk": 18, "Medium Risk": 16, "High Risk": 8 },
  "recommendation_dist": { "Approve": 18, "Review with Conditions": 16, "Reject": 8 }
}
```

---

### `GET /history?limit=50`
Returns recent prediction records.

---

### `GET /model-info`
Returns active model metadata (name, version, metrics).

---

### `POST /train`
Triggers background model retraining. Returns immediately.

```json
{ "message": "Training job started in background.", "status": "accepted" }
```

---

## ML Pipeline

### Dataset
- **German Credit Risk Dataset** — 1,000 applicants, 20 features, binary target.
- Downloaded automatically via [OpenML](https://openml.org/d/31).

### Engineered Features
| Feature | Description |
|---|---|
| `emi_burden_proxy` | Monthly principal adjusted by installment-burden band |
| `loan_to_duration` | Loan size relative to repayment period |
| `credit_age_score` | Quartile-binned applicant age |
| `payment_reliability_score` | Ordinal from credit history (1=worst, 5=best) |
| `employment_stability_score` | Ordinal from employment tenure (0=unemployed, 4=>=7yr) |
| `savings_to_loan_ratio` | Estimated savings midpoint / credit amount |

### Preprocessing
1. Deduplicate rows
2. Fill missing values (mode for categorical, median for numerical)
3. Exclude historical marital/sex and foreign-worker fields
4. Apply feature engineering with observed-data credit-history ordering
5. One-hot encode all categorical columns
6. Sanitize column names (remove `[`, `]`, `<`, `>` for XGBoost compatibility)
7. StandardScaler fit on train, transform on test

---

## Explainable AI

Every prediction returns SHAP values computed by the appropriate explainer:
- **Linear models** → `shap.LinearExplainer`
- **Tree models** → `shap.TreeExplainer`

The frontend renders an interactive horizontal bar chart showing the top 12 most impactful features, coloured:
- 🔴 **Red** = increases default risk (negative for applicant)
- 🟢 **Green** = decreases default risk (positive for applicant)

---

## MLflow Tracking

Each training run logs:
- All model hyperparameters
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix plot (PNG)
- ROC curve plot (PNG)
- SHAP summary plot (for best model)
- The trained model artefact

Access the tracking UI at `http://127.0.0.1:4022`.

---

## Running Tests

```bash
# Unit tests (no server needed)
.venv\Scripts\pytest tests/test_preprocessing.py -v

# Integration tests (start backend first)
uvicorn backend.main:app --port 3022 &
.venv\Scripts\pytest tests/test_api.py -v

# All tests
.venv\Scripts\pytest tests/ -v
```

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.760 | 0.620 | 0.517 | 0.564 | **0.794** |
| Random Forest | 0.715 | 0.519 | 0.700 | 0.596 | 0.798 |
| XGBoost | 0.760 | 0.615 | 0.533 | 0.571 | 0.785 |
| LightGBM | 0.730 | 0.542 | 0.650 | 0.591 | 0.773 |

> *Models within 0.01 of best ROC-AUC are compared by Brier score so the displayed probability remains better calibrated. Sensitive marital/sex and foreign-worker fields are excluded.*

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| MLOps | MLflow |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Database | PostgreSQL, SQLAlchemy |
| Frontend | Streamlit, Plotly, Matplotlib |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| Language | Python 3.12 |

---

*Built as an MCA AI/ML Internship Portfolio Project demonstrating production-quality machine learning engineering practices.*
