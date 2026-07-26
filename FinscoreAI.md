# FinScore AI — Project Guide

## 1. Project status

FinScore AI is a demo-ready, end-to-end credit-risk scoring platform. It accepts 18 model fields plus optional affordability inputs, estimates default probability, converts that probability into a 0–100 credit safety score, assigns a risk category and recommendation, records the result, and explains the prediction with SHAP.

The project supports two execution modes:

- **One-click Windows mode:** SQLite, FastAPI, Streamlit, and MLflow through `finscoreAI_Runner.bat`.
- **Container mode:** PostgreSQL, FastAPI, Streamlit, and MLflow through Docker Compose.
- **Hosted demo mode:** Streamlit Community Cloud with an internal FastAPI service and temporary SQLite history.

This is a portfolio and educational prototype. It is not validated or approved for real lending decisions. See [Responsible-use limits](#14-responsible-use-limits).

## 2. Fastest start on Windows

Double-click:

```text
finscoreAI_Runner.bat
```

The launcher performs the full startup workflow:

1. Uses `.venv` or creates it with Python 3.11+.
2. Verifies required Python packages and installs them if missing.
3. Downloads the OpenML `credit-g` dataset if it is missing.
4. Validates the model, feature-column, and scaler artefacts.
5. Retrains all candidate models if artefacts are missing or incompatible.
6. Reuses verified FinScore services or selects nearby free ports when defaults are occupied.
7. Starts MLflow, the FastAPI server, and the Streamlit dashboard.
8. Waits for API, MLflow, and dashboard health checks to succeed, then opens the dashboard.

Default service URLs (the launcher prints replacements if a port is occupied):

| Service | URL |
|---|---|
| Dashboard | `http://127.0.0.1:2022` |
| API documentation | `http://127.0.0.1:3022/docs` |
| API readiness | `http://127.0.0.1:3022/ready` |
| MLflow | `http://127.0.0.1:4022` |
| PostgreSQL (Docker host) | `127.0.0.1:5022` |

Launcher options:

```bat
finscoreAI_Runner.bat --check
```

Validates setup and runs the in-process test suite without starting services.

```bat
finscoreAI_Runner.bat --setup
```

Forces a dependency installation/update before normal startup.

To stop the local project, close the three command windows titled `FinScore AI - API`, `FinScore AI - Dashboard`, and `FinScore AI - MLflow`.

### Streamlit Community Cloud deployment

Use `frontend/streamlit_app.py` as the Community Cloud entrypoint. Its nearby `frontend/requirements.txt` intentionally contains only serving dependencies, so the hosted demo does not install the heavier MLflow and model-training stack.

The cloud entrypoint starts FastAPI on an internal loopback port and then renders the same dashboard as local mode. Prediction history uses temporary SQLite storage and can disappear when the Community Cloud app sleeps or restarts. Model retraining is disabled in hosted mode; run the Windows launcher or Docker Compose for persistent history, MLflow, and retraining.

Never commit `.env`, Streamlit secrets, local `*.db` files, or `ml_pipeline/mlruns/`. Those runtime files are excluded by `.gitignore`.

## 3. What the application does

For every accepted application, the backend returns:

- `probability_default`: model-estimated probability from 0 to 1.
- `risk_score`: `(1 - probability_default) × 100`, bounded to 0–100.
- `risk_category`: Low, Medium, or High Risk.
- `recommendation`: Approve, Review with Conditions, or Reject.
- `estimated_monthly_payment`: standard amortized payment using the supplied annual rate.
- `affordability_ratio`: estimated payment plus existing obligations divided by supplied monthly income.
- `score_interpretation`, `input_warnings`, and `recommended_actions`: plain-language decision support.
- `shap_values`: per-feature contribution values when the explainer is available.
- `positive_factors`: strongest factors reducing predicted default risk.
- `negative_factors`: strongest factors increasing predicted default risk.

Decision thresholds:

| Safety score | Category | Recommendation |
|---:|---|---|
| 75–100 | Low Risk | Approve |
| 50–74.99 | Medium Risk | Review with Conditions |
| 0–49.99 | High Risk | Reject |

These thresholds are application rules, not thresholds learned or calibrated from lending policy.

## 4. Architecture

```text
Browser
  |
  v
Streamlit dashboard host :2022
  |  REST/JSON
  v
FastAPI backend host :3022
  |-- SQLAlchemy --> SQLite (local) or PostgreSQL (Docker)
  |-- joblib ------> model + feature columns + scaler
  |-- SHAP --------> local prediction explanations
  `-- subprocess --> model retraining pipeline

ML training pipeline --> MLflow tracking host :4022
                     --> refreshed artefacts in models/
```

The Windows runner binds services to `127.0.0.1`, keeping the development services local to the machine. Docker exposes the same ports through container mappings.

## 5. Repository layout

```text
finscore-ai/
|-- backend/
|   |-- main.py                 FastAPI app, scoring, readiness, analytics, training
|   |-- database.py             SQLite/PostgreSQL engine and sessions
|   |-- models.py               SQLAlchemy tables
|   `-- schemas.py              Strict Pydantic request/response models
|-- frontend/
|   `-- app.py                  Four-page Streamlit dashboard
|-- ml_pipeline/
|   |-- preprocessing.py        Cleaning, feature engineering, encoding, scaling
|   `-- train.py                Candidate training, evaluation, MLflow, selection
|-- data/
|   |-- fetch_dataset.py        OpenML dataset fetcher
|   `-- german_credit.csv       Prepared credit-g dataset
|-- models/
|   |-- best_model.joblib       Selected classifier
|   |-- feature_cols.joblib     Training-time feature contract
|   |-- scaler.joblib           Fitted StandardScaler
|   `-- model_metadata.json     Model name and evaluation metrics
|-- tests/
|   |-- test_preprocessing.py   Unit and inference-contract tests
|   |-- test_api_unit.py        In-process API regression tests
|   |-- test_api.py             Optional live-server tests
|   `-- full_api_test.py        Manual endpoint report script
|-- docker/                     Backend and frontend Dockerfiles
|-- docker-compose.yml          PostgreSQL + MLflow + API + dashboard
|-- finscoreAI_Runner.bat       One-click Windows launcher
|-- pytest.ini                  Automated-test discovery rules
|-- requirements.txt            Supported dependency ranges
`-- FinscoreAI.md               This guide
```

## 6. API reference

### System endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and component state; always useful for diagnosis |
| GET | `/ready` | Returns 200 only when database and required model artefacts are ready |
| GET | `/model-info` | Active database model version or local metadata fallback |
| GET | `/docs` | Swagger/OpenAPI interface |
| GET | `/redoc` | ReDoc interface |

### Prediction and analytics endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Validate, score, explain, and store an application |
| GET | `/metrics` | Database-side aggregate prediction metrics |
| GET | `/history?limit=50` | Newest prediction records; limit must be 1–100 |

### MLOps endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/train` | Starts one background training job; returns HTTP 202 |
| GET | `/train/status` | Reports idle/running/succeeded/failed state and timestamps |

Only one training job can run in a backend process at a time. A second request receives HTTP 409.

### Example prediction request

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
  "other_parties": "none",
  "residence_since": 2,
  "property_magnitude": "real estate",
  "age": 30,
  "other_payment_plans": "none",
  "housing": "own",
  "existing_credits": 1,
  "job": "skilled",
  "num_dependents": 1,
  "own_telephone": "none",
  "annual_interest_rate": 10.0,
  "monthly_income": 1500.0,
  "existing_monthly_obligations": 100.0
}
```

Categorical values are restricted to categories present in the training dataset. Loan amount is restricted to the trained range of 250–18,424, duration to 4–72 months, and age to 19–75. Unknown categories, extra fields, and out-of-range values receive HTTP 422 instead of producing unreliable extrapolated scores.

Annual interest, monthly income, and existing monthly obligations are decision-support inputs only. They calculate an amortized payment and affordability ratio and are never presented as ML training features. A payment burden above 50% prevents automatic approval even when model risk is low.

## 7. Frontend pages

The Streamlit application contains four pages:

1. **Risk Scoring:** three plain-language example profiles, grouped fields, friendly category labels, trained-range limits, amortized payment and affordability calculations, safety gauge, category, recommendation, actionable suggestions, and SHAP chart.
2. **Analytics:** platform aggregates, risk/recommendation distributions, dataset class balance, feature distributions, and correlation heatmap.
3. **History:** newest stored predictions, score timeline, and table.
4. **System Info:** API/model health, architecture summary, and retraining action.

The sidebar distinguishes an offline API from a reachable but degraded API with missing database or model components.

## 8. Machine-learning pipeline

### Dataset

- Source: OpenML `credit-g`, version 1 (German Credit dataset).
- Rows: 1,000 applicants.
- Raw predictors: 20.
- Target mapping: `good = 0`, `bad = 1`.
- Training/test split: stratified 80/20 with random state 42.

### Feature processing

The pipeline:

1. Removes duplicate rows.
2. Fills missing categoricals with the mode and numeric values with the median.
3. Removes the historical marital/sex and foreign-worker fields from model inputs.
4. Creates engineered numerical features, including a payment-reliability order checked against observed dataset default rates.
5. One-hot encodes categorical inputs.
6. Sanitizes column names for XGBoost compatibility.
7. Saves the exact feature-column order.
8. Fits `StandardScaler` on the training split and transforms train/test data.

Single-row inference deliberately encodes all observed categorical values before alignment. This is required because applying `drop_first=True` to a single row would discard every observed categorical value.

### Engineered features

- EMI burden proxy
- Loan-to-duration ratio
- Credit age score
- Payment reliability score
- Employment stability score
- Savings-to-loan ratio

### Candidate models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Selection keeps models within 0.01 of the best held-out ROC-AUC, then chooses the lowest Brier score. This prevents a tiny ranking advantage from selecting materially worse default probabilities. The currently supplied metadata identifies Logistic Regression as the selected model:

| Metric | Value |
|---|---:|
| Accuracy | 0.7600 |
| Precision | 0.6200 |
| Recall | 0.5167 |
| F1 | 0.5636 |
| ROC-AUC | 0.7944 |
| Brier score | 0.1578 |

Every retraining run logs parameters, metrics, a confusion matrix, an ROC curve, and the trained model to MLflow. The selected model also receives a SHAP summary plot.

## 9. Data storage

Local mode defaults to the absolute project database `finscore_local.db`. No PostgreSQL installation is required.

Docker mode sets `DATABASE_URL` to PostgreSQL and uses the `postgres_data` volume.

SQLAlchemy tables:

| Table | Contents |
|---|---|
| `predictions` | Input JSON, score, probability, category, recommendation, model version, timestamp |
| `audit_logs` | Prediction and training lifecycle events |
| `model_versions` | Optional registered/deployed model records |
| `users` | Reserved user entity; authentication is not implemented |

`GET /metrics` uses database aggregation rather than loading every prediction row into API memory.

## 10. Manual local development

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python data\fetch_dataset.py
cd ml_pipeline
python train.py
cd ..
```

Terminal 1:

```bat
set DATABASE_URL=sqlite:///./finscore_local.db
.venv\Scripts\python -m uvicorn backend.main:app --reload --port 3022
```

Terminal 2:

```bat
set API_URL=http://127.0.0.1:3022
.venv\Scripts\python -m streamlit run frontend\app.py
```

Terminal 3:

```bat
.venv\Scripts\python -m mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --port 4022
```

## 11. Docker execution

Requirements: Docker Desktop and valid `data/german_credit.csv` plus all three `.joblib` artefacts in `models/`. The Windows runner prepares these automatically; `finscoreAI_Runner.bat --check` is a convenient preflight.

```bat
docker compose up --build
```

Stop and retain PostgreSQL data:

```bat
docker compose down
```

Stop and delete the PostgreSQL volume only when the stored predictions are no longer needed:

```bat
docker compose down -v
```

Container health checks validate MLflow, backend readiness, and Streamlit without relying on unavailable `curl` binaries. The backend uses one worker so a retrained model is consistently hot-reloaded in the serving process.

## 12. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | Project SQLite database | SQLAlchemy connection URL |
| `API_URL` | `http://127.0.0.1:3022` | Frontend backend location |
| `MODEL_PATH` | `models/best_model.joblib` | Selected model path |
| `FEATURE_COLS_PATH` | `models/feature_cols.joblib` | Feature contract path |
| `SCALER_PATH` | `models/scaler.joblib` | Required fitted scaler path |
| `MLFLOW_TRACKING_URI` | Local MLflow SQLite store during direct training | Remote/local tracking target |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |

The model, scaler, and feature columns form one versioned contract. If any required artefact is missing or incompatible, `/ready` returns HTTP 503 and `/predict` refuses to score.

## 13. Testing and verification

Run all automated tests:

```bat
.venv\Scripts\python -m pytest -q
```

Run the same focused readiness suite used by the launcher:

```bat
finscoreAI_Runner.bat --check
```

Run optional live-server tests after starting the API on port 3022:

```bat
.venv\Scripts\python -m pytest tests\test_api.py -q
.venv\Scripts\python tests\full_api_test.py
```

The automated suite covers preprocessing, missing values, sensitive-column exclusion, empirically ordered engineered features, categorical inference, scaler enforcement, strict trained-range validation, probability-to-score bands, amortized payment, affordability, SHAP response structure, aggregate metrics, readiness, and history bounds.

## 14. Responsible-use limits

Do not use this prototype as the sole basis for a real credit decision.

- The dataset is small, old, and tied to a specific historical population and currency context.
- Probability quality is measured with a held-out Brier score, but external and out-of-time calibration has not been evaluated.
- Direct marital/sex and foreign-worker fields are excluded, but a complete fairness assessment across protected and proxy groups is still required.
- Recommendation thresholds are illustrative business rules.
- SHAP explains model behavior; it does not establish causality or legal adverse-action reasons.
- There is no authentication, authorization, encryption-at-rest policy, secrets manager, or personally identifiable information retention policy.
- Monitoring for production drift, data quality, and outcomes is not implemented.

For real financial use, the system would require legal/compliance review, representative data, fairness and calibration studies, human review, documented model governance, security controls, monitoring, and an appeal process.

## 15. Completed readiness improvements

The readiness pass made the following material corrections:

- Fixed the categorical single-row inference bug that discarded all categorical values.
- Made scaling mandatory instead of silently scoring unscaled data.
- Added strict dataset-domain validation and rejected extra request fields.
- Corrected invalid frontend categories and the frontend/API age mismatch.
- Restricted amount, duration, and age to the model's actual training ranges, preventing the 50,000-vs-18,424 extrapolation failure.
- Replaced raw dataset codes with plain-language labels, grouped sections, and three outcome-demonstrating presets.
- Corrected the reversed historical credit-reliability ordering using observed default rates.
- Removed a dimensionally invalid credit-utilization proxy and excluded marital/sex and foreign-worker features.
- Added amortized monthly payment, optional income affordability, a 50% burden guardrail, warnings, and actionable guidance.
- Changed model selection to prioritize Brier calibration among candidates with near-best ROC-AUC.
- Renamed the displayed metric to Credit Safety Score and aligned the lower-risk threshold to 75.
- Preserved the corrected SHAP risk/protective factor sign and added shape normalization across model types.
- Added SQLite as the zero-configuration local database while retaining PostgreSQL overrides.
- Added component-aware `/health` and HTTP-status-aware `/ready` endpoints.
- Added bounded history queries and database-side aggregate calculations.
- Added retraining concurrency protection, timeout, audit status, and `/train/status`.
- Hardened database commit rollback and reduced internal error leakage.
- Fixed Docker health checks, startup dependencies, writable retraining artefacts, and worker consistency.
- Added supported dependency ranges and the missing direct `requests` dependency.
- Added in-process API regression tests and clean pytest discovery.
- Added the full Windows launcher with dependency/data/model preflight and port identity checks.

## 16. Recommended next enhancements

Prioritized next work for moving beyond a portfolio prototype:

1. **Model validation:** probability calibration, cross-validation, threshold optimization, confidence intervals, and out-of-time validation.
2. **Fairness and governance:** protected-group testing, proxy review, model cards, approval workflow, and adverse-action reason mapping.
3. **Security:** authentication, role-based access, restrictive CORS, TLS, rate limiting, secrets management, and PII minimization.
4. **Schema migrations:** use Alembic instead of automatic `create_all` for controlled database evolution.
5. **Durable jobs:** move retraining from in-process background work to a queue/worker system with persistent status.
6. **Model registry deployment:** promote an approved MLflow model version instead of automatically deploying the best training run.
7. **Observability:** structured request IDs, latency/error metrics, centralized logs, data-drift alerts, and outcome monitoring.
8. **Preprocessing contract:** replace separate preprocessing artefacts with one fitted scikit-learn `Pipeline`/`ColumnTransformer` saved alongside the classifier.
9. **CI/CD:** run unit tests, container builds, vulnerability scans, and smoke tests on every pull request.
10. **Frontend workflow:** add authenticated case review, application identifiers, exportable reports, and a clear human override/audit process.

## 17. Troubleshooting

### A preferred port is occupied

The launcher refuses to attach to an unrelated service and automatically selects a nearby free port. Use the URLs printed by the launcher. To recover the standard port, find its current owner:

```powershell
Get-NetTCPConnection -LocalPort 3022 -State Listen | Select-Object OwningProcess
Get-Process -Id <OwningProcess>
```

Use the same command with ports 2022, 4022, or 5022. Stop or reconfigure the conflicting application only if you need the default URL.

### API is degraded

Open `http://127.0.0.1:3022/health`. Check `database_ready`, `model_loaded`, `scaler_loaded`, and `artefact_error`. Then run:

```bat
finscoreAI_Runner.bat --check
```

### Model artefacts are incompatible

Move the old artefacts out of `models/` if they are needed for backup, then run the launcher. It will retrain and generate a compatible model, feature list, scaler, and metadata.

### Dataset download fails

Confirm internet access to OpenML or place a prepared `german_credit.csv` in `data/`. The CSV must contain the expected 20 predictors plus the binary `target` column.

### Docker backend is not ready

```bat
docker compose ps
docker compose logs fastapi_backend
docker compose logs postgres_db
```

Verify that all three `.joblib` files exist in `models/` before starting Compose.
