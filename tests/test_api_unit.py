"""In-process API regression tests; no separately running server is required."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import get_db
from backend.main import app
from backend.models import Base


SAMPLE_PAYLOAD = {
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
}


@pytest.fixture
def client(tmp_path):
    db_path = (tmp_path / "api.db").as_posix()
    test_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    test_engine.dispose()


def test_service_is_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_predict_and_aggregate_metrics(client):
    prediction = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert prediction.status_code == 200, prediction.text
    body = prediction.json()
    assert 0 <= body["probability_default"] <= 1
    assert 0 <= body["risk_score"] <= 100
    assert body["prediction_id"] > 0
    assert body["estimated_monthly_payment"] > 0
    assert body["score_interpretation"]

    metrics = client.get("/metrics").json()
    assert metrics["total_predictions"] == 1
    assert sum(metrics["risk_distribution"].values()) == 1


def test_categorical_inputs_reach_the_model(client):
    baseline = dict(SAMPLE_PAYLOAD, checking_status="<0")
    changed = dict(SAMPLE_PAYLOAD, checking_status="no checking")
    baseline_score = client.post("/predict", json=baseline).json()["risk_score"]
    changed_score = client.post("/predict", json=changed).json()["risk_score"]
    assert baseline_score != changed_score


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 18),
        ("age", 76),
        ("duration", 3),
        ("credit_amount", 50_000),
        ("purpose", "vacation"),
        ("checking_status", "unknown"),
        ("foreign_worker", "yes"),
    ],
)
def test_invalid_domain_values_are_rejected(client, field, value):
    response = client.post("/predict", json=dict(SAMPLE_PAYLOAD, **{field: value}))
    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 101, 1000000])
def test_history_limit_is_bounded(client, limit):
    assert client.get(f"/history?limit={limit}").status_code == 422


def test_affordability_calculation_and_guardrail(client):
    payload = dict(
        SAMPLE_PAYLOAD,
        annual_interest_rate=12.0,
        monthly_income=300.0,
        existing_monthly_obligations=150.0,
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["estimated_monthly_payment"] > 0
    assert body["affordability_ratio"] > 50
    assert body["affordability_status"] == "High burden"
    assert body["recommended_actions"]
