"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI backend.
Run with: pytest tests/ -v
Requires the backend to be running at http://127.0.0.1:3022
or will skip if unreachable.
"""

import os
import sys
import pytest
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:3022")

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


def api_available():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_api():
    if not api_available():
        pytest.skip(f"FastAPI backend is not reachable at {API_URL}. Start it first.")


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{API_URL}/health")
        assert r.status_code == 200

    def test_health_status_field(self):
        r = requests.get(f"{API_URL}/health").json()
        assert r["status"] == "healthy"

    def test_health_has_model_loaded(self):
        r = requests.get(f"{API_URL}/health").json()
        assert "model_loaded" in r


class TestPredictEndpoint:
    def test_predict_returns_200(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_predict_response_structure(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        assert "risk_score" in r
        assert "probability_default" in r
        assert "risk_category" in r
        assert "recommendation" in r

    def test_risk_score_in_range(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        score = r["risk_score"]
        assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_probability_in_range(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        prob = r["probability_default"]
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    def test_risk_category_valid(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        assert r["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_recommendation_valid(self):
        r = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        assert r["recommendation"] in ["Approve", "Review with Conditions", "Reject"]

    def test_prediction_consistency(self):
        """Same input should always give same output."""
        r1 = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        r2 = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD).json()
        assert r1["risk_score"] == r2["risk_score"]

    def test_missing_field_returns_422(self):
        bad_payload = SAMPLE_PAYLOAD.copy()
        del bad_payload["credit_amount"]
        r = requests.post(f"{API_URL}/predict", json=bad_payload)
        assert r.status_code == 422

    def test_invalid_age_returns_422(self):
        bad_payload = SAMPLE_PAYLOAD.copy()
        bad_payload["age"] = -5  # Invalid age
        r = requests.post(f"{API_URL}/predict", json=bad_payload)
        assert r.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        r = requests.get(f"{API_URL}/metrics")
        assert r.status_code == 200

    def test_metrics_has_total(self):
        r = requests.get(f"{API_URL}/metrics").json()
        assert "total_predictions" in r


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        r = requests.get(f"{API_URL}/model-info")
        assert r.status_code == 200

    def test_model_info_structure(self):
        r = requests.get(f"{API_URL}/model-info").json()
        assert "model_name" in r
        assert "model_version" in r
