"""Unit tests for transparent decision-support calculations."""

from backend.main import (
    assess_affordability,
    calculate_monthly_payment,
    classify_risk,
    compute_risk_score,
)


def test_probability_to_safety_score():
    assert compute_risk_score(0.25) == 75.0
    assert compute_risk_score(0.5) == 50.0


def test_evidence_aligned_score_bands():
    assert classify_risk(75) == ("Low Risk", "Approve")
    assert classify_risk(74.99) == ("Medium Risk", "Review with Conditions")
    assert classify_risk(50) == ("Medium Risk", "Review with Conditions")
    assert classify_risk(49.99) == ("High Risk", "Reject")


def test_zero_interest_payment():
    assert calculate_monthly_payment(1200, 0, 12) == 100.0


def test_amortized_payment_and_affordability():
    payment = calculate_monthly_payment(5000, 12, 24)
    assert payment == 235.37
    ratio, status = assess_affordability(payment, 1000, 100)
    assert ratio == 33.54
    assert status == "Comfortable"


def test_affordability_can_be_omitted():
    assert assess_affordability(200, None, 0) == (None, "Not assessed")
