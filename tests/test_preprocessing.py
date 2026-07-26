"""
tests/test_preprocessing.py
─────────────────────────────
Unit tests for the feature engineering and preprocessing pipeline.
Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_pipeline"))
from preprocessing import (
    feature_engineering,
    preprocess_data,
    sanitize_column_names,
    transform_input_for_inference,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_row():
    """A minimal German credit-style row."""
    return {
        "checking_status": "0<=X<200",
        "duration": 24,
        "credit_history": "existing paid",
        "purpose": "radio/tv",
        "credit_amount": 5000.0,
        "savings_status": "<100",
        "employment": "1<=X<4",
        "installment_commitment": 2,
        "personal_status": "male single",
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
        "foreign_worker": "yes",
    }


@pytest.fixture
def sample_df(sample_row):
    rows = [sample_row.copy() for _ in range(10)]
    # Vary some values
    for i, row in enumerate(rows):
        row["credit_amount"] = 1000 + i * 500
        row["duration"] = 12 + i * 3
        row["age"] = 22 + i * 2
        row["target"] = i % 2  # alternating 0/1
    return pd.DataFrame(rows)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestSanitizeColumnNames:
    def test_removes_brackets(self):
        df = pd.DataFrame({"a[1]": [1], "b<2>": [2], "c<=3": [3]})
        sanitized = sanitize_column_names(df)
        for col in sanitized.columns:
            assert "[" not in col and "]" not in col and "<" not in col and ">" not in col

    def test_preserves_safe_names(self):
        df = pd.DataFrame({"age": [1], "credit_amount": [2]})
        result = sanitize_column_names(df)
        assert list(result.columns) == ["age", "credit_amount"]


class TestFeatureEngineering:
    def test_emi_burden_created(self, sample_df):
        result = feature_engineering(sample_df)
        assert "emi_burden_proxy" in result.columns

    def test_emi_burden_values(self, sample_df):
        result = feature_engineering(sample_df)
        expected = (
            sample_df["credit_amount"]
            * sample_df["installment_commitment"]
            / sample_df["duration"].clip(lower=1)
        )
        pd.testing.assert_series_equal(result["emi_burden_proxy"].reset_index(drop=True),
                                        expected.reset_index(drop=True), check_names=False)

    def test_credit_age_score_created(self, sample_df):
        result = feature_engineering(sample_df)
        assert "credit_age_score" in result.columns

    def test_payment_reliability_score(self, sample_df):
        result = feature_engineering(sample_df)
        assert "payment_reliability_score" in result.columns
        # "existing paid" → 3
        assert (result["payment_reliability_score"] == 3).all()

    def test_payment_reliability_uses_observed_dataset_order(self, sample_df):
        sample_df.loc[0, "credit_history"] = "critical/other existing credit"
        sample_df.loc[1, "credit_history"] = "no credits/all paid"
        result = feature_engineering(sample_df)
        assert result.loc[0, "payment_reliability_score"] == 5
        assert result.loc[1, "payment_reliability_score"] == 1

    def test_employment_stability_score(self, sample_df):
        result = feature_engineering(sample_df)
        assert "employment_stability_score" in result.columns
        # "1<=X<4" → 2
        assert (result["employment_stability_score"] == 2).all()

    def test_savings_to_loan_ratio(self, sample_df):
        result = feature_engineering(sample_df)
        assert "savings_to_loan_ratio" in result.columns
        assert (result["savings_to_loan_ratio"] >= 0).all()

    def test_no_target_column_needed(self, sample_df):
        """Feature engineering should work without 'target' column (inference mode)."""
        df_no_target = sample_df.drop(columns=["target"])
        result = feature_engineering(df_no_target)
        assert "emi_burden_proxy" in result.columns

    def test_no_nans_introduced(self, sample_df):
        result = feature_engineering(sample_df)
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert result[num_cols].isnull().sum().sum() == 0


class TestPreprocessData:
    def test_returns_encoded_dataframe(self, sample_df):
        X, y = preprocess_data(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_object_columns_remain(self, sample_df):
        X, y = preprocess_data(sample_df)
        obj_cols = X.select_dtypes(include=["object", "category"]).columns
        assert len(obj_cols) == 0, f"Object columns remain: {list(obj_cols)}"

    def test_target_not_in_X(self, sample_df):
        X, y = preprocess_data(sample_df)
        assert "target" not in X.columns

    def test_sensitive_historical_columns_are_excluded(self, sample_df):
        X, _ = preprocess_data(sample_df)
        assert not any(col.startswith("personal_status") for col in X.columns)
        assert not any(col.startswith("foreign_worker") for col in X.columns)

    def test_y_binary(self, sample_df):
        _, y = preprocess_data(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_removes_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:3]], ignore_index=True)
        X, y = preprocess_data(df_dup)
        assert len(X) <= len(df_dup)

    def test_column_names_valid_for_xgboost(self, sample_df):
        import re
        X, _ = preprocess_data(sample_df)
        for col in X.columns:
            assert not re.search(r"[\[\]<>]", col), f"Invalid column name: {col}"


class TestInferenceTransform:
    def test_single_row_categorical_value_is_preserved(self, tmp_path):
        feature_cols = ["checking_status_no checking"]
        scaler = StandardScaler().fit(
            pd.DataFrame({"checking_status_no checking": [0.0, 1.0]})
        )
        scaler_path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

        baseline = transform_input_for_inference(
            {"checking_status": "<0"}, feature_cols, str(scaler_path)
        )
        non_baseline = transform_input_for_inference(
            {"checking_status": "no checking"}, feature_cols, str(scaler_path)
        )

        assert baseline.iloc[0, 0] != non_baseline.iloc[0, 0]
        assert baseline.iloc[0, 0] < non_baseline.iloc[0, 0]

    def test_missing_scaler_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transform_input_for_inference(
                {"checking_status": "<0"},
                ["checking_status_no checking"],
                str(tmp_path / "missing.joblib"),
            )
