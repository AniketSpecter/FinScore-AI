"""
ml_pipeline/preprocessing.py
──────────────────────────────
Handles data loading, cleaning, feature engineering, encoding and scaling.
Designed to be importable both from training scripts and the FastAPI backend.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import re


# Excluded from model training because these historical fields directly encode
# sensitive personal attributes and are not needed to operate the demo.
MODEL_EXCLUDED_COLUMNS = {"personal_status", "foreign_worker"}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace characters invalid for XGBoost feature names ([], <, >) with underscores."""
    df = df.copy()
    df.columns = [re.sub(r"[\[\]<>]", "_", col).strip() for col in df.columns]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features from raw dataset columns.
    Works even if the 'target' column is absent (inference mode).
    """
    df = df.copy()

    # 1. EMI Burden Proxy — Monthly principal adjusted by installment-rate band
    if all(c in df.columns for c in ["credit_amount", "duration", "installment_commitment"]):
        df["emi_burden_proxy"] = (
            df["credit_amount"]
            * df["installment_commitment"]
            / df["duration"].clip(lower=1)
        )

    # 2. Loan to Duration Ratio — Higher ratio = more aggressive borrowing
    if "credit_amount" in df.columns and "duration" in df.columns:
        df["loan_to_duration"] = df["credit_amount"] / df["duration"].clip(lower=1)

    # 3. Credit Age Score — Binned from applicant age (proxy for credit history length)
    if "age" in df.columns:
        # Use fixed bins so this works for single-row inference too
        df["credit_age_score"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 200],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(float)

    # 4. Payment Reliability Score — Ordinal encoding of credit history quality
    if "credit_history" in df.columns:
        # Ordering follows the observed default rates in OpenML credit-g v1.
        # The historical labels are counterintuitive: "critical/other existing
        # credit" has the lowest observed default rate, while "no credits/all
        # paid" has the highest.
        history_mapping = {
            "no credits/all paid": 1,
            "all paid": 2,
            "existing paid": 3,
            "delayed previously": 3,
            "critical/other existing credit": 5,
        }
        df["payment_reliability_score"] = (
            df["credit_history"].map(history_mapping).fillna(3)
        )

    # 5. Employment Stability Score — Ordinal from employment duration
    if "employment" in df.columns:
        emp_mapping = {
            "unemployed": 0,
            "<1": 1,
            "1<=X<4": 2,
            "4<=X<7": 3,
            ">=7": 4,
        }
        df["employment_stability_score"] = (
            df["employment"].map(emp_mapping).fillna(1)
        )

    # 6. Savings to Loan Ratio — Proxy from savings bucket / credit_amount
    if "savings_status" in df.columns and "credit_amount" in df.columns:
        savings_midpoints = {
            "<100": 50,
            "100<=X<500": 300,
            "500<=X<1000": 750,
            ">=1000": 1500,
            "no known savings": 0,
        }
        df["savings_midpoint"] = df["savings_status"].map(savings_midpoints).fillna(0)
        df["savings_to_loan_ratio"] = df["savings_midpoint"] / df["credit_amount"].clip(lower=1)
        df.drop(columns=["savings_midpoint"], inplace=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame):
    """
    Full preprocessing: dedup, missing value fill, feature engineering, OHE.
    Returns (X_encoded, y) where y may be None in inference mode.
    """
    df = df.drop_duplicates().copy()
    df.drop(columns=list(MODEL_EXCLUDED_COLUMNS), errors="ignore", inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            modes = df[col].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "missing"
            df[col] = df[col].fillna(fill_value)
        else:
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)

    # Feature Engineering
    df = feature_engineering(df)

    # Separate target
    if "target" in df.columns:
        y = df["target"].astype(int)
        X = df.drop(columns=["target"])
    else:
        y = None
        X = df.copy()

    # One-hot encode all remaining categoricals
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Sanitize column names (removes [, ], < which break XGBoost)
    X_encoded = sanitize_column_names(X_encoded)

    return X_encoded, y


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, save_path: str = "../models/scaler.joblib"):
    """
    Fits a StandardScaler on X_train, transforms both splits, saves the scaler.
    Returns (X_train_scaled, X_test_scaled) as DataFrames preserving column names.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved -> {save_path}")

    return X_train_scaled, X_test_scaled


def get_train_test_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_save_path: str = "../models/scaler.joblib",
):
    """
    End-to-end pipeline: load → preprocess → split → scale.
    Returns (X_train, X_test, y_train, y_test, feature_columns_list).
    """
    df = load_data(filepath)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, save_path=scaler_save_path)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Inference Transform (used by backend)
# ──────────────────────────────────────────────────────────────────────────────

def transform_input_for_inference(
    input_dict: dict,
    feature_cols: list,
    scaler_path: str = "../models/scaler.joblib",
) -> pd.DataFrame:
    """
    Takes a raw dict (from API input), applies the same preprocessing as training,
    aligns columns to training feature_cols, then applies the saved scaler.

    Training uses ``drop_first=True`` after seeing the whole dataset. At inference
    time there is only one row, so using ``drop_first=True`` would drop the one
    observed value for *every* categorical feature. Encoding all observed values
    here and then aligning to the saved training columns preserves non-baseline
    categories while naturally discarding training baseline columns.
    """
    if not feature_cols:
        raise ValueError("Training feature columns are unavailable.")

    df = pd.DataFrame([input_dict])
    df.drop(columns=list(MODEL_EXCLUDED_COLUMNS), errors="ignore", inplace=True)
    df_engineered = feature_engineering(df)

    categorical_cols = df_engineered.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df_engineered, columns=categorical_cols, drop_first=False)
    df_encoded = sanitize_column_names(df_encoded)

    # Align to training columns (add missing as 0, drop extras)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_final = df_encoded[feature_cols]

    # Scaling is part of the model contract. Never silently send unscaled values
    # to a model trained on scaled data.
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler artefact not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    df_final = pd.DataFrame(
        scaler.transform(df_final), columns=df_final.columns
    )

    return df_final


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cols = get_train_test_data(
        "../data/german_credit.csv",
        scaler_save_path="../models/scaler.joblib",
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features ({len(cols)}): {cols[:5]} ...")
