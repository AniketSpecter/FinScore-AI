"""
backend/schemas.py
────────────────────
Pydantic v2 models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal


CheckingStatus = Literal["<0", "0<=X<200", ">=200", "no checking"]
CreditHistory = Literal[
    "no credits/all paid", "all paid", "existing paid",
    "delayed previously", "critical/other existing credit",
]
LoanPurpose = Literal[
    "new car", "used car", "furniture/equipment", "radio/tv",
    "domestic appliance", "repairs", "education", "retraining", "business", "other",
]
SavingsStatus = Literal["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"]
Employment = Literal["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
OtherParties = Literal["none", "co applicant", "guarantor"]
PropertyMagnitude = Literal["real estate", "life insurance", "car", "no known property"]
OtherPaymentPlans = Literal["none", "bank", "stores"]
Housing = Literal["rent", "own", "for free"]
Job = Literal["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"]


# ─── Loan Application Input ──────────────────────────────────────────────────

class LoanApplicationInput(BaseModel):
    """Supported model fields plus optional affordability inputs."""

    checking_status: CheckingStatus = Field(
        ...,
        description="Status of existing checking account",
        examples=["<0", "0<=X<200", ">=200", "no checking"],
    )
    duration: int = Field(..., description="Loan duration in months", ge=4, le=72, examples=[24])
    credit_history: CreditHistory = Field(
        ...,
        description="Past credit behaviour",
        examples=["existing paid", "no credits/all paid"],
    )
    purpose: LoanPurpose = Field(..., description="Loan purpose", examples=["new car", "education"])
    credit_amount: float = Field(
        ...,
        description="Loan amount in historical model units (DM); restricted to the trained range",
        ge=250,
        le=18_424,
        examples=[5000.0],
    )
    savings_status: SavingsStatus = Field(
        ...,
        description="Savings account / bond status",
        examples=["<100", ">=1000", "no known savings"],
    )
    employment: Employment = Field(
        ...,
        description="Years at present employer",
        examples=["<1", ">=7", "unemployed"],
    )
    installment_commitment: int = Field(
        ...,
        description="Historical installment-burden category (1=lowest, 4=highest)",
        ge=1,
        le=4,
        examples=[2],
    )
    other_parties: OtherParties = Field(
        ..., description="Guarantors or co-applicants", examples=["none", "guarantor"]
    )
    residence_since: int = Field(
        ..., description="Years at present residence", ge=1, le=4, examples=[2]
    )
    property_magnitude: PropertyMagnitude = Field(
        ...,
        description="Most valuable property owned",
        examples=["real estate", "no known property"],
    )
    age: int = Field(..., description="Applicant age in years", ge=19, le=75, examples=[35])
    other_payment_plans: OtherPaymentPlans = Field(
        ..., description="Other active payment plans", examples=["none", "bank"]
    )
    housing: Housing = Field(..., description="Housing status", examples=["own", "rent"])
    existing_credits: int = Field(
        ..., description="Number of existing credits at this bank", ge=1, le=4, examples=[1]
    )
    job: Job = Field(
        ...,
        description="Employment category",
        examples=["skilled", "high qualif/self emp/mgmt"],
    )
    num_dependents: int = Field(
        ..., description="Number of dependants", ge=1, le=2, examples=[1]
    )
    own_telephone: Literal["yes", "none"] = Field(
        ..., description="Does applicant have a telephone?", examples=["yes", "none"]
    )
    annual_interest_rate: float = Field(
        10.0,
        description="Annual interest rate used only for the payment estimate",
        ge=0,
        le=50,
        examples=[10.0],
    )
    monthly_income: Optional[float] = Field(
        None,
        description="Optional monthly income in the same units as the loan; used for affordability only",
        gt=0,
        le=10_000_000,
        examples=[1500.0],
    )
    existing_monthly_obligations: float = Field(
        0.0,
        description="Existing monthly debt payments, used only for affordability",
        ge=0,
        le=10_000_000,
        examples=[100.0],
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "checking_status": "0<=X<200",
                "duration": 24,
                "credit_history": "existing paid",
                "purpose": "radio/tv",
                "credit_amount": 4000.0,
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
                "existing_monthly_obligations": 100.0,
            }
        }
    }


# ─── Prediction Response ──────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Full response returned by POST /predict."""

    prediction_id: int = Field(..., description="Stored prediction reference")
    model_version: str = Field(..., description="Model version used for this result")
    probability_default: float = Field(..., description="Probability the applicant defaults [0-1]")
    risk_score: float = Field(..., description="Credit safety score [0-100], higher = safer")
    risk_category: str = Field(..., description="Low Risk | Medium Risk | High Risk")
    recommendation: str = Field(..., description="Approve | Review with Conditions | Reject")
    estimated_monthly_payment: float = Field(..., description="Amortized payment estimate")
    affordability_ratio: Optional[float] = Field(
        None, description="Payment plus obligations as a percentage of supplied monthly income"
    )
    affordability_status: str = Field(
        ..., description="Not assessed | Comfortable | Manageable | High burden"
    )
    score_interpretation: str = Field(..., description="Plain-language score explanation")
    input_warnings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP feature impacts")
    positive_factors: Optional[List[str]] = Field(None, description="Features reducing default risk")
    negative_factors: Optional[List[str]] = Field(None, description="Features increasing default risk")


# ─── Model Info ───────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    active: bool
    deployed_at: str
    metrics: Dict[str, Any]
