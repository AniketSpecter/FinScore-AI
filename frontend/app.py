"""
frontend/app.py
─────────────────
FinScore AI — Streamlit Dashboard
Multi-page interactive dashboard with:
  - Loan Application & Prediction
  - Analytics Charts
  - Historical Predictions
  - System Info
"""

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ─── Configuration ────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://127.0.0.1:3022")
CLOUD_MODE = os.getenv("FINSCORE_DEPLOYMENT_MODE", "local").lower() == "cloud"

FRIENDLY_OPTIONS = {
    "checking_status": {
        "<0": "Negative balance",
        "0<=X<200": "Balance from 0 to 199",
        ">=200": "Balance of 200 or more",
        "no checking": "No checking-account balance recorded",
    },
    "savings_status": {
        "<100": "Less than 100",
        "100<=X<500": "100 to 499",
        "500<=X<1000": "500 to 999",
        ">=1000": "1,000 or more",
        "no known savings": "No savings balance recorded",
    },
    "employment": {
        "unemployed": "Currently unemployed",
        "<1": "Less than 1 year",
        "1<=X<4": "1 to 3 years",
        "4<=X<7": "4 to 6 years",
        ">=7": "7 years or more",
    },
    "credit_history": {
        "no credits/all paid": "No established credit / all credit closed",
        "all paid": "All credit at this bank repaid",
        "existing paid": "Current credit paid on time",
        "delayed previously": "Past payment delay",
        "critical/other existing credit": "Established credit with other active accounts",
    },
    "purpose": {
        "new car": "New vehicle",
        "used car": "Used vehicle",
        "furniture/equipment": "Furniture or equipment",
        "radio/tv": "Electronics",
        "domestic appliance": "Home appliance",
        "repairs": "Repairs",
        "education": "Education",
        "retraining": "Professional training",
        "business": "Business",
        "other": "Other purpose",
    },
    "job": {
        "unemp/unskilled non res": "Unemployed / non-resident unskilled",
        "unskilled resident": "Resident unskilled",
        "skilled": "Skilled employee",
        "high qualif/self emp/mgmt": "Highly qualified / self-employed / management",
    },
    "other_parties": {
        "none": "Applicant only",
        "co applicant": "Co-applicant",
        "guarantor": "Guarantor",
    },
    "property_magnitude": {
        "real estate": "Real estate",
        "life insurance": "Life insurance / savings agreement",
        "car": "Vehicle or other property",
        "no known property": "No property recorded",
    },
    "other_payment_plans": {
        "none": "No other payment plan",
        "bank": "Payment plan with a bank",
        "stores": "Payment plan with a store",
    },
    "housing": {"rent": "Renting", "own": "Own home", "for free": "Living without rent"},
    "own_telephone": {"none": "No registered telephone", "yes": "Registered telephone"},
}

PROFILE_PRESETS = {
    "Balanced example": {
        "credit_amount": 4000.0, "duration": 24, "annual_interest_rate": 10.0,
        "monthly_income": 1500.0, "existing_monthly_obligations": 100.0, "age": 35,
        "installment_commitment": 2, "checking_status": "0<=X<200",
        "savings_status": "100<=X<500", "employment": "1<=X<4",
        "credit_history": "existing paid", "purpose": "radio/tv", "job": "skilled",
        "housing": "own", "property_magnitude": "real estate", "other_parties": "none",
        "residence_since": 3, "other_payment_plans": "none", "existing_credits": 1,
        "num_dependents": 1, "own_telephone": "yes",
    },
    "Strong example": {
        "credit_amount": 2000.0, "duration": 12, "annual_interest_rate": 8.0,
        "monthly_income": 1800.0, "existing_monthly_obligations": 0.0, "age": 45,
        "installment_commitment": 1, "checking_status": ">=200",
        "savings_status": ">=1000", "employment": ">=7",
        "credit_history": "critical/other existing credit", "purpose": "used car",
        "job": "high qualif/self emp/mgmt", "housing": "own",
        "property_magnitude": "real estate", "other_parties": "none", "residence_since": 4,
        "other_payment_plans": "none", "existing_credits": 1, "num_dependents": 1,
        "own_telephone": "yes",
    },
    "Higher-risk example": {
        "credit_amount": 15000.0, "duration": 48, "annual_interest_rate": 15.0,
        "monthly_income": 900.0, "existing_monthly_obligations": 250.0, "age": 24,
        "installment_commitment": 4, "checking_status": "<0", "savings_status": "<100",
        "employment": "<1", "credit_history": "delayed previously", "purpose": "business",
        "job": "unskilled resident", "housing": "rent", "property_magnitude": "no known property",
        "other_parties": "co applicant", "residence_since": 1, "other_payment_plans": "bank",
        "existing_credits": 3, "num_dependents": 2, "own_telephone": "none",
    },
}

st.set_page_config(
    page_title="FinScore AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0ff !important; }

    /* Header strip */
    .header-banner {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 24px;
        color: white;
    }
    .header-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .header-banner p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.85; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e3f, #2d2d6b);
        border: 1px solid #444480;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 8px;
    }
    .kpi-card .label { font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card .value { font-size: 2rem; font-weight: 700; margin: 4px 0; }

    /* Risk badges */
    .badge-low    { background:#1a4a1a; color:#4ade80; border:1px solid #4ade80; border-radius:6px; padding:4px 12px; font-weight:600; }
    .badge-medium { background:#4a3a00; color:#facc15; border:1px solid #facc15; border-radius:6px; padding:4px 12px; font-weight:600; }
    .badge-high   { background:#4a1a1a; color:#f87171; border:1px solid #f87171; border-radius:6px; padding:4px 12px; font-weight:600; }

    /* Factor boxes */
    .factor-box { border-radius: 8px; padding: 12px 16px; margin: 4px 0; }
    .factor-positive { background: #0d2a1a; border-left: 4px solid #4ade80; color: #a7f3d0; }
    .factor-negative { background: #2a0d0d; border-left: 4px solid #f87171; color: #fecaca; }

    /* Subtle section dividers */
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #a78bfa;
        border-bottom: 1px solid #333355; padding-bottom: 6px; margin: 16px 0 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        return None, f"API Error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def friendly_feature_name(feature: str) -> str:
    exact_labels = {
        "credit_amount": "Loan amount",
        "duration": "Loan duration",
        "age": "Applicant age",
        "installment_commitment": "Installment burden level",
        "residence_since": "Residence stability",
        "existing_credits": "Existing credits",
        "num_dependents": "Dependants",
        "emi_burden_proxy": "Estimated payment burden",
        "loan_to_duration": "Loan amount per month",
        "credit_age_score": "Credit-age band",
        "payment_reliability_score": "Payment reliability",
        "employment_stability_score": "Employment stability",
        "savings_to_loan_ratio": "Savings compared with loan",
    }
    if feature in exact_labels:
        return exact_labels[feature]

    prefix_labels = {
        "checking_status_": "Checking account",
        "credit_history_": "Credit history",
        "purpose_": "Loan purpose",
        "savings_status_": "Savings profile",
        "employment_": "Employment duration",
        "other_parties_": "Applicant support",
        "property_magnitude_": "Property profile",
        "other_payment_plans_": "Other payment plans",
        "housing_": "Housing",
        "job_": "Job category",
        "own_telephone_": "Telephone record",
    }
    for prefix, label in prefix_labels.items():
        if feature.startswith(prefix):
            return label
    return feature.replace("_", " ").title()


def apply_profile(profile_name: str):
    for field, value in PROFILE_PRESETS[profile_name].items():
        st.session_state[f"loan_{field}"] = value


def select_friendly(container, field: str, label: str, help_text: str = ""):
    options = list(FRIENDLY_OPTIONS[field])
    return container.selectbox(
        label,
        options,
        format_func=lambda value: FRIENDLY_OPTIONS[field][value],
        key=f"loan_{field}",
        help=help_text or None,
    )


def render_gauge(score: float) -> go.Figure:
    color = "#4ade80" if score >= 75 else "#facc15" if score >= 50 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 75, "valueformat": ".1f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Credit Safety Score<br><span style='font-size:0.8em;color:#aaa'>Higher = safer</span>", "font": {"size": 16}},
        number={"font": {"size": 48, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e1e3f",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "#2a0d0d"},
                {"range": [50, 75], "color": "#2a2200"},
                {"range": [75, 100],"color": "#0d2a1a"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#111122",
        font={"color": "#e0e0ff"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def render_shap_bar(shap_vals: dict) -> go.Figure:
    sorted_items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    features = [friendly_feature_name(i[0])[:36] for i in sorted_items]
    values   = [i[1] for i in sorted_items]
    colors   = ["#f87171" if v > 0 else "#4ade80" for v in values]  # red = more risk, green = less risk

    fig = go.Figure(go.Bar(
        x=values[::-1], y=features[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Impact on Default Probability<br><sup>Red = increases risk | Green = reduces risk</sup>",
        paper_bgcolor="#111122",
        plot_bgcolor="#111122",
        font={"color": "#e0e0ff"},
        height=400,
        margin=dict(l=10, r=80, t=60, b=10),
        xaxis=dict(title="SHAP Value", gridcolor="#222244"),
        yaxis=dict(gridcolor="#222244"),
    )
    return fig


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💳 FinScore AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Risk Scoring", "📊 Analytics", "📋 History", "ℹ️ System Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    health = api_get("/health")
    if health and health.get("ready"):
        st.success("🟢 API Online")
        if health.get("model_loaded"):
            st.caption("✅ Model loaded")
        if health.get("explainer_ready"):
            st.caption("✅ SHAP explainer ready")
    elif health:
        st.warning("🟡 API Degraded")
        st.caption("Check System Info for missing database or model artefacts.")
    else:
        st.error("🔴 API Offline")
        st.caption(f"Connecting to: {API_URL}")


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
  <h1>💳 FinScore AI</h1>
  <p>AI-Powered Financial Risk Scoring & Loan Approval Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Risk Scoring
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Risk Scoring":
    st.markdown('<div class="section-title">Easy Loan Assessment</div>', unsafe_allow_html=True)
    st.caption(
        "Start with an example, then change only what you know. Monetary values use the historical "
        "dataset's model units (DM); the supported loan range is 250–18,424."
    )

    if "loan_profile_initialized" not in st.session_state:
        st.session_state["loan_profile_choice"] = "Balanced example"
        apply_profile("Balanced example")
        st.session_state["loan_include_affordability"] = True
        st.session_state["loan_profile_initialized"] = True

    def on_profile_change():
        apply_profile(st.session_state["loan_profile_choice"])

    st.selectbox(
        "Quick-start profile",
        list(PROFILE_PRESETS),
        key="loan_profile_choice",
        on_change=on_profile_change,
        help="These are demonstrative profiles. Select one and adjust the fields below.",
    )

    with st.form("loan_form"):
        st.markdown("#### 1. Loan details")
        c1, c2, c3 = st.columns(3)
        credit_amount = c1.number_input(
            "Loan amount",
            min_value=250.0,
            max_value=18424.0,
            step=100.0,
            key="loan_credit_amount",
            help="The model was trained only on amounts from 250 to 18,424. Out-of-range values are not scored.",
        )
        duration = c2.number_input(
            "Repayment term (months)", min_value=4, max_value=72, step=1,
            key="loan_duration", help="Training range: 4 to 72 months."
        )
        purpose = select_friendly(c3, "purpose", "What is the loan for?")

        c4, c5, c6 = st.columns(3)
        annual_interest_rate = c4.number_input(
            "Annual interest rate (%)", min_value=0.0, max_value=50.0, step=0.25,
            key="loan_annual_interest_rate",
            help="Used for the payment estimate only; it does not alter the ML default probability.",
        )
        age = c5.number_input(
            "Applicant age", min_value=19, max_value=75, step=1, key="loan_age",
            help="The model does not extrapolate outside its trained age range of 19–75."
        )
        installment_commitment = c6.select_slider(
            "Installment burden level",
            options=[1, 2, 3, 4],
            format_func=lambda value: {1: "1 — Low", 2: "2 — Moderate", 3: "3 — Elevated", 4: "4 — High"}[value],
            key="loan_installment_commitment",
            help="This is a historical 1–4 category, not a literal percentage.",
        )

        st.markdown("#### 2. Optional affordability check")
        include_affordability = st.checkbox(
            "Include income and existing monthly payments",
            key="loan_include_affordability",
            help="These values calculate payment burden. They are not fed into the historical ML model.",
        )
        a1, a2 = st.columns(2)
        monthly_income = a1.number_input(
            "Monthly income", min_value=1.0, max_value=10_000_000.0, step=100.0,
            key="loan_monthly_income", disabled=not include_affordability,
            help="Use the same monetary units as the loan amount.",
        )
        existing_monthly_obligations = a2.number_input(
            "Other monthly debt payments", min_value=0.0, max_value=10_000_000.0, step=50.0,
            key="loan_existing_monthly_obligations", disabled=not include_affordability,
        )

        st.markdown("#### 3. Financial profile")
        f1, f2, f3 = st.columns(3)
        checking_status = select_friendly(
            f1, "checking_status", "Checking-account position",
            "Select the historical balance band that best matches the applicant."
        )
        savings_status = select_friendly(f2, "savings_status", "Savings balance")
        employment = select_friendly(f3, "employment", "Time in current employment")

        f4, f5, f6 = st.columns(3)
        credit_history = select_friendly(
            f4, "credit_history", "Credit repayment history",
            "Labels are translated from the historical dataset; choose the closest description."
        )
        job = select_friendly(f5, "job", "Employment type")
        housing = select_friendly(f6, "housing", "Housing situation")

        with st.expander("Additional credit details", expanded=False):
            x1, x2, x3 = st.columns(3)
            property_magnitude = select_friendly(x1, "property_magnitude", "Main property or asset")
            other_parties = select_friendly(x2, "other_parties", "Application support")
            other_payment_plans = select_friendly(x3, "other_payment_plans", "Other payment plans")

            x4, x5, x6, x7 = st.columns(4)
            residence_since = x4.slider(
                "Years at residence", 1, 4, key="loan_residence_since",
                help="Historical dataset band: 4 means four years or more."
            )
            existing_credits = x5.number_input(
                "Existing credits", min_value=1, max_value=4, step=1, key="loan_existing_credits"
            )
            num_dependents = x6.slider("Dependants", 1, 2, key="loan_num_dependents")
            own_telephone = select_friendly(x7, "own_telephone", "Telephone record")

        submitted = st.form_submit_button(
            "Calculate Credit Safety", use_container_width=True, type="primary"
        )

    if submitted:
        payload = dict(
            checking_status=checking_status,
            duration=int(duration),
            credit_history=credit_history,
            purpose=purpose,
            credit_amount=float(credit_amount),
            savings_status=savings_status,
            employment=employment,
            installment_commitment=int(installment_commitment),
            other_parties=other_parties,
            residence_since=int(residence_since),
            property_magnitude=property_magnitude,
            age=int(age),
            other_payment_plans=other_payment_plans,
            housing=housing,
            existing_credits=int(existing_credits),
            job=job,
            num_dependents=int(num_dependents),
            own_telephone=own_telephone,
            annual_interest_rate=float(annual_interest_rate),
            monthly_income=float(monthly_income) if include_affordability else None,
            existing_monthly_obligations=(
                float(existing_monthly_obligations) if include_affordability else 0.0
            ),
        )

        with st.spinner("Scoring loan application..."):
            result, error = api_post("/predict", payload)

        if error:
            st.error(f"**Prediction failed:** {error}")
        else:
            st.markdown("---")
            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

            score    = result["risk_score"]
            risk_cat = result["risk_category"]
            rec      = result["recommendation"]
            prob     = result["probability_default"]

            for warning in result.get("input_warnings", []):
                st.warning(warning)

            st.caption(f"Assessment reference #{result.get('prediction_id')} · Model {result.get('model_version')}")

            # Badge
            badge_class = "badge-low" if score >= 75 else "badge-medium" if score >= 50 else "badge-high"
            rec_icon    = "✅" if rec == "Approve" else "⚠️" if "Conditions" in rec else "❌"

            col_a, col_b, col_c = st.columns([1.2, 1, 1])

            with col_a:
                st.plotly_chart(render_gauge(score), use_container_width=True)

            with col_b:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">Risk Category</div>
                    <div class="value"><span class="{badge_class}">{risk_cat}</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">Recommendation</div>
                    <div class="value">{rec_icon} {rec}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">Default Probability</div>
                    <div class="value">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Estimated Monthly Payment", f"{result['estimated_monthly_payment']:,.2f}")
                ratio = result.get("affordability_ratio")
                ratio_text = f"{ratio:.1f}%" if ratio is not None else "Not supplied"
                st.metric("Total Payment Burden", ratio_text, result.get("affordability_status"))

            with col_c:
                st.markdown("**What you can review**")
                actions = result.get("recommended_actions", [])
                if actions:
                    for action in actions:
                        st.markdown(f"- {action}")
                else:
                    st.success("No major actionable warning was identified for this profile.")

            st.info(result["score_interpretation"])
            st.caption(
                "Score guide: 75–100 = lower model risk · 50–74.99 = moderate model risk · "
                "0–49.99 = higher model risk. Affordability above 50% prevents automatic approval."
            )

            st.markdown("**Model factors**")
            pos = result.get("positive_factors", [])
            neg = result.get("negative_factors", [])
            factor_left, factor_right = st.columns(2)
            with factor_left:
                st.caption("Risk-reducing signals")
                pos = result.get("positive_factors", [])
                for f in (pos or [])[:3]:
                    name = friendly_feature_name(f)
                    st.markdown(f'<div class="factor-box factor-positive">✅ {name}</div>', unsafe_allow_html=True)
            with factor_right:
                st.caption("Risk-increasing signals")
                for f in (neg or [])[:3]:
                    name = friendly_feature_name(f)
                    st.markdown(f'<div class="factor-box factor-negative">⚠️ {name}</div>', unsafe_allow_html=True)

            # SHAP bar chart
            if result.get("shap_values"):
                st.markdown('<div class="section-title">Explainable AI — SHAP Feature Importance</div>', unsafe_allow_html=True)
                st.plotly_chart(render_shap_bar(result["shap_values"]), use_container_width=True)
            else:
                st.info("SHAP explanations are not available for this prediction.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Analytics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown('<div class="section-title">Platform Analytics</div>', unsafe_allow_html=True)

    metrics = api_get("/metrics")

    if not metrics or metrics.get("total_predictions", 0) == 0:
        st.info("No predictions have been recorded yet. Submit a loan application first.")
    else:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predictions", metrics["total_predictions"])
        k2.metric("Avg Safety Score",   f"{metrics['avg_risk_score']:.1f}")
        k3.metric("Avg Default Prob",   f"{metrics['avg_default_prob']:.1%}")

        risk_dist = metrics.get("risk_distribution", {})
        k4.metric("High Risk Count",    risk_dist.get("High Risk", 0))

        st.markdown("---")
        col_left, col_right = st.columns(2)

        # Risk distribution pie
        with col_left:
            if risk_dist:
                colors = {"Low Risk": "#4ade80", "Medium Risk": "#facc15", "High Risk": "#f87171"}
                fig_pie = go.Figure(go.Pie(
                    labels=list(risk_dist.keys()),
                    values=list(risk_dist.values()),
                    marker_colors=[colors.get(k, "#888") for k in risk_dist.keys()],
                    hole=0.45,
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(
                    title="Risk Distribution",
                    paper_bgcolor="#111122",
                    font={"color": "#e0e0ff"},
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(bgcolor="#1e1e3f"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Recommendation distribution bar
        with col_right:
            rec_dist = metrics.get("recommendation_dist", {})
            if rec_dist:
                rec_colors = {
                    "Approve": "#4ade80",
                    "Review with Conditions": "#facc15",
                    "Approve with Conditions": "#facc15",
                    "Reject": "#f87171",
                }
                fig_bar = go.Figure(go.Bar(
                    x=list(rec_dist.keys()),
                    y=list(rec_dist.values()),
                    marker_color=[rec_colors.get(k, "#888") for k in rec_dist.keys()],
                    text=list(rec_dist.values()),
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    title="Loan Approval Distribution",
                    paper_bgcolor="#111122",
                    plot_bgcolor="#111122",
                    font={"color": "#e0e0ff"},
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis=dict(gridcolor="#222244"),
                    yaxis=dict(gridcolor="#222244"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── Dataset Statistics ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dataset Insights</div>', unsafe_allow_html=True)
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "german_credit.csv"))

        col1, col2 = st.columns(2)

        with col1:
            target_counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
            fig_t = go.Figure(go.Pie(
                labels=["No Default", "Default"],
                values=target_counts.values,
                marker_colors=["#4ade80", "#f87171"],
                hole=0.4,
            ))
            fig_t.update_layout(
                title="Target Class Balance (Training Data)",
                paper_bgcolor="#111122",
                font={"color": "#e0e0ff"},
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with col2:
            num_cols = ["credit_amount", "duration", "age", "installment_commitment"]
            existing = [c for c in num_cols if c in df.columns]
            if existing:
                fig_box = go.Figure()
                for col in existing:
                    fig_box.add_trace(go.Box(y=df[col], name=col.replace("_", " ").title(), boxpoints=False))
                fig_box.update_layout(
                    title="Key Feature Distributions",
                    paper_bgcolor="#111122",
                    plot_bgcolor="#111122",
                    font={"color": "#e0e0ff"},
                    height=300,
                    margin=dict(l=10, r=10, t=50, b=10),
                    showlegend=False,
                    xaxis=dict(gridcolor="#222244"),
                    yaxis=dict(gridcolor="#222244"),
                )
                st.plotly_chart(fig_box, use_container_width=True)

        # Correlation heatmap
        st.markdown("**Correlation Heatmap (Numerical Features)**")
        num_df = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore")
        corr = num_df.corr()
        fig_hm = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig_hm.update_layout(
            paper_bgcolor="#111122",
            font={"color": "#e0e0ff"},
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load dataset for analytics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 History":
    st.markdown('<div class="section-title">Recent Prediction History</div>', unsafe_allow_html=True)

    limit = st.slider("Records to display", 5, 100, 20, step=5)
    history = api_get(f"/history?limit={limit}")

    if not history:
        st.info("No prediction history found. The backend may be offline or no predictions have been made.")
    else:
        df_hist = pd.DataFrame(history)
        if not df_hist.empty:
            # Score timeline
            fig_line = go.Figure(go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["risk_score"],
                mode="lines+markers",
                line=dict(color="#a78bfa", width=2),
                marker=dict(
                    color=df_hist["risk_score"],
                    colorscale=[[0, "#f87171"], [0.5, "#facc15"], [1, "#4ade80"]],
                    size=8,
                    cmin=0, cmax=100,
                ),
                hovertemplate="<b>Score: %{y:.1f}</b><br>%{x}<extra></extra>",
            ))
            fig_line.update_layout(
                title="Credit Safety Score Timeline",
                paper_bgcolor="#111122",
                plot_bgcolor="#111122",
                font={"color": "#e0e0ff"},
                height=280,
                margin=dict(l=10, r=10, t=50, b=40),
                xaxis=dict(gridcolor="#222244"),
                yaxis=dict(gridcolor="#222244", range=[0, 105]),
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # Table
            display_cols = ["id", "timestamp", "risk_score", "risk_category", "recommendation", "probability_default"]
            existing = [c for c in display_cols if c in df_hist.columns]
            st.dataframe(
                df_hist[existing].sort_values("id", ascending=False),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — System Info
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ System Info":
    st.markdown('<div class="section-title">System Information</div>', unsafe_allow_html=True)

    health  = api_get("/health")
    m_info  = api_get("/model-info")

    if health:
        h1, h2, h3 = st.columns(3)
        status_str = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Unhealthy"
        h1.metric("API Status", status_str)
        h2.metric("Model Loaded",    str(health.get("model_loaded", "N/A")))
        h3.metric("SHAP Ready",      str(health.get("explainer_ready", "N/A")))

    if m_info:
        st.markdown("**Active Model**")
        st.json(m_info)

    st.markdown("**Architecture Overview**")
    st.markdown("""
    ```
    User Browser (Streamlit host :2022)
         │  HTTP REST JSON
         ▼
    FastAPI Backend (host :3022)
         │  SQLAlchemy ORM
         ├──► PostgreSQL (Docker host :5022) — Predictions, Audit Logs
         │  joblib
         ├──► best_model.joblib   — Trained classifier
         │    feature_cols.joblib — Column alignment
         │    scaler.joblib       — StandardScaler
         │  SHAP
         └──► Explainer           — Feature attributions
    MLflow Server (host :4022)
         └──► mlflow.db           — Experiment tracking
    ```
    """)

    st.markdown("**Retrain Model**")
    if CLOUD_MODE:
        st.info(
            "Retraining is disabled in the public demo because hosted storage is temporary. "
            "Run `finscoreAI_Runner.bat` locally to train and track new model versions."
        )
    elif st.button("🔄  Trigger Model Retraining", type="secondary"):
        resp, err = api_post("/train", {})
        if err:
            st.error(err)
        else:
            st.success("Retraining job started in background. Check backend logs for progress.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem;'>"
    "FinScore AI &nbsp;|&nbsp; MCA AI/ML Internship Portfolio Project &nbsp;|&nbsp; "
    "Built with FastAPI · Streamlit · XGBoost · SHAP · MLflow · PostgreSQL"
    "</div>",
    unsafe_allow_html=True,
)
