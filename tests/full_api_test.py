import requests, json

BASE = 'http://127.0.0.1:3022'
PAYLOAD = {
    'checking_status': '0<=X<200', 'duration': 24, 'credit_history': 'existing paid',
    'purpose': 'radio/tv', 'credit_amount': 5000.0, 'savings_status': '<100',
    'employment': '1<=X<4', 'installment_commitment': 2,
    'other_parties': 'none', 'residence_since': 2, 'property_magnitude': 'real estate',
    'age': 30, 'other_payment_plans': 'none', 'housing': 'own', 'existing_credits': 1,
    'job': 'skilled', 'num_dependents': 1, 'own_telephone': 'none',
    'annual_interest_rate': 10.0, 'monthly_income': 1500.0,
    'existing_monthly_obligations': 100.0
}

sep = "="*60

print(sep)
print("  FINSCORE AI - COMPLETE API TEST REPORT")
print(sep)

# 1. Health
r = requests.get(BASE + "/health")
h = r.json()
print("\n[GET /health] Status:", r.status_code)
print("  status=" + h["status"] + " model_loaded=" + str(h["model_loaded"]) +
      " scaler=" + str(h["scaler_loaded"]) + " shap=" + str(h["explainer_ready"]))

# 2. Model Info
r = requests.get(BASE + "/model-info")
m = r.json()
print("\n[GET /model-info] Status:", r.status_code)
print("  model_name=" + str(m.get("model_name")))
print("  metrics=" + json.dumps(m.get("metrics", {})))

# 3. Predict - Good profile
print("\n[POST /predict] GOOD PROFILE (Low Risk Expected):")
r = requests.post(BASE + "/predict", json=PAYLOAD)
d = r.json()
print("  Status:", r.status_code)
print("  risk_score:", d.get("risk_score"))
print("  risk_category:", d.get("risk_category"))
print("  recommendation:", d.get("recommendation"))
print("  prob_default:", round(d.get("probability_default", 0), 4))
print("  shap_features:", len(d.get("shap_values") or {}))
print("  positive_factors:", d.get("positive_factors", []))
print("  negative_factors:", d.get("negative_factors", []))

# 4. Predict - Bad profile
bad_payload = dict(PAYLOAD)
bad_payload.update({
    'checking_status': '<0', 'credit_amount': 18000.0, 'duration': 60,
    'savings_status': 'no known savings', 'employment': 'unemployed',
    'credit_history': 'critical/other existing credit', 'age': 22
})
print("\n[POST /predict] BAD PROFILE (High Risk Expected):")
r = requests.post(BASE + "/predict", json=bad_payload)
d2 = r.json()
print("  Status:", r.status_code)
print("  risk_score:", d2.get("risk_score"))
print("  risk_category:", d2.get("risk_category"))
print("  recommendation:", d2.get("recommendation"))
print("  prob_default:", round(d2.get("probability_default", 0), 4))

# 5. Predict - Medium profile
med_payload = dict(PAYLOAD)
med_payload.update({'credit_amount': 9000.0, 'savings_status': '<100', 'checking_status': '<0', 'employment': '<1'})
print("\n[POST /predict] MEDIUM PROFILE:")
r = requests.post(BASE + "/predict", json=med_payload)
d3 = r.json()
print("  Status:", r.status_code)
print("  risk_score:", d3.get("risk_score"))
print("  risk_category:", d3.get("risk_category"))
print("  recommendation:", d3.get("recommendation"))

# 6. Metrics
r = requests.get(BASE + "/metrics")
me = r.json()
print("\n[GET /metrics] Status:", r.status_code)
print("  total_predictions:", me.get("total_predictions"))
print("  avg_risk_score:", me.get("avg_risk_score"))
print("  risk_distribution:", me.get("risk_distribution"))
print("  recommendation_dist:", me.get("recommendation_dist"))

# 7. History
r = requests.get(BASE + "/history?limit=5")
hist = r.json()
print("\n[GET /history] Status:", r.status_code)
print("  records returned:", len(hist))
if hist:
    lat = hist[0]
    print("  latest: id=" + str(lat.get("id")) + " score=" + str(lat.get("risk_score")) + " cat=" + str(lat.get("risk_category")))

# 8. Validation - missing field
bad_v = {k: v for k, v in PAYLOAD.items() if k != 'credit_amount'}
r = requests.post(BASE + "/predict", json=bad_v)
print("\n[POST /predict] MISSING FIELD test -> Status:", r.status_code, "(expected 422)")

# 9. Validation - invalid age
inv_v = dict(PAYLOAD)
inv_v['age'] = -5
r = requests.post(BASE + "/predict", json=inv_v)
print("[POST /predict] INVALID AGE (-5) -> Status:", r.status_code, "(expected 422)")

# 10. Swagger docs
r = requests.get(BASE + "/docs")
print("\n[GET /docs] Swagger UI -> Status:", r.status_code, "(expected 200)")

# 11. ReDoc
r = requests.get(BASE + "/redoc")
print("[GET /redoc] ReDoc UI -> Status:", r.status_code, "(expected 200)")

print("\n" + sep)
print("  ALL ENDPOINT TESTS COMPLETE")
print(sep)
