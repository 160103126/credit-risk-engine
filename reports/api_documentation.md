# API Documentation

## Overview
The Credit Risk Engine exposes a REST API for real-time credit risk predictions using FastAPI.

## Endpoints

### GET /healthz
Liveness check. Returns:
```json
{"status": "ok"}
```

### GET /readiness
Readiness probe. Verifies model is loaded and can score a synthetic request. Returns:
```json
{"status": "ready"}
```

### GET /version
Model and deployment metadata. Example:
```json
{
  "model_path": "models/lgb_model.pkl",
  "model_sha256": "...",
  "model_mtime": "2026-01-03T12:34:56",
  "model_size_bytes": 123456,
  "model_class": "LGBMClassifier",
  "feature_names": ["annual_income", "debt_to_income_ratio", ...],
  "threshold": 0.4542
}
```

### POST /predict
Predicts probability of default and returns approve/reject decision.

Request Body:
```json
{
  "annual_income": 50000.0,
  "debt_to_income_ratio": 0.25,
  "credit_score": 700,
  "loan_amount": 200000.0,
  "interest_rate": 4.5,
  "gender": "Female",
  "marital_status": "Single",
  "education_level": "Bachelor's",
  "employment_status": "Employed",
  "loan_purpose": "Home",
  "grade_subgrade": "A1"
}
```

Response:
```json
{
  "probability": 0.123,
  "decision": "APPROVE"
}
```

- `probability`: Probability of class 1 (default). Higher = higher risk.
- `decision`: `REJECT` if probability ≥ threshold else `APPROVE`.

Notes:
- Categorical inputs are converted to pandas Categorical internally; unknown labels are allowed.
- Feature order is aligned to the model’s training order internally.

### POST /explain
Returns probability and SHAP feature contributions (sorted by absolute impact).

Response:
```json
{
  "probability": 0.123,
  "shap": [
    {"feature": "credit_score", "shap_value": -0.042},
    {"feature": "debt_to_income_ratio", "shap_value": 0.033}
  ]
}
```

## Error Responses
- 400: Invalid input data
- 500: Internal server error

## Data Types
- Numerical: `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`
- Categorical: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`

## Authentication
None implemented. For production, add API keys or OAuth.

## Rate Limiting
Not implemented. Consider gateway-level rate limiting for production.

## Deployment
- Local: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
- Docker: `docker-compose up`
- Cloud: Deploy behind a load balancer with health/readiness probes

## Testing
Use curl or Postman:
```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/version
curl -s -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
  -d '{"annual_income":50000.0,"debt_to_income_ratio":0.25,"credit_score":700,"loan_amount":200000.0,"interest_rate":4.5,
       "gender":"Female","marital_status":"Single","education_level":"Bachelor\'s","employment_status":"Employed",
       "loan_purpose":"Home","grade_subgrade":"A1"}'
```

## Monitoring
- Track API latency, error rates
- Log predictions for audit (avoid PII)
- Integrate Prometheus/Grafana where available

## Versioning
If API changes are expected, version via path (e.g., `/v1/predict`).