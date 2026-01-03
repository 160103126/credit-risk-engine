# API Documentation

## Overview
The Credit Risk Engine exposes a REST API for real-time credit risk predictions using FastAPI.

## Endpoints

### POST /predict
Predicts credit risk for a single application.

#### Request Body
```json
{
  "age": 30,
  "annual_income": 50000.0,
  "credit_score": 700,
  "employment_status": "employed",
  "education_level": "bachelor",
  "experience": 5,
  "marital_status": "single",
  "number_of_dependents": 0,
  "loan_purpose": "home",
  "loan_amount": 200000.0,
  "loan_term": 30,
  "interest_rate": 4.5,
  "monthly_debt_payments": 1000.0,
  "credit_card_utilization": 0.3,
  "number_of_credit_inquiries": 2,
  "debt_to_income_ratio": 0.25,
  "home_ownership_status": "owned"
}
```

#### Response
```json
{
  "probability": 0.123,
  "decision": "APPROVE"
}
```

- **probability**: Float between 0-1, higher = higher risk
- **decision**: "APPROVE" if probability < threshold, else "REJECT"

#### Error Responses
- 400: Invalid input data
- 500: Internal server error

## Data Types
- **Numerical**: age, annual_income, etc. as float/int
- **Categorical**: employment_status, etc. as string (will be encoded internally)

## Authentication
None implemented - add API keys for production.

## Rate Limiting
Not implemented - consider adding for production.

## Deployment
- **Local**: `uvicorn api.main:app --reload`
- **Docker**: `docker-compose up`
- **Cloud**: Deploy to AWS/GCP/Azure with load balancer

## Testing
Use tools like Postman or curl:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @request.json
```

## Monitoring
- Log all predictions for audit trail
- Track API performance (latency, errors)
- Integrate with monitoring tools (Prometheus, Grafana)

## Versioning
- API version in URL path (e.g., /v1/predict) for future changes