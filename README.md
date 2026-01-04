# Credit Risk Engine

A machine learning project for credit risk assessment using LightGBM and FastAPI.

## Project Structure

- `data/`: Data files
- `notebooks/`: Jupyter notebooks for EDA, feature engineering, etc.
- `src/`: Source code
- `models/`: Trained models
- `api/`: API for predictions
- `reports/`: Documentation and reports
- `tests/`: Unit tests

## Documentation

- **[End-to-End Guide](reports/end_to_end_guide.md)**: Complete walkthrough of the project
- **[Model Development Document](reports/model_development_document.md)**: Technical specification of model development
- **[Model Validation Report](reports/model_validation.md)**: Performance metrics, KS score, confusion matrix
- **[Decision Policy Document](reports/decision_policy_document.md)**: Threshold selection and decision logic
- **[Explainability Document](reports/explainability.md)**: SHAP analysis, feature importance
- **[Monitoring & Drift Policy](reports/monitoring.md)**: Drift detection, PSI, production monitoring
- **[API Documentation](reports/api_documentation.md)**: API endpoints and usage
- **[MLflow Guide](reports/mlflow_guide.md)**: Experiment tracking setup
- **[Deployment & Versioning Notes](reports/deployment_versioning_notes.md)**: Production deployment details
- **[Production Readiness Checklist](reports/production_readiness_checklist.md)**: What's needed for production
- **[Business Approval Document](reports/business_approval_document.md)**: Business case and approval requirements
- **[ChatGPT Conversation Summary](reports/chatgpt_summary.md)**: Q&A insights on model building, LightGBM, thresholds, and production readiness

## Setup

- Install dependencies:
  - `pip install -r requirements.txt`
- Place data files in `data/raw/`
- Train model (logs to MLflow and saves to models/):
  - `python src/train_pipeline.py`
- Start API locally:
  - `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`

## Docker

Build and run with Docker:

```bash
docker build -t credit-risk .
docker run -p 8000:8000 credit-risk
```

Or with docker-compose:

```bash
docker-compose up
```

## API Overview

Endpoints:
- `GET /healthz` – liveness check
- `GET /readiness` – verifies model is loaded and can score
- `GET /version` – model metadata (path, hash, features, threshold)
- `POST /predict` – returns probability of default and decision
- `POST /explain` – returns probability and SHAP feature contributions

Request schema for `/predict` and `/explain`:
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

Decision logic:
- Model outputs probability of class 1 = default
- Threshold loaded from `models/threshold.json` (default 0.4542)
- Decision = `REJECT` if probability ≥ threshold, else `APPROVE`

## Monitoring

- Data drift: PSI calculations (`src/monitoring/`)
- Model drift: KS statistic over time
- MLflow for experiment tracking

## Tests

- Run tests: `pytest`