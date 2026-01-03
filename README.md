# Credit Risk Engine

A machine learning project for credit risk assessment using LightGBM.

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

1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in `data/raw/`
3. Run training: `python src/train_pipeline.py`
4. Start API: `uvicorn api.main:app --reload`

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

## API Usage

POST to /predict with JSON of features.

Example:

```json
{
  "age": 30,
  "annual_income": 50000,
  "credit_score": 700,
  "employment_status": "employed",
  ...
}
```

Response: {"probability": 0.3, "decision": "APPROVE"}

## Monitoring

- Data drift: Run PSI calculations
- Model drift: Check KS statistic

## Tests

Run tests: `pytest`