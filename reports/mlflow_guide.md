# MLflow Guide

## Overview
MLflow is used for experiment tracking, model versioning, and deployment in the Credit Risk Engine.

## Setup
- **Tracking URI**: `file:./mlruns` (local file system)
- **Experiment**: "credit_risk_experiment"
- **UI**: Run `mlflow ui` to view experiments at http://localhost:5000

## Logged Elements

### Parameters
- Model hyperparameters (n_estimators, learning_rate, etc.)
- Categorical features list
- Monotonic constraints

### Metrics
- AUC (validation)
- KS statistic
- Threshold value
- Confusion matrix elements (optional)

### Artifacts
- Trained model (LightGBM)
- Feature importance plots
- SHAP plots
- Evaluation reports

### Tags
- Dataset version
- Training date
- Model version

## Workflow
1. **Start Run**: `mlflow.start_run()`
2. **Log Params/Metrics**: `mlflow.log_param()`, `mlflow.log_metric()`
3. **Log Model**: `mlflow.lightgbm.log_model()`
4. **End Run**: Automatic on exit

## Model Registry
- Register best models for staging/production
- Version control for rollbacks
- Serve models via MLflow Model Serving

## Integration in Code
```python
import mlflow
import mlflow.lightgbm

with mlflow.start_run():
    mlflow.log_params(params)
    model.fit(...)
    auc = evaluate(...)
    mlflow.log_metric("auc", auc)
    mlflow.lightgbm.log_model(model, "model")
```

## Benefits
- **Reproducibility**: Track exact parameters and data
- **Comparison**: Compare runs across experiments
- **Deployment**: Easy model serving
- **Collaboration**: Share experiments with team

## Production Use
- Log predictions and performance in production
- A/B testing with different model versions
- Automated retraining triggers based on metrics

## Alternatives
- Weights & Biases, Comet ML, or custom logging if needed