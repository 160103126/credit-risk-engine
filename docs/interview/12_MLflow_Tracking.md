# MLflow Tracking & Production Usage

## Overview

**Purpose**: Track experiments, reproduce results, and ship the exact model artifact to production.  
**Backend store**: `file:./mlruns` (local)  
**Experiment**: `credit_risk_experiment`  
**Artifacts**: Trained LightGBM model pickle, optional plots (ROC, feature importance)  
**Metrics**: AUC (primary), KS (optional), CV metrics  
**Params**: LightGBM hyperparameters, data/feature config, monotone constraints

---

## How We Use MLflow Today

- **Tracking runs**: Each training run logs metrics, params, and the trained model artifact under the `credit_risk_experiment` experiment in `mlruns/`.
- **Primary metric**: Validation AUC (and CV AUC when run inside folds). KS can be added the same way.
- **Artifact of record**: The pickled LightGBM model from the best run. This is the model shipped with the API. The business threshold (0.4542) is **not** baked into the artifact; it lives in API config.
- **Local UI**: Launch with `mlflow ui --backend-store-uri file:./mlruns` from repo root to compare runs.

---

## What We Log

- **Parameters**: `n_estimators`, `max_depth`, `num_leaves`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `is_unbalance`, `monotone_constraints`.
- **Metrics**: Validation AUC; CV AUC per fold (when looping), KS if enabled; loss/iteration if using LightGBM callbacks.
- **Artifacts**: Trained model pickle, optional plots (ROC curve, feature importance), and any exportable CV summaries.

---

## Code Pattern (Training Run)

```python
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit_risk_experiment")

with mlflow.start_run(run_name="lgb_credit_risk"):
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        is_unbalance=True,
        monotone_constraints={'credit_score': -1},
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    mlflow.log_params(model.get_params())
    mlflow.log_metric("val_auc", auc)

    # Save model artifact
    mlflow.lightgbm.log_model(model, artifact_path="model")
```

---

## Production Usage

### What We Deploy

- **Model artifact**: LightGBM pickle from the best MLflow run.
- **Threshold**: 0.4542 (business-driven), stored in API config, not inside the model.
- **Schema alignment**: The API uses the same feature order/dtypes as training; categorical columns are coerced with pandas Categorical before inference.

### How to Load by Run ID

```python
import mlflow.lightgbm

run_id = "<best_run_id>"
model_uri = f"runs:/{run_id}/model"
model = mlflow.lightgbm.load_model(model_uri)

# Use model.predict_proba(...) inside the API
```

### Promotion Checklist

1. **Pick winning run**: Highest AUC (and KS) with stable CV.
2. **Record run_id**: Keep it alongside the deployed artifact and threshold.
3. **Export artifact**: Bundle the MLflow model directory (or pickle) into the Docker image used by the API.
4. **Config**: Set `DECISION_THRESHOLD = 0.4542` in the API; keep run_id in deployment metadata for traceability.
5. **Smoke test**: Run `/predict` and `/explain` locally against the packaged model.

### Reproducibility & Rollback

- **Reproduce**: Rerun training with the same params and data version to match the logged run.
- **Rollback**: Swap to a previous run_id artifact if production monitoring degrades.

---

## Monitoring & Drift

- **Track production metrics**: Approval rate (~85%), reject rate (~15%), default rate on approved loans, and periodic AUC/KS on recent data.
- **Compare to training runs**: If AUC/KS drops materially, retrain and promote a new MLflow run.
- **Feature drift**: Watch input distributions; if categories shift, retrain and re-log with MLflow.

---

## FAQs (Interview-Ready)

- **Why MLflow?** Reproducibility, auditability, and easy comparison of tuning experiments.
- **Where are runs stored?** Local `mlruns` directory; backend URI `file:./mlruns`.
- **How do you promote to production?** Select best run → record run_id → package the model artifact with the API → set threshold in config.
- **Is the threshold in the model?** No. The model predicts probabilities; the API applies the 0.4542 threshold (business-owned).
- **How to view runs?** `mlflow ui --backend-store-uri file:./mlruns` then open http://127.0.0.1:5000.

---

## Next Improvements (Optional)

- Add automatic logging of KS and CV metrics per fold to MLflow.
- Store run_id in deployment metadata and expose it via `/version` for traceability.
- Add model validation artifacts (ROC/KS plots) to each run for quicker reviews.
- Consider promoting via a registry when moving beyond local deployments.
