import hashlib
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram

from src.data.load_data import load_config
from src.explainability.shap_cache import get_explainer
from .schema import PredictionRequest, PredictionResponse

router = APIRouter()

# Load configuration and model paths
config = load_config()
models_dir = config.get('paths', {}).get('models', 'models')
model_path = os.path.join(models_dir, 'lgb_model.pkl')
threshold_path = os.path.join(models_dir, 'threshold.json')

# List of categorical columns that need to be converted to pandas Categorical dtype
CATEGORICAL_COLUMNS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None

# Load threshold (fallback to default if missing)
threshold = 0.4542

try:
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            threshold = float(data.get('threshold', threshold))
except Exception:
    pass

_explainer = None
def get_api_explainer():
    global _explainer
    if _explainer is None:
        try:
            _explainer = get_explainer(model)
        except Exception:
            _explainer = None
    return _explainer


# Prometheus metrics
REQUEST_COUNTER = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    labelnames=["endpoint"],
)


@router.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(request: PredictionRequest):
    start = time.perf_counter()
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        df = pd.DataFrame([request.dict()])

        # Convert categorical columns to pandas Categorical
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.Categorical(df[col])
        
        proba = float(model.predict_proba(df)[:, 1][0])
        decision = "REJECT" if proba >= threshold else "APPROVE"
        REQUEST_COUNTER.labels(endpoint="predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
        return PredictionResponse(probability=proba, decision=decision)
    except HTTPException:
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="predict", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - start)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain")
def explain_credit_risk(request: PredictionRequest):
    start = time.perf_counter()
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        df = pd.DataFrame([request.dict()])

        # Convert categorical columns to pandas Categorical
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.Categorical(df[col])

        explainer = get_api_explainer()
        if explainer is None:
            raise HTTPException(status_code=500, detail="SHAP explainer not available")

        shap_values = explainer.shap_values(df)
        # For binary classifiers shap_values is a list: [neg_class_vals, pos_class_vals]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1][0]
        else:
            sv = np.array(shap_values)[0]

        feature_names = df.columns.tolist()
        contributions = [{"feature": f, "shap_value": float(v)} for f, v in zip(feature_names, sv)]
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)

        proba = float(model.predict_proba(df)[:, 1][0])
        REQUEST_COUNTER.labels(endpoint="explain", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="explain").observe(time.perf_counter() - start)
        return {"probability": proba, "shap": contributions}
    except HTTPException:
        REQUEST_COUNTER.labels(endpoint="explain", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="explain").observe(time.perf_counter() - start)
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="explain", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="explain").observe(time.perf_counter() - start)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readiness")
def readiness():
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Build a minimal synthetic row using model feature names if available
        feature_names = list(getattr(model, 'feature_name_', []))
        if not feature_names:
            feature_names = [
                'annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate',
                'gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade'
            ]

        sample = {}
        for f in feature_names:
            if f in CATEGORICAL_COLUMNS:
                sample[f] = 'Unknown'
            else:
                sample[f] = 0

        df = pd.DataFrame([sample])
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.Categorical(df[col])

        # Dry-run prediction
        _ = float(model.predict_proba(df)[:, 1][0])
        return {"status": "ready"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Not ready: {e}")


@router.get("/version")
def version():
    try:
        info = {"model_path": model_path, "threshold": threshold}
        # File stats
        try:
            with open(model_path, "rb") as f:
                data = f.read()
                info["model_sha256"] = hashlib.sha256(data).hexdigest()
        except Exception:
            info["model_sha256"] = None
        try:
            mtime = os.path.getmtime(model_path)
            info["model_mtime"] = datetime.fromtimestamp(mtime).isoformat()
            info["model_size_bytes"] = os.path.getsize(model_path)
        except Exception:
            pass
        # Model details
        if model is not None:
            info["model_class"] = type(model).__name__
            try:
                info["feature_names"] = list(getattr(model, "feature_name_", []))
            except Exception:
                info["feature_names"] = []
        else:
            info["model_class"] = None
            info["feature_names"] = []
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
