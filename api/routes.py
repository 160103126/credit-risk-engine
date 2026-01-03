from fastapi import APIRouter, HTTPException
from .schema import PredictionRequest, PredictionResponse
import pandas as pd
import joblib
import json
import os
from src.data.load_data import load_config
import shap
import numpy as np

router = APIRouter()

# Load configuration and model paths
config = load_config()
models_dir = config.get('paths', {}).get('models', 'models')
model_path = os.path.join(models_dir, 'lgb_model.pkl')
threshold_path = os.path.join(models_dir, 'threshold.json')

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None

# Load threshold (fallback to default if missing)
threshold = 0.4542
if os.path.exists(threshold_path):
    try:
        with open(threshold_path, 'r') as f:
            th = json.load(f)
            if isinstance(th, dict) and 'threshold' in th:
                threshold = float(th['threshold'])
            elif isinstance(th, dict) and 'reject_rate' in th:
                # maintain backward compatibility
                threshold = float(th.get('threshold', threshold))
    except Exception:
        pass

# Lazy SHAP explainer
_explainer = None
def get_explainer():
    global _explainer
    if _explainer is None:
        try:
            _explainer = shap.TreeExplainer(model)
        except Exception:
            _explainer = None
    return _explainer


@router.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        df = pd.DataFrame([request.dict()])
        # Convert categoricals safely
        cat_cols = ['employment_status', 'education_level', 'marital_status', 'loan_purpose', 'home_ownership_status']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        proba = float(model.predict_proba(df)[:, 1][0])
        decision = "REJECT" if proba >= threshold else "APPROVE"
        return PredictionResponse(probability=proba, decision=decision)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain")
def explain_credit_risk(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        df = pd.DataFrame([request.dict()])
        cat_cols = ['employment_status', 'education_level', 'marital_status', 'loan_purpose', 'home_ownership_status']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        explainer = get_explainer()
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
        return {"probability": proba, "shap": contributions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))