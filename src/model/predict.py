import lightgbm as lgb
import numpy as np

def predict_proba(model, X):
    """
    Predict probabilities.
    """
    return model.predict_proba(X)[:, 1]

def predict_class(model, X, threshold=0.5):
    """
    Predict class based on threshold.
    """
    proba = predict_proba(model, X)
    return (proba >= threshold).astype(int)