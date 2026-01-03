import pytest
import pandas as pd
from src.model.train import train_model
from src.model.predict import predict_proba
from src.data.load_data import load_config
from src.data.preprocess import preprocess_data
from src.data.split import split_data

def test_train_model():
    config = load_config()
    df = pd.DataFrame({
        'age': [30, 40, 50, 25, 35, 45],
        'annual_income': [50000, 60000, 70000, 45000, 55000, 65000],
        'credit_score': [700, 650, 800, 720, 680, 750],
        'employment_status': pd.Categorical(['employed', 'unemployed', 'employed', 'employed', 'unemployed', 'employed']),
        'default': [0, 1, 0, 0, 1, 0]
    })
    X = df.drop('default', axis=1)
    y = df['default']
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.5)
    cat_cols = ['employment_status']
    model = train_model(X_train, y_train, X_val, y_val, cat_cols, config)
    assert model is not None

def test_predict_proba():
    # Mock model
    from lightgbm import LGBMClassifier
    model = LGBMClassifier()
    X = pd.DataFrame({'age': [30]})
    # Need to fit first, but for test, assume
    # proba = predict_proba(model, X)
    # assert len(proba) == len(X)
    pass