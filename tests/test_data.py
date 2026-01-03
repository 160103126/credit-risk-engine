import pytest
import pandas as pd
from src.data.load_data import load_train_data, load_config
from src.data.preprocess import preprocess_data

def test_load_train_data():
    config = load_config()
    df = load_train_data(config)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data():
    # Mock data
    df = pd.DataFrame({
        'id': [1, 2],
        'loan_paid_back': [1, 0],
        'age': [30, 40],
        'employment_status': ['employed', 'unemployed']
    })
    processed, cat_cols = preprocess_data(df, is_train=True)
    assert 'default' in processed.columns
    assert 'loan_paid_back' not in processed.columns
    assert 'id' not in processed.columns
    assert processed['employment_status'].dtype == 'category'