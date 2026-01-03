import pandas as pd
import numpy as np

def preprocess_data(df, is_train=True):
    """
    Preprocess the data: create target, drop columns, convert categoricals.
    """
    if is_train:
        # Create 'default' target: 1 if loan not paid back (bad), 0 if paid back (good)
        df['default'] = (df['loan_paid_back'] == 0).astype(int)
        df = df.drop(columns=['loan_paid_back'])

    # Drop 'id' as it's not useful for modeling
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Convert categorical columns to category dtype for LightGBM
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category')

    return df, cat_cols