import pandas as pd
import numpy as np

def build_features(df):
    """
    Build additional features if needed.
    Currently, no additional features.
    """
    # Example: add log of income if skewed, but data is fine.
    # df['log_income'] = np.log1p(df['annual_income'])
    return df