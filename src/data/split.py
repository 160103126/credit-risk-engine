from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and validation sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_stratified_kfold(n_splits=5, shuffle=True, random_state=42):
    """
    Get StratifiedKFold for cross-validation.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)