from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate model with AUC and confusion matrix at threshold.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return {
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def calculate_ks(y_true, y_pred_proba):
    """
    Calculate KS statistic.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_pred_proba})
    df['decile'] = pd.qcut(df['y_prob'], 10, labels=False)
    ks_table = (
        df.groupby('decile')
        .agg(bads=('y_true', 'sum'), total=('y_true', 'count'))
        .sort_index(ascending=False)
    )
    ks_table['goods'] = ks_table['total'] - ks_table['bads']
    ks_table['cum_bad_pct'] = ks_table['bads'].cumsum() / ks_table['bads'].sum()
    ks_table['cum_good_pct'] = ks_table['goods'].cumsum() / ks_table['goods'].sum()
    ks_table['ks'] = np.abs(ks_table['cum_bad_pct'] - ks_table['cum_good_pct'])
    return ks_table['ks'].max()