import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_threshold(y_prob, reject_rate):
    """
    Calculate threshold for given reject rate (e.g., 0.15 for 15%).
    """
    return np.quantile(y_prob, 1 - reject_rate)

def evaluate_reject_rate(y_true, y_prob, reject_rate):
    """
    Evaluate metrics at given reject rate.
    """
    threshold = calculate_threshold(y_prob, reject_rate)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    approval_rate = 1 - reject_rate  # Since reject_rate is the rate of predicted positives

    return {
        'reject_rate': reject_rate,
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'auc': auc,
        'approval_rate': approval_rate,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP
    }