from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {'confusion_matrix': cm, 'classification_report': report}