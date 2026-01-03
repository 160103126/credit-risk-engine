import shap
import matplotlib.pyplot as plt

def individual_shap(model, X, idx, max_display=11):
    """
    Generate SHAP bar plot for individual prediction.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.plots.bar(shap_values[idx], max_display=max_display)
    plt.savefig(f'reports/shap_individual_{idx}.png')
    plt.close()