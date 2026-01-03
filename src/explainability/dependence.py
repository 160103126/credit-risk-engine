import shap
import matplotlib.pyplot as plt

def dependence_plot(model, X, feature, interaction_index=None):
    """
    Generate SHAP dependence plot for a feature.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_index)
    plt.savefig(f'reports/dependence_{feature}.png')
    plt.close()