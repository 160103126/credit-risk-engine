import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

def global_shap(model, X, max_display=15):
    """
    Generate global SHAP summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, max_display=max_display)
    plt.savefig('reports/shap_summary.png')
    plt.close()

def plot_feature_importance(model, max_num_features=15):
    """
    Plot feature importance.
    """
    lgb.plot_importance(model, importance_type='gain', max_num_features=max_num_features)
    plt.savefig('reports/feature_importance.png')
    plt.close()