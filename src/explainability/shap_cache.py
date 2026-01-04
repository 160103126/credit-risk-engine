import shap

# Global SHAP explainer cache for the entire application
_explainer_cache = {}

def get_explainer(model):
    """
    Get cached SHAP explainer or create new one.
    Uses model id as cache key to handle multiple models.
    """
    model_id = id(model)
    if model_id not in _explainer_cache:
        _explainer_cache[model_id] = shap.TreeExplainer(model)
    return _explainer_cache[model_id]

def clear_cache():
    """Clear the explainer cache (useful for memory management)."""
    global _explainer_cache
    _explainer_cache.clear()