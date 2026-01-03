import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib
import os

def get_monotone_constraints(X, cat_cols):
    """
    Set monotonic constraints: DTI +1 (higher risk), credit_score -1 (lower risk)
    """
    monotone_constraints = [0] * X.shape[1]
    if 'debt_to_income_ratio' in X.columns:
        dti_idx = X.columns.get_loc('debt_to_income_ratio')
        monotone_constraints[dti_idx] = 1
    if 'credit_score' in X.columns:
        credit_idx = X.columns.get_loc('credit_score')
        monotone_constraints[credit_idx] = -1
    return monotone_constraints

def train_model(X_train, y_train, X_val, y_val, cat_cols, config):
    """
    Train LightGBM model with MLflow tracking.
    """
    params = config['parameters']['model']
    params['monotone_constraints'] = get_monotone_constraints(X_train, cat_cols)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('cat_features', cat_cols)

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        # Predict and log metrics
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        mlflow.log_metric('auc', auc)

        # Log model
        mlflow.lightgbm.log_model(model, 'model')

        # Save model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/lgb_model.pkl')

        return model