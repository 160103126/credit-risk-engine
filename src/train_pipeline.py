import mlflow
from data.load_data import load_train_data, load_config
from data.preprocess import preprocess_data
from data.split import split_data, get_stratified_kfold
from features.build_features import build_features
from model.train import train_model
from model.evaluate import evaluate_model, calculate_ks
from model.thresholds import evaluate_reject_rate
from explainability.shap_global import global_shap, plot_feature_importance
from explainability.dependence import dependence_plot
import pandas as pd
import numpy as np

def main():
    # Load config
    config = load_config()

    # Set MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit_risk_experiment")

    # Load data
    df = load_train_data(config)
    df, cat_cols = preprocess_data(df, is_train=True)
    df = build_features(df)

    # Split
    y = df['default']
    X = df.drop('default', axis=1)
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, cat_cols, config)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    eval_results = evaluate_model(y_val, y_pred_proba)
    ks = calculate_ks(y_val, y_pred_proba)
    print(f"AUC: {eval_results['auc']}, KS: {ks}")

    # Threshold evaluation
    reject_results = evaluate_reject_rate(y_val, y_pred_proba, 0.15)
    print(reject_results)

    # Explainability
    global_shap(model, X_val)
    plot_feature_importance(model)
    dependence_plot(model, X_val, 'debt_to_income_ratio')
    dependence_plot(model, X_val, 'credit_score')

    print("Training complete!")

if __name__ == "__main__":
    main()