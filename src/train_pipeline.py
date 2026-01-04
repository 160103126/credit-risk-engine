import mlflow
from data.load_data import load_train_data, load_config
from data.preprocess import preprocess_data
from data.split import split_data, get_stratified_kfold
from model.train import train_model
from model.evaluate import evaluate_model, calculate_ks
from model.thresholds import evaluate_reject_rate
from explainability.shap_global import global_shap, plot_feature_importance
from explainability.dependence import dependence_plot
import pandas as pd
import numpy as np

def cross_validate_thresholds(X, y, cat_cols, config, reject_rates=[0.05, 0.10, 0.15, 0.20]):
    """
    Perform cross-validation and evaluate thresholds for different reject rates.
    """
    skf = get_stratified_kfold(n_splits=5)
    results = {rate: [] for rate in reject_rates}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, cat_cols, config, log_mlflow=False)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Evaluate thresholds
        for rate in reject_rates:
            thresh_result = evaluate_reject_rate(y_val, y_pred_proba, rate)
            results[rate].append(thresh_result)
    
    # Aggregate results
    summary = {}
    for rate in reject_rates:
        thresholds = [r['threshold'] for r in results[rate]]
        aucs = [r['auc'] for r in results[rate]]
        approval_rates = [r['approval_rate'] for r in results[rate]]
        
        summary[rate] = {
            'mean_threshold': np.mean(thresholds),
            'std_threshold': np.std(thresholds),
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'mean_approval_rate': np.mean(approval_rates),
            'std_approval_rate': np.std(approval_rates)
        }
    
    return summary

def main():
    # Load config
    config = load_config()

    # Set MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit_risk_experiment")

    # Load data
    df = load_train_data(config)
    df, cat_cols = preprocess_data(df, is_train=True)

    # Full dataset for CV
    y = df['default']
    X = df.drop('default', axis=1)
    
    # Cross-validate thresholds
    print("Performing cross-validation for threshold selection...")
    cv_results = cross_validate_thresholds(X, y, cat_cols, config)
    for rate, stats in cv_results.items():
        print(f"Reject Rate {rate*100}%: Threshold={stats['mean_threshold']:.4f}±{stats['std_threshold']:.4f}, "
              f"AUC={stats['mean_auc']:.4f}±{stats['std_auc']:.4f}, "
              f"Approval={stats['mean_approval_rate']:.4f}±{stats['std_approval_rate']:.4f}")

    # Split for final model training
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Train final model
    model = train_model(X_train, y_train, X_val, y_val, cat_cols, config)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    eval_results = evaluate_model(y_val, y_pred_proba)
    ks = calculate_ks(y_val, y_pred_proba)
    print(f"AUC: {eval_results['auc']}, KS: {ks}")

    # Threshold evaluation for 15%
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