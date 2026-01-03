# Monitoring script
from src.monitoring.drift_report import generate_drift_report
from src.data.load_data import load_train_data, load_config
from src.data.preprocess import preprocess_data
from src.model.predict import predict_proba
import joblib
import pandas as pd

def main():
    config = load_config()
    model = joblib.load('models/lgb_model.pkl')

    # Load current data (assume new data in data/raw/current.csv or something)
    # For demo, use train data as baseline and val as current
    baseline_df = load_train_data(config)
    baseline_df, _ = preprocess_data(baseline_df, is_train=True)
    baseline_X = baseline_df.drop('default', axis=1)
    baseline_y = baseline_df['default']
    baseline_preds = predict_proba(model, baseline_X)

    # Assume current data
    current_df = baseline_df.sample(frac=0.1)  # mock
    current_X = current_df.drop('default', axis=1)
    current_y = current_df['default']
    current_preds = predict_proba(model, current_X)

    report = generate_drift_report(baseline_df, current_df, baseline_preds, current_preds)
    print("Drift report generated:", report)

if __name__ == "__main__":
    main()