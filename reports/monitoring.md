# Monitoring Report

## Overview
This report outlines the monitoring strategies for the credit risk model in production, focusing on data drift, model performance, and operational health.

## Data Drift Detection

### PSI (Population Stability Index)
- **Definition**: Measures distribution shift between baseline (training) and current data.
- **Formula**: PSI = Σ [(Actual% - Expected%) * ln(Actual% / Expected%)]
- **Bins**: 10 equal-frequency bins for continuous features.
- **Thresholds**:
  - PSI < 0.1: No significant change
  - 0.1 < PSI < 0.25: Moderate change, monitor
  - PSI > 0.25: Significant change, investigate/retrain
- **Features Monitored**: All numerical features (age, income, DTI, etc.)
- **Frequency**: Weekly/monthly

### Model Drift Detection
- **KS Statistic**: Monitors separation between good/bad customers.
- **Threshold**: KS drop > 0.1 from baseline triggers alert.
- **AUC Monitoring**: Track AUC on recent data; alert if < 0.80.

## Production Monitoring Setup

### Metrics to Track
- **Prediction Distribution**: Mean probability, percentiles.
- **Approval/Rejection Rates**: Ensure consistent with business rules.
- **Feature Distributions**: Statistical tests (KS test, Chi-square) for categorical.
- **Performance Metrics**: AUC, KS, confusion matrix elements.

### Alerting
- **Automated**: Email/Slack alerts for PSI > 0.25 or KS change > 0.1.
- **Manual Reviews**: Monthly reports for stakeholders.

### Data Collection
- **Baseline**: Training data distributions and model performance.
- **Current**: Batch predictions with feature logging.
- **Storage**: Store in database or cloud storage for historical analysis.

## Implementation
- **Scripts**: `src/monitoring/psi.py`, `ks_monitor.py`, `drift_report.py`
- **Scheduling**: Use cron jobs or Airflow for periodic runs.
- **Logging**: All metrics logged to MLflow or monitoring dashboard.

## Why These Methods?
- **PSI**: Standard for financial models, detects concept drift.
- **KS**: Directly measures model's discriminatory power.
- **Comprehensive**: Covers data, model, and operational aspects.

## Response to Drift
- **Minor Drift**: Log and continue monitoring.
- **Major Drift**: Retrain model with new data.
- **Sudden Changes**: Investigate data quality issues.

## Recommendations
- Implement monitoring from day 1 of production.
- Set up dashboards for real-time visualization.
- Regularly update baseline as model evolves.