# Model Explainability Report

## Overview
This report covers the explainability techniques used to understand the LightGBM credit risk model's predictions, ensuring transparency and regulatory compliance.

## SHAP (SHapley Additive exPlanations)

### What is SHAP?
SHAP assigns each feature an importance value for a particular prediction, based on game theory. It explains how much each feature contributes to pushing the prediction away from the base value.

### Global Explainability
- **Summary Plot**: Shows feature importance across all predictions. Top features: debt_to_income_ratio, credit_score, employment_status.
- **Red (high risk)**: Features pushing prediction towards default (e.g., high DTI).
- **Blue (low risk)**: Features pushing towards non-default (e.g., high credit score).

### Individual Explainability
- **Bar Plots**: For specific customers, shows contribution of each feature to their risk score.
- **Example**: A rejected customer might have high DTI (+0.3) and low credit score (+0.2) outweighing positive factors.

#### How to Generate an Individual SHAP Plot (Offline)
Use the helper in `src/explainability/shap_individual.py` to render a per-record SHAP bar plot and save it under `reports/`.

Example (Python shell or notebook):
```python
import joblib
import pandas as pd
from src.explainability.shap_individual import individual_shap

model = joblib.load('models/lgb_model.pkl')

# df should match the model's feature set and dtypes used in training
df = pd.read_csv('data/samples/validation_sample.csv')  # or your own dataframe

# Convert string columns to pandas Categorical for LightGBM
for col in ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']:
  if col in df.columns:
    df[col] = pd.Categorical(df[col])

# Plot for the first row (index 0); saves to reports/shap_individual_0.png
individual_shap(model, df, idx=0, max_display=11)
```

### Dependence Plots
- **debt_to_income_ratio**: SHAP values increase with DTI, confirming higher DTI → higher risk.
- **credit_score**: SHAP values decrease with credit score, confirming higher score → lower risk.
- **employment_status**: Categorical - "unemployed" increases risk, "employed/retired" decreases.

## Feature Importance (Gain-based)
- **Top Features**:
  1. debt_to_income_ratio
  2. credit_score
  3. employment_status
  4. annual_income
  5. age

## Why Explainability Matters
- **Regulatory**: Credit models require explainable decisions (e.g., GDPR, CCPA).
- **Business**: Understand model behavior for strategy (e.g., focus on DTI reduction programs).
- **Trust**: Builds confidence in automated decisions.

## Implementation
- **Library**: SHAP for Python
- **Integration**: Run after training in pipeline
- **Outputs**: Plots saved in reports/ folder

## Limitations
- SHAP is computationally intensive for large datasets.
- Explanations are model-specific, not causal.

## Recommendations
- Use SHAP for top 10 features in production explanations.
- Monitor if explanations change significantly over time.