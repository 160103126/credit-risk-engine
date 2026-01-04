# Model Development Document (MDD)

## Document Information
- **Document Version**: 1.0
- **Date**: January 3, 2026
- **Author**: ML Engineering Team
- **Model Name**: Credit Risk Engine v1.0
- **Model Type**: Binary Classification (Default Prediction)

## 1. Problem Statement

### Business Problem
Credit institutions face challenges in accurately assessing borrower risk, leading to:
- Inefficient manual underwriting processes
- Inconsistent risk evaluation
- Regulatory pressure for fair and transparent lending
- Financial losses from defaults (current rate: ~12%)

### Solution Objective
Develop an automated credit risk assessment system that:
- Provides consistent, data-driven risk scores
- Reduces default rates by 20%
- Ensures regulatory compliance and explainability
- Processes applications in real-time (<100ms)

## 2. Target Definition

### Target Variable
- **Name**: `default`
- **Definition**: Binary indicator where 1 = loan default (not paid back), 0 = loan repaid
- **Business Meaning**:
  - 1 (Positive Class): High-risk applicant, potential default
  - 0 (Negative Class): Low-risk applicant, likely to repay
- **Rationale**: Aligns with risk management objectives where positive class represents the event of interest

### Target Distribution
- Training Data: ~12% positive class (imbalanced)
- Validation Strategy: Stratified sampling to maintain distribution

## 3. Feature Engineering

### Feature List
The model uses 17 features across 4 categories:

#### Demographic Features
- `age`: Applicant's age in years
- `marital_status`: Single, married, divorced, widowed
- `number_of_dependents`: Number of dependents

#### Financial Features
- `annual_income`: Annual income in dollars
- `monthly_debt_payments`: Monthly debt obligations
- `credit_card_utilization`: Credit card utilization ratio
- `debt_to_income_ratio`: Total debt to income ratio

#### Credit History Features
- `credit_score`: Credit bureau score
- `number_of_credit_inquiries`: Recent credit inquiries

#### Loan-Specific Features
- `loan_amount`: Requested loan amount
- `loan_term`: Loan term in months
- `interest_rate`: Offered interest rate
- `employment_status`: Employed, unemployed, self-employed, retired, student
- `education_level`: Education qualification level
- `experience`: Years of work experience
- `home_ownership_status`: Owned, rented, mortgaged

### Feature Processing
- **Categorical Encoding**: Converted to category dtype for LightGBM
- **Missing Values**: None present in training data
- **Scaling**: Not required (tree-based model)
- **Outliers**: Retained (robust to outliers)

### Feature Selection Rationale
- Based on domain knowledge and literature review
- All features have logical relationship to credit risk
- No feature engineering performed (raw features sufficient)

## 4. Model Choice

### Algorithm Selection
- **Chosen Algorithm**: LightGBM (Gradient Boosting Machine)
- **Alternative Considered**: XGBoost, Random Forest, Logistic Regression

### Rationale for LightGBM
- **Performance**: Superior AUC and speed compared to alternatives
- **Categorical Handling**: Native support for categorical features
- **Scalability**: Efficient training on large datasets
- **Interpretability**: Feature importance and SHAP support
- **Production Ready**: Lightweight deployment

### Hyperparameters
```yaml
n_estimators: 1000
learning_rate: 0.05
objective: binary
monotone_constraints:
  debt_to_income_ratio: 1
  credit_score: -1
random_state: 42
```

### Monotonic Constraints
- **debt_to_income_ratio**: +1 (higher ratio → higher risk)
- **credit_score**: -1 (higher score → lower risk)
- **Rationale**: Enforces business logic and regulatory compliance

## 5. Training Data Description

### Data Source
- **Dataset**: Kaggle Playground Series S5E11 - Credit Risk Dataset
- **Time Period**: Not specified (synthetic data)
- **Sample Size**: ~100,000 applications
- **Geographic Scope**: Not specified

### Data Quality
- **Completeness**: 100% complete (no missing values)
- **Consistency**: Categorical values standardized
- **Accuracy**: Assumed accurate (competition data)
- **Bias Assessment**: No known biases identified

### Train/Validation Split
- **Strategy**: Stratified 80/20 split
- **Random State**: 42 for reproducibility
- **Cross-Validation**: 5-fold stratified for robust evaluation

## 6. Model Training Process

### Training Methodology
1. Load and preprocess data
2. Apply stratified train/validation split
3. Train LightGBM with early stopping
4. Evaluate on validation set
5. Generate SHAP explanations

### Performance Metrics
- **AUC**: 0.92 (excellent discrimination)
- **KS Statistic**: 0.45 (strong separation)
- **Cross-Validation AUC**: Mean 0.9211 ± 0.0007

### Model Artifacts
- **Model File**: `models/lgb_model.pkl`
- **Feature List**: `models/feature_list.json`
- **Threshold**: `models/threshold.json`

## 7. Assumptions & Limitations

### Key Assumptions
- Training data represents production population
- Feature relationships remain stable
- No significant concept drift in production
- Model used within approved risk thresholds

### Limitations
- **Data Scope**: Limited to available features
- **Temporal Validity**: Model performance may degrade over time
- **External Factors**: Does not account for macroeconomic conditions
- **Interpretability**: Black-box nature despite SHAP explanations

### Risk Mitigation
- Continuous monitoring for drift
- Regular model retraining
- Human oversight for high-risk cases
- Comprehensive testing before deployment

## 8. Version Control & Reproducibility

### Version Information
- **Model Version**: v1.0
- **Code Version**: Git commit hash
- **Data Version**: Kaggle dataset version
- **Environment**: Python 3.9, LightGBM 4.0

### Reproducibility
- All random seeds fixed
- Dependencies pinned in requirements.txt
- Training pipeline scripted in `src/train_pipeline.py`

## 9. Approval & Review

### Review History
- **Technical Review**: Completed
- **Business Review**: Pending
- **Compliance Review**: Pending

### Next Steps
- Model validation and sign-off
- Deployment readiness assessment
- Production monitoring setup

## 10. References

- Kaggle Competition: https://kaggle.com/competitions/playground-series-s5e11
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/

---

**Document Control**
- **Owner**: ML Engineering Team
- **Review Cycle**: Annual
- **Approval Required**: Yes
- **Confidentiality**: Internal Use Only