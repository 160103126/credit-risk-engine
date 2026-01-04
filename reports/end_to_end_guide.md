# End-to-End Credit Risk Engine Guide

## Overview
This guide walks through the complete process of building, deploying, and monitoring a production-ready credit risk assessment model using LightGBM.

## 1. Data Preparation

### Raw Data
- **Source**: Kaggle playground dataset (train.csv, test.csv, sample_submission.csv)
- **Location**: `data/raw/`
- **Features**: 17 features including demographics, financials, loan details

### Preprocessing
- **Target Creation**: `default = (loan_paid_back == 0).astype(int)`
- **Categorical Encoding**: Convert object columns to category dtype for LightGBM
- **Cleaning**: Drop 'id', handle no missing values
- **Code**: `src/data/preprocess.py`

### Feature Engineering
- **Additional Features**: None added (data is clean)
- **Scaling**: Not needed (tree-based model)
- **Code**: N/A (no additional features module in current version)

## 2. Model Development

### Algorithm Choice
- **LightGBM**: Fast, handles categorical, good for tabular data
- **Why**: Better performance than XGBoost for this dataset, built-in categorical support

### Training Setup
- **Split**: 80/20 train/validation with stratification + 5-fold CV for thresholds
- **Monotonic Constraints**: `credit_score` (-1)
- **Hyperparameters**: n_estimators=1000, learning_rate=0.05
- **Code**: `src/model/train.py`

### Evaluation
- **Metrics**: AUC, KS statistic, confusion matrix
- **Threshold**: 15% reject rate (probability > 0.4542 → reject)
- **Cross-Validation**: 5-fold stratified for stability
- **Code**: `src/model/evaluate.py`, `src/model/thresholds.py`

## 3. Explainability
- **SHAP**: Global summary, individual bar plots, dependence plots
- **Feature Importance**: Gain-based ranking
- **Why**: Regulatory compliance, business understanding
- **Code**: `src/explainability/`

## 4. Experiment Tracking
- **MLflow**: Log params, metrics, models
- **Benefits**: Reproducibility, comparison, versioning
- **Code**: Integrated in training pipeline

## 5. Deployment
- **API**: FastAPI with Pydantic schemas
- **Containerization**: Docker + docker-compose
- **Serving**: Real-time predictions
- **Health**: `/healthz`, `/readiness`, `/version` endpoints
- **Code**: `api/`, `Dockerfile`

## 6. Monitoring
- **Data Drift**: PSI for feature distributions
- **Model Drift**: KS statistic changes
- **Production Metrics**: Approval rates, prediction distributions
- **Alerting**: Threshold-based notifications
- **Code**: `src/monitoring/`

## 7. Testing
- **Unit Tests**: Data loading, preprocessing, model functions
- **Integration Tests**: Full pipeline
- **Code**: `tests/`

## Workflow Execution

### Development
1. `python src/train_pipeline.py` - Train model with MLflow
2. `mlflow ui` - View experiments
3. `pytest` - Run tests

### Production
1. `docker-compose up` - Start API
2. Send POST requests to /predict
3. `python src/monitor.py` - Check for drift

### Maintenance
- Monitor metrics weekly
- Retrain if drift detected
- Update models via MLflow registry

## Key Decisions

### Metrics Choice
- **AUC**: Threshold-independent, good for imbalanced data
- **KS**: Industry standard for credit scoring
- **Confusion Matrix**: Business-relevant (costs of FP vs FN)

### Threshold Selection
- **15% Reject Rate**: Balances risk and business volume
- **Fixed Rate**: More stable than fixed probability threshold

### Architecture
- **Modular**: Separate concerns (data, model, API)
- **Config-Driven**: YAML for parameters
- **Containerized**: Easy deployment and scaling

## Challenges & Solutions
- **Imbalanced Data**: Stratified sampling, AUC/K S metrics
- **Explainability**: SHAP for transparency
- **Drift**: PSI/KS monitoring
- **Scalability**: API with async processing

## Technologies Used
- **ML**: LightGBM, SHAP, scikit-learn
- **MLOps**: MLflow
- **API**: FastAPI, Pydantic
- **Deployment**: Docker
- **Monitoring**: Custom scripts
- **Testing**: pytest

## Next Steps
- Add CI/CD pipeline
- Implement A/B testing
- Add more advanced drift detection (e.g., alibi-detect)
- Scale to multiple models