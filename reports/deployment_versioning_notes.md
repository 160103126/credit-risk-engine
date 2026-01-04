# Deployment & Versioning Notes

## Document Information
- **Document Version**: 1.0
- **Date**: January 3, 2026
- **Model Version**: Credit Risk Engine v1.0
- **Deployment Environment**: Production

## 1. Model Version Information

### Model Details
- **Model ID**: CRE-2026-001
- **Algorithm**: LightGBM v4.0.0
- **Training Date**: January 3, 2026
- **Training Data Period**: N/A (Kaggle synthetic dataset)
- **Training Sample Size**: ~80,000 applications

### Performance Metrics
- **AUC**: 0.92
- **KS Statistic**: 0.45
- **Cross-Validation AUC**: 0.9211 ± 0.0007

## 2. Feature Schema

### Input Features
Model expects the following features (numeric + categorical). Inference code converts object columns to pandas Categorical automatically and aligns feature order to the model’s training schema.

```json
{
  "annual_income": "float64",
  "debt_to_income_ratio": "float64",
  "credit_score": "int64",
  "loan_amount": "float64",
  "interest_rate": "float64",
  "gender": "category",
  "marital_status": "category",
  "education_level": "category",
  "employment_status": "category",
  "loan_purpose": "category",
  "grade_subgrade": "category"
}
```

### Categorical Handling
Categorical columns are coerced to pandas Categorical without fixed category lists. Unknown labels are allowed and handled by LightGBM at inference.

### Data Types & Ranges
- **Age**: 18-100
- **Annual Income**: 0-1,000,000
- **Credit Score**: 300-850
- **DTI Ratio**: 0-100
- **Loan Amount**: 0-1,000,000

## 3. Model Artifacts

### Files Required for Deployment
- `models/lgb_model.pkl`: Trained LightGBM model
- `models/threshold.json`: Decision threshold configuration
- `models/feature_list.json`: Feature order for model input
- `models/config.pkl`: Model configuration

### Model Signature
- **Input**: Pandas DataFrame with 17 features
- **Output**: Probability score (float 0-1)
- **Preprocessing**: Categorical encoding, no scaling

## 4. Deployment Environment

### Infrastructure Requirements
- **OS**: Linux/Windows
- **Python**: 3.9+
- **Memory**: 2GB RAM minimum
- **Storage**: 500MB for model artifacts
- **Network**: API access for predictions

### Dependencies
```
lightgbm==4.0.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
fastapi==0.100.0
uvicorn==0.23.0
joblib==1.3.0
```

### Containerization
- **Docker Image**: `credit-risk-engine:v1.0`
- **Base Image**: `python:3.9-slim`
- **Port**: 8000
- **Health Checks**: `/healthz` (liveness), `/readiness` (readiness)

## 5. API Specification

### Endpoints
- **GET /healthz**: Liveness check
- **GET /readiness**: Model loaded and dry-run predict
- **GET /version**: Model metadata and threshold
- **POST /predict**: Probability of default and approve/reject decision
- **POST /explain**: Probability with SHAP contributions

### Error Handling
- **400**: Invalid input format
- **422**: Validation error
- **500**: Internal server error
- **Fallback**: Reject on system errors

## 6. Versioning Strategy

### Semantic Versioning
- **Major**: Breaking changes (new features, API changes)
- **Minor**: New features, improvements
- **Patch**: Bug fixes, performance improvements

### Model Versioning
- **v1.0**: Initial production release
- **Version Format**: `CRE-{YEAR}-{SEQUENTIAL}`
- **Backward Compatibility**: Maintained within major versions

### Deployment Strategy
- **Blue-Green**: Zero-downtime deployments
- **Canary**: Gradual rollout (10% → 50% → 100%)
- **Rollback**: < 5 minutes to previous version

## 7. Testing & Validation

### Pre-Deployment Tests
- **Unit Tests**: All functions pass
- **Integration Tests**: API endpoints work
- **Performance Tests**: <100ms response time
- **Load Tests**: 1000 concurrent requests

### Validation Checks
- **Feature Consistency**: API aligns inputs to training schema
- **Output Range**: Probabilities between 0-1
- **Deterministic**: Same input → same output
- **Threshold Application**: Applied in API (not in training)

## 8. Monitoring & Alerting

### Key Metrics
- **Latency**: P95 < 200ms
- **Error Rate**: < 1%
- **Throughput**: 1000 req/min
- **Model Drift**: PSI < 0.25

### Alert Conditions
- **Critical**: Service down, error rate > 5%
- **Warning**: Latency > 500ms, drift detected
- **Info**: Performance degradation

## 9. Backup & Recovery

### Model Backup
- **Storage**: Versioned in MLflow registry
- **Retention**: All versions kept indefinitely
- **Recovery**: < 1 hour to restore any version

### Data Backup
- **Logs**: 30 days retention
- **Predictions**: 1 year retention for audit
- **Disaster Recovery**: Multi-region replication

## 10. Change Management

### Deployment Process
1. **Code Review**: Pull request approval
2. **CI/CD**: Automated testing
3. **Staging**: Deploy to staging environment
4. **Approval**: Business sign-off
5. **Production**: Automated deployment

### Rollback Procedure
1. **Detection**: Alert triggers investigation
2. **Decision**: Business determines rollback
3. **Execution**: Automated rollback to previous version
4. **Verification**: Confirm system stability
5. **Post-mortem**: Root cause analysis

## 11. Security Considerations

### Access Control
- **API Keys**: Required for production access
- **Rate Limiting**: Prevents abuse
- **Input Validation**: Prevents injection attacks

### Data Protection
- **Encryption**: Data at rest and in transit
- **PII Handling**: No personal data stored
- **Compliance**: GDPR/CCPA compliant

## 12. Support & Maintenance

### Support Contacts
- **Technical Issues**: ML Engineering Team
- **Business Issues**: Risk Management Team
- **Customer Disputes**: Customer Service

### Maintenance Schedule
- **Daily**: Automated health checks
- **Weekly**: Performance monitoring
- **Monthly**: Security updates
- **Quarterly**: Model performance review
- **Annually**: Full system audit

## 13. Appendices

### Appendix A: Model Card
- **Model Type**: Binary Classifier
- **Intended Use**: Credit Risk Assessment
- **Limitations**: Synthetic data, no temporal validation
- **Ethical Considerations**: Fair lending compliance

### Appendix B: API Examples
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "annual_income": 50000, ...}'
```

### Appendix C: Troubleshooting
- **High Latency**: Check system resources
- **Prediction Errors**: Validate input format
- **Model Drift**: Retrain with new data

---

**Document Control**
- **Owner**: DevOps Team
- **Review Cycle**: With each deployment
- **Approval Required**: Yes
- **Distribution**: Development, Operations, Risk Teams