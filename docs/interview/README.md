# Credit Risk Engine - Interview Deep Dive Documentation

This folder contains comprehensive, interview-ready documentation on all key concepts and decisions in the Credit Risk Engine project. Each document provides deep understanding with actual project values, metrics, and code.

## Document Roadmap

### 1. **Model Performance Metrics**
   - [01_KS_Statistic.md](01_KS_Statistic.md) - KS = 0.45 (what it is, how calculated, why it matters)
   - [02_AUC_ROC.md](02_AUC_ROC.md) - AUC = 0.92 (interpretation, threshold-independence, why better than accuracy)

### 2. **Threshold & Decision Logic**
   - [03_Threshold_Selection.md](03_Threshold_Selection.md) - How we chose 0.4542 (15% reject rate, cross-validation approach)
   - [04_Decision_Logic.md](04_Decision_Logic.md) - Probability → APPROVE/REJECT (threshold rule, business rules)

### 3. **Model Understanding**
   - [05_LightGBM_Architecture.md](05_LightGBM_Architecture.md) - Why LightGBM (categorical support, speed, performance)
   - [06_Monotonic_Constraints.md](06_Monotonic_Constraints.md) - credit_score: -1 (enforcing business logic)
   - [07_Feature_Importance.md](07_Feature_Importance.md) - Top features (debt_to_income_ratio, credit_score, etc.)

### 4. **Explainability**
   - [08_SHAP_Explainability.md](08_SHAP_Explainability.md) - SHAP values, global/individual/dependence plots
   - [09_Feature_Contributions.md](09_Feature_Contributions.md) - How features affect individual predictions

### 5. **Data & Preprocessing**
   - [10_Categorical_Encoding.md](10_Categorical_Encoding.md) - Why pandas Categorical (no fixed mappings)
   - [11_Data_Preprocessing.md](11_Data_Preprocessing.md) - Target creation (default = 1-loan_paid_back), column drops
   - [12_Imbalanced_Data.md](12_Imbalanced_Data.md) - Class imbalance, why AUC/KS > accuracy

### 6. **Validation & Testing**
   - [13_Cross_Validation.md](13_Cross_Validation.md) - 5-fold stratified CV (why, how stable)
   - [14_Train_Validation_Split.md](14_Train_Validation_Split.md) - 80/20 with stratification

### 7. **API & Deployment**
   - [15_API_Design.md](15_API_Design.md) - /predict, /explain, /healthz, /readiness, /version
   - [16_Categorical_Coercion.md](16_Categorical_Coercion.md) - How API ensures LightGBM compatibility

### 8. **Key Decisions & Trade-offs**
   - [17_Why_These_Choices.md](17_Why_These_Choices.md) - Model selection, metrics, threshold (business vs. ML)

---

## Quick Reference: Project Values

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **AUC (Validation)** | 0.92 | Excellent discrimination |
| **KS Statistic** | 0.45 | 45% maximum separation between good & bad |
| **Cross-Validation AUC** | 0.9211 ± 0.0007 | Stable, generalizes well |
| **Decision Threshold** | 0.4542 | Chosen for 15% reject rate |
| **Model Type** | LightGBM | Fast, handles categorical, monotonic constraints |
| **Features** | 11 | numeric + categorical |
| **Categorical Columns** | 6 | gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade |
| **Monotonic Constraint** | credit_score: -1 | Higher score → lower risk |
| **Cross-Validation** | 5-fold stratified | Ensures stable performance |
| **Train/Val Split** | 80/20 | With stratification |

---

## How to Use This Documentation

1. **For interviews**: Start with README (this file), then pick a topic based on the question
2. **For deep understanding**: Read sequentially: Metrics → Threshold → Model → Data → API
3. **For code walkthrough**: Each document references actual code files and line numbers
4. **For examples**: Each metric uses actual project values, not random examples

---

## Key Concepts at a Glance

### What Makes This Model Production-Ready?

✅ **High AUC (0.92)**: Model ranks defaulters vs. non-defaulters very well  
✅ **Strong KS (0.45)**: Excellent maximum separation at optimal threshold  
✅ **Stable CV (±0.0007)**: Model generalizes across different data splits  
✅ **Explainable (SHAP)**: Every prediction can be explained to stakeholders  
✅ **Monotonic Logic**: Credit score and DTI follow business intuition  
✅ **Categorical Handling**: Works with unknown categories at inference  
✅ **Health Checks**: /healthz, /readiness, /version for production monitoring  

---

## Interview Question Examples

**Q: What's your KS statistic and what does it mean?**
→ See: [01_KS_Statistic.md](01_KS_Statistic.md)

**Q: Why did you choose that threshold?**
→ See: [03_Threshold_Selection.md](03_Threshold_Selection.md)

**Q: How does your model handle categorical features?**
→ See: [10_Categorical_Encoding.md](10_Categorical_Encoding.md)

**Q: Why LightGBM over other models?**
→ See: [05_LightGBM_Architecture.md](05_LightGBM_Architecture.md)

**Q: How do you ensure the model makes fair decisions?**
→ See: [06_Monotonic_Constraints.md](06_Monotonic_Constraints.md)

**Q: What does your AUC mean and how is it calculated?**
→ See: [02_AUC_ROC.md](02_AUC_ROC.md)

**Q: How can you explain a single prediction to a customer?**
→ See: [08_SHAP_Explainability.md](08_SHAP_Explainability.md)

---

**Created**: January 6, 2026  
**Project**: Credit Risk Engine  
**Purpose**: Interview preparation & deep technical understanding
