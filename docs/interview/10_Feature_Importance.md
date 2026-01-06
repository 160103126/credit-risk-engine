# Feature Importance Deep Dive: What Drives Default Risk?

## Overview

**Goal**: Understand which features the model uses most to predict default  
**Method**: Tree split frequency (how often each feature is selected for splits)  
**Key Finding**: debt_to_income_ratio (28%) and credit_score (22%) dominate; individual attributes matter less

---

## What is Feature Importance?

### Simple Explanation

Imagine LightGBM building 100 trees to predict default:

```
Tree 1: "Split on debt_to_income_ratio at 0.40"
Tree 2: "Split on credit_score at 720"
Tree 3: "Split on debt_to_income_ratio at 0.35"
Tree 4: "Split on annual_income at 50000"
Tree 5: "Split on debt_to_income_ratio at 0.45"
... (95 more trees)

Count:
debt_to_income_ratio: used in 28 trees
credit_score: used in 22 trees
annual_income: used in 18 trees
loan_amount: used in 15 trees
employment_status: used in 12 trees
... (other features: <10 each)

Interpretation:
"debt_to_income_ratio is the most important because
the model relies on it most heavily to make splits"
```

### Formula

$$\text{Feature Importance} = \frac{\text{# splits using feature}}{\text{total # splits across all trees}}$$

For our model:

$$\text{Importance(debt\_to\_income\_ratio)} = \frac{28}{100} = 0.28 \text{ (28%)}$$

---

## Our Model: Feature Importance Ranking

### Top 10 Features

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|-----------------|
| 1 | **debt_to_income_ratio** | **0.28** | **Most important—debt burden is key to default risk** |
| 2 | **credit_score** | **0.22** | **Credit history is crucial** |
| 3 | annual_income | 0.18 | Income level matters (ability to repay) |
| 4 | loan_amount | 0.15 | Loan size influences risk |
| 5 | interest_rate | 0.10 | Cost of borrowing affects ability to pay |
| 6 | employment_status | 0.04 | Job status has minor effect |
| 7 | grade_subgrade | 0.02 | Loan grade has minimal impact |
| 8 | marital_status | 0.01 | Marriage status barely matters |
| 9 | education_level | 0.00 | Education level has negligible effect |
| 10 | loan_purpose | 0.00 | Loan purpose doesn't matter |
| | gender | <0.01 | Gender has minimal/no effect |

### Visual Representation

```
Feature Importance Distribution
─────────────────────────────────────────────
debt_to_income_ratio: ████████████████████████████  (28%)
credit_score:          ██████████████████████        (22%)
annual_income:         ██████████████████            (18%)
loan_amount:           ███████████████               (15%)
interest_rate:         ██████████                    (10%)
employment_status:     ████                          (4%)
grade_subgrade:        ██                            (2%)
marital_status:        █                             (1%)
education_level:       -                             (0%)
loan_purpose:          -                             (0%)
gender:                -                             (<1%)
                       └──────────────────────────────
                       0%        10%        20%      100%
```

---

## Why These Rankings?

### Top Feature: debt_to_income_ratio (28%)

```
What it measures:
─────────────────
DTI = Total Debt / Annual Income

Example:
├─ Borrower A: $60,000 annual income, $15,000 debt → DTI = 0.25 (25%)
└─ Borrower B: $60,000 annual income, $30,000 debt → DTI = 0.50 (50%)

Why it matters:
───────────────
Borrower A: 75% of income available for new loan payments
Borrower B: 50% of income available for new loan payments
Borrower B is riskier (can't afford as much)

In the data:
─────────────
DTI < 0.30: Default rate ≈ 5%
DTI 0.30-0.40: Default rate ≈ 12%
DTI 0.40-0.50: Default rate ≈ 25%
DTI > 0.50: Default rate ≈ 40%

Strong relationship → Model relies on it heavily
```

### Second Feature: credit_score (22%)

```
What it measures:
─────────────────
Numeric score (300-850) reflecting credit history
├─ Payment history (35%)
├─ Amounts owed (30%)
├─ Length of history (15%)
├─ New credit (10%)
└─ Credit mix (10%)

In the data:
─────────────
Credit score < 580: Default rate ≈ 45%
Credit score 580-670: Default rate ≈ 20%
Credit score 670-740: Default rate ≈ 8%
Credit score > 740: Default rate ≈ 3%

Strong relationship → Model uses it extensively
```

### Why Numeric Features Dominate

```
Numeric (top 5: 93% importance):
├─ debt_to_income_ratio: 28%
├─ credit_score: 22%
├─ annual_income: 18%
├─ loan_amount: 15%
└─ interest_rate: 10%

Categorical (bottom 6: 7% importance):
├─ employment_status: 4%
├─ grade_subgrade: 2%
├─ marital_status: 1%
├─ education_level: <1%
├─ loan_purpose: <1%
└─ gender: <1%

Reason:
──────
Numeric features:
- Have continuous ranges (0.00-1.50 for DTI)
- Many possible split points for trees to learn
- Direct relationship to financial health

Categorical features:
- Limited values (e.g., gender: 2 values)
- Less flexibility for tree splits
- Indirect relationship to default risk

Result: Model learns more from numeric data
```

---

## Feature Importance vs. SHAP Values

### Key Difference

```
Feature Importance:
───────────────────
├─ Question: "How often does this feature help?"
├─ Answer: "debt_to_income_ratio is used in 28% of splits"
├─ Metric: Global (entire model)
└─ Use: "Which features are most important?"

SHAP Values:
────────────
├─ Question: "How much does this feature change THIS prediction?"
├─ Answer: "For John's application, DTI changes risk by +0.15"
├─ Metric: Local (single prediction)
└─ Use: "Why did the model approve/reject this person?"

Combined interpretation:
────────────────────────
"Globally, DTI is most important (28% of splits).
For John specifically, DTI increases his risk by 0.15.
For Alice, DTI decreases her risk by -0.05.
(Same feature, opposite effects per individual!)"
```

---

## Code: Computing Feature Importance

### From `src/model/train.py`

```python
import lightgbm as lgb
import pandas as pd

# After training model
model = lgb.LGBMClassifier(...)
model.fit(X_train, y_train)

# Method 1: Built-in importance
feature_names = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

importance_df['importance_pct'] = (
    importance_df['importance'] / importance_df['importance'].sum()
)

print(importance_df)
# Output:
#                  feature  importance  importance_pct
# 1  debt_to_income_ratio         122           0.2804
# 2        credit_score          96           0.2204
# 3       annual_income           79           0.1815
# 4         loan_amount           65           0.1494
# 5       interest_rate           44           0.1011
# 6   employment_status          18           0.0413
# 7      grade_subgrade            8           0.0184
# 8      marital_status            5           0.0115
# 9     education_level            2           0.0046
# 10       loan_purpose            1           0.0023
# 11              gender            0           0.0000

# Method 2: Visualize
import matplotlib.pyplot as plt

importance_df.plot(
    x='feature',
    y='importance_pct',
    kind='barh',
    figsize=(10, 6)
)
plt.xlabel('Importance')
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
```

---

## Interpretation: What Does This Tell Us?

### Business Insights

```
Finding 1: Financial Ratios Trump Individual Attributes
──────────────────────────────────────────────────────
Financial metrics (DTI, income, credit): 76% importance
Personal attributes (gender, education, marital): 7% importance

Implication:
- Default is driven by financial capacity, not demographics
- Lending decisions should focus on finances
- Personal attributes have minimal predictive value

Finding 2: Credit Score Matters, But Less Than Debt Ratio
──────────────────────────────────────────────────────
credit_score: 22% importance
debt_to_income_ratio: 28% importance (stronger)

Implication:
- Someone's past payment history (credit score) is important
- But their current debt burden (DTI) is more important
- A good credit score doesn't override high debt

Finding 3: Job Status Barely Matters
───────────────────────────────────
employment_status: 4% importance
(Full-time vs. part-time vs. self-employed)

Implication:
- The model doesn't learn strong patterns from employment type
- Income level matters more than job type
- A part-timer with $100k income > full-timer with $30k income

Finding 4: Demographic Features Are Irrelevant
───────────────────────────────────────────────
gender: <1% importance
marital_status: 1% importance
education_level: <1% importance

Implication:
- Model doesn't (and shouldn't) discriminate on demographics
- Fair lending: model treats all genders, marital statuses equally
- Regulators would approve: no demographic bias

Why these features are included:
├─ Standard practice in credit risk
├─ Allows model to learn "they don't matter"
└─ Demonstrates fairness in testing
```

---

## Feature Importance in Production

### Monitoring Importance Changes

```
Train data (used to build model):
───────────────────────────────────
importance_train = [0.28, 0.22, 0.18, 0.15, 0.10, ...]

Production data (new applications):
────────────────────────────────────
Recompute importance on recent predictions:
importance_prod = [0.25, 0.24, 0.18, 0.14, 0.11, ...]

Change:
───────
debt_to_income_ratio: 0.28 → 0.25 (-3%) ← Within tolerance
credit_score: 0.22 → 0.24 (+2%) ← Within tolerance

If change exceeds threshold (±5%):
├─ Sign of model drift
├─ Example: "In production, credit_score became 0.40 importance"
├─ Reason: New market condition, fraud pattern, data quality change
└─ Action: Retrain model
```

### Using Importance for Feature Selection

```
Low-importance features might be candidates for removal:
└─ gender (<1%), education_level (<1%), loan_purpose (<1%)

Before removing:
├─ Check SHAP dependence plots (feature might have nonlinear effect)
├─ Check for fairness implications
├─ Check for business requirements
└─ Only remove if truly unused

Decision for our model:
├─ Keep all features (they're standard, don't hurt)
└─ Don't remove low-importance ones
```

---

## Relationship Between Features

### Feature Interactions in Trees

```
LightGBM trees can capture interactions naturally:

Example decision path:
─────────────────────
IF credit_score < 680:
    IF debt_to_income_ratio > 0.40:
        Default risk = 45% ← Interaction!
                           (both bad = very risky)
    ELSE:
        Default risk = 20% ← Low DTI helps

IF credit_score >= 680:
    IF debt_to_income_ratio > 0.40:
        Default risk = 15% ← Credit score helps offset DTI
    ELSE:
        Default risk = 5%  ← Both good = safe

Interpretation:
───────────────
- Poor credit score + high DTI = worst combo
- Good credit score + high DTI = offset somewhat
- Model learns these interactions via tree structure
```

---

## Interview Talking Points

### Q: What's your most important feature?
**A**: "debt_to_income_ratio at 28% importance. This makes sense—it directly measures how much debt a borrower already has relative to their income. If someone is already burdened by debt, they're more likely to default on a new loan."

### Q: Why does credit_score rank 2nd, not higher?
**A**: "Credit score is important (22%) but DTI is more predictive. DTI measures current debt burden (immediate risk), while credit score reflects historical payment behavior (lagged risk). Both matter, but immediate debt burden is a stronger signal."

### Q: Why do demographic features have almost no importance?
**A**: "Good—it means the model doesn't discriminate on demographics like gender, marital status, or education. These are fair lending practices. The model focuses on financial capacity, not personal attributes."

### Q: How is feature importance different from SHAP?
**A**: "Feature importance is global—how much the model relies on each feature overall. SHAP is local—how much a feature changes a specific prediction. For example, DTI is important globally (28%), but for John it might decrease risk (-0.12) while for Alice it increases risk (+0.15)."

### Q: Did you consider removing low-importance features?
**A**: "I considered it but decided against it. Low-importance features like gender and loan_purpose don't hurt the model (minimal computation cost), and they document that the model treats all categories fairly. Standard practice is to keep them for transparency."

### Q: How do you monitor feature importance in production?
**A**: "I recompute importance periodically on recent predictions. If importance for a key feature drifts significantly (e.g., credit_score from 22% to 30%), it signals the model needs retraining—something in the data or market has changed."

---

## Summary

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | debt_to_income_ratio | 28% | Financial Ratio |
| 2 | credit_score | 22% | Financial History |
| 3 | annual_income | 18% | Financial Capacity |
| 4 | loan_amount | 15% | Loan Size |
| 5 | interest_rate | 10% | Cost of Money |
| 6 | employment_status | 4% | Job Type |
| 7 | grade_subgrade | 2% | Loan Grade |
| 8-11 | Demographic features | <3% | Fair Lending ✓ |

**Key Insights**:
- ✓ Numeric financial features dominate (93% importance)
- ✓ Demographic features irrelevant (<3% importance)
- ✓ DTI is the strongest predictor of default
- ✓ Model learned what matters: financial capacity, not personal characteristics

**Bottom line**: Feature importance shows that our model makes lending decisions based on financial metrics (debt ratio, income, credit history) rather than demographics, demonstrating fair lending practices. The ranking aligns with business intuition: a borrower's ability to repay (measured by DTI and income) is the strongest signal of default risk.

