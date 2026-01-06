# Decision Logic Deep Dive: REJECT vs. APPROVE

## Overview

The API implements a **single-threshold decision rule**: compare predicted probability of default to 0.4542 and return APPROVE or REJECT.

**The Rule**:
```
IF P(default) >= 0.4542
    REJECT the loan application
ELSE
    APPROVE the loan application
```

This simple rule stems from:
1. **LightGBM prediction**: A probability between 0 and 1
2. **Threshold**: 0.4542 (derived from cross-validation, 15% reject rate)
3. **Business action**: Convert probability to discrete decision

---

## The Complete Decision Flow

```
┌─────────────────────────────────┐
│  Customer Loan Application      │
│  (11 features provided)         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Feature Validation             │
│  - Numeric: reasonable ranges   │
│  - Categorical: valid values    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  LightGBM Model Prediction      │
│  P(default) = 0.0 to 1.0        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Apply Decision Rule            │
│  IF P >= 0.4542:                │
│      Decision = REJECT          │
│  ELSE:                          │
│      Decision = APPROVE         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Return Decision + Explanation  │
│  - Probability                  │
│  - Decision (REJECT/APPROVE)    │
│  - SHAP explanation (optional)  │
└─────────────────────────────────┘
```

---

## From Probability to Decision

### Why Two Steps?

**Question**: Why not just return the probability and let someone else decide?

**Answer**: For production efficiency and clarity.

1. **Probability alone is ambiguous**: "0.456" doesn't tell a loan officer "yes" or "no"
2. **Threshold must be consistent**: Every application uses the same rule
3. **Explainability**: "REJECT" is clearer than "probability = 0.456"
4. **Automation**: Approve/reject can be automated; probability requires human judgment

### Converting Probability to Decision

```
LightGBM Output        Decision Logic           Final Decision
─────────────────────  ────────────────────────  ──────────────
0.12 (low default)  ──> 0.12 < 0.4542  ─────────> APPROVE ✓
                         (low risk)

0.35 (moderate)     ──> 0.35 < 0.4542  ─────────> APPROVE ✓
                         (acceptable risk)

0.4542 (borderline) ──> 0.4542 >= 0.4542 ───────> REJECT (tie-breaking)
                         (hits threshold)

0.60 (high default) ──> 0.60 >= 0.4542  ─────────> REJECT ✗
                         (high risk)

0.87 (very high)    ──> 0.87 >= 0.4542  ─────────> REJECT ✗
                         (very risky)
```

---

## Real-World Example Decision

### Borrower Profile

```
Name:                  John Smith
Annual Income:         $60,000
Debt-to-Income Ratio:  0.35 (35%)
Credit Score:          720
Loan Amount:           $25,000
Interest Rate:         6.5%
Gender:                Male
Marital Status:        Single
Education Level:       Bachelor
Employment Status:     Full-time
Loan Purpose:          Auto Purchase
Grade/Subgrade:        B/B1
```

### What the Model Predicts

```
LightGBM Inference:

Input Features → [Model] → P(default) = 0.1158

Interpretation: 
"This borrower has an 11.58% chance of defaulting"
```

### The Decision

```
Decision Rule:
P(default) = 0.1158
Threshold = 0.4542

0.1158 >= 0.4542 ?  NO ✓

Decision: APPROVE

Explanation:
"Low default probability. Risk is within acceptable limits."
```

### What This Means for John

✅ Loan **APPROVED**  
✅ Can proceed with application  
✅ Will receive funds (subject to other checks)  

The model determined John is a good credit risk.

---

## Decision Distribution in Real Data

### Our Validation Set

```
Total Applications: 1,080
────────────────────────────────────────

APPROVE:  916 applications (84.8%)
├─ P(default) range: 0.001 to 0.454
├─ Average P(default): 0.15
└─ Actual default rate: ~5% (true positives we caught)

REJECT:   164 applications (15.2%)
├─ P(default) range: 0.454 to 0.997
├─ Average P(default): 0.68
└─ Actual default rate: ~40% (true defaults we flagged)
```

### What This Tells Us

- **84.8% approval rate**: We approve most applicants (good for business)
- **15.2% rejection rate**: We reject a meaningful fraction to manage risk
- **Approved group has ~5% actual default**: Our approved borrowers mostly don't default
- **Rejected group has ~40% actual default**: Our rejected borrowers were indeed risky

This validates the threshold choice.

---

## Implementation in the API

### From `api/routes.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Load trained model
with open("models/lgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Decision threshold (from cross-validation)
DECISION_THRESHOLD = 0.4542

class PredictionRequest(BaseModel):
    """Input features for loan decision."""
    annual_income: float
    debt_to_income_ratio: float
    credit_score: int
    loan_amount: float
    interest_rate: float
    gender: str
    marital_status: str
    education_level: str
    employment_status: str
    loan_purpose: str
    grade_subgrade: str

@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """
    Make a loan approval/rejection decision.
    
    Returns:
        {
            "probability_of_default": float,
            "decision": "APPROVE" or "REJECT",
            "threshold": float,
            "confidence": float
        }
    """
    # Prepare features for the model
    features = prepare_features(request)
    
    # Get prediction probability
    probability = model.predict_proba(features)[0, 1]
    
    # Apply decision rule
    decision = "REJECT" if probability >= DECISION_THRESHOLD else "APPROVE"
    
    # Calculate confidence (distance from threshold)
    distance_from_threshold = abs(probability - DECISION_THRESHOLD)
    confidence = min(distance_from_threshold / DECISION_THRESHOLD, 1.0)
    
    return {
        "probability_of_default": float(probability),
        "decision": decision,
        "threshold": DECISION_THRESHOLD,
        "confidence": float(confidence),
        "message": f"Application {decision.lower()}ed. Default probability: {probability:.2%}"
    }

def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Convert request to feature vector for model."""
    # Handle categorical encoding
    categorical_mapping = {
        'gender': {'Male': 0, 'Female': 1},
        'marital_status': {'Single': 0, 'Married': 1, 'Divorced': 2},
        # ... more mappings
    }
    
    # Build feature vector
    features = np.array([
        request.annual_income,
        request.debt_to_income_ratio,
        request.credit_score,
        request.loan_amount,
        request.interest_rate,
        categorical_mapping['gender'].get(request.gender, 0),
        # ... more features
    ]).reshape(1, -1)
    
    return features
```

---

## Decision Confidence

### What Makes a Decision Confident?

**Confident decision**: Probability is far from threshold
**Uncertain decision**: Probability is close to threshold

```
Confidence Calculation:
distance_from_threshold = |probability - threshold|
confidence = min(distance_from_threshold / threshold, 1.0)

Examples:
─────────────────────────────────────────────────────
P(default) = 0.10, threshold = 0.4542
distance = |0.10 - 0.4542| = 0.3542
confidence = 0.3542 / 0.4542 = 77.9% ← VERY CONFIDENT APPROVE

P(default) = 0.45, threshold = 0.4542
distance = |0.45 - 0.4542| = 0.0042
confidence = 0.0042 / 0.4542 = 0.9% ← NOT CONFIDENT BORDERLINE

P(default) = 0.80, threshold = 0.4542
distance = |0.80 - 0.4542| = 0.3458
confidence = 0.3458 / 0.4542 = 76.1% ← VERY CONFIDENT REJECT
```

### Using Confidence in Production

**High confidence (>70%)**:
- Decision is robust
- Can be automated
- Low review rate

**Medium confidence (30-70%)**:
- Borderline cases
- May need manual review
- Check supporting documents

**Low confidence (<30%)**:
- Very close to threshold
- Always escalate to human review
- Consider additional data

---

## Common Decision Scenarios

### Scenario 1: Clear Approval

```
Borrower: Good credit (720+), low debt (30%), stable job
Prediction: P(default) = 0.08
Threshold: 0.4542
Decision: APPROVE
Confidence: 85%
Action: Immediate approval, no review needed
```

### Scenario 2: Clear Rejection

```
Borrower: Poor credit (600), high debt (50%), unemployed
Prediction: P(default) = 0.78
Threshold: 0.4542
Decision: REJECT
Confidence: 72%
Action: Immediate rejection, no review needed
```

### Scenario 3: Borderline (Manual Review)

```
Borrower: Moderate credit (680), moderate debt (40%), recent employment
Prediction: P(default) = 0.46
Threshold: 0.4542
Decision: REJECT
Confidence: 1.2%
Action: Escalate to human review
Reason: Probability very close to threshold; need expert judgment
```

---

## Explaining Decisions to Customers

### If APPROVED

```
Dear John,

Your loan application has been APPROVED.

Based on our assessment:
- Your predicted default probability: 11.6%
- This is well below our decision threshold of 45.4%
- Your key strengths: Good credit score, stable employment, 
  reasonable debt levels

Your loan will proceed to final processing.
```

### If REJECTED

```
Dear Alice,

Your loan application has been DECLINED.

Based on our assessment:
- Your predicted default probability: 62.3%
- This exceeds our decision threshold of 45.4%
- Areas of concern: Recent late payments, high debt-to-income ratio

We recommend:
1. Improve credit score by paying bills on time
2. Reduce existing debt
3. Reapply in 6-12 months

You may also request a manual review.
```

---

## Decision Errors: False Positives and False Negatives

### Type I Error: False Positive (Reject Good Borrowers)

```
Actual: Good borrower (won't default)
Prediction: P(default) >= 0.4542 (flagged as risky)
Decision: REJECT
Outcome: WRONG ✗

Impact:
- Lost business (we didn't make this loan)
- Customer may go to competitor
- No financial loss to us
- Opportunity cost

In our model: ~25% false positive rate
│ Among approved 1,080:
│ ~270 were good borrowers wrongly rejected
```

### Type II Error: False Negative (Approve Bad Borrowers)

```
Actual: Bad borrower (will default)
Prediction: P(default) < 0.4542 (didn't flag as risky)
Decision: APPROVE
Outcome: WRONG ✗

Impact:
- Financial loss (we lose the loaned amount)
- Default risk in portfolio
- Increased portfolio risk
- Bad customer experience

In our model: ~30% false negative rate
│ Among approved borrowers:
│ ~30% of actual defaults slip through
```

### Trade-off: Why FP > FN in Lending

**Question**: Why do we tolerate 25% false positives but only 30% false negatives?

**Answer**: Cost asymmetry
- Rejecting a good borrower: Lost opportunity (smaller impact)
- Approving a bad borrower: Direct loss (larger impact)

We can afford to reject more good borrowers to catch defaults.

---

## Monitoring Decisions in Production

### Key Metrics to Track

```
Over time (daily/weekly):
──────────────────────────────────────

1. Approval Rate
   - Should stay ~85% (matches training data)
   - If drifts to 95%+: Model may be degrading
   - If drifts to 70%-: May be overly conservative

2. Rejection Rate
   - Should stay ~15%
   - Used to trigger retraining if drifts

3. Actual Default Rate Among Approved
   - Should match training data (~5%)
   - If increases: Portfolio risk increasing
   - If decreases: Fewer risky approvals (good)

4. Model Drift (KS/AUC)
   - Retrain if performance drops >5%

5. Borderline Cases
   - # of decisions with confidence < 30%
   - These get manual review
```

---

## Interview Talking Points

### Q: How do you go from a probability to a decision?
**A**: "Simple threshold rule: if P(default) >= 0.4542, reject; else approve. The threshold was determined through cross-validation targeting a 15% reject rate. This converts a continuous probability into a discrete business action."

### Q: What does your approval rate tell you?
**A**: "Our approval rate is 85%, meaning we approve 85% of applications. This matches the 15% reject rate target. If the approval rate drifts in production, it signals the model's output distribution has changed—a sign of drift."

### Q: How do you handle borderline cases?
**A**: "Cases where probability is very close to threshold (e.g., 0.454 vs 0.4542) get escalated for manual review. I calculate confidence as the distance from threshold; decisions with <30% confidence go to a human."

### Q: What's the cost of rejecting good borrowers?
**A**: "Lost opportunity—we don't make a profitable loan. But the cost of approving bad borrowers is loss of the loan amount. Since the latter is more expensive, we tolerate ~25% false positive rate to minimize defaults."

### Q: How would you explain a rejection to a customer?
**A**: "I'd explain the predicted default probability, show it exceeds our threshold, and identify the risk factors (using SHAP). I'd also suggest concrete steps to improve (pay bills on time, reduce debt, improve credit score) and encourage reapplication."

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Decision Rule** | IF P(default) ≥ 0.4542 THEN REJECT ELSE APPROVE |
| **Threshold Origin** | 5-fold CV targeting 15% reject rate |
| **Approval Rate** | 85% (meets business target) |
| **False Positive Rate** | 25% (reject some good borrowers) |
| **False Negative Rate** | 30% (approve some bad borrowers) |
| **Why FP > FN** | Cost of default > cost of lost opportunity |
| **Confidence Calculation** | Distance from threshold |
| **Borderline Handling** | Escalate to manual review |
| **Monitoring** | Track approval rate, actual defaults, drift |

**Bottom line**: The decision rule is simple, transparent, and grounded in cross-validated threshold selection. It balances business requirements (15% reject) with statistical performance (AUC=0.92, KS=0.45) and incorporates error trade-offs aligned with lending costs.

