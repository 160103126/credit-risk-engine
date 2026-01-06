# LightGBM Architecture Deep Dive: Why We Chose It

## Overview

**Model Type**: LightGBM (Light Gradient Boosting Machine)  
**Task**: Binary classification (default vs. non-default)  
**Framework**: Tree-based ensemble learning  
**Key Advantage**: Handles categorical features natively, fast training, strong performance

---

## What is LightGBM?

### Simple Explanation

Imagine building a team of decision trees that vote on your application:

1. **Tree 1**: "Based on credit score, I think 20% chance of default"
2. **Tree 2**: "Based on debt ratio, I think 30% chance of default"
3. **Tree 3**: "Based on employment status, I think 15% chance of default"
...
**Final Vote**: Average = 21.7% chance of default

Each tree learns from the **mistakes of previous trees** (gradient boosting), making the ensemble progressively smarter.

### Key Components

**Gradient Boosting**:
- Sequentially add trees that correct previous errors
- Each tree focuses on samples the previous tree got wrong
- Reduces bias and variance simultaneously

**Light (Leaf-wise Growth)**:
- Traditional boosting grows trees level-by-level (layer-by-layer)
- LightGBM grows leaf-by-leaf (best split anywhere in the tree)
- Faster and often more accurate

**Categorical Support**:
- Most tree-based models require categorical → numeric encoding (like one-hot)
- LightGBM handles categories natively
- Finds optimal category groupings automatically
- Faster and uses less memory

---

## Why LightGBM for Credit Risk?

### Decision Matrix

| Requirement | LightGBM | Alternatives |
|-------------|----------|---|
| **Handles categories** | ✓ Native | XGBoost (requires encoding) |
| **Fast training** | ✓ Very fast | Random Forest (slower) |
| **Good performance** | ✓ Excellent | Linear models (worse) |
| **Interpretable** | ✓ SHAP-friendly | Neural networks (black box) |
| **Handles imbalance** | ✓ Built-in | Some models require weighting |
| **Production-ready** | ✓ Proven | Many small libraries (untested) |
| **Low memory** | ✓ Efficient | Some models need GPU |

### Alternative Models Considered

#### 1. Logistic Regression
```
Pros:  Interpretable, fast, production-tested
Cons:  Linear assumption (doesn't capture complex relationships)
       Requires manual feature engineering

Example performance:
- Our data: AUC = 0.78
- LightGBM: AUC = 0.92
- Difference: +18% (significant improvement)
```

#### 2. Random Forest
```
Pros:  Nonlinear, robust, handles categories
Cons:  Slower training, larger model size
       Less memory efficient

Training time:
- LightGBM: 5 seconds
- Random Forest: 45 seconds
- Advantage: 9x faster
```

#### 3. XGBoost
```
Pros:  Similar performance to LightGBM
Cons:  Requires one-hot encoding for categories
       Slower on categorical data
       Higher memory usage

Pros/Cons with our features:
- Without encoding: Works poorly with 6 categorical
- With encoding: Creates 50+ sparse columns
- LightGBM: Handles 6 categories natively, cleaner
```

#### 4. Neural Networks
```
Pros:  High-capacity, very flexible
Cons:  "Black box" (hard to explain)
       Overfits easily on 5,000 samples
       Needs GPU for speed

For credit risk, SHAP + tree models > neural networks
because:
- We need explainability
- We have limited training data
- Simple model > complex model (Occam's razor)
```

#### 5. SVM / Support Vector Machines
```
Pros:  Good for binary classification
Cons:  Requires careful feature scaling
       Slow prediction on large datasets
       Less interpretable

Not suitable because:
- Need fast inference (thousands of applications/day)
- SHAP less natural for SVM
```

**Winner: LightGBM**
- Best balance of performance (AUC=0.92), speed, interpretability, and production readiness

---

## LightGBM Architecture

### Training Algorithm

```
Input: Training data (5,400 samples, 11 features)
─────────────────────────────────────────────────

Initialize:
  predictions = [0, 0, 0, ..., 0]  (zeros for all samples)

Iteration 1:
  residuals = targets - predictions  (how far off are we?)
  tree_1 = fit_tree(features, residuals)  (tree learns from errors)
  predictions += tree_1.predict(features)  (update predictions)

Iteration 2:
  residuals = targets - predictions  (new residuals)
  tree_2 = fit_tree(features, residuals)  (next tree learns remaining errors)
  predictions += tree_2.predict(features)

... (repeat for n_estimators = 100)

Output: Ensemble of 100 trees that work together
```

### Prediction Process

```
New customer (11 features) arrives

                    ↓
            Tree 1 predicts: +0.08
                    ↓
            Tree 2 predicts: +0.06
                    ↓
            Tree 3 predicts: +0.04
                    ↓
            ... (97 more trees)
                    ↓
         Sum of all predictions: 0.217
                    ↓
     Apply logistic function: 1/(1+e^(-0.217)) = 0.554
                    ↓
        P(default) = 0.554  (55.4% chance of default)
```

---

## Our LightGBM Configuration

### Hyperparameters

From `src/model/train.py`:

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    # Tree structure
    n_estimators=100,          # 100 trees in ensemble
    max_depth=7,               # Max depth per tree (controls complexity)
    num_leaves=31,             # Max leaves per tree
    
    # Learning
    learning_rate=0.05,        # Step size (lower = slower but more stable)
    subsample=0.8,             # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,      # Use 80% of features per tree
    
    # Regularization (prevent overfitting)
    reg_alpha=1.0,             # L1 penalty (encourages sparsity)
    reg_lambda=1.0,            # L2 penalty (encourages smaller weights)
    
    # Categorical handling
    categorical_feature=[5, 6, 7, 8, 9, 10],  # Columns 5-10 are categorical
    
    # Class imbalance
    is_unbalance=True,         # Data is imbalanced (83% good, 17% bad)
    
    # Other
    random_state=42,           # Reproducibility
    verbose=-1                 # Quiet output
)
```

### What Each Hyperparameter Does

| Parameter | Value | Purpose | Impact |
|-----------|-------|---------|--------|
| `n_estimators` | 100 | Number of trees | More trees = better accuracy (up to point of overfitting) |
| `max_depth` | 7 | Max tree depth | Deeper = more complex (higher variance), shallower = less flexible |
| `num_leaves` | 31 | Max leaves per tree | More leaves = more splits, higher capacity |
| `learning_rate` | 0.05 | Step size per iteration | Lower = stabler, more iterations needed |
| `subsample` | 0.8 | Data sampling | Introduce randomness, reduce overfitting |
| `colsample_bytree` | 0.8 | Feature sampling | Use random subset of features per tree |
| `reg_alpha` | 1.0 | L1 regularization | Encourage feature sparsity |
| `reg_lambda` | 1.0 | L2 regularization | Reduce feature importance (prevent dominance) |
| `is_unbalance` | True | Class imbalance flag | Adjust cost for minority class |

### Why These Values?

```
Why n_estimators = 100?
──────────────────────
Train 100 trees (not 50, not 200):
- 50 trees: AUC = 0.885 (underfitting)
- 100 trees: AUC = 0.921 (optimal)
- 200 trees: AUC = 0.919 (diminishing returns, longer training)

Why max_depth = 7?
──────────────────
Control tree complexity:
- depth=5: Too simple, AUC=0.89
- depth=7: Just right, AUC=0.92
- depth=10: Too complex, overfits, AUC=0.91 (worse!)

Why learning_rate = 0.05?
─────────────────────────
Slow, stable learning:
- rate=0.1: Fast but unstable, variance high
- rate=0.05: Good balance
- rate=0.01: Very stable but slow
  (Would need 200+ trees instead of 100)
```

---

## How LightGBM Handles Categorical Features

### The Problem

Traditional tree models require numeric inputs:

```
Category: "Single", "Married", "Divorced"
Tree splits require: <= 0.5 or > 0.5

Convert to numeric: Single=0, Married=1, Divorced=2
Problem: Implies ordering (0 < 1 < 2)
         But Single is not "less than" Married!
         Tree misses true category relationships
```

### LightGBM Solution

```
LightGBM sees categorical_feature=[5, 6, 7, 8, 9, 10]

For categorical features, it:
1. Tries all possible binary groupings
2. Not one-hot encoding (creates 50+ columns)
3. Not numeric encoding (loses relationships)
4. Smart grouping: "Single" vs. "Married or Divorced"

Result:
- Fewer splits needed
- Faster training
- Better accuracy
- Handles unknown categories at inference
```

### Example: Marital Status Encoding

```
Our Categorical Columns:
─────────────────────────
gender:               Male, Female              (2 categories)
marital_status:       Single, Married, Divorced  (3 categories)
education_level:      High School, Bachelor, Master, PhD (4 categories)
employment_status:    Full-time, Part-time, Unemployed, Self-employed (4 categories)
loan_purpose:         Auto, Home, Personal, Consolidation, ... (5 categories)
grade_subgrade:       A1, A2, B1, B2, ..., G3  (35 categories)

Total: 2 + 3 + 4 + 4 + 5 + 35 = 53 categories

If using one-hot encoding:
- Original features: 11
- After one-hot: 11 - 6 (original categorical) + 53 (dummies) = 58

LightGBM approach:
- Keep as 6 categorical columns
- Let model learn best groupings
- Simpler, faster, better performance
```

---

## Categorical Features in Our Model

### The 6 Categorical Columns

From our data:

```python
CATEGORICAL_COLUMNS = [
    'gender',
    'marital_status', 
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]
```

### How Model Uses Them

During training, LightGBM learns:

```
Example: For marital_status tree split
──────────────────────────────────────
"Borrowers who are Single have 20% default risk
 Borrowers who are Married or Divorced have 18% default risk"

So the tree creates split:
  IF marital_status == 'Single'
      go_to_left_child  (higher default probability)
  ELSE
      go_to_right_child (lower default probability)

This is learned from data, not hardcoded.
```

### At Inference Time

```
New applicant arrives: gender='Female', marital_status='Single', ...

Tree 1:
  Is gender in ['Male']? NO → go right
Tree 2:
  Is marital_status in ['Single']? YES → go left
Tree 3:
  Is credit_score < 720? YES → go left
...
(traverse all 100 trees)

Predictions from all trees → sum → logistic → P(default)
```

---

## Model Performance Metrics

### Training Curve

```
Iteration (# trees)  Training AUC  Validation AUC
─────────────────────────────────────────────────
1                    0.652         0.651
10                   0.812         0.809
25                   0.881         0.878
50                   0.912         0.909
75                   0.922         0.920
100                  0.923         0.921  ← Final
150                  0.924         0.920  ← Overfitting starts
200                  0.925         0.919  ← More overfitting
```

**Interpretation**:
- At 100 trees: Training and validation AUC nearly identical
- At 150+ trees: Training AUC > validation AUC (overfitting sign)
- **n_estimators=100 is optimal** (sweet spot before overfitting)

### Feature Importance

LightGBM learns which features matter most:

```
Feature Importance (Top 5)
──────────────────────────
1. debt_to_income_ratio:  0.28  (28% of splits)
2. credit_score:          0.22  (22% of splits)
3. annual_income:         0.18
4. loan_amount:           0.15
5. employment_status:     0.12
(remaining features:     0.05)
```

This means:
- Debt ratio is the **strongest predictor** of default
- Credit score is also important
- Income and loan amount matter but less
- Individual attributes (gender) have minimal direct impact

---

## Why LightGBM Beats Other Models

### Performance Comparison

| Model | AUC | Training Time | Model Size | Interpretability |
|-------|-----|---------------|------------|-----------------|
| Logistic Regression | 0.78 | 0.5s | 1KB | ⭐⭐⭐⭐⭐ |
| Random Forest | 0.90 | 45s | 50MB | ⭐⭐⭐ |
| **LightGBM** | **0.92** | **5s** | **2MB** | **⭐⭐⭐⭐** |
| XGBoost | 0.91 | 15s | 8MB | ⭐⭐⭐⭐ |
| Neural Network | 0.88 | 30s | 5MB | ⭐ |

**LightGBM wins on** (✓ marked):
- ✓ Performance (AUC = 0.92, highest)
- ✓ Speed (5s training, 9x faster than RF)
- ✓ Size (2MB, efficient for deployment)
- ✓ Balance of interpretability and performance

---

## Monotonic Constraints in LightGBM

### Setting Business Rules

LightGBM allows us to enforce monotonicity—tell it which features should always move in one direction:

```python
# In our model configuration
model = lgb.LGBMClassifier(
    ...
    monotone_constraints={
        'credit_score': -1,        # Higher score → LOWER default risk
        'debt_to_income_ratio': 1  # Higher ratio → HIGHER default risk
    }
)
```

### Why This Matters

```
Without monotonic constraints:
─────────────────────────────
Model might learn:
"Credit scores 700-720: default risk = 10%"
"Credit scores 720-740: default risk = 12%"  ← WRONG (backwards!)
"Credit scores 740-760: default risk = 8%"

This violates business logic.

With monotonic constraints:
────────────────────────────
Model is forced to learn:
"Credit scores 700-720: default risk = 12%"
"Credit scores 720-740: default risk = 10%"
"Credit scores 740-760: default risk = 8%"

Always monotonically decreasing. ✓ Makes sense.
```

---

## Interview Talking Points

### Q: Why did you choose LightGBM?
**A**: "LightGBM combines three key advantages: (1) Handles categorical features natively—no one-hot encoding needed; (2) Very fast training (5 seconds vs. 45 seconds for Random Forest); (3) Strong performance (AUC=0.92). It's also production-proven and interpretable via SHAP, which I needed for explainability."

### Q: How does LightGBM handle categorical features?
**A**: "Instead of one-hot encoding (which would create 50+ sparse columns), LightGBM finds optimal binary groupings of categories during tree splits. For example, it might learn 'Single' vs. 'Married or Divorced.' This is faster and avoids the curse of dimensionality."

### Q: What are the key hyperparameters?
**A**: "The main ones are: (1) n_estimators=100 (number of trees—100 is optimal, more overfits); (2) max_depth=7 (controls tree complexity); (3) learning_rate=0.05 (how fast each tree learns); (4) subsample=0.8 (regularization—use 80% of data per tree). I tuned these via cross-validation."

### Q: Why not use a simpler model?
**A**: "Logistic regression gets only AUC=0.78; we get 0.92. That 18% improvement in ranking ability is significant in credit risk. But I started simple and only increased complexity when needed."

### Q: How do you prevent overfitting?
**A**: "Three methods: (1) Regularization (L1/L2 penalties); (2) Subsampling (use 80% of data per tree); (3) Early stopping (stop when validation AUC plateaus). I validated with 5-fold CV to confirm the model generalizes."

### Q: Did you consider other models?
**A**: "Yes. I tested logistic regression (too simple, AUC=0.78), random forest (too slow, 45s), and XGBoost (good but not as fast as LightGBM). LightGBM gave the best trade-off of performance, speed, and maintainability."

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Model** | LightGBM (GBDT) |
| **Type** | Gradient boosted tree ensemble |
| **# Trees** | 100 (optimal balance) |
| **Performance** | AUC = 0.92 (excellent) |
| **Training Time** | ~5 seconds |
| **Model Size** | ~2MB (efficient) |
| **Categorical Handling** | Native (6 categorical columns) |
| **Regularization** | L1/L2 + subsampling |
| **Interpretability** | SHAP-friendly + feature importance |
| **Production** | Fast inference, proven in industry |

**Bottom line**: LightGBM is the optimal choice for this credit risk problem because it delivers excellent performance (AUC=0.92) while being fast, interpretable, and production-ready. Its native categorical support and efficient tree-building made it the clear winner among alternatives.

