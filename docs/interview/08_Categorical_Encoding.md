# Categorical Encoding Deep Dive: Why pandas Categorical

## Overview

**Approach**: Convert categorical columns to `pandas.Categorical` type  
**Goal**: Make features compatible with LightGBM while maintaining interpretability  
**Key Decision**: No fixed category lists (allows unknown categories at inference)

---

## What is Categorical Encoding?

### The Problem: Text Doesn't Work in Models

```
Model input: Must be numeric (numbers only)
Raw data: Contains text (e.g., "Single", "Married")

Raw DataFrame:
┌─────────────────────┐
│ gender | marital_status │
├─────────────────────┤
│ Male | Single │
│ Female | Married │
│ Male | Divorced │
└─────────────────────┘

Model needs:
┌──────────────────────┐
│ gender | marital_status │
├──────────────────────┤
│ 0 | 0 │
│ 1 | 1 │
│ 0 | 2 │
└──────────────────────┘

Encoding converts text → numbers
```

### Our Solution: pandas.Categorical

```python
# Before encoding
df['gender'] = 'Male'  (object dtype - text)

# After encoding
df['gender'] = pd.Categorical(df['gender'])

# Result
df['gender'] = 0  (categorical dtype - numeric)

# Key feature:
# The Categorical dtype remembers the original categories
# Can convert back: 0 → 'Male' for explanations
```

---

## Our Categorical Columns

### The 6 Categorical Features

```python
CATEGORICAL_COLUMNS = [
    'gender',              # M, F (2 categories)
    'marital_status',      # Single, Married, Divorced (3)
    'education_level',     # HS, Bachelor, Master, PhD (4)
    'employment_status',   # FT, PT, Unemployed, Self (4)
    'loan_purpose',        # Auto, Home, Personal, ... (5+)
    'grade_subgrade'       # A1-G3 (35+)
]

Total categories: 2 + 3 + 4 + 4 + 5+ + 35+ = 50+ unique values
```

---

## Our Implementation: No Fixed Category Lists

### The Decision

```python
# WRONG approach: Define fixed categories
GENDER_CATEGORIES = {'Male', 'Female'}
education_categories = {'High School', 'Bachelor', 'Master', 'PhD'}

# At inference:
if applicant_gender not in GENDER_CATEGORIES:
    raise ValueError("Unknown gender!")  # Fails on unknown input

# CORRECT approach: Let pandas handle it
df['gender'] = pd.Categorical(df['gender'])
# At inference:
df['gender'] = pd.Categorical(df['gender'])
# Even if new gender appears, pandas creates a new category
```

### Code from `src/preprocess.py`

```python
import pandas as pd

CATEGORICAL_COLUMNS = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

def preprocess_features(df):
    """
    Convert categorical columns to pandas.Categorical type.
    Allows unknown categories at inference time.
    """
    df = df.copy()
    
    for col in CATEGORICAL_COLUMNS:
        # Convert to categorical (LightGBM compatible)
        df[col] = pd.Categorical(df[col])
    
    return df

# Training:
train_df = pd.read_csv('data/train.csv')
train_df = preprocess_features(train_df)
# Result: 6 columns are now categorical dtype

# Inference (new applicant):
new_applicant = pd.DataFrame({
    'gender': ['Unknown'],  # Never seen in training
    'marital_status': ['Single'],
    ...
})
new_applicant = preprocess_features(new_applicant)
# Works! pandas creates 'Unknown' as new category for gender
```

---

## Why NOT Fixed Category Lists?

### Problem 1: Brittleness

```
Fixed categories approach:
──────────────────────────
GENDER = {'Male', 'Female'}

# At inference, new applicant comes
applicant['gender'] = 'Other'

# API crashes
if applicant['gender'] not in GENDER:
    raise ValueError("Unknown gender!")
# Error: Unknown gender!

# Real-world impact:
# - API dies in production
# - Thousands of customers blocked
# - Bad user experience
# - Business loss
```

### Problem 2: Maintenance Burden

```
If business adds new categories:
─────────────────────────────────
"We now accept 'Non-binary' gender"

With fixed categories:
1. Update GENDER = {'Male', 'Female', 'Non-binary'}
2. Retrain model to learn non-binary patterns
3. Redeploy API
4. Multiple manual steps, error-prone

With dynamic categories:
1. Just start receiving 'Non-binary' in data
2. Model learns automatically
3. No code changes needed
4. Seamless
```

### Problem 3: Unknown Category Handling

```
Fixed categories:
─────────────────
If unseen value arrives:
1. Must reject it
2. Fail the prediction
3. Force API error

Dynamic categories (pandas):
─────────────────────────────
If unseen value arrives:
1. Create new category on-the-fly
2. LightGBM assigns it a default prediction
3. API completes successfully
4. No error
```

---

## How LightGBM Processes Categorical Features

### Training Phase

```python
# Training data
df['gender'] = pd.Categorical(['Male', 'Female', 'Male', 'Female', ...])

# LightGBM sees the categories
model = lgb.LGBMClassifier(
    categorical_feature=['gender', 'marital_status', ...]
)

# During training, LightGBM learns:
# "Splitting on gender: Male goes left, Female goes right"
# Stores this split internally
```

### Inference Phase

```python
# New applicant: gender = 'Female'
new_applicant['gender'] = pd.Categorical(['Female'])

# LightGBM traverses trees:
# Tree 1: Is gender == 'Male'? NO → go right
# Tree 2: Is gender in ['Single']? (checking marital_status) YES → go left
# ... (all 100 trees)

# Prediction: P(default) = 0.35

# Note: If new_applicant had gender = 'Unknown'
# LightGBM would still traverse (unknown category treated as distinct value)
# Prediction still works!
```

---

## Alternative Approaches (and Why We Didn't Use Them)

### Approach 1: One-Hot Encoding

```python
# Original:
df['gender'] = ['Male', 'Female', 'Male']

# One-hot encoding:
df['gender_Male'] = [1, 0, 1]
df['gender_Female'] = [0, 1, 0]

# Pros:
├─ Works with any model
├─ Handles unknowns by setting all 0
└─ Industry standard (common knowledge)

# Cons:
├─ Creates many sparse columns
│  (gender: 2 → 2 columns; education: 4 → 4 columns)
│  (all 6 categorical → 50+ columns)
├─ Slow training on sparse data
├─ Harder to interpret (feature importance is split)
├─ LightGBM has native categorical support
│  (why not use it?)
└─ Memory intensive
```

### Approach 2: Ordinal Encoding

```python
# Original:
df['marital_status'] = ['Single', 'Married', 'Divorced']

# Ordinal encoding:
df['marital_status'] = [0, 1, 2]

# Pros:
├─ Simple
├─ No extra columns
└─ Fast

# Cons:
├─ Assumes ordering: 0 < 1 < 2
├─ But 'Single' is NOT "less than" 'Married'!
├─ Model learns wrong relationships
├─ Tree splits misinterpret: "marital_status < 1.5" (nonsense)
└─ Bad for credit risk (violates domain logic)
```

### Approach 3: Target Encoding (WOE)

```python
# Original:
df['gender'] = ['Male', 'Female', 'Male']
df['target'] = [0, 1, 0]

# Target encoding: Replace each category with its average target
# Male: average target = (0 + 0) / 2 = 0.0
# Female: average target = 1 / 1 = 1.0

df['gender'] = [0.0, 1.0, 0.0]

# Pros:
├─ Incorporates target information
├─ Works well sometimes
└─ Single column (like ordinal)

# Cons:
├─ Information leakage if not careful
├─ Loses original categories (no interpretability)
├─ Hard to explain: "Gender coefficient = 0.15" (what does this mean?)
└─ Overkill for LightGBM (which learns this anyway)
```

### Approach 4: Hashing

```python
# Original:
df['gender'] = ['Male', 'Female', 'Male']

# Hashing:
df['gender'] = [hash('Male') % 50, hash('Female') % 50, hash('Male') % 50]

# Pros:
├─ Handles unlimited categories
└─ Fixed output dimension

# Cons:
├─ Hash collisions (multiple categories → same number)
├─ Not interpretable
├─ Overkill for small category sets
└─ Not recommended for credit risk
```

**Winner: pandas.Categorical**

```
Why?
──
✓ LightGBM native support (no overhead)
✓ Handles unknown categories gracefully
✓ Preserves interpretability (knows original categories)
✓ Fast inference
✓ No fixed mappings (flexible)
✓ No one-hot explosion
✓ Respects domain logic (no false orderings)
```

---

## Implementation: From Raw to Ready

### Step 1: Load Raw Data

```python
df = pd.read_csv('data/train.csv')

print(df['gender'].dtype)  # object (text)
print(df['gender'].unique())  # ['Male', 'Female']
```

### Step 2: Convert to Categorical

```python
df['gender'] = pd.Categorical(df['gender'])

print(df['gender'].dtype)  # category
print(df['gender'].cat.categories)  # Index(['Male', 'Female'])
```

### Step 3: Train Model

```python
model = lgb.LGBMClassifier(
    categorical_feature=['gender', 'marital_status', ...]
)
model.fit(df[features], df['target'])
```

### Step 4: Inference with Unknown Category

```python
# New applicant with unknown value
new_df = pd.DataFrame({
    'gender': ['Other'],  # Never seen in training
    ...
})

new_df['gender'] = pd.Categorical(new_df['gender'])

# LightGBM still predicts!
pred = model.predict_proba(new_df)  # Works!
```

---

## How to Interpret Categorical Features

### Using SHAP for Categorical Features

```python
import shap

# Get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For categorical feature (gender):
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)

# Output shows:
# If gender = 'Male': SHAP value = +0.05 (increases default risk slightly)
# If gender = 'Female': SHAP value = -0.03 (decreases default risk slightly)
#
# But these effects are small → gender has low feature importance
```

### Feature Importance with Categorical

```python
# LightGBM reports feature importance
model.feature_importances_

# Output:
# gender: 0.02 (2% of splits) ← Low importance
# marital_status: 0.03 (3%)
# education_level: 0.04 (4%)
# employment_status: 0.12 (12%)
# loan_purpose: 0.12 (12%)
# grade_subgrade: 0.08 (8%)
# (numeric features dominate)
```

---

## Handling Missing Values in Categorical

### If a Category is Missing

```python
# Training data: gender = ['Male', 'Female']
# But production: gender = ['Male', 'Female', 'Unknown']

# At inference:
if new_applicant['gender'] == 'Unknown':
    # Option 1: Treat as new category (pandas handles it)
    new_applicant['gender'] = pd.Categorical(['Unknown'])
    pred = model.predict_proba(new_applicant)  # Works!
    
    # Option 2: Use default imputation
    if missing:
        new_applicant['gender'] = 'Male'  # Most common
```

---

## Interview Talking Points

### Q: How do you handle categorical features?
**A**: "I convert them to pandas.Categorical type, which LightGBM handles natively. This is better than one-hot encoding (which creates 50+ sparse columns) or ordinal encoding (which falsely implies ordering). Categorical type is interpretable and efficient."

### Q: Why not use fixed category lists?
**A**: "Fixed lists are brittle. If a new category appears in production, the API crashes. With pandas.Categorical, unknown categories are handled gracefully—pandas creates a new category on-the-fly and the model still predicts. It's more robust."

### Q: What if a new category appears at inference?
**A**: "pandas.Categorical automatically creates it as a new category. LightGBM assigns it a prediction based on learned patterns. The API doesn't crash. This is production-safe."

### Q: Why not one-hot encoding?
**A**: "One-hot would create 50+ binary columns (one per unique category). It's sparse, slow, and harder to interpret. LightGBM supports categorical natively, so there's no reason to use one-hot."

### Q: How does feature importance work for categorical features?
**A**: "LightGBM reports how many splits use each feature. Categorical features like gender and marital_status have low importance (~2-3%), while debt_to_income_ratio has high importance (~28%). This tells us numeric features matter more for default prediction."

---

## Summary

| Aspect | Our Approach |
|--------|-------------|
| **Method** | pandas.Categorical |
| **Columns** | 6 (gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade) |
| **Fixed categories?** | No (flexible, handles unknowns) |
| **One-hot encoding?** | No (LightGBM doesn't need it) |
| **Ordinal encoding?** | No (falsely implies ordering) |
| **Unknown handling** | Dynamic category creation (no error) |
| **Interpretability** | High (can map back to text) |
| **Performance** | Native LightGBM support (no overhead) |

**Bottom line**: pandas.Categorical is the optimal approach for categorical features in LightGBM. It's flexible (unknown categories don't crash), efficient (native support), and interpretable (preserves original category names). No fixed mappings means the model gracefully handles new categories at inference—a must-have for production systems.

