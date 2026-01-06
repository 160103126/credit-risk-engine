# Data Preprocessing Deep Dive: From Raw to Model-Ready

## Overview

**Goal**: Transform raw loan data into a format suitable for LightGBM  
**Key Steps**: Target creation, column selection, categorical conversion, handling missing values  
**Result**: Clean, normalized data ready for training

---

## The Raw Data

### Source Data File

```
data/raw/train.csv  (6,000 rows × 15 columns)

Columns:
├─ Numerical Features
│  ├─ annual_income
│  ├─ debt_to_income_ratio
│  ├─ credit_score
│  ├─ loan_amount
│  └─ interest_rate
│
├─ Categorical Features
│  ├─ gender
│  ├─ marital_status
│  ├─ education_level
│  ├─ employment_status
│  ├─ loan_purpose
│  └─ grade_subgrade
│
├─ Target-Related
│  └─ loan_paid_back (1 = no default, 0 = default)
│
└─ To Exclude
   ├─ customer_id
   ├─ application_date
   ├─ bank_branch
   └─ unknown_field

Total: 6,000 samples
Good borrowers: ~5,000 (83%)
Defaults: ~1,000 (17%)
```

---

## Step 1: Target Creation

### The Raw Target Variable

```
Column name: loan_paid_back
Values: 1, 0
Meaning:
├─ 1 = Borrower paid back loan (good)
└─ 0 = Borrower defaulted (bad)

Problem:
┌──────────────────────────────────────────┐
│ Our model predicts probability of DEFAULT│
│ But column is 1 = paid back              │
│ Need to invert!                          │
└──────────────────────────────────────────┘
```

### Target Inversion

From `src/data/preprocess.py`:

```python
def create_target(df):
    """
    Create binary target for classification.
    
    Input: loan_paid_back (1 = good, 0 = bad)
    Output: target (1 = bad/default, 0 = good/non-default)
    """
    # Invert: default = 1 - loan_paid_back
    df['target'] = 1 - df['loan_paid_back']
    
    return df

# Example
df['loan_paid_back'] = [1, 1, 0, 1, 0]
df['target'] = 1 - df['loan_paid_back']
# Result: [0, 0, 1, 0, 1]

# Interpretation:
# customer 0: loan_paid_back=1 (good) → target=0 (no default)
# customer 2: loan_paid_back=0 (bad) → target=1 (default)
# customer 4: loan_paid_back=0 (bad) → target=1 (default)
```

### Verification

```python
# Check target distribution
print(df['target'].value_counts())

# Output:
# 0    4968  (83% non-default)
# 1    1032  (17% default)

# Good! Matches original distribution
# 83:17 ratio indicates imbalanced data
# We use AUC/KS instead of accuracy
```

---

## Step 2: Column Selection

### Which Columns to Keep

```
Total columns in raw data: 15

Keep (11 features used by model):
├─ annual_income
├─ debt_to_income_ratio
├─ credit_score
├─ loan_amount
├─ interest_rate
├─ gender
├─ marital_status
├─ education_level
├─ employment_status
├─ loan_purpose
└─ grade_subgrade

Drop (4 columns not used):
├─ customer_id (identifier, no predictive value)
├─ application_date (time index, causes leakage risk)
├─ bank_branch (location, not in model spec)
└─ loan_paid_back (used to create target, must drop)
```

### Why Drop These?

```
customer_id:
├─ Reason: Identifier only, no information about risk
├─ Risk: Using it would overfit (each customer = unique)
└─ Action: Drop it

application_date:
├─ Reason: Model should predict based on attributes,
│  not when application was made
├─ Risk: Date might correlate with external factors
│  (market conditions, economic cycles) → data leakage
├─ Pattern: "Applications in Jan 2020 default more"
│  might be true but due to COVID, not applicant quality
└─ Action: Drop it for model purity

bank_branch:
├─ Reason: Not in model specification
├─ Risk: If included, would learn branch-specific patterns
│  which might not generalize
└─ Action: Drop it (can add later if needed)

loan_paid_back:
├─ Reason: This IS our target
├─ Risk: We already used it to create 'target'
│ Keeping it would cause information leakage
│ (we'd be predicting from the answer)
└─ Action: Drop it
```

---

## Step 3: Feature Validation and Cleaning

### Numeric Features

From `src/data/preprocess.py`:

```python
NUMERIC_FEATURES = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

def validate_numeric_features(df):
    """
    Check numeric features for validity.
    Handle outliers and missing values.
    """
    for col in NUMERIC_FEATURES:
        print(f"\n{col}")
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Missing: {df[col].isna().sum()}")
        
        # Remove rows with missing values
        df = df.dropna(subset=[col])
        
        # Check for outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > 0:
            print(f"  Outliers: {outliers}")
            # Keep outliers (they're real, not errors)
    
    return df

# Output:
# annual_income
#   Min: 15000
#   Max: 500000
#   Missing: 0
#
# debt_to_income_ratio
#   Min: 0.05
#   Max: 1.23
#   Missing: 0
#   Outliers: 45 (kept as real observations)
#
# credit_score
#   Min: 300
#   Max: 850
#   Missing: 0
#
# ... (remaining features)
```

### Categorical Features

```python
CATEGORICAL_FEATURES = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

def validate_categorical_features(df):
    """Check categorical features for validity."""
    for col in CATEGORICAL_FEATURES:
        print(f"\n{col}")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing: {df[col].isna().sum()}")
        print(f"  Values:\n{df[col].value_counts()}")

# Output:
# gender
#   Unique values: 2
#   Missing: 0
#   Values:
#   Male     3000
#   Female   3000
#
# marital_status
#   Unique values: 3
#   Missing: 0
#   Values:
#   Single     1800
#   Married    1900
#   Divorced    300
#
# ... (remaining features)
```

---

## Step 4: Missing Value Handling

### Check for Missing Values

```python
# Check all columns
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print("Missing Values:")
print(missing[missing > 0])  # Only show columns with missing values

# If our data has no missing:
# (empty output)

# If our data had missing (example):
# annual_income        5  (0.08%)
# employment_status   10  (0.17%)
```

### Handling Strategy

```python
def handle_missing_values(df):
    """
    Handle missing values by column type.
    """
    # Numeric features: Fill with median
    for col in NUMERIC_FEATURES:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical features: Fill with mode
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Result: No missing values remain
```

**Our data**: No missing values (already clean) ✓

---

## Step 5: Categorical Encoding

### Convert to pandas.Categorical

```python
def encode_categorical_features(df):
    """
    Convert categorical columns to pandas.Categorical type.
    Compatible with LightGBM's native categorical support.
    """
    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col])
    
    return df

# Before:
df['gender'].dtype  # object (text)

# After:
df['gender'].dtype  # category
```

---

## Step 6: Feature Scaling (Optional for Tree Models)

### Do We Need Scaling?

```
Tree-based models (like LightGBM):
──────────────────────────────────
✗ DON'T require scaling
├─ Trees split based on thresholds, not distances
├─ A split on annual_income=50000 vs. credit_score=720
│  is equally valid regardless of scale
└─ Feature magnitude doesn't affect splits

Linear models (like logistic regression):
───────────────────────────────────────────
✓ DO require scaling
├─ Coefficients are magnitudes
├─ If annual_income is 50,000 and credit_score is 720,
│  raw coefficients become incomparable
└─ Need to normalize to [0,1] or [-1,1]

Decision for our model:
├─ Using LightGBM (tree-based)
└─ ✗ No scaling needed
```

---

## Complete Preprocessing Pipeline

### From Raw to Ready

From `src/data/preprocess.py`:

```python
import pandas as pd

NUMERIC_FEATURES = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

CATEGORICAL_FEATURES = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

def preprocess_data(df, is_train=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        is_train: If True, create target. If False, skip target creation.
    
    Returns:
        Processed DataFrame ready for model
    """
    # Step 1: Create target (if training)
    if is_train:
        df['target'] = 1 - df['loan_paid_back']
    
    # Step 2: Select columns
    columns_to_keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if is_train:
        columns_to_keep += ['target']
    
    df = df[columns_to_keep]
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Encode categorical features
    for col in CATEGORICAL_FEATURES:
        df[col] = pd.Categorical(df[col])
    
    # Note: No scaling (tree models don't need it)
    
    return df

def preprocess_train_data(df):
    """Preprocess training data."""
    return preprocess_data(df, is_train=True)

def preprocess_inference_data(df):
    """Preprocess inference data (e.g., new applicant)."""
    return preprocess_data(df, is_train=False)

# Usage
train_df = pd.read_csv('data/raw/train.csv')
train_df = preprocess_train_data(train_df)

# Result shape: (6000, 11) raw features + 1 target = (6000, 12)
```

---

## Data Quality Checks

### After Preprocessing

```python
def quality_checks(df):
    """Verify data quality after preprocessing."""
    
    print("Data Shape:", df.shape)
    print("Columns:", list(df.columns))
    
    # Check for NaN
    print("\nMissing values:", df.isnull().sum().sum())
    
    # Check target distribution
    print("\nTarget distribution:")
    print(df['target'].value_counts(normalize=True))
    
    # Check feature types
    print("\nFeature types:")
    print(df.dtypes)
    
    # Check numeric feature ranges
    print("\nNumeric feature ranges:")
    for col in NUMERIC_FEATURES:
        print(f"  {col}: [{df[col].min():.1f}, {df[col].max():.1f}]")
    
    # Check categorical feature cardinality
    print("\nCategorical feature cardinality:")
    for col in CATEGORICAL_FEATURES:
        print(f"  {col}: {df[col].nunique()} categories")

# Expected output:
# Data Shape: (6000, 12)
# Columns: [annual_income, ..., target]
# Missing values: 0
#
# Target distribution:
# 0    0.83  (non-default)
# 1    0.17  (default)
#
# Feature types:
# annual_income        float64
# credit_score         int64
# gender               category
# ... (mixed types)
#
# Numeric feature ranges:
#   annual_income: [15000.0, 500000.0]
#   credit_score: [300, 850]
# ...
#
# Categorical feature cardinality:
#   gender: 2 categories
#   marital_status: 3 categories
#   ...
```

---

## Data Split: Train/Validation

### Stratified Split

```python
from sklearn.model_selection import train_test_split

def split_train_validation(df, test_size=0.2, random_state=42):
    """
    Split data into train (80%) and validation (20%).
    Stratify by target to preserve class distribution.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target'],
        random_state=random_state
    )
    
    return train_df, val_df

# Usage
train_df, val_df = split_train_validation(df)

print("Train:", train_df.shape[0], "samples")
print("  - Good: ", (train_df['target'] == 0).sum())
print("  - Bad:  ", (train_df['target'] == 1).sum())

print("Validation:", val_df.shape[0], "samples")
print("  - Good: ", (val_df['target'] == 0).sum())
print("  - Bad:  ", (val_df['target'] == 1).sum())

# Output:
# Train: 4800 samples
#   - Good: 3984 (83%)
#   - Bad: 816 (17%)
#
# Validation: 1200 samples
#   - Good: 984 (82%)
#   - Bad: 216 (18%)
#
# Note: Both folds have similar 83:17 ratio ✓
```

---

## Interview Talking Points

### Q: How did you preprocess the data?
**A**: "Six main steps: (1) Create target by inverting loan_paid_back (1 = default, 0 = non-default); (2) Select 11 features (drop customer_id, application_date, bank_branch); (3) Handle missing values (none in this dataset); (4) Convert categorical columns to pandas.Categorical; (5) No scaling (LightGBM doesn't need it); (6) Stratified 80/20 train/validation split to preserve class distribution."

### Q: Why invert loan_paid_back?
**A**: "The raw column is 1 = loan paid back (good) and 0 = default (bad). But our model predicts probability of default, so we need 1 = default (bad). Inverting makes the target align with model semantics."

### Q: Which columns did you drop and why?
**A**: "I dropped customer_id (no predictive value), application_date (risks data leakage—external factors like economic conditions could bias the model), bank_branch (not in the model specification), and loan_paid_back (used to create target, must drop to avoid information leakage)."

### Q: Did you scale features?
**A**: "No, because LightGBM is a tree-based model. Trees split based on thresholds, not distances, so feature scale doesn't matter. Scaling is important for linear models but not for trees."

### Q: How did you handle the class imbalance?
**A**: "By using stratified splits (preserve 83:17 ratio in train/validation) and using AUC/KS metrics instead of accuracy. I also set is_unbalance=True in LightGBM to adjust costs for the minority class."

---

## Summary

| Step | Action | Details |
|------|--------|---------|
| **1. Target** | Invert loan_paid_back | 1 = default, 0 = non-default |
| **2. Columns** | Keep 11 features, drop 4 | Drop identifiers, leakage risks, unneeded columns |
| **3. Missing** | Check and handle | No missing values in this dataset |
| **4. Categorical** | Convert to pandas.Categorical | 6 columns, LightGBM native support |
| **5. Scaling** | None (not needed) | Tree models don't require scaling |
| **6. Split** | Stratified 80/20 | Preserve 83:17 class ratio |
| **Result** | 6,000 samples, 11 features | Ready for model training |

**Bottom line**: Preprocessing transforms raw loan data into a clean, model-ready format. Key decisions include inverting the target (to align with default prediction), selecting relevant features (dropping identifiers and leakage risks), using categorical encoding compatible with LightGBM, and stratified splitting to handle class imbalance. No scaling is needed for tree models.

