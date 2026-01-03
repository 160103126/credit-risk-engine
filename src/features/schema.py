# Feature schema definitions

# List of features used in modeling
FEATURES = [
    'age',
    'annual_income',
    'credit_score',
    'employment_status',
    'education_level',
    'experience',
    'marital_status',
    'number_of_dependents',
    'loan_purpose',
    'loan_amount',
    'loan_term',
    'interest_rate',
    'monthly_debt_payments',
    'credit_card_utilization',
    'number_of_credit_inquiries',
    'debt_to_income_ratio',
    'home_ownership_status'
]

# Data types
DTYPES = {
    'age': 'int64',
    'annual_income': 'float64',
    'credit_score': 'int64',
    'employment_status': 'category',
    'education_level': 'category',
    'experience': 'int64',
    'marital_status': 'category',
    'number_of_dependents': 'int64',
    'loan_purpose': 'category',
    'loan_amount': 'float64',
    'loan_term': 'int64',
    'interest_rate': 'float64',
    'monthly_debt_payments': 'float64',
    'credit_card_utilization': 'float64',
    'number_of_credit_inquiries': 'int64',
    'debt_to_income_ratio': 'float64',
    'home_ownership_status': 'category'
}

TARGET = 'default'