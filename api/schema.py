from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    employment_status: str
    education_level: str
    experience: int
    marital_status: str
    number_of_dependents: int
    loan_purpose: str
    loan_amount: float
    loan_term: int
    interest_rate: float
    monthly_debt_payments: float
    credit_card_utilization: float
    number_of_credit_inquiries: int
    debt_to_income_ratio: float
    home_ownership_status: str

class PredictionResponse(BaseModel):
    probability: float
    decision: str