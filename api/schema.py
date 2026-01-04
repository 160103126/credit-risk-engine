from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
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

class PredictionResponse(BaseModel):
    probability: float
    decision: str