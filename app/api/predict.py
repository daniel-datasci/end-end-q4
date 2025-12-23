from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from app.ml.predictor import predict_churn
from app.core.mlflow_utils import log_llm_run

router = APIRouter()

class ChurnRequest(BaseModel):
    age: int
    tenure_days: int
    avg_time_spent: float
    avg_transaction_value: float
    avg_frequency_login_days: float
    points_in_wallet: float
    used_special_discount: str
    offer_application_preference: str
    past_complaint: str
    complaint_status: str
    membership_category: str
    region_category: str

class ChurnResponse(BaseModel):
    churn_probability: float
    risk_segment: str


@router.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    # Convert request to DataFrame
    df = pd.DataFrame([request.dict()])

    # IMPORTANT:
    # Here, in Phase C, we will reuse the SAME feature engineering logic
    # For now assume model-ready features are already aligned

    churn_score = float(predict_churn(df)[0])

    risk_segment = (
        "High Risk" if churn_score >= 0.7
        else "Medium Risk" if churn_score >= 0.4
        else "Low Risk"
    )

    log_llm_run(request.dict(), churn_score)

    return ChurnResponse(
        churn_probability=churn_score,
        risk_segment=risk_segment
    )
