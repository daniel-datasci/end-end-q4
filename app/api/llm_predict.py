from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import asyncio

from app.llm.hf_client import explain_churn, LLMError
from app.core.mlflow_utils import log_llm_run

router = APIRouter()

class LLMExplainRequest(BaseModel):
    features: Dict[str, Any]
    churn_score: float

class LLMExplainResponse(BaseModel):
    explanation: str

@router.post("/llm/explain", response_model=LLMExplainResponse)
async def llm_explain(req: LLMExplainRequest):
    try:
        # Run the async HF explanation
        explanation = await explain_churn(req.features, req.churn_score)
        
        # Log MLflow run
        log_llm_run(req.features, req.churn_score, explanation)

        return {"explanation": explanation}
    except LLMError as e:
        raise HTTPException(status_code=500, detail=f"LLM explanation failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
