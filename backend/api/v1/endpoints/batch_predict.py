# backend/api/v1/endpoints/batch_predict.py
from fastapi import APIRouter
from services.inference import batch_predict
from api.v1.schemas.prediction import PredictionRequest
from core.response import success_response, error_response
from typing import List

router = APIRouter()

@router.post("")
async def batch_predict_single(request: List[PredictionRequest]):
    try:
        # Extract text data from the request list of PredictionRequest objects
        texts = [item.text for item in request]
        
        # Use the batch prediction method
        predictions = await batch_predict(texts)
        
        return success_response({"predictions": predictions})
    except Exception as e:
        return error_response(str(e))