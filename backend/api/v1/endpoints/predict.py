from fastapi import APIRouter
from backend.services.inference import predict
from backend.api.v1.schemas.prediction import PredictionRequest
from backend.core.response import success_response

router = APIRouter()

@router.post("")
async def predict_single(request: PredictionRequest):
    result = await predict(request.text)
    return success_response({"prediction": result})
