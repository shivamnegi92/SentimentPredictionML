"""
This module contains the API endpoints for making predictions.
"""
from fastapi import APIRouter  # Import API router class from FastAPI
from backend.services.inference import predict
from backend.api.v1.schemas.prediction import PredictionRequest
from backend.core.response import success_response

# Initialize a router object
router = APIRouter()  # type: ignore
"""
This router handles API endpoints for predictions.
"""

# @router: This part is a decorator that's used in FastAPI for defining API
# endpoints (routes) within a FastAPI application. It associates the function that
# follows it with the router object (router in this case).

@router.post("")
async def predict_single(request: PredictionRequest):
    """
    This function takes a PredictionRequest object as input,
    which contains the text data to be predicted on.
    It calls the predict function from the inference service to make the prediction
    and returns a success response with the prediction result.
    """
    result = await predict(request.text)  # Call predict function from inference service and wait for the result
    return success_response({"prediction": result})  # Return a success response with the prediction