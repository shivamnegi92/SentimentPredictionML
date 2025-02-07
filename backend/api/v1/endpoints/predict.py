"""
This module contains the API endpoints for making predictions.
"""
from fastapi import APIRouter  # Import API router class from FastAPI
from backend.services.inference import predict
from backend.api.v1.schemas.prediction import PredictionRequest
from backend.core.response import success_response
import logging

# Initialize a router object
router = APIRouter()  # type: ignore
"""
This router handles API endpoints for predictions.
"""

logger = logging.getLogger(__name__)

# @router: This part is a decorator that's used in FastAPI for defining API
# endpoints (routes) within a FastAPI application. It associates the function that
# follows it with the router object (router in this case).

@router.post("")
async def predict_single(request: PredictionRequest):
    """
    This function takes a PredictionRequest object as input,
    which contains the text data to be predicted on.
    """
    # Add debug prints
    print("=== DEBUG ===")
    print(f"Endpoint accessed: predict_single")
    print(f"Request received: {request}")
    print("=============")
    
    logger.info(f"Received prediction request for text: {request.text[:50]}...")
    try:
        result = await predict(request.text)
        logger.info(f"Predict endpoint- Prediction successful: {result}")
        print(f"Predict endpoint- Prediction result: {result}")  # Debug print
        return success_response({"prediction": result})
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        print(f"Error occurred: {str(e)}")  # Debug print
        raise