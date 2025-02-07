"""
This module contains functions and classes for handling inference
using the SentimentModel.
"""

from backend.models.sentiment_model import SentimentModel
from typing import List
import logging

logger = logging.getLogger(__name__)
model = SentimentModel()

async def predict(text: str) -> str:
    """Predict sentiment for single text"""
    logger.info("Processing prediction request")
    logger.debug("Input text: %s", text[:50])
    try:
        result = model.predict(text)
        logger.info("Prediction completed successfully")
        logger.debug("Prediction result: %s", result)
        return result
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise

async def batch_predict(texts: List[str]) -> List[str]:
    """Batch prediction for multiple texts"""
    logger.info("Processing batch prediction request")
    logger.debug("Number of texts: %d", len(texts))
    try:
        results = model.batch_predict(texts)
        logger.info("Batch prediction completed successfully")
        return results
    except Exception as e:
        logger.error("Batch prediction failed: %s", str(e), exc_info=True)
        raise

# # backend/services/inference.py
# async def batch_predict(texts):
#     # Simulate inference logic (replace with your actual implementation)
#     return ["1" if "awesome" in text else "0" for text in texts]
