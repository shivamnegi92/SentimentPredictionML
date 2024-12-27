"""
This module contains functions and classes for handling inference
using the SentimentModel.
"""

from backend.models.sentiment_model import SentimentModel
from typing import List

# Load the sentiment model
model = SentimentModel()

async def predict(text: str) -> str:
    """Predict the output based on the input text.

    Args:
        text (str): The input text for prediction.

    Returns:
        str: The predicted output.
    """
    return model.predict(text)


async def batch_predict(texts: List[str]) -> List[str]:
    """
    Batch prediction function that processes a list of texts and returns predictions.
    
    Args:
        texts (List[str]): A list of texts for sentiment prediction.
    
    Returns:
        List[str]: A list of predictions (e.g., "Positive", "Negative").
    """
    return model.batch_predict(texts)
