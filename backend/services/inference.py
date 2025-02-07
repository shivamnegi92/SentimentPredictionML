"""
This module contains functions and classes for handling inference
using the SentimentModel.
"""

from backend.models.sentiment_model import SentimentModel
from typing import List
from backend.core.logger import setup_logging
import torch

# Load the sentiment model
model = SentimentModel()

logger = setup_logging()

async def predict(text: str) -> str:
    """Predict the output based on the input text.

    Args:
        text (str): The input text for prediction.

    Returns:
        str: The predicted output.
    """
    logger.debug("=== Prediction Pipeline Debug ===")
    logger.debug(f"1. Input text: {text}")
    try:
        # Log tokenization
        inputs = model.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        print(f"2. Tokenized input: {inputs}")
        
        # Log model input shape
        print(f"3. Input tensor shape: {inputs['input_ids'].shape}")
        
        # Log raw model output
        with torch.no_grad():
            outputs = model.model(**inputs)
        print(f"4. Raw model output: {outputs.logits}")
        
        # Log probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        print(f"5. Probabilities: {probs}")
        
        # Get prediction
        result = model.predict(text)
        print(f"6. Final prediction: {result}")        
        return result
    
    
    
    
    except Exception as e:
        logger.error("Model prediction failed: %s", str(e), exc_info=True)
        raise


async def batch_predict(texts: List[str]) -> List[str]:
    """
    Batch prediction function that processes a list of texts and returns predictions.
    
    Args:
        texts (List[str]): A list of texts for sentiment prediction.
    
    Returns:
        List[str]: A list of predictions (e.g., "Positive", "Negative").
    """
    return model.batch_predict(texts)


# # backend/services/inference.py
# async def batch_predict(texts):
#     # Simulate inference logic (replace with your actual implementation)
#     return ["1" if "awesome" in text else "0" for text in texts]
