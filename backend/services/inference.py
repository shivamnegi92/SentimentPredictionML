from backend.models.sentiment_model import SentimentModel
from typing import List

# Initialize the model
model = SentimentModel()

async def predict(text: str) -> str:
    return model.predict(text)

async def batch_predict(texts: List[str]) -> List[str]:
    return model.batch_predict(texts)
