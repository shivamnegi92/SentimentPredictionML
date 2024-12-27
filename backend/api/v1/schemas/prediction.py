from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]
