import os

# Define the folder structure
folders = [
    "backend/api/v1/endpoints",
    "backend/api/v1/schemas",
    "backend/api/v2/endpoints",
    "backend/api/v2/schemas",
    "backend/services",
    "backend/models",
    "backend/utils",
    "backend/config",
    "backend/models_store",
    "backend/core",
    "tests",
]

# Define the files to create
files = {
    "backend/main.py": '''from fastapi import FastAPI
from backend.api.v1.endpoints import predict

app = FastAPI()

# Include the prediction routes
app.include_router(predict.router, prefix="/predict", tags=["Predict"])
''',

    "backend/api/v1/endpoints/predict.py": '''from fastapi import APIRouter
from backend.services.inference import predict
from backend.api.v1.schemas import prediction

router = APIRouter()

@router.post("/single")
async def predict_single(request: prediction.PredictionRequest):
    prediction = await predict(request.text)
    return {"prediction": prediction}
''',

    "backend/api/v1/endpoints/batch_predict.py": '''from fastapi import APIRouter
from backend.services.inference import batch_predict
from backend.api.v1.schemas import prediction

router = APIRouter()

@router.post("/batch")
async def predict_batch(request: prediction.BatchPredictionRequest):
    predictions = await batch_predict(request.texts)
    return {"predictions": predictions}
''',

    "backend/api/v1/endpoints/health.py": '''from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}
''',

    "backend/api/v1/schemas/prediction.py": '''from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]
''',

    "backend/services/inference.py": '''from backend.models.sentiment_model import SentimentModel
from typing import List

# Initialize the model
model = SentimentModel()

async def predict(text: str) -> str:
    return model.predict(text)

async def batch_predict(texts: List[str]) -> List[str]:
    return model.batch_predict(texts)
''',

    "backend/models/sentiment_model.py": '''import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentModel:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.model_name = "bert-base-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "positive" if prediction == 1 else "negative"
    
    def batch_predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            prediction = self.predict(text)
            predictions.append(prediction)
        return predictions
''',

    "backend/utils/__init__.py": "",

    "backend/config/config.py": '''import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings (add more as needed)
MODEL_PATH = os.getenv("MODEL_PATH", "path_to_your_model")
''',

    "backend/core/logger.py": '''import logging

def setup_logging():
    logger = logging.getLogger("uvicorn")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
setup_logging()
''',

    "backend/core/device.py": '''# You can add device-related utilities here if needed, like managing GPUs.
''',

    "backend/core/response.py": '''# Utility functions for formatting responses, error handling, etc.
''',

    "backend/core/health.py": '''# Health check related utilities can be placed here.
''',

    "backend/requirements.txt": '''fastapi
uvicorn
pydantic
transformers
torch
python-dotenv
pytest
httpx
''',

    "backend/.env": '''MODEL_PATH=path_to_your_model
''',

    "tests/test_api.py": '''from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_single_prediction():
    response = client.post("/predict/single", json={"text": "I love this movie!"})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_batch_prediction():
    response = client.post("/predict/batch", json={"texts": ["I love this movie!", "This is bad"]})
    assert response.status_code == 200
    assert "predictions" in response.json()
''',

    "tests/test_services.py": '''from backend.services.inference import predict, batch_predict

def test_predict():
    result = predict("I love this movie!")
    assert result == "positive"

def test_batch_predict():
    result = batch_predict(["I love this movie!", "This is bad"])
    assert result == ["positive", "negative"]
''',

    "tests/test_inference.py": '''from backend.models.sentiment_model import SentimentModel

model = SentimentModel()

def test_model_predict():
    result = model.predict("I love this movie!")
    assert result == "positive"

def test_model_batch_predict():
    result = model.batch_predict(["I love this movie!", "This is bad"])
    assert result == ["positive", "negative"]
''',

    "README.md": '''# Sentiment Prediction API

This project provides an API for sentiment analysis using a pre-trained model.

## Endpoints

- `POST /predict/single`: Get sentiment prediction for a single text.
- `POST /predict/batch`: Get sentiment predictions for a list of texts.
- `GET /health`: Check if the API is healthy.
''',

    "run.py": '''import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
'''
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files and add content
for file, content in files.items():
    with open(file, 'w') as f:
        f.write(content)

print("Backend folder structure and files have been created successfully!")
