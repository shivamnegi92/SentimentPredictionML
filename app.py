import mlflow
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List
import logging

# Initialize FastAPI app
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model and tokenizer paths
MODEL_PATH = "./final_model"
TOKENIZER_PATH = "./final_tokenizer"

# Load model and tokenizer during startup (this is done only once)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Set the model to evaluation mode
model.eval()

# Function to make inference with the model
def predict(texts: List[str]):
    # Tokenize input texts (handle batch processing)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Move tensors to the correct device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # Return the predicted classes
    return predictions.cpu().numpy()

# Define Pydantic models for request and response validation
class InferenceRequest(BaseModel):
    texts: List[str]

class InferenceResponse(BaseModel):
    predictions: List[int]

class BatchInferenceRequest(BaseModel):
    batch_texts: List[List[str]]  # List of text lists for batch prediction

class BatchInferenceResponse(BaseModel):
    predictions: List[List[int]]  # List of predictions for each text list in the batch

# FastAPI route for single inference
@app.post("/predict", response_model=InferenceResponse)
async def predict_endpoint(request: InferenceRequest):
    # Extract texts from request
    texts = request.texts

    # Input validation
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Perform prediction
        predictions = predict(texts)
        return InferenceResponse(predictions=predictions.tolist())
    
    except Exception as e:
        # Log the error and raise an HTTP exception
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# FastAPI route for batch inference
@app.post("/predict_batch", response_model=BatchInferenceResponse)
async def batch_predict_endpoint(request: BatchInferenceRequest):
    # Extract batch texts from request
    batch_texts = request.batch_texts

    # Input validation
    if not batch_texts or len(batch_texts) == 0:
        raise HTTPException(status_code=400, detail="No batch texts provided")

    try:
        # Perform batch prediction for each set of texts in the batch
        batch_predictions = [predict(texts) for texts in batch_texts]
        return BatchInferenceResponse(predictions=[pred.tolist() for pred in batch_predictions])
    
    except Exception as e:
        # Log the error and raise an HTTP exception
        logger.error(f"Error during batch inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check route
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Main entry for application start (when running with uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
