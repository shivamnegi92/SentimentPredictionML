# backend/main.py
from dotenv import load_dotenv
import os

from fastapi import FastAPI
from api.v1.endpoints import predict, batch_predict, health
from core.logger import setup_logging


load_dotenv()
print("PYTHONPATH:", os.getenv("PYTHONPATH"))



# Initialize logging
logger = setup_logging()
logger.info("=== Starting Sentiment Analysis API ===")

# Create FastAPI app
app = FastAPI(title="Sentiment Prediction API", version="1.0.0")

# Log inclusion for debugging
print("Including batch_predict router")

# Include routers with correct prefixes (remove trailing slashes)
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(batch_predict.router, prefix="/batch", tags=["Batch Prediction"])
app.include_router(health.router, prefix="/health", tags=["Health"])

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application is starting up")
