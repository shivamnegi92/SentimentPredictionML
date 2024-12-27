# backend/main.py
from fastapi import FastAPI
from backend.api.v1.endpoints import predict, batch_predict, health
from backend.core.logger import setup_logging

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(title="Sentiment Prediction API", version="1.0.0")

# Log inclusion for debugging
print("Including batch_predict router")

# Include routers with correct prefixes
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(batch_predict.router, prefix="/batch", tags=["Batch Prediction"])
app.include_router(health.router, prefix="/health", tags=["Health"])
