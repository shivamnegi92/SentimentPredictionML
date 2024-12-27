# backend/api/v1/health.py

from fastapi import APIRouter

# Define a FastAPI APIRouter instance
router = APIRouter()

# Define a simple health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "ok"}
