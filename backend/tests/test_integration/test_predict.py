
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pytest

from services.inference import predict

@pytest.mark.asyncio
async def test_inference_service():  
    text = "FastAPI is great!"
    result = await predict(text)  
    assert result in ["0", "1"], "Prediction should be 0 or 1"