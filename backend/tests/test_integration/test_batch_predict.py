import sys
import os
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


from services.inference import batch_predict

@pytest.mark.asyncio
async def test_batch_inference_service():
    texts = ["FastAPI is awesome!", "I dislike bugs."]
    results = await batch_predict(texts)  # Await the coroutine
    assert len(results) == len(texts), "Results should match input size"
    for result in results:
        assert result in ["0", "1"], "Each prediction should be 0 or 1"