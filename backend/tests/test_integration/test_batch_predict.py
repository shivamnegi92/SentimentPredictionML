import pytest
from backend.services.inference import batch_predict

@pytest.mark.asyncio
async def test_batch_inference_service():
    texts = ["FastAPI is awesome!", "I dislike bugs."]
    results = await batch_predict(texts)  # Await the coroutine
    assert len(results) == len(texts), "Results should match input size"
    for result in results:
        assert result in ["0", "1"], "Each prediction should be 0 or 1"