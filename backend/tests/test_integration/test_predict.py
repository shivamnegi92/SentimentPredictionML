from backend.services.inference import predict

def test_inference_service():
    text = "FastAPI is great!"
    result = predict(text)
    assert result in ["0", "1"], "Prediction should be 0 or 1"
