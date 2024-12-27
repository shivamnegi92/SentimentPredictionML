from fastapi.testclient import TestClient
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
