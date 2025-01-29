import json

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_predict_endpoint():
    response = client.post("/predict", json={"text": "FastAPI makes testing fun!"})

    # Print the response for debugging
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions (adjust based on your expected response structure)
    assert response.status_code == 200, "Status code should be 200"
    assert response.json()["success"] is True, "Response should indicate success"
    assert "prediction" in response.json()["data"], "Response should contain 'prediction'"