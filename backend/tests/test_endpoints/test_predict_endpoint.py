import sys
import os

# Dynamically resolve the backend directory path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, BASE_DIR)

from main import app  # Import FastAPI app
from fastapi.testclient import TestClient

client = TestClient(app)  # ✅ Correct way to initialize TestClient

def test_predict_endpoint():
    response = client.post("/predict", json={"text": "FastAPI makes testing fun!"})

    # Debugging prints
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions
    assert response.status_code == 200, "Status code should be 200"
    assert response.json()["success"] is True, "Response should indicate success"
    assert "prediction" in response.json()["data"], "Response should contain 'prediction'"
