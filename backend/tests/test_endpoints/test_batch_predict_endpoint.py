# pylint: disable=import-error
import json
import sys
import os
from fastapi.testclient import TestClient


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from main import app




client = TestClient(app)


def test_batch_predict_endpoint():
    # Sending test data to the endpoint
    response = client.post("/batch", json=[
  { "text": "The movie was amazing!" },
  { "text": "I did not enjoy the film." }
]
)

    # Print the response for debugging
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Assertions (adjust based on your expected response structure)
    assert response.status_code == 200, "Status code should be 200"
    assert "data" in response.json(), "Response should contain 'data'"
    data = response.json()["data"]
    assert "predictions" in data, "Response should contain 'predictions'"

    # Check predictions
    predictions = data["predictions"]
    assert len(predictions) == 2, "There should be two predictions"
    assert isinstance(predictions, list), "Predictions should be a list"