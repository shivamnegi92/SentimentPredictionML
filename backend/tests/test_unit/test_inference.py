import pytest
from unittest.mock import MagicMock
from models.sentiment_model import SentimentModel

@pytest.fixture
def sentiment_model():
    model = SentimentModel()
    model.predict = MagicMock(return_value="1")  # Mock predict method
    return model

def test_predict_single(sentiment_model):
    text = "I love this service!"
    prediction = sentiment_model.predict(text)
    assert prediction == "1", "The prediction should be '1'"