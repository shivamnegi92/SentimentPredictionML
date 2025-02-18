import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.sentiment_model import SentimentModel




def test_model_loads_correctly():
    model = SentimentModel()
    assert model.model is not None, "Model should load successfully"
    assert model.tokenizer is not None, "Tokenizer should load successfully"
