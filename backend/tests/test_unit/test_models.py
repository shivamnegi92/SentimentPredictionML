from models.sentiment_model import SentimentModel

def test_model_loads_correctly():
    model = SentimentModel()
    assert model.model is not None, "Model should load successfully"
    assert model.tokenizer is not None, "Tokenizer should load successfully"
