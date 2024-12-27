from backend.models.sentiment_model import SentimentModel

model = SentimentModel()

def test_model_predict():
    result = model.predict("I love this movie!")
    assert result == "positive"

def test_model_batch_predict():
    result = model.batch_predict(["I love this movie!", "This is bad"])
    assert result == ["positive", "negative"]
