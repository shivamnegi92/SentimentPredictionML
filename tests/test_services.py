from backend.services.inference import predict, batch_predict

def test_predict():
    result = predict("I love this movie!")
    assert result == "positive"

def test_batch_predict():
    result = batch_predict(["I love this movie!", "This is bad"])
    assert result == ["positive", "negative"]
