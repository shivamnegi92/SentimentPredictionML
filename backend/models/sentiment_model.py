from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
from backend.config import MODEL_PATH, TOKENIZER_PATH  # Import config values
import torch
class SentimentModel:
    def __init__(self):
        # Load tokenizer from the correct path
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

        # Load model weights from safetensors
        state_dict = load_file(f"{MODEL_PATH}/model.safetensors")

        # Load the model configuration and model weights from the correct path
        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,  # Use the model path for configuration and model weights
            state_dict=state_dict  # Load the model weights from safetensors
        )
        self.model.eval()  # Set model to evaluation mode

    def batch_predict(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=1)
        
        
        return [str(pred.item()) for pred in predictions]
    def predict(self, text: str) -> str:
        """Predict the sentiment of the given text.

        Args:
            text (str): The input text for sentiment prediction.

        Returns:
            str: The predicted sentiment.
        """
        return self.batch_predict([text])[0]  # For single prediction, return the first result