from typing import List  # <-- Import List here

import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self):
        # Load the model and tokenizer from models_store
        self.model = BertForSequenceClassification.from_pretrained('backend/models_store/final_model')  # Load the fine-tuned model
        self.model.eval()  # Set model to evaluation mode
        
        # Load tokenizer from the saved location
        self.tokenizer = BertTokenizer.from_pretrained('backend/models_store/final_tokenizer')  # Load the tokenizer

    def batch_predict(self, texts: List[str]) -> List[str]:
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted class labels
        predictions = outputs.logits.argmax(dim=1)  # Get the index of the highest probability (class prediction)
        
        # Return predictions as a list of strings (e.g., "Positive", "Negative")
        return [str(pred.item()) for pred in predictions]

    def predict(self, text: str) -> str:
        # For a single text, we can just call batch_predict with one element
        return self.batch_predict([text])[0]  # Return first prediction for single text
