import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
from backend.config import MODEL_PATH, TOKENIZER_PATH, HF_MODEL_REPO
import logging

logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self):
        logger.info("Initializing SentimentModel...")
        
        # Load or download tokenizer
        if not os.path.exists(TOKENIZER_PATH):
            logger.warning("Tokenizer not found locally. Downloading from Hugging Face...")
            self.tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO + "/final_tokenizer")
            self.tokenizer.save_pretrained(TOKENIZER_PATH)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        
        # Load or download model
        model_file = f"{MODEL_PATH}/model.safetensors"
        if not os.path.exists(model_file):
            logger.warning("Model file not found locally. Downloading from Hugging Face...")
            self.model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO + "/final_model")
            self.model.save_pretrained(MODEL_PATH)
        else:
            state_dict = load_file(model_file)
            self.model = BertForSequenceClassification.from_pretrained(
                MODEL_PATH,
                state_dict=state_dict
            )
        
        self.model.eval()
        logger.info("SentimentModel initialized successfully")

    def batch_predict(self, texts: list[str]) -> list[str]:
        """Batch prediction for multiple texts"""
        logger.debug("Processing batch of %d texts", len(texts))
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=1)
        
        results = [str(pred.item()) for pred in predictions]
        logger.debug("Batch predictions completed")
        return results

    def predict(self, text: str) -> str:
        """Predict sentiment for single text"""
        logger.debug("Processing single text: %s", text[:50])
        result = self.batch_predict([text])[0]
        logger.debug("Prediction result: %s", result)
        return result
