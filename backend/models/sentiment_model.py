from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import os
import torch
from config import MODEL_PATH, TOKENIZER_PATH, HF_MODEL_REPO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self):
        logger.info("Initializing SentimentModel...")
        logger.info(f"Loading model from: {MODEL_PATH}")
        logger.info(f"Loading tokenizer from: {TOKENIZER_PATH}")
        
        try:
            model_path = Path(MODEL_PATH)
            tokenizer_path = Path(TOKENIZER_PATH)

            
            # Load tokenizer (check local first, else download from HF)
            if not os.path.exists(TOKENIZER_PATH):
                logger.warning("Tokenizer not found locally. Downloading from Hugging Face...")
                self.tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO, subfolder="final_tokenizer")
                self.tokenizer.save_pretrained(TOKENIZER_PATH)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))

            # Load model (check local first, else download from HF)
            model_file = f"{MODEL_PATH}/model.safetensors"
            if not os.path.exists(model_file):
                logger.warning("Model file not found locally. Downloading from Hugging Face...")
                self.model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO, subfolder="final_model")
                self.model.save_pretrained(MODEL_PATH)
            else:
                state_dict = load_file(model_file)
                self.model = BertForSequenceClassification.from_pretrained(
                    str(model_path),
                    state_dict=state_dict
                )

            self.model.eval()
            logger.info("SentimentModel initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, text: str) -> str:
        """Predict sentiment for a single text."""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        return str(prediction)  # Convert to string as per your API contract

    def batch_predict(self, texts: list[str]) -> list[str]:
        """Predict sentiment for a batch of texts."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1).tolist()
        return [str(pred) for pred in predictions]  # Convert to list of strings
