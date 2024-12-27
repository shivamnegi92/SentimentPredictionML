import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "backend/models_store/final_model")  # Default value is set to path in backend
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "backend/models_store/final_tokenizer")  # Tokenizer path
