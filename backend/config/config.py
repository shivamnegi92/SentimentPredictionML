import os
from dotenv import load_dotenv

load_dotenv()

# Update paths to point to the directories
MODEL_PATH = os.getenv("MODEL_PATH", 
                       "backend/models_store/final_model")  # Path to the model directory
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", 
                           "backend/models_store/final_tokenizer")  # Path to the tokenizer directory

HF_MODEL_REPO = "Shivamnegi92/sentimentanalysisml"  # Replace with your repo name
