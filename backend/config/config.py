import os
from dotenv import load_dotenv

load_dotenv()

# Update paths to point to the new directories
MODEL_PATH = os.getenv("MODEL_PATH", 
                      "models_store/final_model")  # Removed 'backend/' prefix

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", 
                          "models_store/final_tokenizer")  # Removed 'backend/' prefix

# HuggingFace repo without the subfolder paths
HF_MODEL_REPO = "Shivamnegi92/sentimentanalysisml"
