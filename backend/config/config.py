import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings (add more as needed)
MODEL_PATH = os.getenv("MODEL_PATH", "path_to_your_model")
