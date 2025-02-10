import os
import sys
from pathlib import Path

# Get the project root (assuming this conftest.py is at the project root)
ROOT_DIR = Path(__file__).resolve().parent
backend_dir = ROOT_DIR / "backend"

# Change working directory to the backend folder
os.chdir(backend_dir)

# Optionally, add the backend directory to PYTHONPATH
sys.path.insert(0, str(backend_dir))

print(f"Working directory set to: {os.getcwd()}") 