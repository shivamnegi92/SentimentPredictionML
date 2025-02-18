import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure the root logger
    logging.basicConfig(level=log_level)
    
    # Create logger
    logger = logging.getLogger("sentiment-api")
    logger.setLevel(log_level)
    
    # Format for logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "logs/api.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Ensure logger propagates to root
    logger.propagate = True
    
    return logger
