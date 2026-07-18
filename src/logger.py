import logging
import sys
from pathlib import Path

# Resolve root path to save logs in the project root
ROOT_DIR = Path(__file__).parent.parent.resolve()
LOG_FILE = ROOT_DIR / "weather_analysis.log"

def setup_logger(name: str = "weather_analysis") -> logging.Logger:
    """Sets up a logger with both File and Stream handlers."""
    logger = logging.getLogger(name)
    
    # If the logger has already been configured, do not add handlers again
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Failed to initialize file logger: {e}", file=sys.stderr)
        
    return logger

# Create default logger
logger = setup_logger("weather_analysis")
