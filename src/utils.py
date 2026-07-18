import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from src.logger import logger

def save_model(model: Any, filepath: Path) -> None:
    """Saves a machine learning model or pipeline to disk using joblib."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Successfully saved model to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")
        raise

def load_model(filepath: Path) -> Any:
    """Loads a machine learning model or pipeline from disk using joblib."""
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        model = joblib.load(filepath)
        logger.info(f"Successfully loaded model from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise

def save_metrics_json(metrics: Dict[str, float], filepath: Path) -> None:
    """Saves model metrics to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Successfully saved metrics JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON to {filepath}: {e}")
        raise

def save_metrics_csv(metrics: Dict[str, float], filepath: Path) -> None:
    """Saves model metrics to a CSV file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Convert dictionary to DataFrame
        df = pd.DataFrame([metrics])
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved metrics CSV to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save metrics CSV to {filepath}: {e}")
        raise
