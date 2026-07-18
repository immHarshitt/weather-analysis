import numpy as np
import pandas as pd
from typing import Dict, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import logger
from src.utils import save_metrics_json, save_metrics_csv

def calculate_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Computes regression performance metrics:
    - Root Mean Square Error (RMSE)
    - Mean Absolute Error (MAE)
    - R-squared (R²) Score
    - Mean Absolute Percentage Error (MAPE)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    
    # Handle division by zero for MAPE
    denom = np.where(y_true_arr == 0.0, 1e-5, y_true_arr)
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape
    }
    
    logger.info(f"Calculated metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    return metrics

def export_evaluation_results(metrics: Dict[str, float], metrics_csv_path: Any, metrics_json_path: Any) -> None:
    """Exports metrics to both CSV and JSON formats."""
    save_metrics_json(metrics, metrics_json_path)
    save_metrics_csv(metrics, metrics_csv_path)
    logger.info("Successfully exported evaluation results.")
