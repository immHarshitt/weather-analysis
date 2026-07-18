import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    DEFAULT_LATITUDE, 
    DEFAULT_LONGITUDE, 
    DEFAULT_TRAIN_RATIO, 
    MODEL_PATH, 
    RESULTS_DIR, 
    PLOTS_DIR
)
from src.logger import logger
from src.utils import load_model
from src.data_loader import fetch_historical_weather
from src.preprocessing import preprocess_weather_data
from src.feature_engineering import generate_features, get_feature_columns
from src.metrics import calculate_metrics, export_evaluation_results
from src.forecasting import generate_iterative_forecast
from src.visualization import generate_all_plots

def run_evaluation(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    use_sample: bool = False
) -> None:
    """Evaluates the trained model, exports metrics, and generates 10 plots."""
    logger.info("Starting evaluation pipeline...")
    
    # 1. Load Model Pipeline
    model = load_model(MODEL_PATH)
    
    # 2. Fetch & Preprocess Data
    raw_df = fetch_historical_weather(
        latitude=latitude, 
        longitude=longitude, 
        start_date=start_date, 
        end_date=end_date,
        use_sample=use_sample
    )
    cleaned_df = preprocess_weather_data(raw_df)
    feature_df = generate_features(cleaned_df)
    
    # Extract features & target
    feature_cols = get_feature_columns()
    X = feature_df[feature_cols]
    y = feature_df["TempMean"]
    
    # 3. Chronological Train-Test Split
    split_idx = int(len(feature_df) * train_ratio)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # 4. Predict on Test Set
    logger.info(f"Generating predictions on test set of size {len(X_test)}...")
    y_pred = model.predict(X_test)
    
    # 5. Compute Metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # 6. Export Metrics
    metrics_csv_path = RESULTS_DIR / "metrics.csv"
    metrics_json_path = RESULTS_DIR / "evaluation.json"
    export_evaluation_results(metrics, metrics_csv_path, metrics_json_path)
    
    # 7. Generate 14-Day Forecast for visualization purposes
    forecast_df = generate_iterative_forecast(model, cleaned_df, 14)
    
    # 8. Generate and save the 10 plots
    logger.info("Generating evaluation and data analysis plots...")
    test_df = cleaned_df.iloc[split_idx + 7:] # account for dropped lags/rolling rows in feature engineering
    
    generate_all_plots(
        df_clean=feature_df,
        test_df=test_df,
        y_test=y_test,
        y_pred=y_pred,
        forecast_df=forecast_df,
        model_pipeline=model,
        plots_dir=PLOTS_DIR
    )
    
    logger.info("Evaluation pipeline execution finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Weather Predictor model.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE, help="Latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE, help="Longitude")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--sample", action="store_true", help="Use offline sample data")
    
    args = parser.parse_args()
    
    run_evaluation(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=args.end,
        use_sample=args.sample
    )
