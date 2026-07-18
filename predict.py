import argparse
import pandas as pd
from pathlib import Path

from src.config import (
    DEFAULT_LATITUDE, 
    DEFAULT_LONGITUDE, 
    DEFAULT_FORECAST_HORIZON, 
    MODEL_PATH, 
    RESULTS_DIR
)
from src.logger import logger
from src.utils import load_model
from src.data_loader import fetch_historical_weather
from src.preprocessing import preprocess_weather_data
from src.forecasting import generate_iterative_forecast

def run_prediction(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    history_start: str = "2025-12-01", 
    history_end: str = "2025-12-31",
    use_sample: bool = False
) -> pd.DataFrame:
    """Loads model, fetches recent history, runs iterative forecasting, and exports to CSV."""
    logger.info("Starting prediction/forecasting script...")
    
    # 1. Load model pipeline
    model = load_model(MODEL_PATH)
    
    # 2. Fetch history required to build lags/rolling averages (at least 7 days needed)
    history_raw = fetch_historical_weather(
        latitude=latitude, 
        longitude=longitude, 
        start_date=history_start, 
        end_date=history_end,
        use_sample=use_sample
    )
    
    # 3. Preprocess history
    history_clean = preprocess_weather_data(history_raw)
    
    # 4. Generate forecast iteratively
    forecast_df = generate_iterative_forecast(model, history_clean, horizon)
    
    # 5. Export predictions
    predictions_path = RESULTS_DIR / "predictions.csv"
    forecast_export = forecast_df[["TempMean", "TempMax", "TempMin"]].copy()
    forecast_export.rename(columns={"TempMean": "predicted_mean", "TempMax": "predicted_max", "TempMin": "predicted_min"}, inplace=True)
    forecast_export.to_csv(predictions_path, index=True)
    
    logger.info(f"Forecast predictions exported successfully to {predictions_path}")
    return forecast_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weather prediction forecast.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE, help="Latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE, help="Longitude")
    parser.add_argument("--horizon", type=int, default=DEFAULT_FORECAST_HORIZON, help="Forecast horizon (3, 7, 14 days)")
    parser.add_argument("--sample", action="store_true", help="Use offline sample data")
    
    args = parser.parse_args()
    
    run_prediction(
        latitude=args.lat,
        longitude=args.lon,
        horizon=args.horizon,
        use_sample=args.sample
    )
