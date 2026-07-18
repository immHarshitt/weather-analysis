import pandas as pd
import numpy as np
from typing import Any
from src.logger import logger
from src.feature_engineering import get_feature_columns

def generate_iterative_forecast(model: Any, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Generates an iterative multi-day temperature forecast using the trained model.
    Predicts mean temperature recursively, while estimating max/min temperatures
    based on the prior day's temperature spread.
    """
    if len(history_df) < 7:
        raise ValueError(f"History DataFrame must contain at least 7 records to construct lag and rolling features. Got: {len(history_df)}")
        
    logger.info(f"Generating iterative temperature forecast for a {horizon}-day horizon...")
    
    # Work on a copy of the historical data
    df = history_df[["TempMean", "TempMax", "TempMin"]].copy()
    
    # Determine the date range for the forecast
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    
    feature_cols = get_feature_columns()
    forecast_records = []
    
    for current_date in forecast_dates:
        # 1. Lag features
        temp_lag_1 = df["TempMean"].iloc[-1]
        prev_max = df["TempMax"].iloc[-1]
        prev_min = df["TempMin"].iloc[-1]
        temp_lag_7 = df["TempMean"].iloc[-7]
        
        # 2. Rolling window features
        temp_ma_3 = df["TempMean"].iloc[-3:].mean()
        temp_ma_7 = df["TempMean"].iloc[-7:].mean()
        temp_std_7 = df["TempMean"].iloc[-7:].std()
        
        # 3. Calendar/temporal features
        day_of_week = current_date.dayofweek
        month = current_date.month
        day_of_year = current_date.dayofyear
        season = (month % 12) // 3
        
        # Build feature dictionary in correct order
        feat_dict = {
            "temp_lag_1": temp_lag_1,
            "prev_max": prev_max,
            "prev_min": prev_min,
            "temp_lag_7": temp_lag_7,
            "temp_ma_3": temp_ma_3,
            "temp_ma_7": temp_ma_7,
            "temp_std_7": temp_std_7,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_year": day_of_year,
            "season": season
        }
        
        # Create input DataFrame for model prediction
        X_pred = pd.DataFrame([feat_dict])
        X_pred = X_pred[feature_cols]  # Ensure exact column alignment
        
        # Generate mean prediction
        pred_mean = float(model.predict(X_pred)[0])
        
        # Heuristic for max/min spread
        spread = prev_max - prev_min
        # Clip spread to prevent unrealistic convergence
        spread = max(1.0, spread)
        
        pred_max = pred_mean + spread / 2.0
        pred_min = pred_mean - spread / 2.0
        
        # Append prediction to history to feed next steps
        new_row = pd.DataFrame({
            "TempMean": [pred_mean],
            "TempMax": [pred_max],
            "TempMin": [pred_min]
        }, index=[current_date])
        df = pd.concat([df, new_row])
        
        # Save record
        record = {
            "Date": current_date,
            "TempMean": pred_mean,
            "TempMax": pred_max,
            "TempMin": pred_min
        }
        forecast_records.append(record)
        
    forecast_df = pd.DataFrame(forecast_records)
    forecast_df.set_index("Date", inplace=True)
    logger.info("Iterative forecast generation completed successfully.")
    
    return forecast_df
