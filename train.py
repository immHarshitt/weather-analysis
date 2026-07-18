import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import (
    DEFAULT_LATITUDE, 
    DEFAULT_LONGITUDE, 
    RANDOM_STATE, 
    DEFAULT_TRAIN_RATIO, 
    MODEL_PATH
)
from src.logger import logger
from src.data_loader import fetch_historical_weather
from src.preprocessing import preprocess_weather_data
from src.feature_engineering import generate_features, get_feature_columns
from src.utils import save_model

def run_training_pipeline(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    use_sample: bool = False
) -> Pipeline:
    """Trains the Linear Regression pipeline on historical weather data."""
    logger.info("Starting training pipeline...")
    
    # 1. Load Data
    raw_df = fetch_historical_weather(
        latitude=latitude, 
        longitude=longitude, 
        start_date=start_date, 
        end_date=end_date,
        use_sample=use_sample
    )
    
    # 2. Preprocess Data
    cleaned_df = preprocess_weather_data(raw_df)
    
    # 3. Feature Engineering
    feature_df = generate_features(cleaned_df)
    
    # Extract features and target
    feature_cols = get_feature_columns()
    X = feature_df[feature_cols]
    y = feature_df["TempMean"]
    
    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    
    # 4. Chronological Train-Test Split (Time-Series constraint)
    split_idx = int(len(feature_df) * train_ratio)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Chronological split - Train size: {len(X_train)} ({X_train.index[0]} to {X_train.index[-1]})")
    logger.info(f"Chronological split - Test size: {len(X_test)} ({X_test.index[0]} to {X_test.index[-1]})")
    
    # 5. Define ML Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    
    # 6. Time-Series Cross Validation (on Train Set)
    logger.info("Performing TimeSeriesSplit Cross Validation on training set...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        pipeline, 
        X_train, 
        y_train, 
        cv=tscv, 
        scoring="r2"
    )
    
    logger.info(f"Cross-Validation R² Scores: {cv_scores}")
    logger.info(f"Mean CV R² Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # 7. Model Training (Full training set)
    logger.info("Training final pipeline model...")
    pipeline.fit(X_train, y_train)
    
    # Quick sanity check on test set R2
    test_r2 = pipeline.score(X_test, y_test)
    logger.info(f"Quick Test R² evaluation: {test_r2:.4f}")
    
    # 8. Save Model
    save_model(pipeline, MODEL_PATH)
    logger.info("Training pipeline execution finished successfully.")
    
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weather Predictor model.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE, help="Latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE, help="Longitude")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--sample", action="store_true", help="Use offline sample data")
    
    args = parser.parse_args()
    
    run_training_pipeline(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=args.end,
        use_sample=args.sample
    )
