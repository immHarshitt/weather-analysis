import pandas as pd
import numpy as np
from src.logger import logger

def preprocess_weather_data(df: pd.DataFrame, outlier_std_threshold: float = 3.0) -> pd.DataFrame:
    """
    Cleans raw weather data:
    1. Ensures datetime index is sorted.
    2. Imputes missing entries using forward-fill followed by backward-fill.
    3. Handles statistical outliers by clipping them to standard deviation bounds.
    """
    df = df.copy()
    
    # Check that required columns exist
    if "TempMean" not in df.columns:
        raise ValueError("Input DataFrame must contain 'TempMean' column.")
        
    logger.info("Starting data preprocessing...")
    
    # Sort index
    df = df.sort_index()
    
    # Impute missing values across all temperature columns
    for col in ["TempMean", "TempMax", "TempMin"]:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {col}. Forward-filling...")
                df[col] = df[col].ffill().bfill()
                
    # Detect and handle outliers using standard deviation bounds on TempMean
    mean_temp = df["TempMean"].mean()
    std_temp = df["TempMean"].std()
    
    lower_bound = mean_temp - outlier_std_threshold * std_temp
    upper_bound = mean_temp + outlier_std_threshold * std_temp
    
    # Find outliers
    outliers = df[(df["TempMean"] < lower_bound) | (df["TempMean"] > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        logger.warning(f"Detected {outlier_count} temperature outliers outside standard deviation range [{lower_bound:.2f}°C, {upper_bound:.2f}°C].")
        # Clip all columns to reasonable bounds relative to mean bounds
        df["TempMean"] = df["TempMean"].clip(lower=lower_bound, upper=upper_bound)
        if "TempMax" in df.columns:
            df["TempMax"] = df["TempMax"].clip(lower=lower_bound)
        if "TempMin" in df.columns:
            df["TempMin"] = df["TempMin"].clip(upper=upper_bound)
        logger.info("Outliers clipped to standard deviation bounds.")
    else:
        logger.info(f"No statistical outliers detected within {outlier_std_threshold} standard deviations.")
        
    return df
