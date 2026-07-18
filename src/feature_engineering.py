import pandas as pd
import numpy as np
from src.logger import logger

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calendar-based features to the DataFrame based on index dates."""
    df = df.copy()
    
    # Day of week: Monday=0, Sunday=6
    df["day_of_week"] = df.index.dayofweek
    
    # Month: 1 to 12
    df["month"] = df.index.month
    
    # Day of year: 1 to 366
    df["day_of_year"] = df.index.dayofyear
    
    # Season: 0=Winter (Dec, Jan, Feb), 1=Spring (Mar, Apr, May), 2=Summer (Jun, Jul, Aug), 3=Autumn (Sep, Oct, Nov)
    df["season"] = (df.index.month % 12) // 3
    
    return df

def generate_features(df: pd.DataFrame, drop_nans: bool = True) -> pd.DataFrame:
    """
    Generates lag and rolling window features for weather prediction:
    - Previous day mean temperature (temp_lag_1 / PrevMean)
    - Previous day max temperature (prev_max / PrevMax)
    - Previous day min temperature (prev_min / PrevMin)
    - 7-day mean temperature persistence (temp_lag_7)
    - 3-day rolling mean temperature (temp_ma_3 / MA3)
    - 7-day rolling mean temperature (temp_ma_7 / MA7)
    - 7-day rolling standard deviation (temp_std_7)
    - Calendar/temporal features (day_of_week, month, day_of_year, season)
    """
    df = df.copy()
    
    # 1. Lags
    df["temp_lag_1"] = df["TempMean"].shift(1)
    df["prev_max"] = df["TempMax"].shift(1)
    df["prev_min"] = df["TempMin"].shift(1)
    df["temp_lag_7"] = df["TempMean"].shift(7)
    
    # 2. Rolling window features
    df["temp_ma_3"] = df["TempMean"].rolling(window=3).mean()
    df["temp_ma_7"] = df["TempMean"].rolling(window=7).mean()
    df["temp_std_7"] = df["TempMean"].rolling(window=7).std()
    
    # 3. Calendar/temporal features
    df = add_temporal_features(df)
    
    if drop_nans:
        initial_len = len(df)
        df = df.dropna()
        dropped_len = initial_len - len(df)
        logger.info(f"Generated features and dropped {dropped_len} early rows containing NaNs.")
    else:
        logger.info("Generated features (kept rows with NaNs).")
        
    return df

def get_feature_columns() -> list:
    """Returns the list of features in the exact order they are engineered."""
    return [
        "temp_lag_1",
        "prev_max",
        "prev_min",
        "temp_lag_7",
        "temp_ma_3",
        "temp_ma_7",
        "temp_std_7",
        "day_of_week",
        "month",
        "day_of_year",
        "season"
    ]
