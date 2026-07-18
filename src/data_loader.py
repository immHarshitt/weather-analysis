import time
import json
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Optional
from src.config import (
    OPEN_METEO_ARCHIVE_URL, 
    API_TIMEOUT_SECONDS, 
    API_MAX_RETRIES, 
    API_BACKOFF_FACTOR,
    DATA_DIR,
    SAMPLE_DATA_PATH
)
from src.logger import logger

def load_sample_data() -> pd.DataFrame:
    """Loads sample offline demo weather data."""
    try:
        if not SAMPLE_DATA_PATH.exists():
            raise FileNotFoundError(f"Sample data file not found at {SAMPLE_DATA_PATH}")
        df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"]).set_index("Date")
        logger.info(f"Loaded sample weather data from {SAMPLE_DATA_PATH} ({len(df)} records)")
        return df.dropna()
    except Exception as e:
        logger.error(f"Failed to load sample weather data: {e}")
        raise

def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    use_sample: bool = False,
    cache_filename: str = "historical_weather_cache.csv"
) -> pd.DataFrame:
    """
    Fetches daily mean, max, and min temperatures.
    - If use_sample is True, loads from sample CSV.
    - If use_sample is False, fetches from Open-Meteo Archive API.
    - Supports fallback to local cache on API failure.
    """
    if use_sample:
        df_sample = load_sample_data()
        if days:
            df_sample = df_sample.tail(days)
        return df_sample

    # Resolve dates if days window is specified
    if days is not None:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        start_date_str = start.isoformat()
        end_date_str = end.isoformat()
    else:
        if not start_date or not end_date:
            raise ValueError("Must provide either 'days' or both 'start_date' and 'end_date'.")
        start_date_str = start_date
        end_date_str = end_date

    cache_path = DATA_DIR / cache_filename
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": "auto"
    }

    logger.info(f"Querying weather API for Lat: {latitude}, Lon: {longitude} from {start_date_str} to {end_date_str}")
    
    response = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = requests.get(
                OPEN_METEO_ARCHIVE_URL, 
                params=params, 
                timeout=API_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            logger.info("API request completed successfully.")
            break
        except requests.RequestException as e:
            logger.warning(f"API request attempt {attempt} failed: {e}")
            if attempt < API_MAX_RETRIES:
                sleep_time = API_BACKOFF_FACTOR * (2 ** (attempt - 1))
                logger.info(f"Retrying API call in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("All API retry attempts failed.")

    # Process response if successful
    if response and response.status_code == 200:
        try:
            data = response.json()
            daily = data.get("daily")
            if not daily or "temperature_2m_mean" not in daily:
                raise ValueError("Response JSON does not contain daily temperature fields.")
            
            df = pd.DataFrame({
                "Date": pd.to_datetime(daily["time"]),
                "TempMean": daily["temperature_2m_mean"],
                "TempMax": daily["temperature_2m_max"],
                "TempMin": daily["temperature_2m_min"]
            }).set_index("Date").dropna()
            
            # Cache the successfully retrieved data
            try:
                df.to_csv(cache_path, index=True)
                logger.info(f"Successfully cached weather data to {cache_path}")
            except Exception as cache_err:
                logger.warning(f"Failed to cache weather data: {cache_err}")
                
            return df
        except Exception as parse_err:
            logger.error(f"Error parsing API response: {parse_err}")

    # Fallback to cache if API calls failed
    logger.warning("Attempting to load data from offline cache fallback...")
    if cache_path.exists():
        try:
            df_cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
            logger.info(f"Successfully loaded offline cached data from {cache_path} ({len(df_cached)} records).")
            
            # Filter the cached data for requested date range
            mask = (df_cached.index >= pd.to_datetime(start_date_str)) & (df_cached.index <= pd.to_datetime(end_date_str))
            df_filtered = df_cached.loc[mask]
            
            if df_filtered.empty:
                logger.warning("Cached data exists but contains no records in the requested date range.")
            else:
                return df_filtered
        except Exception as cache_read_err:
            logger.error(f"Error reading offline cache: {cache_read_err}")
            
    # As a last resort, return sample data to prevent crash
    logger.warning("No cache available. Falling back to sample data as last resort.")
    return load_sample_data().tail(days if days else 365)
