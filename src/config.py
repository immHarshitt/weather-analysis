import os
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure dirs exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
API_TIMEOUT_SECONDS = 30
API_MAX_RETRIES = 3
API_BACKOFF_FACTOR = 1.0

# Default Cities (Preserving user list)
CITIES = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "Mumbai": (19.0760, 72.8777),
    "Sydney": (-33.8688, 151.2093),
}
DEFAULT_CITY = "New York"
DEFAULT_LATITUDE = CITIES[DEFAULT_CITY][0]
DEFAULT_LONGITUDE = CITIES[DEFAULT_CITY][1]

# Prediction & time-series parameters
RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_FORECAST_HORIZON = 7
ALLOWED_FORECAST_HORIZONS = [3, 7, 14]

# File names
MODEL_FILENAME = "linear_regression.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
SAMPLE_DATA_PATH = DATA_DIR / "sample_weather.csv"
