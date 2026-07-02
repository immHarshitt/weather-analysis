# Weather Data Analysis & Prediction

Analyze historical weather data and predict future temperature trends using time series features and linear regression.

## Setup

```bash
cd ~/Projects/weather-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

Pick a city, choose a history window, and click **Analyze & Predict**.

## How it works

- Fetches daily temperature data from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api)
- Builds features: previous day temps, 3/7-day moving averages, day-of-year seasonality
- Trains linear regression on 80% of data, evaluates on the last 20%
- Forecasts the next 3–14 days using iterative predictions
