import json
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SAMPLE_DATA = Path(__file__).parent / "data" / "sample_weather.csv"


def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DATA, parse_dates=["Date"]).set_index("Date")
    return df.dropna()


def fetch_weather(
    latitude: float,
    longitude: float,
    days: int = 730,
) -> pd.DataFrame:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }

    url = f"{ARCHIVE_URL}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError(
            "Could not fetch live weather data. Try demo mode or check your connection."
        ) from exc

    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        raise ValueError("No weather data returned for this location.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily["time"]),
            "TempMean": daily["temperature_2m_mean"],
            "TempMax": daily["temperature_2m_max"],
            "TempMin": daily["temperature_2m_min"],
        }
    ).set_index("Date")

    return df.dropna()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["PrevMean"] = out["TempMean"].shift(1)
    out["PrevMax"] = out["TempMax"].shift(1)
    out["PrevMin"] = out["TempMin"].shift(1)
    out["MA3"] = out["TempMean"].rolling(3).mean()
    out["MA7"] = out["TempMean"].rolling(7).mean()
    out["DayOfYear"] = out.index.dayofyear
    out["Target"] = out["TempMean"].shift(-1)
    return out.dropna()


def train_and_predict(
    latitude: float,
    longitude: float,
    days: int = 730,
    forecast_days: int = 7,
    use_sample: bool = False,
):
    if use_sample:
        raw = load_sample_data().tail(days)
    else:
        raw = fetch_weather(latitude, longitude, days)
    data = build_features(raw)

    features = ["PrevMean", "PrevMax", "PrevMin", "MA3", "MA7", "DayOfYear"]
    X = data[features]
    y = data["Target"]

    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    history = pd.DataFrame(
        {
            "Date": y_test.index,
            "Actual": y_test.values,
            "Predicted": test_pred,
        }
    )

    forecast_rows = []
    rolling = raw.copy()
    cursor = raw.index[-1]

    for step in range(1, forecast_days + 1):
        next_date = cursor + timedelta(days=1)
        prev = rolling.iloc[-1]
        recent = rolling["TempMean"].tail(7)

        ma3 = recent.tail(3).mean()
        ma7 = recent.mean()
        day_of_year = next_date.timetuple().tm_yday

        features_row = pd.DataFrame(
            [
                {
                    "PrevMean": prev["TempMean"],
                    "PrevMax": prev["TempMax"],
                    "PrevMin": prev["TempMin"],
                    "MA3": ma3,
                    "MA7": ma7,
                    "DayOfYear": day_of_year,
                }
            ]
        )

        predicted_mean = float(model.predict(features_row)[0])
        spread = prev["TempMax"] - prev["TempMin"]
        predicted_max = predicted_mean + spread / 2
        predicted_min = predicted_mean - spread / 2

        forecast_rows.append(
            {
                "Date": next_date,
                "TempMean": predicted_mean,
                "TempMax": predicted_max,
                "TempMin": predicted_min,
            }
        )

        rolling = pd.concat(
            [
                rolling,
                pd.DataFrame(
                    [
                        {
                            "TempMean": predicted_mean,
                            "TempMax": predicted_max,
                            "TempMin": predicted_min,
                        }
                    ],
                    index=[next_date],
                ),
            ]
        )
        cursor = next_date

    forecast = pd.DataFrame(forecast_rows).set_index("Date")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "use_sample": use_sample,
        "last_temp": float(raw["TempMean"].iloc[-1]),
        "next_temp": float(forecast["TempMean"].iloc[0]),
        "mae": mae,
        "rmse": rmse,
        "history": history,
        "raw": raw,
        "forecast": forecast,
    }
