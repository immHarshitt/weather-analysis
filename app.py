import matplotlib.pyplot as plt
import streamlit as st

from model import train_and_predict

CITIES = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "Mumbai": (19.0760, 72.8777),
    "Sydney": (-33.8688, 151.2093),
}

st.set_page_config(page_title="Weather Analysis", page_icon="🌡️", layout="centered")

st.title("Weather Data Analysis & Prediction")
st.caption("Historical temperature trends with linear regression forecasting.")

city = st.selectbox("City", list(CITIES.keys()), index=0)
use_custom = st.checkbox("Use custom coordinates")

if use_custom:
    latitude = st.number_input("Latitude", value=CITIES[city][0], format="%.4f")
    longitude = st.number_input("Longitude", value=CITIES[city][1], format="%.4f")
else:
    latitude, longitude = CITIES[city]

days = st.selectbox("History window", [365, 730, 1095], index=1, format_func=lambda d: f"{d} days")
forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
use_sample = st.checkbox("Demo mode (offline sample data)", value=False)

if st.button("Analyze & Predict", type="primary"):
    label = f"{city} (demo)" if use_sample else city
    with st.spinner(f"Loading weather data for {label}..."):
        try:
            result = train_and_predict(
                latitude, longitude, days, forecast_days, use_sample=use_sample
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    change = result["next_temp"] - result["last_temp"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Mean Temp", f"{result['last_temp']:.1f}°C")
    c2.metric("Predicted Tomorrow", f"{result['next_temp']:.1f}°C", f"{change:+.1f}°C")
    c3.metric("Test MAE", f"{result['mae']:.2f}°C")

    st.write(f"Test RMSE: **{result['rmse']:.2f}°C**")

    st.subheader("Historical trend")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(result["raw"].index, result["raw"]["TempMean"], label="Daily mean", alpha=0.8)
    ax1.plot(result["raw"].index, result["raw"]["TempMax"], label="Daily max", alpha=0.5)
    ax1.plot(result["raw"].index, result["raw"]["TempMin"], label="Daily min", alpha=0.5)
    ax1.set_title(f"{city} — Historical Temperature")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    st.subheader("Model fit (test set)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(result["history"]["Date"], result["history"]["Actual"], label="Actual")
    ax2.plot(result["history"]["Date"], result["history"]["Predicted"], label="Predicted")
    ax2.set_title(f"{city} — Test Set Predictions")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend()
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    st.subheader(f"{forecast_days}-day forecast")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    recent = result["raw"].tail(30)
    ax3.plot(recent.index, recent["TempMean"], label="Recent actual", color="C0")
    ax3.plot(
        result["forecast"].index,
        result["forecast"]["TempMean"],
        label="Forecast mean",
        color="C1",
        marker="o",
    )
    ax3.fill_between(
        result["forecast"].index,
        result["forecast"]["TempMin"],
        result["forecast"]["TempMax"],
        alpha=0.2,
        color="C1",
        label="Forecast range",
    )
    ax3.set_title(f"{city} — Temperature Forecast")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Temperature (°C)")
    ax3.legend()
    fig3.autofmt_xdate()
    st.pyplot(fig3)

    st.dataframe(
        result["forecast"].reset_index().rename(columns={"index": "Date"}),
        use_container_width=True,
    )

    st.info(
        "Educational demo using Open-Meteo historical data and simple linear regression. "
        "Weather is complex — use professional forecasts for real decisions."
    )
