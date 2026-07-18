import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path
import json

from src.config import (
    CITIES,
    ALLOWED_FORECAST_HORIZONS,
    MODEL_PATH,
    RESULTS_DIR
)
from src.logger import logger
from src.data_loader import fetch_historical_weather
from src.preprocessing import preprocess_weather_data
from src.feature_engineering import generate_features, get_feature_columns
from src.forecasting import generate_iterative_forecast
from src.metrics import calculate_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Weather Analytics & Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }
    .stMetric label {
        color: #475569 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .section-header {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to run the full training and prediction pipeline dynamically
def run_dynamic_pipeline(latitude: float, longitude: float, days: int, forecast_days: int, use_sample: bool):
    # 1. Load Data
    raw = fetch_historical_weather(
        latitude=latitude,
        longitude=longitude,
        days=days,
        use_sample=use_sample
    )
    
    # 2. Preprocess Data
    cleaned = preprocess_weather_data(raw)
    
    # 3. Feature Engineering
    feature_df = generate_features(cleaned)
    
    # Extract features and target
    feature_cols = get_feature_columns()
    X = feature_df[feature_cols]
    y = feature_df["TempMean"]
    
    # 4. Split chronologically
    split = int(len(feature_df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 5. Fit pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    # 6. Predict on test set
    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    history = pd.DataFrame({
        "Date": y_test.index,
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    
    # 7. Generate iterative forecast
    forecast = generate_iterative_forecast(pipeline, cleaned, forecast_days)
    
    # Auto-save results to output files
    try:
        from src.utils import save_model
        save_model(pipeline, MODEL_PATH)
        
        # Save forecast predictions to results
        predictions_path = RESULTS_DIR / "predictions.csv"
        forecast_export = forecast[["TempMean", "TempMax", "TempMin"]].copy()
        forecast_export.rename(columns={"TempMean": "predicted_mean", "TempMax": "predicted_max", "TempMin": "predicted_min"}, inplace=True)
        forecast_export.to_csv(predictions_path, index=True)
        
        # Save metrics
        metrics_csv_path = RESULTS_DIR / "metrics.csv"
        metrics_json_path = RESULTS_DIR / "evaluation.json"
        from src.metrics import export_evaluation_results
        export_evaluation_results(metrics, metrics_csv_path, metrics_json_path)
    except Exception as save_err:
        logger.warning(f"Could not auto-save evaluation files: {save_err}")
        
    return {
        "latitude": latitude,
        "longitude": longitude,
        "use_sample": use_sample,
        "last_temp": float(cleaned["TempMean"].iloc[-1]),
        "next_temp": float(forecast["TempMean"].iloc[0]),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2_score": metrics["r2_score"],
        "mape": metrics["mape"],
        "history": history,
        "raw": cleaned,
        "forecast": forecast,
        "feature_df": feature_df,
        "pipeline": pipeline,
        "y_test": y_test,
        "y_pred": y_pred
    }

# Title
st.title("🌦️ Weather Data Analysis & Prediction Dashboard")
st.markdown("An end-to-end Machine Learning pipeline utilizing Linear Regression to predict daily mean temperatures and generate recursive multi-day forecasts.")

# Sidebar Settings
st.sidebar.header("📍 Location & Date Settings")
city = st.sidebar.selectbox("City", list(CITIES.keys()), index=0)
use_custom = st.sidebar.checkbox("Use custom coordinates")

if use_custom:
    latitude = st.sidebar.number_input("Latitude", value=CITIES[city][0], format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=CITIES[city][1], format="%.4f")
else:
    latitude, longitude = CITIES[city]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Data & Horizon Settings")
days = st.sidebar.selectbox("History Window", [365, 730, 1095], index=1, format_func=lambda d: f"{d} days")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 3, 14, 7)
use_sample = st.sidebar.checkbox("Demo Mode (Offline Sample Data)", value=False)

st.sidebar.markdown("---")
# Retrain action button
analyze_clicked = st.sidebar.button("📊 Analyze & Predict", type="primary", use_container_width=True)

# Run pipeline when clicked
if analyze_clicked or "pipeline_results" not in st.session_state:
    label = f"{city} (demo)" if use_sample else city
    with st.spinner(f"Running ML pipeline for {label}..."):
        try:
            results = run_dynamic_pipeline(latitude, longitude, days, forecast_days, use_sample=use_sample)
            st.session_state["pipeline_results"] = results
            st.sidebar.success("Analysis complete!")
        except Exception as exc:
            st.error(f"Error executing pipeline: {exc}")
            st.stop()

# Retrieve results from state
res = st.session_state["pipeline_results"]

# Metric Cards Display
st.markdown("<div class='section-header'>📊 Model Performance Metrics</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Latest Temp Mean", value=f"{res['last_temp']:.1f} °C")
with col2:
    change = res["next_temp"] - res["last_temp"]
    st.metric(label="Predicted Tomorrow", value=f"{res['next_temp']:.1f} °C", delta=f"{change:+.1f} °C")
with col3:
    st.metric(label="Test Set RMSE", value=f"{res['rmse']:.3f} °C")
with col4:
    st.metric(label="Test Set R² Score", value=f"{res['r2_score']:.4f}")

# Tabs for visual categories
tab_forecast, tab_history, tab_diagnostics = st.tabs([
    "🔮 Multi-Day Forecast", 
    "📈 Historical Trend Analysis", 
    "🛠️ Model Diagnostics & Importances"
])

# --- Tab 1: Multi-day Forecast ---
with tab_forecast:
    st.markdown("<h3 style='color: #1e3a8a;'>Forecast and Test set Predictions</h3>", unsafe_allow_html=True)
    
    # 1. Forecast Plot with range shading
    fig_fc = go.Figure()
    # Plot recent history (last 30 days)
    hist_subset = res["raw"].tail(30)
    fig_fc.add_trace(go.Scatter(
        x=hist_subset.index, y=hist_subset["TempMean"],
        mode="lines+markers", name="Recent Historical Actual",
        line=dict(color="#1e3a8a", width=2),
        marker=dict(size=6)
    ))
    # Plot forecast
    fig_fc.add_trace(go.Scatter(
        x=res["forecast"].index, y=res["forecast"]["TempMean"],
        mode="lines+markers", name="Forecast Mean",
        line=dict(color="#f59e0b", width=2),
        marker=dict(symbol="square", size=6)
    ))
    # Forecast range shading
    fig_fc.add_trace(go.Scatter(
        x=list(res["forecast"].index) + list(res["forecast"].index)[::-1],
        y=list(res["forecast"]["TempMax"]) + list(res["forecast"]["TempMin"])[::-1],
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Forecast Range"
    ))
    fig_fc.update_layout(
        title=f"Recursive {forecast_days}-Day Temperature Forecast",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Dataframe display
    st.markdown("<h4 style='color: #1e3a8a;'>Forecast Value Table</h4>", unsafe_allow_html=True)
    st.dataframe(
        res["forecast"].reset_index().rename(columns={"index": "Date"}),
        use_container_width=True,
    )
    
    # Downloads
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        # Download predictions CSV
        csv_forecast = res["forecast"].reset_index().rename(columns={"Date": "date"}).to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast predictions.csv",
            data=csv_forecast,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        # Download metrics CSV
        metrics_dict = {"rmse": res["rmse"], "mae": res["mae"], "r2_score": res["r2_score"], "mape": res["mape"]}
        csv_metrics = pd.DataFrame([metrics_dict]).to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics metrics.csv",
            data=csv_metrics,
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- Tab 2: Historical Analysis ---
with tab_history:
    st.markdown("<h3 style='color: #1e3a8a;'>Historical Weather Observations</h3>", unsafe_allow_html=True)
    
    # 1. Historical temperature trend
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=res["raw"].index, y=res["raw"]["TempMean"], name="Daily Mean Temp", line=dict(color="#1e3a8a", width=1.5)))
    fig_hist.add_trace(go.Scatter(x=res["raw"].index, y=res["raw"]["TempMax"], name="Daily Max Temp", line=dict(color="#ef4444", width=1), opacity=0.4))
    fig_hist.add_trace(go.Scatter(x=res["raw"].index, y=res["raw"]["TempMin"], name="Daily Min Temp", line=dict(color="#3b82f6", width=1), opacity=0.4))
    fig_hist.update_layout(
        title=f"Historical Temperature Trend ({city})",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # 2. Moving Average Overlay (Zoomed in on last 365 days)
    st.markdown("---")
    st.markdown("<h4 style='color: #1e3a8a;'>7-Day and 3-Day Moving Average Overlays</h4>", unsafe_allow_html=True)
    zoom_len = min(365, len(res["feature_df"]))
    zoom_df = res["feature_df"].tail(zoom_len)
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=zoom_df.index, y=zoom_df["TempMean"], name="Daily Mean", line=dict(color="#94a3b8", width=1), opacity=0.6))
    fig_ma.add_trace(go.Scatter(x=zoom_df.index, y=zoom_df["temp_ma_3"], name="3-Day Moving Avg", line=dict(color="#3b82f6", width=1.5)))
    fig_ma.add_trace(go.Scatter(x=zoom_df.index, y=zoom_df["temp_ma_7"], name="7-Day Moving Avg", line=dict(color="#ef4444", width=2)))
    fig_ma.update_layout(
        title=f"Daily Temperature with Moving Averages (Last {zoom_len} Days)",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # 3. Monthly distribution (Boxplot & Monthly Averages)
    st.markdown("---")
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        monthly_avg = res["feature_df"].groupby("month")["TempMean"].mean().reset_index()
        months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg["month_name"] = [months_names[m-1] for m in monthly_avg["month"]]
        
        fig_m_avg = px.line(
            monthly_avg, x="month_name", y="TempMean", markers=True,
            title="Average Temperature by Month",
            color_discrete_sequence=["#8b5cf6"]
        )
        fig_m_avg.update_layout(xaxis_title="Month", yaxis_title="Mean Temperature (°C)")
        st.plotly_chart(fig_m_avg, use_container_width=True)
        
    with col_hist2:
        df_box = res["feature_df"].reset_index().copy()
        df_box["month_name"] = [months_names[m-1] for m in df_box["month"]]
        fig_box = px.box(
            df_box, x="month_name", y="TempMean", color="month_name",
            title="Temperature Distribution Box Plot by Month",
            color_discrete_sequence=px.colors.qualitative.Flare
        )
        fig_box.update_layout(xaxis_title="Month", yaxis_title="Temperature (°C)", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# --- Tab 3: Model Diagnostics & Importances ---
with tab_diagnostics:
    st.markdown("<h3 style='color: #1e3a8a;'>Diagnostics and Feature Coefficients</h3>", unsafe_allow_html=True)
    
    # Residuals calculations
    y_test_vals = res["y_test"].values
    residuals = y_test_vals - res["y_pred"]
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        # Residuals Scatter
        fig_res = px.scatter(
            x=res["y_pred"], y=residuals,
            title="Residuals vs. Predicted Temperature",
            labels={"x": "Predicted Temperature (°C)", "y": "Residuals (°C)"},
            color_discrete_sequence=["#84cc16"],
            opacity=0.6
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="#b91c1c", line_width=2)
        st.plotly_chart(fig_res, use_container_width=True)
        
    with col_diag2:
        # Error Distribution
        import scipy.stats as stats
        fig_err = px.histogram(
            residuals, nbins=30, histnorm='probability density',
            title="Error (Residuals) Distribution Histogram",
            labels={"value": "Prediction Error (°C)"},
            color_discrete_sequence=["#06b6d4"],
            opacity=0.7
        )
        mu, std = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(np.min(residuals), np.max(residuals), 100)
        y_norm = stats.norm.pdf(x_norm, mu, std)
        fig_err.add_trace(go.Scatter(
            x=x_norm, y=y_norm, mode="lines", name="Normal Fit",
            line=dict(color="black", width=2)
        ))
        fig_err.update_layout(showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)
        
    st.markdown("---")
    col_diag3, col_diag4 = st.columns(2)
    
    with col_diag3:
        # Correlation Heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        feature_cols = get_feature_columns()
        features_and_target = ["TempMean"] + feature_cols
        cols_to_use = [col for col in features_and_target if col in res["feature_df"].columns]
        corr_matrix = res["feature_df"][cols_to_use].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr, cbar=False)
        ax_corr.set_title("Correlation Heatmap Matrix", fontweight="bold", pad=15)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        
    with col_diag4:
        # Feature Importance / Coefficients
        try:
            coefs = res["pipeline"].named_steps["regressor"].coef_
        except AttributeError:
            coefs = res["pipeline"].coef_
            
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": coefs,
            "Absolute": np.abs(coefs)
        }).sort_values(by="Absolute", ascending=True)
        
        fig_feat = px.bar(
            importance_df, x="Coefficient", y="Feature", orientation="h",
            title="Feature Importance (Linear Regression Coefficients)",
            color="Coefficient",
            color_continuous_scale=px.colors.diverging.Coolwarm,
            text_auto=".4f"
        )
        fig_feat.update_layout(xaxis_title="Coefficient Value", yaxis_title="Feature")
        st.plotly_chart(fig_feat, use_container_width=True)

# Footer
st.info(
    "🌦️ Upgraded Weather Analysis and Prediction project using Scikit-Learn Linear Regression pipeline and recursive forecasting. "
    "Features include statistical outlier handling, moving averages, standard deviation, and seasonal encodings."
)
