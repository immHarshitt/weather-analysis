# Weather Data Analysis & Prediction

An end-to-end production-quality Machine Learning project designed to analyze historical meteorological trends and forecast daily mean temperatures using **Linear Regression**. This project serves as a portfolio piece and internship report submission, demonstrating standard software engineering practices in machine learning, modular design, type hints, proper logging, robust error handling, and web dashboard implementation.

---

## Project Overview

Accurate local weather forecasting is vital for agricultural planning, resource management, and climate studies. While numerical weather prediction (NWP) models require massive supercomputing power, statistical and machine learning approaches can model daily temperature transitions rapidly and interpretably.

This project implements a single-step autoregressive framework for multi-day temperature forecasting. The model retrieves historical daily temperature averages from the **Open-Meteo API**, applies extensive feature engineering (rolling means, lag terms, standard deviations, and calendar season indicators), trains a standardized Linear Regression pipeline, and uses recursive forecasting to project temperature trends 3, 7, or 14 days into the future.

---

## Features

- **Robust Data Acquisition**: Custom Open-Meteo Archive API client supporting request retries with exponential backoff, request timeouts, and an offline cached fallback.
- **Advanced Feature Engineering**:
  - *Lags*: 1-day and 7-day prior temperature observations to capture short-term and weekly persistence.
  - *Rolling Windows*: 3-day and 7-day rolling temperature averages to smooth out daily fluctuations, alongside a 7-day rolling standard deviation.
  - *Cyclic/Temporal Encoding*: Extracting Day of Week, Month, Day of Year, and a customized 4-season encoding.
- **ML Pipeline & Cross Validation**: Time-series aware data splitting (chronological 80/20 train/test split) to prevent data leakage. The model pipeline encapsulates a `StandardScaler` and a `LinearRegression` estimator. Model performance is cross-validated using `TimeSeriesSplit`.
- **Iterative Forecasting**: Multi-day predictions are computed recursively, feeding back predicted values as lag inputs for subsequent steps.
- **Comprehensive Visual Diagnostics**: Generates 10 premium visual plots covering historical trends, actual vs. predicted validation, residuals analysis, error distributions, box plots, monthly trends, and feature coefficients.
- **Interactive Streamlit Web Dashboard**: Sleek UI containing custom sidebar inputs, city presets, metric cards, Plotly-based interactive charts, model retraining options, and prediction/metrics CSV file downloads.

---

## Project Architecture

The system follows a three-layered architecture designed to separate presentation, processing, and data access concerns:

1. **Presentation Layer**: Streamlit web dashboard rendering interactive plots and exposing sidebar controls.
2. **Processing Layer**: Contains the preprocessing, feature engineering, training, forecasting, and evaluation engines.
3. **Data Access Layer**: Communicates with the Open-Meteo REST API and writes/reads cached CSV files.

### Directory Structure

```
weather-analysis/
│
├── app.py                       # Streamlit web application
├── train.py                     # Standalone CLI training script
├── predict.py                   # Standalone CLI prediction/forecasting script
├── evaluate.py                  # Standalone CLI model evaluation & plotting script
│
├── src/                         # Core Python modules
│   ├── config.py                # Configuration constants and path settings
│   ├── logger.py                # Centralized project logging setup
│   ├── utils.py                 # File serialization and CSV/JSON save utilities
│   ├── data_loader.py           # API client and cache fallback loader
│   ├── preprocessing.py         # Missing data imputation and outlier clipping
│   ├── feature_engineering.py   # Lags, rolling windows, and temporal features
│   ├── forecasting.py           # Recursive forecasting algorithm
│   ├── metrics.py               # RMSE, MAE, R², and MAPE calculators
│   └── visualization.py         # Matplotlib and Seaborn plot generator
│
├── models/                      # Saved trained models
│   └── linear_regression.pkl    # Serialized Scikit-Learn Pipeline
│
├── results/                     # Evaluation results and artifacts
│   ├── predictions.csv          # Forecasted values output
│   ├── metrics.csv              # Calculated test metrics (CSV)
│   ├── evaluation.json          # Calculated test metrics (JSON)
│   └── plots/                   # 10 saved PNG plots
│
├── data/                        # Local data and offline API cache
│   └── sample_weather.csv       # Preloaded sample weather data (offline demo)
│
├── notebooks/                   # Jupyter notebooks for exploratory research
├── requirements.txt             # Python project dependencies
├── .gitignore                   # Files excluded from git
└── README.md                    # Project documentation
```

---

## Workflow Diagram

The training and prediction pipeline follows these distinct sequential steps:

1. **Configure Parameters**: User defines location coordinates, date ranges, and forecast horizon.
2. **Fetch Data**: The API Client queries Open-Meteo API. If unsuccessful, it attempts fallback to the local cache, or falls back to demo mode.
3. **Preprocess**: Resolves missing entries via forward-fill and clips anomalies outside 3 standard deviations.
4. **Engineer Features**: Computes lag features, rolling windows, and temporal features. Drops boundary rows.
5. **Split Data**: Partitions data chronologically (80% train / 20% test).
6. **Train Model**: Runs 5-split time-series cross-validation, trains the `StandardScaler` + `LinearRegression` pipeline, and saves the `.pkl` artifact.
7. **Iterative Forecast**: Projects future temperatures recursively step-by-step for the specified horizon (3/7/14 days).
8. **Evaluate & Render**: Outputs evaluation metrics and saves 10 diagnostic plots. Renders the interactive components inside the dashboard.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Pip (Python Package Installer)

### Installation Steps

1. Navigate to the project directory:
   ```bash
   cd ~/Projects/weather-analysis
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the Streamlit Dashboard

To launch the interactive dashboard in your web browser:
```bash
streamlit run app.py
```
By default, the server runs on `http://localhost:8501`.

### 2. Standalone Command Line Execution

You can run each module of the pipeline independently from the command line:

- **Train the Model**:
  ```bash
  python train.py --lat 40.7128 --lon -74.0060 --start 2021-01-01 --end 2025-12-31
  ```
  This trains the model using data for the specified coordinates and date range, saving the model file to `models/linear_regression.pkl`. Add `--sample` to run in offline demo mode using sample data.

- **Evaluate and Generate Plots**:
  ```bash
  python evaluate.py --lat 40.7128 --lon -74.0060 --start 2021-01-01 --end 2025-12-31
  ```
  This evaluates the trained model on the test set, saves evaluation results (`results/metrics.csv` and `results/evaluation.json`), and automatically saves the 10 diagnostic plots in `results/plots/`.

- **Generate Forecasts**:
  ```bash
  python predict.py --lat 40.7128 --lon -74.0060 --horizon 14
  ```
  This loads the trained model and performs a 14-day recursive forecast, saving the predictions to `results/predictions.csv`.

---

## Evaluation Plots

The evaluation pipeline automatically saves 10 high-quality plots in `results/plots/`:

1. `historical_temperature_trend.png` - Continuous daily temperature observations.
2. `moving_average_trend.png` - Daily temperature overlaid with 3-day and 7-day rolling means.
3. `actual_vs_predicted.png` - Model predictions plotted against actual temperature on the last 100 days of the test set.
4. `forecast_14_days.png` - 14-day recursive temperature projections extending from the last known historical date.
5. `residual_plot.png` - Scatter plot showing residual values vs. predicted target values to verify homoscedasticity.
6. `residual_distribution.png` - Density histogram of residuals overlaid with a fitted Normal distribution.
7. `monthly_average_temperature.png` - Multi-year mean temperatures grouped by calendar month to show macro-seasonality.
8. `monthly_temperature_boxplot.png` - Box plots showing dispersion, medians, and outliers of temperatures for each month.
9. `correlation_heatmap.png` - Correlation coefficient matrix heatmap of features and targets.
10. `feature_importance.png` - horizontal bar chart showing Scikit-Learn linear regression coefficient signs and weights.

---

## Future Scope & Extensions

This project has been architected to support future scaling and extensions:
- **Advanced Estimator Integration**: Placeholders inside the training pipelines can easily swap the `LinearRegression` estimator for advanced time-series algorithms (such as Random Forest Regressor, XGBoost, or deep learning models like LSTM) by updating the model instantiation inside `train.py`.
- **Multi-Location Forecasting**: Extending the dashboard configuration to support multiple parallel coordinates or batch region evaluations.
- **Exogenous Features**: Incorporating additional meteorological fields (like relative humidity, wind speed, solar radiation, or pressure) from the Open-Meteo API to build a multi-variable regression model.

---

## License

This project is licensed under the MIT License.
