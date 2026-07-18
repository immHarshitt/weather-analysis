import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any

from src.logger import logger
from src.feature_engineering import get_feature_columns

# Set theme and style settings globally
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 15,
    'figure.dpi': 150
})

def plot_historical_temperature_trend(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 1: Historical Temperature Trend."""
    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df['TempMean'], color='#1e3a8a', alpha=0.8, label='Daily Mean Temp', linewidth=1)
    if 'TempMax' in df.columns:
        plt.plot(df.index, df['TempMax'], color='#f87171', alpha=0.4, label='Daily Max Temp', linewidth=1)
    if 'TempMin' in df.columns:
        plt.plot(df.index, df['TempMin'], color='#60a5fa', alpha=0.4, label='Daily Min Temp', linewidth=1)
        
    plt.title('Historical Daily Temperature Trend', fontweight='bold', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_moving_average_trend(df: pd.DataFrame, save_path: Path, days_to_zoom: int = 365) -> None:
    """Plot 2: Moving Average Trend (Zoomed in on last year of data)."""
    plt.figure(figsize=(11, 5))
    zoom_df = df.tail(days_to_zoom)
    plt.plot(zoom_df.index, zoom_df['TempMean'], color='#94a3b8', alpha=0.5, label='Daily Temp', linewidth=1)
    
    if 'temp_ma_3' in zoom_df.columns:
        plt.plot(zoom_df.index, zoom_df['temp_ma_3'], color='#3b82f6', linewidth=1.5, label='3-Day Moving Avg')
    if 'temp_ma_7' in zoom_df.columns:
        plt.plot(zoom_df.index, zoom_df['temp_ma_7'], color='#ef4444', linewidth=2, label='7-Day Moving Avg')
        
    plt.title(f'Daily Temperature and Moving Averages (Last {days_to_zoom} Days)', fontweight='bold', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, save_path: Path, last_n: int = 100) -> None:
    """Plot 3: Actual vs Predicted (Last 100 days of the test set)."""
    plt.figure(figsize=(11, 5))
    
    dates = y_true.index[-last_n:]
    true_vals = y_true.values[-last_n:]
    pred_vals = y_pred[-last_n:]
    
    plt.plot(dates, true_vals, color='#10b981', marker='o', label='Actual Temperature', alpha=0.8, linewidth=1.5)
    plt.plot(dates, pred_vals, color='#ef4444', linestyle='--', marker='x', label='Predicted Temperature', alpha=0.8, linewidth=1.5)
    
    plt.title(f'Actual vs. Predicted Temperature (Last {last_n} Days of Test Set)', fontweight='bold', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_forecast_14_days(history_df: pd.DataFrame, forecast_df: pd.DataFrame, save_path: Path) -> None:
    """Plot 4: 14 Days Forecast vs recent history with min/max bounds shading."""
    plt.figure(figsize=(11, 5))
    
    # Take last 30 days of history
    history_subset = history_df.tail(30)
    
    plt.plot(history_subset.index, history_subset['TempMean'], color='#1e3a8a', marker='o', label='Recent Historical Actual')
    plt.plot(forecast_df.index, forecast_df['TempMean'], color='#f59e0b', marker='s', linestyle='-', label=f'{len(forecast_df)}-Day Forecast Mean')
    
    if 'TempMax' in forecast_df.columns and 'TempMin' in forecast_df.columns:
        plt.fill_between(
            forecast_df.index,
            forecast_df['TempMin'],
            forecast_df['TempMax'],
            color='#f59e0b',
            alpha=0.15,
            label='Forecast Range'
        )
        
    plt.title(f'{len(forecast_df)}-Day Weather Temperature Forecast Horizon', fontweight='bold', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_residual(y_pred: np.ndarray, residuals: np.ndarray, save_path: Path) -> None:
    """Plot 5: Residual Plot."""
    plt.figure(figsize=(9, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color='#84cc16', edgecolors='#3f6212', s=35)
    plt.axhline(0, color='#b91c1c', linestyle='--', linewidth=2)
    plt.title('Residuals vs. Predicted Temperature', fontweight='bold', pad=15)
    plt.xlabel('Predicted Temperature (°C)')
    plt.ylabel('Residuals (Actual - Predicted) (°C)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_residual_distribution(residuals: np.ndarray, save_path: Path) -> None:
    """Plot 6: Residual Distribution."""
    from scipy.stats import norm
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, color='#06b6d4', bins=30, stat='density', alpha=0.6)
    
    # Overlay normal distribution fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, 'k', linewidth=2, label='Normal Dist Fit')
    
    plt.title('Error (Residuals) Distribution Histogram', fontweight='bold', pad=15)
    plt.xlabel('Prediction Error (°C)')
    plt.ylabel('Density')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_monthly_average_temperature(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 7: Monthly Average Temperature."""
    monthly_avg = df.groupby('month')['TempMean'].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', color='#8b5cf6', linewidth=2.5, markersize=8)
    plt.title('Average Temperature by Month', fontweight='bold', pad=15)
    plt.xlabel('Month')
    plt.ylabel('Mean Temperature (°C)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_monthly_temperature_boxplot(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 8: Monthly Temperature Box Plot."""
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df.reset_index(), x='month', y='TempMean', hue='month', palette='flare', legend=False)
    plt.title('Temperature Distribution Box Plot by Month', fontweight='bold', pad=15)
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 9: Correlation Heatmap."""
    features_and_target = ['TempMean'] + get_feature_columns()
    cols_to_use = [col for col in features_and_target if col in df.columns]
    
    corr_matrix = df[cols_to_use].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix Heatmap', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def plot_feature_importance(model_pipeline: Any, save_path: Path) -> None:
    """Plot 10: Feature Importance (Linear Regression coefficients)."""
    feature_names = get_feature_columns()
    
    try:
        coefs = model_pipeline.named_steps['regressor'].coef_
    except AttributeError:
        coefs = model_pipeline.coef_
        
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Absolute Magnitude': np.abs(coefs)
    }).sort_values(by='Absolute Magnitude', ascending=True)
    
    plt.figure(figsize=(9, 5))
    colors = ['#ef4444' if c < 0 else '#3b82f6' for c in importance_df['Coefficient']]
    
    bars = plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, edgecolor='none', height=0.6)
    plt.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.gca().text(
            width + (0.01 if width >= 0 else -0.1), 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', 
            va='center', 
            ha='left' if width >= 0 else 'right',
            fontsize=8, 
            fontweight='bold',
            color='#1e293b'
        )
        
    plt.title('Feature Importance (Linear Regression Coefficients)', fontweight='bold', pad=15)
    plt.xlabel('Coefficient Value (Red=Negative, Blue=Positive Impact)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {save_path}")

def generate_all_plots(
    df_clean: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    forecast_df: pd.DataFrame,
    model_pipeline: Any,
    plots_dir: Path
) -> None:
    """Generates and saves all 10 required plots to results/plots/."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    residuals = y_test.values - y_pred
    
    plot_historical_temperature_trend(df_clean, plots_dir / "historical_temperature_trend.png")
    plot_moving_average_trend(df_clean, plots_dir / "moving_average_trend.png")
    plot_actual_vs_predicted(y_test, y_pred, plots_dir / "actual_vs_predicted.png")
    plot_forecast_14_days(df_clean, forecast_df, plots_dir / "forecast_14_days.png")
    plot_residual(y_pred, residuals, plots_dir / "residual_plot.png")
    plot_residual_distribution(residuals, plots_dir / "residual_distribution.png")
    plot_monthly_average_temperature(df_clean, plots_dir / "monthly_average_temperature.png")
    plot_monthly_temperature_boxplot(df_clean, plots_dir / "monthly_temperature_boxplot.png")
    plot_correlation_heatmap(df_clean, plots_dir / "correlation_heatmap.png")
    plot_feature_importance(model_pipeline, plots_dir / "feature_importance.png")
    
    logger.info("Successfully generated and saved all 10 plots.")
