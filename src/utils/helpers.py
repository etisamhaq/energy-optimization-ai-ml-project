"""
Utility functions for Energy Optimization ML project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_date_range(start_date: str, end_date: str, max_days: int = 365) -> Tuple[datetime, datetime]:
    """
    Validate and parse date range.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_days: Maximum allowed days in range
    
    Returns:
        Tuple of parsed start and end dates
    
    Raises:
        ValueError: If dates are invalid or range exceeds max_days
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")
    
    if end < start:
        raise ValueError("End date must be after start date")
    
    if (end - start).days > max_days:
        raise ValueError(f"Date range cannot exceed {max_days} days")
    
    return start, end


def calculate_energy_metrics(consumption_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various energy consumption metrics.
    
    Args:
        consumption_data: DataFrame with 'timestamp' and 'energy_consumption_kwh' columns
    
    Returns:
        Dictionary of calculated metrics
    """
    if 'timestamp' not in consumption_data.columns:
        consumption_data['timestamp'] = pd.to_datetime(consumption_data.index)
    else:
        consumption_data['timestamp'] = pd.to_datetime(consumption_data['timestamp'])
    
    consumption_col = 'energy_consumption_kwh'
    if consumption_col not in consumption_data.columns:
        consumption_col = 'predicted_consumption_kwh'
    
    # Add time features
    consumption_data['hour'] = consumption_data['timestamp'].dt.hour
    consumption_data['day'] = consumption_data['timestamp'].dt.date
    consumption_data['weekday'] = consumption_data['timestamp'].dt.weekday
    
    metrics = {
        'total_consumption_kwh': consumption_data[consumption_col].sum(),
        'average_hourly_kwh': consumption_data[consumption_col].mean(),
        'peak_consumption_kwh': consumption_data[consumption_col].max(),
        'min_consumption_kwh': consumption_data[consumption_col].min(),
        'std_consumption_kwh': consumption_data[consumption_col].std(),
        
        # Peak hours metrics (7 AM - 7 PM)
        'peak_hours_consumption_kwh': consumption_data[
            consumption_data['hour'].between(7, 19)
        ][consumption_col].sum(),
        
        'off_peak_consumption_kwh': consumption_data[
            ~consumption_data['hour'].between(7, 19)
        ][consumption_col].sum(),
        
        # Daily metrics
        'average_daily_kwh': consumption_data.groupby('day')[consumption_col].sum().mean(),
        'max_daily_kwh': consumption_data.groupby('day')[consumption_col].sum().max(),
        
        # Weekend vs weekday
        'weekday_avg_kwh': consumption_data[
            consumption_data['weekday'] < 5
        ][consumption_col].mean(),
        
        'weekend_avg_kwh': consumption_data[
            consumption_data['weekday'] >= 5
        ][consumption_col].mean()
    }
    
    # Calculate load factor
    if metrics['peak_consumption_kwh'] > 0:
        metrics['load_factor'] = metrics['average_hourly_kwh'] / metrics['peak_consumption_kwh']
    else:
        metrics['load_factor'] = 0
    
    return metrics


def create_performance_report(
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None,
    model_name: str = "model"
) -> Dict:
    """
    Create a performance report for model predictions.
    
    Args:
        predictions: DataFrame with predictions
        actuals: Optional DataFrame with actual values
        model_name: Name of the model
    
    Returns:
        Dictionary containing performance metrics and statistics
    """
    report = {
        'model_name': model_name,
        'prediction_period': {
            'start': predictions['timestamp'].min().isoformat(),
            'end': predictions['timestamp'].max().isoformat(),
            'duration_days': (predictions['timestamp'].max() - predictions['timestamp'].min()).days + 1
        },
        'prediction_statistics': calculate_energy_metrics(predictions)
    }
    
    if actuals is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Merge predictions and actuals
        merged = pd.merge(
            predictions[['timestamp', 'predicted_consumption_kwh']],
            actuals[['timestamp', 'energy_consumption_kwh']],
            on='timestamp',
            how='inner'
        )
        
        if not merged.empty:
            y_true = merged['energy_consumption_kwh']
            y_pred = merged['predicted_consumption_kwh']
            
            report['accuracy_metrics'] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            report['residuals'] = {
                'mean': (y_pred - y_true).mean(),
                'std': (y_pred - y_true).std(),
                'max_overestimate': (y_pred - y_true).max(),
                'max_underestimate': (y_pred - y_true).min()
            }
    
    return report


def save_model_artifacts(
    model,
    model_name: str,
    artifacts_dir: str,
    metrics: Optional[Dict] = None,
    feature_importance: Optional[pd.DataFrame] = None
):
    """
    Save model artifacts including model file, metrics, and feature importance.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        artifacts_dir: Directory to save artifacts
        metrics: Optional dictionary of model metrics
        feature_importance: Optional feature importance DataFrame
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = artifacts_path / f"{model_name}_{timestamp}.pkl"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    if metrics:
        metrics_path = artifacts_path / f"{model_name}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance
    if feature_importance is not None:
        importance_path = artifacts_path / f"{model_name}_importance_{timestamp}.csv"
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Create metadata file
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'model_file': model_path.name,
        'metrics_file': metrics_path.name if metrics else None,
        'importance_file': importance_path.name if feature_importance is not None else None
    }
    
    metadata_path = artifacts_path / f"{model_name}_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def plot_consumption_patterns(
    consumption_data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Create visualization of energy consumption patterns.
    
    Args:
        consumption_data: DataFrame with consumption data
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    consumption_data = consumption_data.copy()
    consumption_data['timestamp'] = pd.to_datetime(consumption_data['timestamp'])
    consumption_data['hour'] = consumption_data['timestamp'].dt.hour
    consumption_data['day_of_week'] = consumption_data['timestamp'].dt.dayofweek
    consumption_data['date'] = consumption_data['timestamp'].dt.date
    
    consumption_col = 'energy_consumption_kwh'
    if consumption_col not in consumption_data.columns:
        consumption_col = 'predicted_consumption_kwh'
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Energy Consumption Patterns', fontsize=16)
    
    # Hourly pattern
    hourly_avg = consumption_data.groupby('hour')[consumption_col].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, linewidth=2)
    axes[0, 0].set_title('Average Hourly Consumption')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Consumption (kWh)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Daily pattern
    daily_total = consumption_data.groupby('date')[consumption_col].sum()
    axes[0, 1].bar(range(len(daily_total)), daily_total.values)
    axes[0, 1].set_title('Daily Total Consumption')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Total Consumption (kWh)')
    
    # Day of week pattern
    dow_avg = consumption_data.groupby('day_of_week')[consumption_col].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(dow_avg.index, dow_avg.values)
    axes[1, 0].set_title('Average Consumption by Day of Week')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(days)
    axes[1, 0].set_ylabel('Average Consumption (kWh)')
    
    # Heatmap of hourly consumption by day
    pivot_data = consumption_data.pivot_table(
        values=consumption_col,
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )
    
    im = axes[1, 1].imshow(pivot_data.values, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_title('Consumption Heatmap (Hour vs Day)')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Hour of Day')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(days)
    axes[1, 1].set_yticks(range(0, 24, 4))
    axes[1, 1].set_yticklabels(range(0, 24, 4))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('Consumption (kWh)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()


def format_currency(value: float, currency: str = "$") -> str:
    """Format value as currency string."""
    return f"{currency}{value:,.2f}"


def format_energy(value: float, unit: str = "kWh") -> str:
    """Format energy value with appropriate unit."""
    if value >= 1000000:
        return f"{value/1000000:.2f} MWh"
    elif value >= 1000:
        return f"{value/1000:.2f} MWh"
    else:
        return f"{value:.2f} {unit}"


def calculate_carbon_emissions(
    energy_kwh: float,
    carbon_intensity: float = 0.4  # kg CO2/kWh default
) -> Dict[str, float]:
    """
    Calculate carbon emissions from energy consumption.
    
    Args:
        energy_kwh: Energy consumption in kWh
        carbon_intensity: Carbon intensity factor (kg CO2/kWh)
    
    Returns:
        Dictionary with emissions in different units
    """
    emissions_kg = energy_kwh * carbon_intensity
    
    return {
        'emissions_kg': emissions_kg,
        'emissions_tons': emissions_kg / 1000,
        'emissions_lbs': emissions_kg * 2.20462,
        'equivalent_trees': emissions_kg / 21.77,  # kg CO2 absorbed per tree per year
        'equivalent_miles_driven': emissions_kg / 0.404  # kg CO2 per mile for average car
    }