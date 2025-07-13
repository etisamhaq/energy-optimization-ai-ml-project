"""
Time series models for energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseEnergyModel


class ProphetModel(BaseEnergyModel):
    """Facebook Prophet model for time series forecasting."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10,
            'holidays_prior_scale': 10,
            'seasonality_mode': 'multiplicative',
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'yearly_seasonality': True
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('prophet', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Prophet model."""
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': X.index if isinstance(X.index, pd.DatetimeIndex) else X['timestamp'],
            'y': y.values
        })
        
        # Add regressors if available
        regressor_cols = []
        for col in ['temperature_c', 'humidity_percent', 'is_weekend', 'is_business_hour']:
            if col in X.columns:
                prophet_df[col] = X[col].values
                regressor_cols.append(col)
        
        # Initialize Prophet
        self.model = Prophet(**self.model_params)
        
        # Add regressors
        for col in regressor_cols:
            self.model.add_regressor(col)
        
        # Fit model
        self.model.fit(prophet_df)
        
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare future dataframe
        future_df = pd.DataFrame({
            'ds': X.index if isinstance(X.index, pd.DatetimeIndex) else X['timestamp']
        })
        
        # Add regressors
        for col in ['temperature_c', 'humidity_percent', 'is_weekend', 'is_business_hour']:
            if col in X.columns:
                future_df[col] = X[col].values
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values


class ARIMAModel(BaseEnergyModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'order': (2, 1, 2),  # (p, d, q)
            'seasonal_order': (1, 1, 1, 24),  # (P, D, Q, s) - hourly seasonality
            'trend': 'c'
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('arima', default_params)
        self.fitted_values = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit ARIMA model."""
        # For ARIMA, we primarily use the time series data
        # External regressors can be added through SARIMAX
        
        if 'seasonal_order' in self.model_params:
            # Use SARIMAX for seasonal patterns
            exog_vars = []
            exog_data = None
            
            # Add external regressors if available
            for col in ['temperature_c', 'humidity_percent', 'is_weekend']:
                if col in X.columns:
                    exog_vars.append(col)
            
            if exog_vars:
                exog_data = X[exog_vars]
            
            self.model = SARIMAX(
                y,
                exog=exog_data,
                order=self.model_params['order'],
                seasonal_order=self.model_params['seasonal_order'],
                trend=self.model_params.get('trend', 'c')
            )
        else:
            # Use basic ARIMA
            self.model = ARIMA(
                y,
                order=self.model_params['order']
            )
        
        # Fit the model
        self.fitted_model = self.model.fit(disp=False)
        self.fitted_values = self.fitted_model.fittedvalues
        
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with ARIMA."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        
        # Get exogenous variables if used in training
        exog_data = None
        if hasattr(self.model, 'exog') and self.model.exog is not None:
            exog_vars = [col for col in ['temperature_c', 'humidity_percent', 'is_weekend'] 
                        if col in X.columns]
            if exog_vars:
                exog_data = X[exog_vars]
        
        # Make forecast
        forecast = self.fitted_model.forecast(steps=n_periods, exog=exog_data)
        
        return np.array(forecast)


class ExponentialSmoothingModel(BaseEnergyModel):
    """Exponential Smoothing model for time series forecasting."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'trend': 'add',
            'seasonal': 'add',
            'seasonal_periods': 24  # Hourly seasonality
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('exponential_smoothing', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Exponential Smoothing model."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Exponential smoothing doesn't use external features
        # It's purely based on the time series
        self.model = ExponentialSmoothing(
            y,
            trend=self.model_params.get('trend'),
            seasonal=self.model_params.get('seasonal'),
            seasonal_periods=self.model_params.get('seasonal_periods', 24)
        )
        
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        self.feature_names = ['time_series_only']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Exponential Smoothing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_periods = len(X)
        forecast = self.fitted_model.forecast(steps=n_periods)
        
        return np.array(forecast)