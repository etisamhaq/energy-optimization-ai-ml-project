"""
Machine learning models for energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

from .base_model import BaseEnergyModel


class XGBoostModel(BaseEnergyModel):
    """XGBoost model for energy prediction."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': 42
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('xgboost', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds=50, verbose=False):
        """Fit XGBoost model."""
        self.feature_names = X.columns.tolist()
        
        # Early stopping needs a validation set, so only enable it when one is given
        params = dict(self.model_params)
        if eval_set is not None:
            params.setdefault('early_stopping_rounds', early_stopping_rounds)
        self.model = xgb.XGBRegressor(**params)
        
        if eval_set is not None:
            self.model.fit(
                X, y, 
                eval_set=eval_set,
                verbose=verbose
            )
        else:
            self.model.fit(X, y, verbose=verbose)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_features(X)
        X = self.prepare_features(X)
        
        return self.model.predict(X)


class LightGBMModel(BaseEnergyModel):
    """LightGBM model for energy prediction."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'n_estimators': 300,
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'regression',
            'metric': 'rmse',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('lightgbm', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds=50, verbose=False):
        """Fit LightGBM model."""
        self.feature_names = X.columns.tolist()
        
        # Initialize model
        self.model = lgb.LGBMRegressor(**self.model_params)
        
        # Fit model
        callbacks = None if verbose else [lgb.log_evaluation(0)]
        
        if eval_set is not None:
            self.model.fit(
                X, y, 
                eval_set=eval_set,
                eval_metric='rmse',
                callbacks=callbacks
            )
        else:
            self.model.fit(X, y, callbacks=callbacks)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_features(X)
        X = self.prepare_features(X)
        
        return self.model.predict(X)


class RandomForestModel(BaseEnergyModel):
    """Random Forest model for energy prediction."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('random_forest', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Random Forest model."""
        self.feature_names = X.columns.tolist()
        
        # Initialize model
        self.model = RandomForestRegressor(**self.model_params)
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_features(X)
        X = self.prepare_features(X)
        
        return self.model.predict(X)


class LinearRegressionModel(BaseEnergyModel):
    """Linear Regression model for baseline predictions."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'fit_intercept': True
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('linear_regression', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Linear Regression model."""
        self.feature_names = X.columns.tolist()
        
        # Initialize model
        self.model = LinearRegression(**self.model_params)
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance (coefficients)
        self.feature_importance = np.abs(self.model.coef_)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Linear Regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_features(X)
        X = self.prepare_features(X)
        
        return self.model.predict(X)


class RidgeModel(BaseEnergyModel):
    """Ridge Regression model with L2 regularization."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'random_state': 42
        }
        if model_params:
            default_params.update(model_params)
        
        super().__init__('ridge', default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Ridge Regression model."""
        self.feature_names = X.columns.tolist()
        
        # Initialize model
        self.model = Ridge(**self.model_params)
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance (coefficients)
        self.feature_importance = np.abs(self.model.coef_)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Ridge Regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_features(X)
        X = self.prepare_features(X)
        
        return self.model.predict(X)