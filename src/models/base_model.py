"""
Base model class for Energy Optimization ML models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseEnergyModel(ABC):
    """Abstract base class for energy prediction models."""
    
    def __init__(self, model_name: str, model_params: Optional[Dict] = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.feature_importance = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = self.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_params = model_data['model_params']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance')
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if self.feature_importance is None:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
    
    def validate_features(self, X: pd.DataFrame):
        """Validate that input features match trained model."""
        if self.feature_names is None:
            return
        
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
    
    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        if self.feature_names is not None:
            # Ensure feature order matches training
            X = X[self.feature_names]
        return X