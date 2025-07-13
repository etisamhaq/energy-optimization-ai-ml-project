"""
Tests for machine learning models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

import sys
sys.path.append('..')
from src.models import (
    XGBoostModel, LightGBMModel, RandomForestModel,
    ProphetModel, LinearRegressionModel, EnsembleModel
)


class TestModels:
    """Test cases for ML models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        n_samples = 1000
        n_features = 10
        
        # Create features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some relationship to features
        y = pd.Series(
            X['feature_0'] * 2 + X['feature_1'] * 1.5 + 
            np.random.randn(n_samples) * 0.1 + 100
        )
        
        return X, y
    
    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data for Prophet."""
        n_days = 30
        dates = pd.date_range('2024-01-01', periods=n_days * 24, freq='H')
        
        X = pd.DataFrame({
            'timestamp': dates,
            'temperature_c': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24),
            'is_weekend': (dates.weekday >= 5).astype(int)
        })
        
        # Create energy consumption with pattern
        y = pd.Series(
            100 + 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) +
            np.random.randn(len(dates)) * 5
        )
        
        return X, y
    
    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Initialize and train model
        model = XGBoostModel({'n_estimators': 50, 'max_depth': 3})
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.feature_names == X_train.columns.tolist()
        assert model.feature_importance is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0  # Should have some predictive power
    
    def test_lightgbm_model(self, sample_data):
        """Test LightGBM model."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Initialize and train model
        model = LightGBMModel({'n_estimators': 50, 'num_leaves': 10})
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.feature_importance is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Check feature importance
        importance_df = model.get_feature_importance()
        assert importance_df is not None
        assert len(importance_df) == len(X_train.columns)
    
    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Initialize and train model
        model = RandomForestModel({'n_estimators': 50, 'max_depth': 5})
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        assert model.feature_importance is not None
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_linear_regression_model(self, sample_data):
        """Test Linear Regression model."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Initialize and train model
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_prophet_model(self, time_series_data):
        """Test Prophet model."""
        X, y = time_series_data
        
        # Initialize and train model
        model = ProphetModel({
            'changepoint_prior_scale': 0.05,
            'daily_seasonality': True
        })
        model.fit(X, y)
        
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_ensemble_model(self, sample_data):
        """Test Ensemble model."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Initialize ensemble with fast models
        model = EnsembleModel(
            models=['xgboost', 'lightgbm', 'random_forest'],
            model_params={
                'xgboost': {'n_estimators': 20, 'max_depth': 3},
                'lightgbm': {'n_estimators': 20, 'num_leaves': 10},
                'random_forest': {'n_estimators': 20, 'max_depth': 3}
            }
        )
        
        # Train ensemble
        model.fit(X_train, y_train, validation_data=(X_test, y_test))
        
        assert model.is_fitted
        assert len(model.models) == 3
        assert len(model.weights) == 3
        assert sum(model.weights.values()) == pytest.approx(1.0)
        
        # Make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Get individual model predictions
        individual_preds = model.get_model_predictions(X_test)
        assert len(individual_preds) == 3
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert importance is not None
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y = sample_data
        
        # Train model
        model = XGBoostModel({'n_estimators': 20})
        model.fit(X, y)
        
        # Save model
        model_path = str(tmp_path / "test_model.pkl")
        model.save_model(model_path)
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = XGBoostModel()
        loaded_model.load_model(model_path)
        
        assert loaded_model.is_fitted
        assert loaded_model.feature_names == model.feature_names
        
        # Check predictions are the same
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)
        assert np.allclose(pred_original, pred_loaded)
    
    def test_ensemble_save_load(self, sample_data, tmp_path):
        """Test ensemble model saving and loading."""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Train ensemble
        model = EnsembleModel(
            models=['xgboost', 'lightgbm'],
            model_params={
                'xgboost': {'n_estimators': 20},
                'lightgbm': {'n_estimators': 20}
            }
        )
        model.fit(X_train, y_train, validation_data=(X_test, y_test))
        
        # Save ensemble
        ensemble_dir = str(tmp_path / "test_ensemble")
        model.save_ensemble(ensemble_dir)
        assert os.path.exists(ensemble_dir)
        
        # Load ensemble
        loaded_model = EnsembleModel()
        loaded_model.load_ensemble(ensemble_dir)
        
        assert loaded_model.is_fitted
        assert loaded_model.weights == model.weights
        
        # Check predictions are similar
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        assert np.allclose(pred_original, pred_loaded, rtol=1e-5)