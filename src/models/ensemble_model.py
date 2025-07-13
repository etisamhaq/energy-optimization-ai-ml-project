"""
Ensemble model combining multiple models for robust predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
import logging

from .base_model import BaseEnergyModel
from .time_series_models import ProphetModel, ARIMAModel
from .ml_models import XGBoostModel, LightGBMModel, RandomForestModel
from .deep_learning_models import LSTMModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseEnergyModel):
    """Ensemble model combining multiple models with weighted averaging."""
    
    def __init__(self, models: Optional[List[str]] = None, model_params: Optional[Dict] = None):
        # Handle case where models are passed in model_params
        if model_params and isinstance(model_params, dict) and 'models' in model_params:
            models = model_params['models']
            model_params = {k: v for k, v in model_params.items() if k != 'models'}
        
        super().__init__('ensemble', model_params or {})
        
        # Default models to include
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest']
        
        self.model_names = models
        self.models = {}
        self.weights = {}
        self.validation_scores = {}
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual models based on model names."""
        model_mapping = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'random_forest': RandomForestModel,
            'prophet': ProphetModel,
            'arima': ARIMAModel,
            'lstm': LSTMModel
        }
        
        for model_name in self.model_names:
            if model_name in model_mapping:
                # Get model-specific params if provided
                model_params = self.model_params.get(model_name, {})
                self.models[model_name] = model_mapping[model_name](model_params)
                logger.info(f"Initialized {model_name} model")
            else:
                logger.warning(f"Unknown model: {model_name}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data=None, use_cv=True, cv_folds=3):
        """Fit all models in the ensemble."""
        logger.info(f"Training ensemble with models: {self.model_names}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                if model_name in ['xgboost', 'lightgbm'] and validation_data is not None:
                    # These models support validation set for early stopping
                    model.fit(X, y, eval_set=[validation_data], early_stopping_rounds=50)
                elif model_name == 'lstm' and validation_data is not None:
                    # LSTM needs special handling for validation data
                    model.fit(X, y, validation_data=validation_data)
                else:
                    # Other models
                    model.fit(X, y)
                
                # Evaluate model performance
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_score = model.evaluate(X_val, y_val)
                    self.validation_scores[model_name] = val_score['rmse']
                    logger.info(f"{model_name} validation RMSE: {val_score['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                # Remove failed model from ensemble
                del self.models[model_name]
        
        # Calculate weights based on validation scores
        if self.validation_scores:
            self._calculate_weights()
        else:
            # Equal weights if no validation scores
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        self.is_fitted = True
        logger.info(f"Ensemble training complete. Model weights: {self.weights}")
    
    def _calculate_weights(self):
        """Calculate model weights based on validation performance."""
        # Use inverse of RMSE as weight (better models get higher weights)
        inverse_scores = {name: 1.0 / score for name, score in self.validation_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        # Normalize weights
        self.weights = {name: score / total_inverse for name, score in inverse_scores.items()}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        # Weighted average of predictions
        if not predictions:
            raise ValueError("No models produced predictions")
        
        # Stack predictions and weights
        pred_array = np.column_stack([predictions[name] for name in predictions.keys()])
        weights_array = np.array([self.weights.get(name, 0) for name in predictions.keys()])
        
        # Ensure weights sum to 1
        weights_array = weights_array / weights_array.sum()
        
        # Calculate weighted average
        ensemble_predictions = np.average(pred_array, axis=1, weights=weights_array)
        
        return ensemble_predictions
    
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual predictions from each model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                predictions[model_name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {str(e)}")
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from models that support it."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                # Get importance for this model
                importance = model.get_feature_importance()
                if importance is not None:
                    for _, row in importance.iterrows():
                        feature = row['feature']
                        if feature not in importance_dict:
                            importance_dict[feature] = {}
                        importance_dict[feature][model_name] = row['importance']
        
        # Create aggregated importance dataframe
        if not importance_dict:
            return None
        
        # Calculate weighted average importance
        aggregated_importance = []
        for feature, model_scores in importance_dict.items():
            # Weight by model performance
            weighted_sum = sum(
                score * self.weights.get(model_name, 0) 
                for model_name, score in model_scores.items()
            )
            aggregated_importance.append({
                'feature': feature,
                'importance': weighted_sum,
                'models_with_importance': list(model_scores.keys())
            })
        
        importance_df = pd.DataFrame(aggregated_importance)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_ensemble(self, directory: str):
        """Save all models in the ensemble."""
        from pathlib import Path
        
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble metadata
        ensemble_data = {
            'model_names': self.model_names,
            'weights': self.weights,
            'validation_scores': self.validation_scores,
            'feature_names': self.feature_names
        }
        
        import joblib
        joblib.dump(ensemble_data, save_dir / 'ensemble_metadata.pkl')
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = save_dir / f'{model_name}_model.pkl'
            model.save_model(str(model_path))
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load_ensemble(self, directory: str):
        """Load ensemble from saved models."""
        from pathlib import Path
        import joblib
        
        load_dir = Path(directory)
        
        # Load ensemble metadata
        ensemble_data = joblib.load(load_dir / 'ensemble_metadata.pkl')
        self.model_names = ensemble_data['model_names']
        self.weights = ensemble_data['weights']
        self.validation_scores = ensemble_data['validation_scores']
        self.feature_names = ensemble_data['feature_names']
        
        # Re-initialize models
        self._initialize_models()
        
        # Load individual models
        for model_name in self.model_names:
            model_path = load_dir / f'{model_name}_model.pkl'
            if model_path.exists():
                self.models[model_name].load_model(str(model_path))
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        self.is_fitted = True
        logger.info(f"Ensemble loaded from {directory}")