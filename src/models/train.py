#!/usr/bin/env python
"""
Training script for Energy Optimization ML models.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
import optuna
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import (
    XGBoostModel, LightGBMModel, RandomForestModel,
    ProphetModel, ARIMAModel, LSTMModel, EnsembleModel
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and hyperparameter optimization."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Available models
        self.model_classes = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'random_forest': RandomForestModel,
            'prophet': ProphetModel,
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'ensemble': EnsembleModel
        }
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Load the latest processed datasets."""
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            # Find latest file for this split
            files = list(self.processed_dir.glob(f"processed_{split}_*.csv"))
            if not files:
                raise FileNotFoundError(f"No processed {split} data found. Run preprocess.py first.")
            
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading {split} data from: {latest_file}")
            datasets[split] = pd.read_csv(latest_file)
            
            # Convert timestamp to datetime
            datasets[split]['timestamp'] = pd.to_datetime(datasets[split]['timestamp'])
        
        return datasets
    
    def load_feature_info(self) -> Dict:
        """Load feature information."""
        feature_info_path = self.processed_dir / "feature_info.json"
        if not feature_info_path.exists():
            raise FileNotFoundError("feature_info.json not found. Run preprocess.py first.")
        
        with open(feature_info_path, 'r') as f:
            return json.load(f)
    
    def prepare_data_for_training(self, datasets: Dict[str, pd.DataFrame], 
                                 feature_info: Dict) -> Dict:
        """Prepare data for model training."""
        prepared_data = {}
        
        # Get feature columns
        feature_cols = feature_info['all_features']
        target_col = feature_info['target']
        
        for split in ['train', 'validation', 'test']:
            df = datasets[split]
            
            # Separate features and target
            X = df[feature_cols]
            y = df[target_col]
            
            prepared_data[f'X_{split}'] = X
            prepared_data[f'y_{split}'] = y
        
        return prepared_data
    
    def train_single_model(self, model_name: str, model_params: Optional[Dict] = None,
                          prepared_data: Optional[Dict] = None) -> Dict:
        """Train a single model."""
        logger.info(f"Training {model_name} model...")
        
        # Load data if not provided
        if prepared_data is None:
            datasets = self.load_processed_data()
            feature_info = self.load_feature_info()
            prepared_data = self.prepare_data_for_training(datasets, feature_info)
        
        # Initialize model
        model_class = self.model_classes[model_name]
        model = model_class(model_params)
        
        # Prepare validation data for models that support it
        validation_data = None
        if model_name in ['xgboost', 'lightgbm', 'lstm']:
            validation_data = (prepared_data['X_validation'], prepared_data['y_validation'])
        
        # Train model
        if model_name == 'ensemble':
            # Ensemble needs special handling
            model.fit(
                prepared_data['X_train'], 
                prepared_data['y_train'],
                validation_data=validation_data
            )
        elif model_name in ['xgboost', 'lightgbm']:
            # These models use eval_set parameter
            model.fit(
                prepared_data['X_train'], 
                prepared_data['y_train'],
                eval_set=[validation_data]
            )
        elif model_name == 'lstm':
            # LSTM uses validation_data parameter
            model.fit(
                prepared_data['X_train'], 
                prepared_data['y_train'],
                validation_data=validation_data
            )
        else:
            model.fit(prepared_data['X_train'], prepared_data['y_train'])
        
        # Evaluate on all splits
        results = {
            'model_name': model_name,
            'model': model,
            'metrics': {}
        }
        
        for split in ['train', 'validation', 'test']:
            metrics = model.evaluate(prepared_data[f'X_{split}'], prepared_data[f'y_{split}'])
            results['metrics'][split] = metrics
            logger.info(f"{model_name} {split} metrics: {metrics}")
        
        return results
    
    def optimize_hyperparameters(self, model_name: str, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        # Load data
        datasets = self.load_processed_data()
        feature_info = self.load_feature_info()
        prepared_data = self.prepare_data_for_training(datasets, feature_info)
        
        def objective(trial):
            # Define hyperparameter search space
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
                }
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
            
            # Train model with suggested parameters
            results = self.train_single_model(model_name, params, prepared_data)
            
            # Return validation RMSE as objective to minimize
            return results['metrics']['validation']['rmse']
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation RMSE: {study.best_value:.4f}")
        
        # Train final model with best parameters
        best_results = self.train_single_model(model_name, study.best_params, prepared_data)
        best_results['optimization_study'] = study
        
        return best_results
    
    def train_ensemble(self, models: List[str] = None) -> Dict:
        """Train ensemble model."""
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest']
        
        logger.info(f"Training ensemble with models: {models}")
        
        # Load data
        datasets = self.load_processed_data()
        feature_info = self.load_feature_info()
        prepared_data = self.prepare_data_for_training(datasets, feature_info)
        
        # Initialize ensemble model with the list of models
        from src.models import EnsembleModel
        model = EnsembleModel(models=models)
        
        # Prepare validation data
        validation_data = (prepared_data['X_validation'], prepared_data['y_validation'])
        
        # Train ensemble
        model.fit(
            prepared_data['X_train'], 
            prepared_data['y_train'],
            validation_data=validation_data
        )
        
        # Evaluate on all splits
        results = {
            'model_name': 'ensemble',
            'model': model,
            'metrics': {}
        }
        
        for split in ['train', 'validation', 'test']:
            metrics = model.evaluate(prepared_data[f'X_{split}'], prepared_data[f'y_{split}'])
            results['metrics'][split] = metrics
            logger.info(f"ensemble {split} metrics: {metrics}")
        
        return results
    
    def save_model(self, results: Dict, experiment_name: str = None):
        """Save trained model and log to MLflow."""
        model = results['model']
        model_name = results['model_name']
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Save model
        if model_name == 'ensemble':
            model.save_ensemble(str(self.models_dir / f"ensemble_{timestamp}"))
        else:
            model.save_model(str(model_path))
        
        # Log to MLflow
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model.model_params)
            
            # Log metrics
            for split, metrics in results['metrics'].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split}_{metric_name}", value)
            
            # Log model
            if model_name != 'ensemble':
                mlflow.sklearn.log_model(model.model, "model")
            
            # Log feature importance if available
            importance = model.get_feature_importance()
            if importance is not None:
                # Save importance as artifact
                importance_path = self.models_dir / f"{model_name}_importance_{timestamp}.csv"
                importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
        
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description='Train Energy Optimization ML models')
    parser.add_argument(
        '--model',
        choices=['xgboost', 'lightgbm', 'random_forest', 'prophet', 
                'arima', 'lstm', 'ensemble', 'all'],
        default='ensemble',
        help='Which model to train'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials for hyperparameter optimization'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='energy_optimization',
        help='MLflow experiment name'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.data_dir, args.models_dir)
    
    # Train models
    if args.model == 'all':
        # Train all models
        all_results = {}
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'prophet']:
            if args.optimize and model_name in ['xgboost', 'lightgbm', 'random_forest']:
                results = trainer.optimize_hyperparameters(model_name, args.n_trials)
            else:
                results = trainer.train_single_model(model_name)
            
            all_results[model_name] = results
            trainer.save_model(results, args.experiment_name)
        
        # Train ensemble
        ensemble_results = trainer.train_ensemble()
        all_results['ensemble'] = ensemble_results
        trainer.save_model(ensemble_results, args.experiment_name)
        
    elif args.model == 'ensemble':
        results = trainer.train_ensemble()
        trainer.save_model(results, args.experiment_name)
        
    else:
        # Train single model
        if args.optimize and args.model in ['xgboost', 'lightgbm', 'random_forest']:
            results = trainer.optimize_hyperparameters(args.model, args.n_trials)
        else:
            results = trainer.train_single_model(args.model)
        
        trainer.save_model(results, args.experiment_name)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()