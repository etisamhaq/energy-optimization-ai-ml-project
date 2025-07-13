#!/usr/bin/env python
"""
Prediction script for Energy Optimization ML models.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import (
    XGBoostModel, LightGBMModel, RandomForestModel,
    ProphetModel, ARIMAModel, LSTMModel, EnsembleModel
)
from src.features.feature_engineering import EnergyFeatureEngineer
from src.data.data_loader import EnergyDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyPredictor:
    """Handles energy consumption predictions."""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        
        self.loaded_models = {}
        self.feature_engineer = self._load_feature_engineer()
        self.data_loader = EnergyDataLoader(data_dir)
        
        # Model classes mapping
        self.model_classes = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'random_forest': RandomForestModel,
            'prophet': ProphetModel,
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'ensemble': EnsembleModel
        }
    
    def load_latest_model(self, model_name: str) -> any:
        """Load the latest saved model."""
        if model_name == 'ensemble':
            # Find latest ensemble directory
            ensemble_dirs = [d for d in self.models_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('ensemble_')]
            if not ensemble_dirs:
                raise FileNotFoundError(f"No ensemble models found in {self.models_dir}")
            
            latest_dir = max(ensemble_dirs, key=lambda d: d.stat().st_mtime)
            logger.info(f"Loading ensemble from: {latest_dir}")
            
            model = EnsembleModel()
            model.load_ensemble(str(latest_dir))
            
        else:
            # Find latest model file
            model_files = list(self.models_dir.glob(f"{model_name}_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No {model_name} models found in {self.models_dir}")
            
            latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading model from: {latest_file}")
            
            model_class = self.model_classes[model_name]
            model = model_class()
            model.load_model(str(latest_file))
        
        self.loaded_models[model_name] = model
        return model
    
    def load_feature_info(self) -> Dict:
        """Load feature information."""
        feature_info_path = self.processed_dir / "feature_info.json"
        if not feature_info_path.exists():
            raise FileNotFoundError("feature_info.json not found. Run preprocess.py first.")
        
        with open(feature_info_path, 'r') as f:
            return json.load(f)
    
    def _load_feature_engineer(self) -> EnergyFeatureEngineer:
        """Load the saved feature engineer with fitted label encoders."""
        engineer_path = self.processed_dir / "feature_engineer.pkl"
        if engineer_path.exists():
            import pickle
            with open(engineer_path, 'rb') as f:
                logger.info(f"Loading feature engineer from {engineer_path}")
                return pickle.load(f)
        else:
            logger.warning("No saved feature engineer found, creating new instance")
            return EnergyFeatureEngineer()
    
    def prepare_prediction_data(self, building_id: str, start_date: str, 
                              end_date: str, use_actual_weather: bool = False) -> pd.DataFrame:
        """Prepare data for prediction."""
        logger.info(f"Preparing prediction data for building {building_id} from {start_date} to {end_date}")
        
        # Generate timestamp range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        timestamps = pd.date_range(start=start, end=end, freq='H')
        
        # Create base dataframe
        prediction_df = pd.DataFrame({
            'timestamp': timestamps,
            'building_id': building_id
        })
        
        # Get building metadata
        building_metadata = self.data_loader.load_building_metadata([building_id])
        
        # Get weather data
        if use_actual_weather:
            weather_data = self.data_loader.load_weather_data(start_date, end_date, "New York")
        else:
            # Use synthetic weather for demonstration
            weather_data = self.data_loader._generate_synthetic_weather_data(start_date, end_date)
        
        # Merge data
        prediction_df = pd.merge(prediction_df, weather_data, on='timestamp', how='left')
        prediction_df = pd.merge(prediction_df, building_metadata, on='building_id', how='left')
        
        # Add placeholder for historical consumption (needed for lag features)
        # In production, this would come from actual historical data
        prediction_df['energy_consumption_kwh'] = 0  # Will be updated with predictions
        
        return prediction_df
    
    def predict_single_timestep(self, model: any, features: pd.DataFrame) -> float:
        """Make prediction for a single timestep."""
        prediction = model.predict(features)
        return prediction[0] if isinstance(prediction, np.ndarray) else prediction
    
    def predict_range(self, building_id: str, start_date: str, end_date: str,
                     model_name: str = 'ensemble', use_actual_weather: bool = False) -> pd.DataFrame:
        """Predict energy consumption for a date range."""
        # Load model if not already loaded
        if model_name not in self.loaded_models:
            self.load_latest_model(model_name)
        
        model = self.loaded_models[model_name]
        
        # Prepare prediction data
        base_df = self.prepare_prediction_data(
            building_id, start_date, end_date, use_actual_weather
        )
        
        # Apply feature engineering
        prediction_df = self.feature_engineer.create_all_features(base_df)
        
        # Get feature columns
        feature_info = self.load_feature_info()
        feature_cols = feature_info['all_features']
        
        # For time series models, we need to predict sequentially
        predictions = []
        
        if model_name in ['prophet', 'arima', 'lstm']:
            # These models need special handling
            # For simplicity, we'll use the model's predict method directly
            available_features = [col for col in feature_cols if col in prediction_df.columns]
            X = prediction_df[available_features]
            
            try:
                pred_values = model.predict(X)
                predictions = pred_values
            except Exception as e:
                logger.warning(f"Error with {model_name} prediction: {e}")
                # Fallback to simple prediction
                predictions = np.full(len(prediction_df), 150.0)  # Default value
        else:
            # For ML models, we can predict all at once
            # Filter to available features
            available_features = [col for col in feature_cols if col in prediction_df.columns]
            X = prediction_df[available_features]
            
            # Fill missing features with 0
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Ensure correct feature order
            X = X[feature_cols]
            
            predictions = model.predict(X)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'timestamp': prediction_df['timestamp'],
            'building_id': building_id,
            'predicted_consumption_kwh': predictions,
            'model': model_name
        })
        
        # Add additional useful information
        results_df['hour'] = results_df['timestamp'].dt.hour
        results_df['day_of_week'] = results_df['timestamp'].dt.day_name()
        results_df['date'] = results_df['timestamp'].dt.date
        
        # Calculate daily totals
        daily_totals = results_df.groupby('date')['predicted_consumption_kwh'].sum().reset_index()
        daily_totals.columns = ['date', 'daily_total_kwh']
        
        return results_df, daily_totals
    
    def predict_multiple_buildings(self, building_ids: List[str], start_date: str, 
                                 end_date: str, model_name: str = 'ensemble') -> pd.DataFrame:
        """Make predictions for multiple buildings."""
        all_predictions = []
        
        for building_id in building_ids:
            logger.info(f"Predicting for building {building_id}")
            predictions, _ = self.predict_range(building_id, start_date, end_date, model_name)
            all_predictions.append(predictions)
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def compare_models(self, building_id: str, start_date: str, end_date: str,
                      models: List[str] = None) -> pd.DataFrame:
        """Compare predictions from multiple models."""
        if models is None:
            models = ['xgboost', 'lightgbm', 'random_forest', 'ensemble']
        
        comparison_results = []
        
        for model_name in models:
            try:
                predictions, _ = self.predict_range(
                    building_id, start_date, end_date, model_name
                )
                comparison_results.append(predictions)
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
        
        if not comparison_results:
            raise ValueError("No models produced predictions")
        
        # Combine results
        comparison_df = comparison_results[0][['timestamp', 'building_id']].copy()
        
        for i, model_name in enumerate(models):
            if i < len(comparison_results):
                comparison_df[f'{model_name}_prediction'] = comparison_results[i]['predicted_consumption_kwh']
        
        # Calculate statistics
        prediction_cols = [col for col in comparison_df.columns if '_prediction' in col]
        comparison_df['mean_prediction'] = comparison_df[prediction_cols].mean(axis=1)
        comparison_df['std_prediction'] = comparison_df[prediction_cols].std(axis=1)
        comparison_df['min_prediction'] = comparison_df[prediction_cols].min(axis=1)
        comparison_df['max_prediction'] = comparison_df[prediction_cols].max(axis=1)
        
        return comparison_df
    
    def save_predictions(self, predictions: pd.DataFrame, filename: str = None):
        """Save predictions to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"predictions_{timestamp}.csv"
        
        output_path = self.data_dir / "predictions" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        return output_path


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description='Make energy consumption predictions')
    parser.add_argument(
        '--building-id',
        type=str,
        default='B001',
        help='Building ID to predict for'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for predictions (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for predictions (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--model',
        choices=['xgboost', 'lightgbm', 'random_forest', 'prophet', 
                'arima', 'lstm', 'ensemble'],
        default='ensemble',
        help='Model to use for predictions'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare predictions from multiple models'
    )
    parser.add_argument(
        '--multiple-buildings',
        type=str,
        help='Comma-separated list of building IDs'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory for data'
    )
    
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    if not args.start_date:
        args.start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize predictor
    predictor = EnergyPredictor(args.models_dir, args.data_dir)
    
    # Make predictions
    if args.compare_models:
        # Compare multiple models
        logger.info("Comparing model predictions...")
        comparison_df = predictor.compare_models(
            args.building_id, args.start_date, args.end_date
        )
        
        # Display comparison
        print("\nModel Comparison Results:")
        print(comparison_df.head(24))  # First 24 hours
        
        # Summary statistics
        print("\nSummary Statistics:")
        for col in comparison_df.columns:
            if '_prediction' in col:
                print(f"{col}: Mean={comparison_df[col].mean():.2f}, Std={comparison_df[col].std():.2f}")
        
        if args.save_predictions:
            predictor.save_predictions(comparison_df, f"comparison_{args.building_id}.csv")
    
    elif args.multiple_buildings:
        # Predict for multiple buildings
        building_ids = args.multiple_buildings.split(',')
        logger.info(f"Predicting for buildings: {building_ids}")
        
        all_predictions = predictor.predict_multiple_buildings(
            building_ids, args.start_date, args.end_date, args.model
        )
        
        # Display summary
        print("\nPrediction Summary by Building:")
        summary = all_predictions.groupby('building_id')['predicted_consumption_kwh'].agg(['mean', 'sum', 'std'])
        print(summary)
        
        if args.save_predictions:
            predictor.save_predictions(all_predictions, "multiple_buildings_predictions.csv")
    
    else:
        # Single building prediction
        predictions, daily_totals = predictor.predict_range(
            args.building_id, args.start_date, args.end_date, args.model
        )
        
        # Display results
        print(f"\nPredictions for Building {args.building_id}")
        print(f"Model: {args.model}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print("\nFirst 24 hours:")
        print(predictions.head(24))
        
        print("\nDaily Totals:")
        print(daily_totals)
        
        print(f"\nTotal predicted consumption: {predictions['predicted_consumption_kwh'].sum():.2f} kWh")
        print(f"Average hourly consumption: {predictions['predicted_consumption_kwh'].mean():.2f} kWh")
        print(f"Peak hourly consumption: {predictions['predicted_consumption_kwh'].max():.2f} kWh")
        
        if args.save_predictions:
            predictor.save_predictions(predictions)


if __name__ == "__main__":
    main()