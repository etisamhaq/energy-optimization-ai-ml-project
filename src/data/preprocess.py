#!/usr/bin/env python
"""
Data preprocessing script for Energy Optimization ML project.
Handles data cleaning, validation, and feature engineering.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.feature_engineering import EnergyFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and preparation for modeling."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.engineer = EnergyFeatureEngineer()
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_latest_integrated_data(self) -> pd.DataFrame:
        """Load the most recent integrated dataset."""
        # Find latest integrated data file
        integrated_files = list(self.processed_dir.glob("integrated_data_*.csv"))
        if not integrated_files:
            raise FileNotFoundError("No integrated data files found. Run collect_data.py first.")
        
        latest_file = max(integrated_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading integrated data from: {latest_file}")
        return pd.read_csv(latest_file)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        logger.info("Starting data cleaning")
        initial_shape = df.shape
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['building_id', 'timestamp'])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns.difference(['timestamp', 'building_id'])
        
        # Fill missing numeric values with interpolation
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df.groupby('building_id')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
        
        # Fill missing categorical values with mode
        for col in categorical_columns:
            if df[col].isna().any():
                df[col] = df.groupby('building_id')[col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown')
                )
        
        # Remove outliers in energy consumption (beyond 5 std dev)
        energy_mean = df['energy_consumption_kwh'].mean()
        energy_std = df['energy_consumption_kwh'].std()
        df = df[
            (df['energy_consumption_kwh'] >= energy_mean - 5 * energy_std) &
            (df['energy_consumption_kwh'] <= energy_mean + 5 * energy_std)
        ]
        
        # Validate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Data cleaning complete. Shape: {initial_shape} -> {df.shape}")
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               val_size: float = 0.1) -> dict:
        """Create temporal train/validation/test split."""
        logger.info("Creating train/validation/test split")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate split points
        n_samples = len(df)
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Validation shape: {val_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        logger.info("Applying feature engineering")
        
        # Apply all feature engineering steps
        df = self.engineer.create_all_features(df)
        
        # Only drop rows where the target variable is NaN
        initial_rows = len(df)
        
        # For lag features, we'll only drop rows where lag_168 is NaN
        # This preserves most of the data while ensuring we have sufficient history
        lag_columns = [col for col in df.columns if 'lag_168' in col]
        if lag_columns:
            # Drop rows where the longest lag feature is NaN
            df = df.dropna(subset=lag_columns)
        else:
            # If no lag features, just drop rows where target is NaN
            df = df.dropna(subset=['energy_consumption_kwh'])
        
        logger.info(f"Dropped {initial_rows - len(df)} rows with insufficient history")
        
        # Fill any remaining NaN values in other columns
        # For numeric columns, forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df.groupby('building_id')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        return df
    
    def save_processed_data(self, datasets: dict, prefix: str = ""):
        """Save processed datasets."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for split_name, df in datasets.items():
            filename = f"{prefix}_{split_name}_{timestamp}.csv" if prefix else f"{split_name}_{timestamp}.csv"
            output_path = self.processed_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} data to {output_path}")
    
    def create_feature_list(self, df: pd.DataFrame) -> dict:
        """Create lists of different feature types."""
        # Get all columns
        all_columns = df.columns.tolist()
        
        # Identify feature types
        feature_lists = {
            'target': 'energy_consumption_kwh',
            'timestamp': 'timestamp',
            'id': 'building_id',
            'numerical': [],
            'categorical': [],
            'temporal': [],
            'lag': [],
            'weather': [],
            'building': []
        }
        
        for col in all_columns:
            if col in [feature_lists['target'], feature_lists['timestamp'], feature_lists['id']]:
                continue
            
            # Categorize features
            if 'lag' in col or 'rolling' in col:
                feature_lists['lag'].append(col)
            elif col in ['hour', 'day_of_week', 'month', 'is_business_hour', 'is_weekend', 
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
                feature_lists['temporal'].append(col)
            elif col in ['temperature_c', 'humidity_percent', 'wind_speed_kmh', 
                        'solar_radiation_wm2', 'precipitation_mm', 'heating_degree_days',
                        'cooling_degree_days', 'discomfort_index']:
                feature_lists['weather'].append(col)
            elif col in ['floor_area_sqm', 'year_built', 'energy_star_rating', 
                        'building_age', 'energy_intensity', 'age_efficiency_factor']:
                feature_lists['building'].append(col)
            elif '_encoded' in col:
                feature_lists['categorical'].append(col)
            elif df[col].dtype in ['float64', 'int64']:
                feature_lists['numerical'].append(col)
        
        # All features for modeling
        feature_lists['all_features'] = (
            feature_lists['temporal'] + 
            feature_lists['lag'] + 
            feature_lists['weather'] + 
            feature_lists['building'] + 
            feature_lists['categorical'] +
            [col for col in feature_lists['numerical'] 
             if col not in feature_lists['temporal'] + feature_lists['lag'] + 
                          feature_lists['weather'] + feature_lists['building']]
        )
        
        return feature_lists
    
    def run_preprocessing_pipeline(self, test_size: float = 0.2, 
                                 val_size: float = 0.1) -> dict:
        """Run complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_latest_integrated_data()
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Clean data
        df = self.clean_data(df)
        
        # Apply feature engineering
        df = self.prepare_features(df)
        
        # Create train/val/test split
        datasets = self.create_train_test_split(df, test_size, val_size)
        
        # Save processed datasets
        self.save_processed_data(datasets, prefix="processed")
        
        # Create and save feature lists
        feature_lists = self.create_feature_list(df)
        
        # Save feature information
        import json
        import pickle
        
        feature_info_path = self.processed_dir / "feature_info.json"
        with open(feature_info_path, 'w') as f:
            json.dump(feature_lists, f, indent=2)
        logger.info(f"Saved feature information to {feature_info_path}")
        
        # Save the feature engineer with fitted label encoders
        engineer_path = self.processed_dir / "feature_engineer.pkl"
        with open(engineer_path, 'wb') as f:
            pickle.dump(self.engineer, f)
        logger.info(f"Saved feature engineer to {engineer_path}")
        
        logger.info("Preprocessing pipeline complete")
        
        return {
            'datasets': datasets,
            'feature_lists': feature_lists
        }


def main():
    """Main entry point for data preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess data for Energy Optimization ML')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proportion of data for validation set'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.data_dir)
    
    # Run preprocessing pipeline
    results = preprocessor.run_preprocessing_pipeline(
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Print summary
    logger.info("\nPreprocessing Summary:")
    logger.info(f"Train samples: {len(results['datasets']['train'])}")
    logger.info(f"Validation samples: {len(results['datasets']['validation'])}")
    logger.info(f"Test samples: {len(results['datasets']['test'])}")
    logger.info(f"Total features: {len(results['feature_lists']['all_features'])}")
    logger.info(f"Temporal features: {len(results['feature_lists']['temporal'])}")
    logger.info(f"Lag features: {len(results['feature_lists']['lag'])}")
    logger.info(f"Weather features: {len(results['feature_lists']['weather'])}")
    logger.info(f"Building features: {len(results['feature_lists']['building'])}")


if __name__ == "__main__":
    main()