import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnergyFeatureEngineer:
    """Feature engineering for energy consumption prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business hours indicator
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Season
        df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           lag_periods: List[int] = [1, 24, 168]) -> pd.DataFrame:
        """Create lag features for time series prediction."""
        df = df.copy()
        
        # Sort by building and timestamp
        df = df.sort_values(['building_id', 'timestamp'])
        
        # Create lag features for each building
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df.groupby('building_id')[target_col].shift(lag)
        
        # Rolling statistics with min_periods to avoid excessive NaN
        for window in [24, 168]:  # Daily and weekly
            # Use smaller min_periods to keep more data
            min_periods = min(window // 4, 12)  # At least 1/4 of window or 12 hours
            df[f'{target_col}_rolling_mean_{window}'] = df.groupby('building_id')[target_col].transform(
                lambda x: x.rolling(window, min_periods=min_periods).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = df.groupby('building_id')[target_col].transform(
                lambda x: x.rolling(window, min_periods=min_periods).std()
            )
        
        return df
    
    def create_weather_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from weather-energy interactions."""
        df = df.copy()
        
        # Temperature-based features
        df['temp_squared'] = df['temperature_c'] ** 2
        df['heating_degree_days'] = np.maximum(18 - df['temperature_c'], 0)
        df['cooling_degree_days'] = np.maximum(df['temperature_c'] - 18, 0)
        
        # Comfort index
        df['discomfort_index'] = df['temperature_c'] + 0.5 * df['humidity_percent']
        
        # Weather change features
        df['temp_change_1h'] = df.groupby('building_id')['temperature_c'].diff()
        df['humidity_change_1h'] = df.groupby('building_id')['humidity_percent'].diff()
        
        return df
    
    def create_building_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create building efficiency metrics."""
        df = df.copy()
        
        # Energy intensity (consumption per square meter)
        df['energy_intensity'] = df['energy_consumption_kwh'] / df['floor_area_sqm']
        
        # Building age
        current_year = pd.Timestamp.now().year
        df['building_age'] = current_year - df['year_built']
        
        # Age-efficiency interaction
        df['age_efficiency_factor'] = df['building_age'] / df['energy_star_rating']
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Handle unseen labels by assigning them to the most common class
                    # or a default value
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    except ValueError as e:
                        # Get the unique values in the current data
                        unique_values = df[col].unique()
                        
                        # Create a mapping for unseen values
                        known_classes = self.label_encoders[col].classes_
                        encoded_col = df[col].copy()
                        
                        for val in unique_values:
                            if val not in known_classes:
                                # Use the first known class as default
                                # In production, you might want to use the most common class
                                encoded_col = encoded_col.replace(val, known_classes[0])
                                logger.warning(f"Unseen label '{val}' in column '{col}', replacing with '{known_classes[0]}'")
                        
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(encoded_col)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      feature_cols: List[str], 
                      fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, 'energy_consumption_kwh')
        
        # Create weather interaction features
        if 'temperature_c' in df.columns:
            df = self.create_weather_interaction_features(df)
        
        # Create building efficiency features
        if 'floor_area_sqm' in df.columns:
            df = self.create_building_efficiency_features(df)
        
        # Encode categorical features
        categorical_cols = ['building_type', 'occupancy_type', 'hvac_type', 'season']
        df = self.encode_categorical_features(df, categorical_cols)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def get_feature_importance_candidates(self) -> List[str]:
        """Get list of features for importance analysis."""
        return [
            # Temporal features
            'hour', 'day_of_week', 'month', 'is_business_hour', 'is_weekend',
            
            # Lag features
            'energy_consumption_kwh_lag_1', 'energy_consumption_kwh_lag_24',
            'energy_consumption_kwh_rolling_mean_24', 'energy_consumption_kwh_rolling_mean_168',
            
            # Weather features
            'temperature_c', 'humidity_percent', 'heating_degree_days', 'cooling_degree_days',
            
            # Building features
            'floor_area_sqm', 'building_age', 'energy_star_rating', 'energy_intensity',
            
            # Encoded categorical features
            'building_type_encoded', 'hvac_type_encoded'
        ]


if __name__ == "__main__":
    # Example usage
    engineer = EnergyFeatureEngineer()
    
    # Load sample data
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.data.data_loader import EnergyDataLoader
    
    loader = EnergyDataLoader()
    df = loader.load_energy_consumption_data()
    
    # Apply feature engineering
    df_featured = engineer.create_all_features(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_featured.columns)}")
    print(f"\nNew features created: {set(df_featured.columns) - set(df.columns)}")