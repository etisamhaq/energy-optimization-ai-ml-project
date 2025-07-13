"""
Tests for feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from src.features.feature_engineering import EnergyFeatureEngineer


class TestEnergyFeatureEngineer:
    """Test cases for EnergyFeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create EnergyFeatureEngineer instance."""
        return EnergyFeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 24 * 7  # One week of hourly data
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'building_id': 'B001',
            'energy_consumption_kwh': np.random.uniform(100, 200, n_samples),
            'temperature_c': 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'humidity_percent': np.random.uniform(40, 80, n_samples),
            'floor_area_sqm': 5000,
            'year_built': 2010,
            'building_type': 'office',
            'occupancy_type': 'single_tenant',
            'hvac_type': 'central',
            'energy_star_rating': 75
        })
        
        return df
    
    def test_create_temporal_features(self, engineer, sample_data):
        """Test temporal feature creation."""
        df = engineer.create_temporal_features(sample_data)
        
        # Check new columns exist
        temporal_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                        'is_business_hour', 'is_weekend', 'season']
        
        for col in temporal_cols:
            assert col in df.columns
        
        # Check value ranges
        assert df['hour'].between(0, 23).all()
        assert df['day_of_week'].between(0, 6).all()
        assert df['month'].between(1, 12).all()
        assert df['is_business_hour'].isin([0, 1]).all()
        assert df['is_weekend'].isin([0, 1]).all()
        
        # Check cyclical encoding
        assert df['hour_sin'].between(-1, 1).all()
        assert df['hour_cos'].between(-1, 1).all()
    
    def test_create_lag_features(self, engineer, sample_data):
        """Test lag feature creation."""
        target_col = 'energy_consumption_kwh'
        lag_periods = [1, 24]
        
        df = engineer.create_lag_features(sample_data, target_col, lag_periods)
        
        # Check lag columns exist
        for lag in lag_periods:
            assert f'{target_col}_lag_{lag}' in df.columns
        
        # Check rolling statistics
        assert f'{target_col}_rolling_mean_24' in df.columns
        assert f'{target_col}_rolling_std_24' in df.columns
        
        # Verify lag values
        # After sorting, lag_1 should be the previous value
        df_sorted = df.sort_values(['building_id', 'timestamp'])
        lag_1_col = f'{target_col}_lag_1'
        
        # Check that lag values match expected (excluding first row due to shift)
        for i in range(1, len(df_sorted) - 1):
            if not pd.isna(df_sorted.iloc[i][lag_1_col]):
                assert df_sorted.iloc[i][lag_1_col] == df_sorted.iloc[i-1][target_col]
    
    def test_create_weather_interaction_features(self, engineer, sample_data):
        """Test weather interaction feature creation."""
        df = engineer.create_weather_interaction_features(sample_data)
        
        # Check new columns exist
        weather_cols = ['temp_squared', 'heating_degree_days', 'cooling_degree_days',
                       'discomfort_index', 'temp_change_1h', 'humidity_change_1h']
        
        for col in weather_cols:
            assert col in df.columns
        
        # Check calculations
        assert (df['temp_squared'] == df['temperature_c'] ** 2).all()
        assert (df['heating_degree_days'] >= 0).all()
        assert (df['cooling_degree_days'] >= 0).all()
    
    def test_create_building_efficiency_features(self, engineer, sample_data):
        """Test building efficiency feature creation."""
        df = engineer.create_building_efficiency_features(sample_data)
        
        # Check new columns exist
        efficiency_cols = ['energy_intensity', 'building_age', 'age_efficiency_factor']
        
        for col in efficiency_cols:
            assert col in df.columns
        
        # Check calculations
        expected_intensity = df['energy_consumption_kwh'] / df['floor_area_sqm']
        assert np.allclose(df['energy_intensity'], expected_intensity)
        
        current_year = pd.Timestamp.now().year
        expected_age = current_year - df['year_built']
        assert (df['building_age'] == expected_age).all()
    
    def test_encode_categorical_features(self, engineer, sample_data):
        """Test categorical feature encoding."""
        categorical_cols = ['building_type', 'occupancy_type', 'hvac_type']
        
        df = engineer.encode_categorical_features(sample_data, categorical_cols)
        
        # Check encoded columns exist
        for col in categorical_cols:
            assert f'{col}_encoded' in df.columns
            assert df[f'{col}_encoded'].dtype in [np.int64, np.int32]
    
    def test_scale_features(self, engineer, sample_data):
        """Test feature scaling."""
        feature_cols = ['energy_consumption_kwh', 'temperature_c', 'humidity_percent']
        
        # Test fit and transform
        df_scaled = engineer.scale_features(sample_data.copy(), feature_cols, fit=True)
        
        # Check that features are scaled (mean ~0, std ~1)
        for col in feature_cols:
            assert abs(df_scaled[col].mean()) < 0.1
            assert abs(df_scaled[col].std() - 1) < 0.1
        
        # Test transform only
        df_scaled_2 = engineer.scale_features(sample_data.copy(), feature_cols, fit=False)
        assert np.allclose(df_scaled[feature_cols], df_scaled_2[feature_cols])
    
    def test_create_all_features(self, engineer, sample_data):
        """Test complete feature engineering pipeline."""
        df = engineer.create_all_features(sample_data)
        
        # Check that dataframe has more columns than original
        assert len(df.columns) > len(sample_data.columns)
        
        # Check specific feature groups are present
        assert 'hour_sin' in df.columns  # Temporal
        assert 'energy_consumption_kwh_lag_1' in df.columns  # Lag
        assert 'temp_squared' in df.columns  # Weather
        assert 'energy_intensity' in df.columns  # Building
        assert 'building_type_encoded' in df.columns  # Categorical
    
    def test_get_feature_importance_candidates(self, engineer):
        """Test feature importance candidates list."""
        candidates = engineer.get_feature_importance_candidates()
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        
        # Check some expected features are in the list
        expected_features = ['hour', 'temperature_c', 'energy_consumption_kwh_lag_1']
        for feature in expected_features:
            assert feature in candidates