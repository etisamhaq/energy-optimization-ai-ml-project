"""
Tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from src.data.data_loader import EnergyDataLoader


class TestEnergyDataLoader:
    """Test cases for EnergyDataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def data_loader(self, temp_data_dir):
        """Create EnergyDataLoader instance."""
        return EnergyDataLoader(temp_data_dir)
    
    def test_initialization(self, data_loader, temp_data_dir):
        """Test data loader initialization."""
        assert data_loader.data_dir == Path(temp_data_dir)
        assert data_loader.raw_dir.exists()
        assert data_loader.processed_dir.exists()
        assert data_loader.external_dir.exists()
    
    def test_generate_synthetic_energy_data(self, data_loader):
        """Test synthetic energy data generation."""
        n_buildings = 3
        n_days = 7
        
        df = data_loader._generate_synthetic_energy_data(n_buildings, n_days)
        
        # Check shape
        expected_rows = n_buildings * n_days * 24
        assert len(df) == expected_rows
        
        # Check columns
        expected_columns = ['building_id', 'timestamp', 'energy_consumption_kwh', 'peak_demand_kw']
        assert all(col in df.columns for col in expected_columns)
        
        # Check data types
        assert df['energy_consumption_kwh'].dtype == np.float64
        assert df['peak_demand_kw'].dtype == np.float64
        
        # Check values are positive
        assert (df['energy_consumption_kwh'] >= 0).all()
        assert (df['peak_demand_kw'] >= 0).all()
        
        # Check building IDs
        assert len(df['building_id'].unique()) == n_buildings
    
    def test_generate_synthetic_weather_data(self, data_loader):
        """Test synthetic weather data generation."""
        start_date = "2024-01-01"
        end_date = "2024-01-07"
        
        df = data_loader._generate_synthetic_weather_data(start_date, end_date)
        
        # Check shape
        expected_hours = 7 * 24  # 7 days * 24 hours
        assert len(df) == expected_hours
        
        # Check columns
        expected_columns = ['timestamp', 'temperature_c', 'humidity_percent', 
                          'wind_speed_kmh', 'solar_radiation_wm2', 'precipitation_mm']
        assert all(col in df.columns for col in expected_columns)
        
        # Check value ranges
        assert df['temperature_c'].between(-30, 50).all()
        assert df['humidity_percent'].between(0, 100).all()
        assert (df['wind_speed_kmh'] >= 0).all()
        assert (df['solar_radiation_wm2'] >= 0).all()
        assert (df['precipitation_mm'] >= 0).all()
    
    def test_load_building_metadata(self, data_loader):
        """Test building metadata loading."""
        building_ids = ['B001', 'B002', 'B003']
        
        df = data_loader.load_building_metadata(building_ids)
        
        # Check shape
        assert len(df) == len(building_ids)
        
        # Check columns
        expected_columns = ['building_id', 'building_type', 'floor_area_sqm', 
                          'year_built', 'occupancy_type', 'hvac_type', 'energy_star_rating']
        assert all(col in df.columns for col in expected_columns)
        
        # Check building IDs match
        assert set(df['building_id']) == set(building_ids)
    
    def test_integrate_datasets(self, data_loader):
        """Test dataset integration."""
        # Generate test data
        energy_df = data_loader._generate_synthetic_energy_data(2, 2)
        weather_df = data_loader._generate_synthetic_weather_data("2024-01-01", "2024-01-02")
        building_ids = energy_df['building_id'].unique()
        metadata_df = data_loader.load_building_metadata(building_ids)
        
        # Integrate datasets
        integrated_df = data_loader.integrate_datasets(energy_df, weather_df, metadata_df)
        
        # Check that all data is integrated
        assert len(integrated_df) == len(energy_df)
        
        # Check columns from all datasets are present
        energy_cols = ['energy_consumption_kwh', 'peak_demand_kw']
        weather_cols = ['temperature_c', 'humidity_percent']
        metadata_cols = ['building_type', 'floor_area_sqm']
        
        all_cols = energy_cols + weather_cols + metadata_cols
        assert all(col in integrated_df.columns for col in all_cols)
    
    def test_save_processed_data(self, data_loader):
        """Test saving processed data."""
        # Create test dataframe
        df = pd.DataFrame({
            'building_id': ['B001', 'B001'],
            'timestamp': pd.date_range('2024-01-01', periods=2, freq='H'),
            'energy_consumption_kwh': [100, 110]
        })
        
        # Save data
        filename = "test_data.csv"
        data_loader.save_processed_data(df, filename)
        
        # Check file exists
        output_path = data_loader.processed_dir / filename
        assert output_path.exists()
        
        # Load and verify
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)