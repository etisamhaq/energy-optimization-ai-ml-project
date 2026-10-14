import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import zlib
from datetime import datetime, timedelta
import logging

from src.data.open_meteo import fetch_weather

logger = logging.getLogger(__name__)

ENERGY_COLUMNS = ['building_id', 'timestamp', 'energy_consumption_kwh']


def hourly_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Hourly timestamps covering every hour from start_date through end_date (inclusive)."""
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize() + pd.Timedelta(hours=23)
    return pd.date_range(start=start, end=end, freq='H')


def validate_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check that energy data has the expected columns and plausible values."""
    missing = set(ENERGY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Energy data is missing required columns: {sorted(missing)}")
    
    df = df.copy()
    df['building_id'] = df['building_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['energy_consumption_kwh'] = pd.to_numeric(df['energy_consumption_kwh'], errors='coerce')
    if (df['energy_consumption_kwh'] < 0).any():
        raise ValueError("Energy data contains negative consumption values")
    if df.duplicated(['building_id', 'timestamp']).any():
        raise ValueError("Energy data contains duplicate building_id/timestamp rows")
    return df


class EnergyDataLoader:
    """Handles loading and integration of various energy-related datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def site_file(self) -> Path:
        return self.external_dir / "site.json"
    
    @property
    def metadata_file(self) -> Path:
        return self.external_dir / "building_metadata.csv"
    
    def load_energy_consumption_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load building energy consumption data.
        
        file_path: CSV with building_id, timestamp and energy_consumption_kwh columns
        (hourly, local time). Without a file, synthetic demo data is generated.
        """
        if file_path:
            df = validate_energy_data(pd.read_csv(file_path))
        else:
            # Generate synthetic data for demonstration
            df = self._generate_synthetic_energy_data()
        
        logger.info(f"Loaded energy data with shape: {df.shape}")
        return df
    
    def save_site(self, site: Dict):
        """Record where the data comes from, so predictions use matching weather."""
        with open(self.site_file, 'w') as f:
            json.dump(site, f, indent=2)
    
    def load_site(self) -> Dict:
        """Location and weather source of the collected data (synthetic if none was saved)."""
        if self.site_file.exists():
            with open(self.site_file) as f:
                return json.load(f)
        return {'name': 'synthetic', 'weather_source': 'synthetic'}
    
    def load_weather_data(self, start_date: str, end_date: str, location: Optional[str] = None) -> pd.DataFrame:
        """Hourly weather for every hour of the date range at the data's site."""
        site = self.load_site()
        if site.get('weather_source') == 'open-meteo':
            df = fetch_weather(
                site['latitude'], site['longitude'], site['timezone'],
                start_date, end_date, cache_dir=self.external_dir / "weather_cache"
            )
        else:
            df = self._generate_synthetic_weather_data(start_date, end_date)
        logger.info(f"Loaded {site.get('weather_source')} weather for {site.get('name')} "
                    f"from {start_date} to {end_date}")
        return df
    
    def save_building_metadata(self, metadata: pd.DataFrame):
        """Store building metadata so predictions use the same characteristics as training."""
        metadata.to_csv(self.metadata_file, index=False)
    
    def list_buildings(self) -> List[str]:
        """IDs of the buildings with stored metadata."""
        if not self.metadata_file.exists():
            return []
        return pd.read_csv(self.metadata_file)['building_id'].astype(str).tolist()
    
    def load_building_metadata(self, building_ids: List[str]) -> pd.DataFrame:
        """
        Load building characteristics and metadata.
        
        Uses the metadata saved at collection time when present (raising ValueError for
        unknown buildings); otherwise generates synthetic demo metadata.
        """
        if self.metadata_file.exists():
            stored = pd.read_csv(self.metadata_file)
            stored['building_id'] = stored['building_id'].astype(str)
            unknown = set(map(str, building_ids)) - set(stored['building_id'])
            if unknown:
                raise ValueError(f"Unknown building(s): {sorted(unknown)}")
            return stored.set_index('building_id').loc[list(map(str, building_ids))].reset_index()
        
        # Generate synthetic building metadata, seeded by building ID so the same
        # building always gets the same characteristics across calls and processes
        metadata = []
        for building_id in building_ids:
            rng = np.random.default_rng(zlib.crc32(str(building_id).encode()))
            metadata.append({
                'building_id': building_id,
                'building_type': rng.choice(['office', 'residential', 'retail', 'industrial']),
                'floor_area_sqm': int(rng.integers(1000, 50000)),
                'year_built': int(rng.integers(1970, 2020)),
                'occupancy_type': rng.choice(['single_tenant', 'multi_tenant']),
                'hvac_type': rng.choice(['central', 'distributed', 'hybrid']),
                'energy_star_rating': int(rng.integers(1, 100))
            })
        
        return pd.DataFrame(metadata)
    
    def integrate_datasets(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame, 
                          metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Integrate multiple datasets into a single dataframe."""
        # Merge energy data with weather data on timestamp
        integrated_df = pd.merge(
            energy_df,
            weather_df,
            on='timestamp',
            how='left'
        )
        
        # Merge with building metadata
        integrated_df = pd.merge(
            integrated_df,
            metadata_df,
            on='building_id',
            how='left'
        )
        
        logger.info(f"Integrated dataset shape: {integrated_df.shape}")
        return integrated_df
    
    def _generate_synthetic_energy_data(self, n_buildings: int = 5, 
                                      n_days: int = 365) -> pd.DataFrame:
        """Generate synthetic energy consumption data for testing."""
        data = []
        
        # Start at midnight so timestamps fall exactly on the hour and line up with weather data
        start_date = (datetime.now() - timedelta(days=n_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        for building_id in [f"B{str(i).zfill(3)}" for i in range(1, n_buildings + 1)]:
            base_consumption = np.random.uniform(100, 500)  # Base kWh
            
            for day in range(n_days):
                current_date = start_date + timedelta(days=day)
                
                for hour in range(24):
                    timestamp = current_date.replace(hour=hour, minute=0, second=0)
                    
                    # Add patterns: higher during business hours, lower at night
                    hour_factor = 1.5 if 8 <= hour <= 18 else 0.7
                    
                    # Add weekly pattern: lower on weekends
                    day_factor = 0.6 if current_date.weekday() >= 5 else 1.0
                    
                    # Add seasonal pattern
                    season_factor = 1.2 if current_date.month in [6, 7, 8, 12, 1, 2] else 0.9
                    
                    # Add some noise
                    noise = np.random.normal(0, 0.1)
                    
                    consumption = base_consumption * hour_factor * day_factor * season_factor * (1 + noise)
                    
                    data.append({
                        'building_id': building_id,
                        'timestamp': timestamp,
                        'energy_consumption_kwh': max(0, consumption),
                        'peak_demand_kw': max(0, consumption * np.random.uniform(0.8, 1.2))
                    })
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic weather data for testing."""
        date_range = hourly_range(start_date, end_date)
        
        data = []
        for timestamp in date_range:
            # Temperature pattern: seasonal + daily variation
            base_temp = 15 + 10 * np.sin((timestamp.dayofyear - 80) * 2 * np.pi / 365)
            daily_variation = 5 * np.sin((timestamp.hour - 6) * 2 * np.pi / 24)
            temperature = base_temp + daily_variation + np.random.normal(0, 2)
            
            # Humidity: inverse correlation with temperature
            humidity = 70 - 0.5 * temperature + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 95)
            
            data.append({
                'timestamp': timestamp,
                'temperature_c': temperature,
                'humidity_percent': humidity,
                'wind_speed_kmh': np.random.gamma(2, 2),
                'solar_radiation_wm2': max(0, 500 * np.sin((timestamp.hour - 6) * np.pi / 12)) if 6 <= timestamp.hour <= 18 else 0,
                'precipitation_mm': np.random.exponential(0.5) if np.random.random() < 0.1 else 0
            })
        
        return pd.DataFrame(data)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to the processed directory."""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = EnergyDataLoader()
    
    # Load synthetic data (weather follows the saved site, synthetic by default)
    energy_data = loader.load_energy_consumption_data()
    weather_data = loader.load_weather_data("2023-01-01", "2023-12-31")
    building_metadata = loader.load_building_metadata(energy_data['building_id'].unique())
    
    # Integrate datasets
    integrated_data = loader.integrate_datasets(energy_data, weather_data, building_metadata)
    
    # Save processed data
    loader.save_processed_data(integrated_data, "integrated_energy_data.csv")