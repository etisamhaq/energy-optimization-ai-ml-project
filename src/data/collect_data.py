#!/usr/bin/env python
"""
Data collection script for Energy Optimization ML project.
Collects data from various sources including energy consumption, weather, and building metadata.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append('../..')

from data_loader import EnergyDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self, data_dir: str = "data"):
        self.loader = EnergyDataLoader(data_dir)
        self.data_dir = Path(data_dir)
    
    def collect_energy_data(self, source: str = "synthetic", **kwargs):
        """Collect energy consumption data."""
        logger.info(f"Collecting energy data from source: {source}")
        
        if source == "synthetic":
            # Generate synthetic data
            n_buildings = kwargs.get('n_buildings', 10)
            n_days = kwargs.get('n_days', 365)
            df = self.loader._generate_synthetic_energy_data(n_buildings, n_days)
        elif source == "file":
            # Load from file
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path required when source='file'")
            df = self.loader.load_energy_consumption_data(file_path)
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Save raw data
        output_path = self.loader.raw_dir / f"energy_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved energy data to {output_path}")
        return df
    
    def collect_weather_data(self, start_date: str, end_date: str, 
                           location: str = "New York", source: str = "synthetic"):
        """Collect weather data."""
        logger.info(f"Collecting weather data for {location} from {start_date} to {end_date}")
        
        if source == "synthetic":
            df = self.loader._generate_synthetic_weather_data(start_date, end_date)
        elif source == "api":
            # In production, this would call NOAA or other weather API
            raise NotImplementedError("Weather API integration not yet implemented")
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Save raw data
        output_path = self.loader.raw_dir / f"weather_data_{location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved weather data to {output_path}")
        return df
    
    def collect_building_metadata(self, building_ids: list, source: str = "synthetic"):
        """Collect building metadata."""
        logger.info(f"Collecting metadata for {len(building_ids)} buildings")
        
        if source == "synthetic":
            df = self.loader.load_building_metadata(building_ids)
        elif source == "database":
            # In production, this would query building database
            raise NotImplementedError("Database integration not yet implemented")
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Save raw data
        output_path = self.loader.raw_dir / f"building_metadata_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved building metadata to {output_path}")
        return df
    
    def collect_all_data(self, start_date: str = None, end_date: str = None, 
                        n_buildings: int = 10, n_days: int = 365):
        """Collect all data sources."""
        logger.info("Starting comprehensive data collection")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')
        
        # Collect energy data
        energy_df = self.collect_energy_data(
            source="synthetic",
            n_buildings=n_buildings,
            n_days=n_days
        )
        
        # Get unique building IDs
        building_ids = energy_df['building_id'].unique().tolist()
        
        # Collect weather data
        weather_df = self.collect_weather_data(
            start_date=start_date,
            end_date=end_date,
            location="New York",
            source="synthetic"
        )
        
        # Collect building metadata
        metadata_df = self.collect_building_metadata(
            building_ids=building_ids,
            source="synthetic"
        )
        
        # Integrate all datasets
        integrated_df = self.loader.integrate_datasets(energy_df, weather_df, metadata_df)
        
        # Save integrated dataset
        output_path = self.loader.processed_dir / f"integrated_data_{datetime.now().strftime('%Y%m%d')}.csv"
        integrated_df.to_csv(output_path, index=False)
        logger.info(f"Saved integrated dataset to {output_path}")
        
        return {
            'energy': energy_df,
            'weather': weather_df,
            'metadata': metadata_df,
            'integrated': integrated_df
        }


def main():
    """Main entry point for data collection."""
    parser = argparse.ArgumentParser(description='Collect data for Energy Optimization ML')
    parser.add_argument(
        '--source',
        choices=['all', 'energy', 'weather', 'metadata'],
        default='all',
        help='Which data source to collect'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data collection (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data collection (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--n-buildings',
        type=int,
        default=10,
        help='Number of buildings to generate data for'
    )
    parser.add_argument(
        '--n-days',
        type=int,
        default=365,
        help='Number of days of historical data to generate'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to store data'
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = DataCollector(args.data_dir)
    
    # Collect data based on source
    if args.source == 'all':
        datasets = collector.collect_all_data(
            start_date=args.start_date,
            end_date=args.end_date,
            n_buildings=args.n_buildings,
            n_days=args.n_days
        )
        logger.info("Data collection completed successfully")
        logger.info(f"Energy data shape: {datasets['energy'].shape}")
        logger.info(f"Weather data shape: {datasets['weather'].shape}")
        logger.info(f"Metadata shape: {datasets['metadata'].shape}")
        logger.info(f"Integrated data shape: {datasets['integrated'].shape}")
    
    elif args.source == 'energy':
        df = collector.collect_energy_data(
            n_buildings=args.n_buildings,
            n_days=args.n_days
        )
        logger.info(f"Energy data collected: {df.shape}")
    
    elif args.source == 'weather':
        if not args.start_date or not args.end_date:
            raise ValueError("start-date and end-date required for weather data collection")
        df = collector.collect_weather_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        logger.info(f"Weather data collected: {df.shape}")
    
    elif args.source == 'metadata':
        # For metadata, we need building IDs, so collect energy data first
        energy_df = collector.collect_energy_data(
            n_buildings=args.n_buildings,
            n_days=1  # Just need building IDs
        )
        building_ids = energy_df['building_id'].unique().tolist()
        df = collector.collect_building_metadata(building_ids)
        logger.info(f"Metadata collected: {df.shape}")


if __name__ == "__main__":
    main()