#!/usr/bin/env python
"""
Data collection script for Energy Optimization ML project.

Sources:
  synthetic  Generated demo data (energy, weather and metadata). Default.
  bdg2       Real hourly electricity data from the Building Data Genome Project 2,
             with real Open-Meteo weather for the site's location.
  file       Your own meter data and building metadata CSVs, with Open-Meteo weather
             for the given coordinates.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.data import bdg2
from src.data.data_loader import EnergyDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

METADATA_COLUMNS = ['building_id', 'building_type', 'floor_area_sqm', 'year_built',
                    'occupancy_type', 'hvac_type', 'energy_star_rating']


class DataCollector:
    """Orchestrates data collection from multiple sources."""

    def __init__(self, data_dir: str = "data"):
        self.loader = EnergyDataLoader(data_dir)
        self.data_dir = Path(data_dir)

    def collect_all_data(self, start_date: str = None, end_date: str = None,
                        n_buildings: int = 10, n_days: int = 365):
        """Generate a complete synthetic dataset for demos and tests."""
        logger.info("Generating synthetic data")

        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')

        self.loader.save_site({'name': 'synthetic', 'weather_source': 'synthetic'})
        # Clear stored metadata so synthetic metadata is generated for the new buildings
        self.loader.metadata_file.unlink(missing_ok=True)

        energy_df = self.loader._generate_synthetic_energy_data(n_buildings, n_days)
        weather_df = self.loader.load_weather_data(start_date, end_date)
        metadata_df = self.loader.load_building_metadata(energy_df['building_id'].unique().tolist())
        return self._save_datasets(energy_df, weather_df, metadata_df)

    def collect_bdg2_data(self, site_id: str = "Rat", n_buildings: int = 10):
        """Collect real BDG2 meter data and matching Open-Meteo weather."""
        energy_df, metadata_df, site = bdg2.load_site(
            site_id, n_buildings, cache_dir=self.loader.external_dir / "bdg2"
        )
        self.loader.save_site(site)
        weather_df = self._weather_for(energy_df)
        return self._save_datasets(energy_df, weather_df, metadata_df)

    def collect_from_files(self, energy_file: str, metadata_file: str,
                           latitude: float, longitude: float, timezone: str, site_name: str = "custom"):
        """Collect your own meter data and building metadata, with Open-Meteo weather."""
        energy_df = self.loader.load_energy_consumption_data(energy_file)

        metadata_df = pd.read_csv(metadata_file)
        missing = set(METADATA_COLUMNS) - set(metadata_df.columns)
        if missing:
            raise ValueError(f"Building metadata is missing required columns: {sorted(missing)}")
        metadata_df['building_id'] = metadata_df['building_id'].astype(str)
        unknown = set(energy_df['building_id']) - set(metadata_df['building_id'])
        if unknown:
            raise ValueError(f"No metadata for building(s): {sorted(unknown)}")

        self.loader.save_site({
            'name': site_name, 'latitude': latitude, 'longitude': longitude,
            'timezone': timezone, 'weather_source': 'open-meteo'
        })
        weather_df = self._weather_for(energy_df)
        return self._save_datasets(energy_df, weather_df, metadata_df[METADATA_COLUMNS])

    def _weather_for(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        start = energy_df['timestamp'].min().strftime('%Y-%m-%d')
        end = energy_df['timestamp'].max().strftime('%Y-%m-%d')
        return self.loader.load_weather_data(start, end)

    def _save_datasets(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame,
                       metadata_df: pd.DataFrame) -> dict:
        """Save raw datasets, stored metadata and the integrated dataset."""
        stamp = datetime.now().strftime('%Y%m%d')
        energy_df.to_csv(self.loader.raw_dir / f"energy_data_{stamp}.csv", index=False)
        weather_df.to_csv(self.loader.raw_dir / f"weather_data_{stamp}.csv", index=False)
        metadata_df.to_csv(self.loader.raw_dir / f"building_metadata_{stamp}.csv", index=False)
        self.loader.save_building_metadata(metadata_df)

        integrated_df = self.loader.integrate_datasets(energy_df, weather_df, metadata_df)
        output_path = self.loader.processed_dir / f"integrated_data_{stamp}.csv"
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
        choices=['synthetic', 'all', 'bdg2', 'file'],
        default='synthetic',
        help="Data source ('all' is an alias for 'synthetic')"
    )
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store data')

    synthetic = parser.add_argument_group('synthetic source')
    synthetic.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    synthetic.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    synthetic.add_argument('--n-days', type=int, default=365, help='Days of history to generate')

    real = parser.add_argument_group('bdg2 source')
    real.add_argument('--site', type=str, default='Rat', help='BDG2 site ID')
    parser.add_argument('--n-buildings', type=int, default=10, help='Number of buildings (synthetic, bdg2)')

    files = parser.add_argument_group('file source')
    files.add_argument('--energy-file', type=str, help='CSV: building_id, timestamp, energy_consumption_kwh')
    files.add_argument('--metadata-file', type=str, help=f"CSV: {', '.join(METADATA_COLUMNS)}")
    files.add_argument('--latitude', type=float, help='Site latitude, for weather')
    files.add_argument('--longitude', type=float, help='Site longitude, for weather')
    files.add_argument('--timezone', type=str, help='Site IANA timezone, e.g. America/New_York')

    args = parser.parse_args()
    collector = DataCollector(args.data_dir)

    if args.source in ('synthetic', 'all'):
        datasets = collector.collect_all_data(
            start_date=args.start_date,
            end_date=args.end_date,
            n_buildings=args.n_buildings,
            n_days=args.n_days
        )
    elif args.source == 'bdg2':
        datasets = collector.collect_bdg2_data(args.site, args.n_buildings)
    else:
        required = ['energy_file', 'metadata_file', 'latitude', 'longitude', 'timezone']
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--source file requires {', '.join(missing)}")
        datasets = collector.collect_from_files(
            args.energy_file, args.metadata_file, args.latitude, args.longitude, args.timezone
        )

    logger.info("Data collection completed successfully")
    for name, df in datasets.items():
        logger.info(f"{name} data shape: {df.shape}")


if __name__ == "__main__":
    main()
