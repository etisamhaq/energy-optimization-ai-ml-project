"""
Real building energy data from the Building Data Genome Project 2 (BDG2).

BDG2 holds two years (2016-2017) of hourly electricity meter readings for about 1,600
non-residential buildings, with building metadata. Licensed CC BY 4.0:
https://github.com/buds-lab/building-data-genome-project-2

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome
Project 2, energy meter data from the ASHRAE Great Energy Predictor III
competition. Sci Data 7, 368 (2020).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {
    'metadata': "metadata/metadata.csv",
    'electricity': "meters/cleaned/electricity_cleaned.csv",
}

# Buildings with more missing hours than this are skipped
MAX_MISSING_FRACTION = 0.05

# Energy Star scores are missing for most BDG2 buildings; use the midpoint of the 1-100 scale
DEFAULT_ENERGY_STAR_RATING = 50

BUILDING_TYPE_MAP = {
    'lodging/residential': 'residential',
    'manufacturing/industrial': 'industrial',
}


def download(name: str, cache_dir: Path) -> Path:
    """Download a BDG2 file once and return its local path."""
    path = Path(cache_dir) / Path(FILES[name]).name
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{FILES[name]}"
    logger.info(f"Downloading {url}")
    partial = path.with_suffix('.part')
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(partial, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    partial.rename(path)
    return path


def to_building_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    """Map BDG2 metadata columns onto the project's building metadata schema."""
    building_type = raw['primaryspaceusage'].fillna('unknown').str.lower()
    return pd.DataFrame({
        'building_id': raw['building_id'],
        'building_type': building_type.replace(BUILDING_TYPE_MAP),
        'floor_area_sqm': raw['sqm'].astype(float),
        'year_built': raw['yearbuilt'].astype(int),
        'occupancy_type': 'unknown',
        'hvac_type': raw['heatingtype'].fillna('unknown').str.lower(),
        'energy_star_rating': raw['energystarscore'].fillna(DEFAULT_ENERGY_STAR_RATING).astype(int),
    }).reset_index(drop=True)


def to_energy_data(meters: pd.DataFrame) -> pd.DataFrame:
    """Convert BDG2's wide meter table (one column per building) to long format."""
    long = meters.melt(id_vars='timestamp', var_name='building_id', value_name='energy_consumption_kwh')
    long['timestamp'] = pd.to_datetime(long['timestamp'])
    return long.sort_values(['building_id', 'timestamp']).reset_index(drop=True)


def load_site(site_id: str, n_buildings: int, cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load real electricity data for up to n_buildings buildings at one BDG2 site.

    Returns (energy_df, metadata_df, site), where site holds the location used to
    fetch matching weather. Timestamps are local time at the site.
    """
    metadata = pd.read_csv(download('metadata', cache_dir))
    candidates = metadata[
        (metadata['site_id'] == site_id)
        & metadata['electricity'].notna()
        & metadata['sqm'].notna()
        & metadata['yearbuilt'].notna()
    ]
    if candidates.empty:
        sites = sorted(metadata.loc[metadata['yearbuilt'].notna(), 'site_id'].unique())
        raise ValueError(f"No usable buildings at BDG2 site '{site_id}'. Sites with metadata: {sites}")

    site_info = candidates.iloc[0]
    if pd.isna(site_info['lat']):
        raise ValueError(f"BDG2 site '{site_id}' has no coordinates, so weather can't be matched")

    meters = pd.read_csv(
        download('electricity', cache_dir),
        usecols=lambda c: c == 'timestamp' or c in set(candidates['building_id'])
    )

    # Prefer the most complete meters; skip buildings with long gaps or mostly-zero readings
    readings = meters.drop(columns='timestamp')
    missing = readings.isna().mean()
    usable = (missing <= MAX_MISSING_FRACTION) & ((readings > 0).mean() >= 1 - MAX_MISSING_FRACTION)
    selected = missing[usable].sort_values(kind='stable').index[:n_buildings].tolist()
    if not selected:
        raise ValueError(f"No buildings at site '{site_id}' have complete enough electricity data")
    logger.info(f"Selected {len(selected)} buildings at site {site_id}")

    energy = to_energy_data(meters[['timestamp'] + selected])
    building_metadata = to_building_metadata(candidates.set_index('building_id').loc[selected].reset_index())
    site = {
        'name': f"BDG2 {site_id}",
        'latitude': float(site_info['lat']),
        'longitude': float(site_info['lng']),
        'timezone': site_info['timezone'],
        'weather_source': 'open-meteo',
    }
    return energy, building_metadata, site
