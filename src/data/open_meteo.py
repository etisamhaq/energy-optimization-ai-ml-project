"""
Real hourly weather from Open-Meteo (https://open-meteo.com), free and keyless.

Past dates come from the historical archive (ERA5 reanalysis, available up to about
five days ago); recent and future dates come from the forecast API (about 92 days
back to 15 days ahead). Archive responses are cached on disk because they don't change.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Archive data lags real time by a few days; newer dates come from the forecast API
ARCHIVE_DELAY_DAYS = 6
FORECAST_HORIZON_DAYS = 15
REQUEST_TIMEOUT_SECONDS = 60

# Open-Meteo variable -> project column
VARIABLES = {
    'temperature_2m': 'temperature_c',
    'relative_humidity_2m': 'humidity_percent',
    'wind_speed_10m': 'wind_speed_kmh',
    'shortwave_radiation': 'solar_radiation_wm2',
    'precipitation': 'precipitation_mm',
}


def _request(url: str, params: dict) -> pd.DataFrame:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    payload = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
    if response.status_code != 200 or payload.get('error'):
        reason = payload.get('reason', response.text[:200])
        raise ValueError(f"Weather data unavailable for {params['start_date']}..{params['end_date']}: {reason}")

    hourly = payload['hourly']
    df = pd.DataFrame({column: hourly[variable] for variable, column in VARIABLES.items()})
    df.insert(0, 'timestamp', pd.to_datetime(hourly['time']))
    return df


def fetch_weather(latitude: float, longitude: float, timezone: str,
                  start_date: str, end_date: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Hourly weather in local time for every hour from start_date through end_date.

    Raises ValueError when the range reaches beyond the forecast horizon.
    """
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = date.today()
    if end > today + timedelta(days=FORECAST_HORIZON_DAYS):
        raise ValueError(
            f"Weather forecasts only reach {FORECAST_HORIZON_DAYS} days ahead "
            f"(until {today + timedelta(days=FORECAST_HORIZON_DAYS)}); requested end date {end}"
        )

    base_params = {
        'latitude': latitude,
        'longitude': longitude,
        'timezone': timezone,
        'hourly': ','.join(VARIABLES),
        'wind_speed_unit': 'kmh',
    }
    archive_end = min(end, today - timedelta(days=ARCHIVE_DELAY_DAYS))
    parts = []

    if start <= archive_end:
        parts.append(_fetch_archive(base_params, start, archive_end, cache_dir))
    if end > archive_end:
        forecast_start = max(start, archive_end + timedelta(days=1))
        parts.append(_request(FORECAST_URL, {
            **base_params, 'start_date': forecast_start.isoformat(), 'end_date': end.isoformat()
        }))

    df = pd.concat(parts, ignore_index=True)

    # Local time repeats an hour when DST ends and skips one when it starts:
    # keep one row per timestamp and interpolate onto a regular hourly grid
    hours = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq='H')
    df = df.drop_duplicates('timestamp').set_index('timestamp').reindex(hours)
    df = df.interpolate(limit_direction='both').rename_axis('timestamp').reset_index()
    return df


def _fetch_archive(base_params: dict, start: date, end: date, cache_dir: Optional[Path]) -> pd.DataFrame:
    cache_file = None
    if cache_dir is not None:
        key = f"{base_params['latitude']:.4f}_{base_params['longitude']:.4f}_{base_params['timezone'].replace('/', '-')}"
        cache_file = Path(cache_dir) / f"open_meteo_{key}_{start}_{end}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=['timestamp'])

    logger.info(f"Fetching archived weather {start}..{end} from Open-Meteo")
    df = _request(ARCHIVE_URL, {**base_params, 'start_date': start.isoformat(), 'end_date': end.isoformat()})

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
    return df
