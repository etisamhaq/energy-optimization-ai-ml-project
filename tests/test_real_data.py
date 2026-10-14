"""
Tests for real-data sources (BDG2, Open-Meteo, user CSV files). No network access:
HTTP calls are replaced with canned responses.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data import bdg2, open_meteo
from src.data.collect_data import DataCollector
from src.data.data_loader import EnergyDataLoader, validate_energy_data


def _open_meteo_payload(times):
    n = len(times)
    return {
        'hourly': {
            'time': times,
            'temperature_2m': [float(i) for i in range(n)],
            'relative_humidity_2m': [50.0] * n,
            'wind_speed_10m': [10.0] * n,
            'shortwave_radiation': [0.0] * n,
            'precipitation': [0.0] * n,
        }
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {'content-type': 'application/json'}
        self.text = str(payload)

    def json(self):
        return self._payload


class TestOpenMeteo:
    def test_fetch_maps_columns_and_fills_dst_gap(self, monkeypatch):
        # Spring-forward day: local 02:00 doesn't exist, so only 23 hours are returned
        times = [f"2016-03-13T{h:02d}:00" for h in range(24) if h != 2]
        monkeypatch.setattr(open_meteo.requests, 'get', lambda url, params, timeout: FakeResponse(_open_meteo_payload(times)))

        df = open_meteo.fetch_weather(38.9, -77.0, 'US/Eastern', '2016-03-13', '2016-03-13')

        assert len(df) == 24
        assert list(df.columns) == ['timestamp', 'temperature_c', 'humidity_percent', 'wind_speed_kmh',
                                    'solar_radiation_wm2', 'precipitation_mm']
        assert df['temperature_c'].notna().all()

    def test_fetch_uses_archive_cache(self, monkeypatch, tmp_path):
        times = [f"2016-01-01T{h:02d}:00" for h in range(24)]
        calls = []

        def fake_get(url, params, timeout):
            calls.append(url)
            return FakeResponse(_open_meteo_payload(times))

        monkeypatch.setattr(open_meteo.requests, 'get', fake_get)
        open_meteo.fetch_weather(38.9, -77.0, 'US/Eastern', '2016-01-01', '2016-01-01', cache_dir=tmp_path)
        open_meteo.fetch_weather(38.9, -77.0, 'US/Eastern', '2016-01-01', '2016-01-01', cache_dir=tmp_path)

        assert calls == [open_meteo.ARCHIVE_URL]

    def test_rejects_dates_beyond_forecast_horizon(self):
        too_far = (date.today() + timedelta(days=60)).isoformat()
        with pytest.raises(ValueError, match="15 days ahead"):
            open_meteo.fetch_weather(38.9, -77.0, 'US/Eastern', too_far, too_far)

    def test_api_error_becomes_value_error(self, monkeypatch):
        monkeypatch.setattr(open_meteo.requests, 'get', lambda url, params, timeout: FakeResponse(
            {'error': True, 'reason': 'out of range'}, status_code=400))
        with pytest.raises(ValueError, match="out of range"):
            open_meteo.fetch_weather(38.9, -77.0, 'US/Eastern', '2016-01-01', '2016-01-01')


class TestBDG2:
    def test_metadata_mapping(self):
        raw = pd.DataFrame({
            'building_id': ['Rat_lodging_A', 'Rat_office_B'],
            'primaryspaceusage': ['Lodging/residential', 'Office'],
            'sqm': [1000.0, 2000.0],
            'yearbuilt': [1990.0, 2005.0],
            'heatingtype': [None, 'Gas'],
            'energystarscore': [None, 80.0],
        })
        meta = bdg2.to_building_metadata(raw)

        assert meta['building_type'].tolist() == ['residential', 'office']
        assert meta['hvac_type'].tolist() == ['unknown', 'gas']
        assert meta['energy_star_rating'].tolist() == [bdg2.DEFAULT_ENERGY_STAR_RATING, 80]
        assert meta['year_built'].dtype.kind == 'i'

    def test_energy_wide_to_long(self):
        wide = pd.DataFrame({
            'timestamp': ['2016-01-01 00:00:00', '2016-01-01 01:00:00'],
            'Rat_office_B': [10.0, 11.0],
            'Rat_lodging_A': [1.0, 2.0],
        })
        long = bdg2.to_energy_data(wide)

        assert len(long) == 4
        assert set(long.columns) == {'timestamp', 'building_id', 'energy_consumption_kwh'}
        assert long['timestamp'].dtype.kind == 'M'


class TestUserData:
    def test_validate_energy_data_rejects_bad_input(self):
        with pytest.raises(ValueError, match="missing required columns"):
            validate_energy_data(pd.DataFrame({'building_id': ['A'], 'timestamp': ['2024-01-01']}))

        negative = pd.DataFrame({'building_id': ['A'], 'timestamp': ['2024-01-01'], 'energy_consumption_kwh': [-1]})
        with pytest.raises(ValueError, match="negative"):
            validate_energy_data(negative)

        duplicated = pd.DataFrame({'building_id': ['A', 'A'], 'timestamp': ['2024-01-01'] * 2,
                                   'energy_consumption_kwh': [1, 2]})
        with pytest.raises(ValueError, match="duplicate"):
            validate_energy_data(duplicated)

    def test_collect_from_files_stores_site_and_metadata(self, tmp_path, monkeypatch):
        energy = pd.DataFrame({
            'building_id': 'HQ',
            'timestamp': pd.date_range('2024-01-01', periods=48, freq='H'),
            'energy_consumption_kwh': 100.0,
        })
        energy.to_csv(tmp_path / 'energy.csv', index=False)
        pd.DataFrame([{
            'building_id': 'HQ', 'building_type': 'office', 'floor_area_sqm': 5000, 'year_built': 2000,
            'occupancy_type': 'single_tenant', 'hvac_type': 'central', 'energy_star_rating': 70
        }]).to_csv(tmp_path / 'meta.csv', index=False)

        times = [t.strftime('%Y-%m-%dT%H:%M') for t in pd.date_range('2024-01-01', periods=48, freq='H')]
        monkeypatch.setattr(open_meteo.requests, 'get', lambda url, params, timeout: FakeResponse(_open_meteo_payload(times)))

        data_dir = tmp_path / 'data'
        datasets = DataCollector(str(data_dir)).collect_from_files(
            str(tmp_path / 'energy.csv'), str(tmp_path / 'meta.csv'), 38.9, -77.0, 'US/Eastern'
        )

        assert datasets['integrated']['temperature_c'].notna().all()
        loader = EnergyDataLoader(str(data_dir))
        assert loader.load_site()['weather_source'] == 'open-meteo'
        assert loader.list_buildings() == ['HQ']
        assert loader.load_building_metadata(['HQ']).iloc[0]['building_type'] == 'office'
        with pytest.raises(ValueError, match="Unknown building"):
            loader.load_building_metadata(['NOPE'])
