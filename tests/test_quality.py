"""Tests for meter-outage detection and repair."""

import numpy as np
import pandas as pd

from src.data.quality import flag_outages, repair_outages, repair_outages_by_building


def _daily_profile(days: int, base: float = 100.0) -> pd.Series:
    index = pd.date_range('2017-01-01', periods=days * 24, freq='H')
    hours = index.hour.to_numpy()
    values = base + 50 * np.sin(np.pi * hours / 24)
    return pd.Series(values, index=index)


def test_multi_day_outage_is_flagged_throughout():
    series = _daily_profile(40)
    outage = slice(30 * 24, 34 * 24)          # four days near zero, like the BDG2 logger failure
    series.iloc[outage] = 0.5

    flags = flag_outages(series)

    assert flags.iloc[outage].all()
    assert not flags.iloc[:30 * 24].any()
    assert not flags.iloc[34 * 24:].any()


def test_normal_low_hours_are_not_flagged():
    series = _daily_profile(40)
    series.iloc[35 * 24:36 * 24] *= 0.3       # a quiet holiday is real consumption, not a fault

    assert not flag_outages(series).any()


def test_repair_uses_same_hour_last_week():
    series = _daily_profile(40)
    original = series.copy()
    series.iloc[30 * 24:32 * 24] = 0.0

    repaired = repair_outages(series)

    np.testing.assert_allclose(repaired.iloc[30 * 24:32 * 24], original.iloc[23 * 24:25 * 24])
    pd.testing.assert_series_equal(repaired.iloc[:30 * 24], original.iloc[:30 * 24])


def test_repair_by_building_marks_and_fixes_only_the_failed_meter():
    healthy = _daily_profile(40)
    broken = _daily_profile(40, base=500.0)
    broken.iloc[30 * 24:31 * 24] = 0.0
    df = pd.concat([
        pd.DataFrame({'building_id': 'A', 'timestamp': healthy.index, 'energy_consumption_kwh': healthy.values}),
        pd.DataFrame({'building_id': 'B', 'timestamp': broken.index, 'energy_consumption_kwh': broken.values}),
    ])

    out = repair_outages_by_building(df)

    assert out.loc[out.building_id == 'A', 'meter_outage'].sum() == 0
    assert out.loc[out.building_id == 'B', 'meter_outage'].sum() == 24
    assert (out.loc[out.building_id == 'B', 'energy_consumption_kwh'] > 100).all()
