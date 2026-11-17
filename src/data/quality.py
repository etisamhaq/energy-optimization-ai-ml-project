"""
Meter-outage detection and repair.

Real meter feeds fail. In the BDG2 Washington, DC site, four buildings drop to near
zero at the same hour and recover together days later (17 days in 2017) - a shared
logger outage, not real consumption. Left in, those readings poison every forecast
whose lag features see them and every score computed against them.

A reading counts as an outage when it is below OUTAGE_FRACTION of the building's
trailing median. The median window (4 weeks) is long so that a multi-day outage
cannot drag the median down to its own level and hide itself; it only uses past
readings, so the check works the same way live as in a backtest.
"""

import numpy as np
import pandas as pd

OUTAGE_FRACTION = 0.05
REFERENCE_WINDOW_HOURS = 28 * 24
MIN_REFERENCE_HOURS = 48
SEASONAL_LAG_HOURS = 168


def flag_outages(series: pd.Series) -> pd.Series:
    """True where an hourly reading is implausibly low against the building's recent level."""
    reference = series.shift(1).rolling(REFERENCE_WINDOW_HOURS, min_periods=MIN_REFERENCE_HOURS).median()
    return (series < OUTAGE_FRACTION * reference).fillna(False)


def repair_outages(series: pd.Series, mask: pd.Series = None) -> pd.Series:
    """
    Replace outage readings with the same hour one week earlier (repeating back further
    if that hour was also an outage), falling back to the trailing median.

    The series must be on a regular hourly index so a 168-step shift is one week.
    """
    if mask is None:
        mask = flag_outages(series)
    repaired = series.where(~mask)
    for weeks in range(1, 5):
        repaired = repaired.fillna(repaired.shift(SEASONAL_LAG_HOURS * weeks))
    reference = series.where(~mask).rolling(REFERENCE_WINDOW_HOURS, min_periods=1).median()
    return repaired.fillna(reference).fillna(series)


def flag_outages_by_building(df: pd.DataFrame, value_col: str = 'energy_consumption_kwh') -> pd.Series:
    """flag_outages applied per building; df must be sorted by building then timestamp."""
    return df.groupby('building_id', group_keys=False)[value_col].apply(flag_outages).astype(bool)


def repair_outages_by_building(df: pd.DataFrame, value_col: str = 'energy_consumption_kwh') -> pd.DataFrame:
    """Repair outage readings per building; adds a boolean 'meter_outage' column."""
    df = df.sort_values(['building_id', 'timestamp']).copy()
    df['meter_outage'] = flag_outages_by_building(df, value_col)
    for _, idx in df.groupby('building_id').groups.items():
        df.loc[idx, value_col] = repair_outages(df.loc[idx, value_col], df.loc[idx, 'meter_outage']).to_numpy()
    return df


def outage_share(mask: pd.Series) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0
