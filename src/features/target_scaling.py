"""
Per-building target scaling.

Buildings differ in consumption by orders of magnitude, so one model predicting raw
kWh pulls small buildings toward the average. Instead the model predicts consumption
relative to the building's own mean over the previous week; that scale only uses past
values, so it is available at prediction time and leaks nothing.
"""

import re
from typing import List

import numpy as np
import pandas as pd

# Model features computed from past consumption, e.g. energy_consumption_kwh_lag_24
TARGET_HISTORY_FEATURE = re.compile(r'energy_consumption_kwh_(lag|rolling_mean|rolling_std)_(\d+)')

SCALE_FEATURE = 'energy_consumption_kwh_rolling_mean_168'

# Floor for the scale so near-zero meters don't divide by zero
MIN_SCALE = 1e-3


def target_history_columns(columns) -> List[str]:
    return [c for c in columns if TARGET_HISTORY_FEATURE.fullmatch(c)]


def get_scale(X: pd.DataFrame) -> np.ndarray:
    """Each row's scale: the building's mean consumption over the previous week."""
    return np.maximum(X[SCALE_FEATURE].to_numpy(dtype=float), MIN_SCALE)


def scale_features(X: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    """Express lag and rolling consumption features relative to the scale."""
    X = X.copy()
    columns = target_history_columns(X.columns)
    X[columns] = X[columns].div(scale, axis=0)
    return X
