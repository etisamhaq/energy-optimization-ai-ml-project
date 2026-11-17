#!/usr/bin/env python
"""
Rolling-origin backtest: the experiment behind the headline forecast numbers.

For every building, the trained model makes recursive 48-hour forecasts starting at
midnight on dates spread evenly across the held-out test period. Each forecast only
sees observed consumption before its start, exactly as it would in production. The
naive baseline predicts the same hour one week earlier.

Writes reports/backtest_predictions.csv (one row per forecast hour) and
reports/backtest_metrics.json, which the business analysis and README read.

    python3 src/evaluation/backtest.py --origins-per-building 20
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict import EnergyPredictor  # noqa: E402
from src.data.quality import flag_outages, repair_outages  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HORIZON_HOURS = 48
NAIVE_LAG_HOURS = 168


def wape(actual, predicted) -> float:
    """Total absolute error as a percentage of total consumption."""
    actual = np.asarray(actual, dtype=float)
    return float(np.abs(actual - np.asarray(predicted, dtype=float)).sum() / np.abs(actual).sum() * 100)


def forecast_origins(test_start: pd.Timestamp, test_end: pd.Timestamp, n: int) -> list:
    """n midnights spread evenly over the test period, each leaving room for a full horizon."""
    first = test_start.normalize() + pd.Timedelta(days=1)
    last = (test_end - pd.Timedelta(hours=HORIZON_HOURS - 1)).normalize()
    if last < first:
        raise ValueError("Test period is shorter than one forecast horizon")
    days = pd.date_range(first, last, freq='D')
    picks = np.linspace(0, len(days) - 1, n).round().astype(int)
    return [days[i] for i in sorted(set(picks))]


def run_backtest(data_dir: str, models_dir: str, origins_per_building: int,
                 model_name: str = 'ensemble') -> pd.DataFrame:
    predictor = EnergyPredictor(models_dir, data_dir)
    test_files = sorted(Path(data_dir, 'processed').glob('processed_test_*.csv'),
                        key=lambda p: p.stat().st_mtime)
    if not test_files:
        raise FileNotFoundError("No processed test split. Run preprocess.py first.")
    test = pd.read_csv(test_files[-1], usecols=['timestamp'], parse_dates=['timestamp'])
    origins = forecast_origins(test['timestamp'].min(), test['timestamp'].max(), origins_per_building)
    logger.info(f"Test period {test['timestamp'].min()} to {test['timestamp'].max()}; "
                f"origins: {[o.date().isoformat() for o in origins]}")

    buildings = predictor.data_loader.list_buildings()
    rows = []
    for building_id in buildings:
        actual = predictor.load_consumption_history(building_id)
        actual = actual[~actual.index.duplicated()].asfreq('H')
        # Hours where the meter itself failed can't score a forecast; the naive baseline
        # gets the same repaired history the model sees
        outage = flag_outages(actual)
        repaired = repair_outages(actual.interpolate(limit_direction='both'), outage)
        for origin in origins:
            end = origin + pd.Timedelta(hours=HORIZON_HOURS - 1)
            forecast, _ = predictor.predict_range(
                building_id, origin.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), model_name)
            forecast = forecast.set_index('timestamp')['predicted_consumption_kwh'].loc[origin:end]
            frame = pd.DataFrame({
                'building_id': building_id,
                'origin': origin,
                'timestamp': forecast.index,
                'horizon_hour': np.arange(1, len(forecast) + 1),
                'actual_kwh': actual.reindex(forecast.index).to_numpy(),
                'forecast_kwh': forecast.to_numpy(),
                'naive_kwh': repaired.reindex(forecast.index - pd.Timedelta(hours=NAIVE_LAG_HOURS)).to_numpy(),
                'meter_outage': outage.reindex(forecast.index).fillna(False).to_numpy(),
            })
            rows.append(frame)
        logger.info(f"{building_id}: {len(origins)} forecasts done")

    results = pd.concat(rows, ignore_index=True)
    outage_hours = int(results['meter_outage'].sum())
    if outage_hours:
        logger.warning(f"Not scoring {outage_hours} forecast hours where the meter had failed")
    results = results[~results['meter_outage']].drop(columns='meter_outage')
    results.attrs['outage_hours'] = outage_hours
    dropped = results[['actual_kwh', 'naive_kwh']].isna().any(axis=1).sum()
    if dropped:
        logger.warning(f"Dropping {dropped} forecast hours with a missing meter reading")
    return results.dropna(subset=['actual_kwh', 'naive_kwh']).reset_index(drop=True)


def bootstrap_ci(results: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    95% intervals for both WAPEs and their gap, resampling whole forecasts (building x
    origin) because hours inside one forecast are not independent.
    """
    rng = np.random.default_rng(seed)
    sums = results.assign(
        err_model=(results.actual_kwh - results.forecast_kwh).abs(),
        err_naive=(results.actual_kwh - results.naive_kwh).abs(),
        total=results.actual_kwh.abs(),
    ).groupby(['building_id', 'origin'])[['err_model', 'err_naive', 'total']].sum().to_numpy()
    draws = sums[rng.integers(0, len(sums), size=(n_boot, len(sums)))].sum(axis=1)
    model = draws[:, 0] / draws[:, 2] * 100
    naive = draws[:, 1] / draws[:, 2] * 100
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 2), round(float(np.percentile(a, 97.5)), 2)]
    return {'ensemble_wape_ci95': ci(model), 'naive_wape_ci95': ci(naive),
            'improvement_points_ci95': ci(naive - model), 'n_boot': n_boot}


def summarize(results: pd.DataFrame) -> dict:
    per_building = results.groupby('building_id').apply(
        lambda g: pd.Series({'ensemble_wape': wape(g.actual_kwh, g.forecast_kwh),
                             'naive_wape': wape(g.actual_kwh, g.naive_kwh)}))
    return {
        **bootstrap_ci(results),
        'n_buildings': int(results['building_id'].nunique()),
        'n_forecasts': int(results.groupby(['building_id', 'origin']).ngroups),
        'horizon_hours': HORIZON_HOURS,
        'forecast_hours': int(len(results)),
        'excluded_meter_outage_hours': int(results.attrs.get('outage_hours', 0)),
        'origins': sorted(pd.to_datetime(results['origin']).dt.date.astype(str).unique().tolist()),
        'ensemble_wape': wape(results.actual_kwh, results.forecast_kwh),
        'naive_wape': wape(results.actual_kwh, results.naive_kwh),
        'buildings_beating_naive': int((per_building.ensemble_wape < per_building.naive_wape).sum()),
        'per_building': per_building.round(2).to_dict(orient='index'),
    }


def main():
    parser = argparse.ArgumentParser(description='Rolling-origin backtest of the forecasting model')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--origins-per-building', type=int, default=20)
    parser.add_argument('--model', default='ensemble')
    args = parser.parse_args()

    results = run_backtest(args.data_dir, args.models_dir, args.origins_per_building, args.model)
    summary = summarize(results)

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / 'backtest_predictions.csv', index=False)
    with open(reports / 'backtest_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{summary['n_forecasts']} forecasts x {HORIZON_HOURS} h over {summary['n_buildings']} buildings")
    print(f"Ensemble WAPE: {summary['ensemble_wape']:.1f}%   Naive WAPE: {summary['naive_wape']:.1f}%")
    print(f"95% CI: ensemble {summary['ensemble_wape_ci95']}, naive {summary['naive_wape_ci95']}, "
          f"improvement {summary['improvement_points_ci95']} points")
    print(f"Ensemble beats naive on {summary['buildings_beating_naive']}/{summary['n_buildings']} buildings")


if __name__ == '__main__':
    main()
