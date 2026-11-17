#!/usr/bin/env python
"""
Builds site/data.json for the static results page (site/index.html) from reports/.

The page is static so it can be hosted anywhere (Vercel serves the site/ folder):
no model runs in the browser or on the server, every number comes from the latest
`make backtest report` run.

    python3 src/evaluation/build_site.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.quality import flag_outages, repair_outages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports'
SITE = ROOT / 'site'

# The first backtest, before meter outages were handled (17 Nov 2026, same 200 forecasts,
# model trained on unrepaired data). Kept so the page can show what the fix changed.
BEFORE_OUTAGE_FIX = {'ensemble_wape': 12.9, 'naive_wape': 13.5, 'improvement_points_ci95': [-3.66, 4.11],
                     'buildings_beating_naive': 7}

OUTAGE_EXAMPLE = {'building_id': 'Rat_office_Colby', 'start': '2017-08-24', 'end': '2017-09-04'}


def short_name(building_id: str) -> str:
    """Rat_education_Kelsey -> Kelsey (education)"""
    _, use, name = building_id.split('_', 2)
    return f"{name} ({use})"


def forecasts(results: pd.DataFrame) -> dict:
    out = {}
    for building, g in results.groupby('building_id'):
        out[building] = {}
        for origin, f in g.groupby('origin'):
            f = f.sort_values('horizon_hour')
            out[building][str(origin)[:10]] = {
                'h': f['horizon_hour'].tolist(),
                'actual': f['actual_kwh'].round(1).tolist(),
                'model': f['forecast_kwh'].round(1).tolist(),
                'naive': f['naive_kwh'].round(1).tolist(),
            }
    return out


def outage_example() -> dict:
    files = sorted((ROOT / 'data' / 'processed').glob('integrated_data_*.csv'))
    df = pd.read_csv(files[-1], usecols=['building_id', 'timestamp', 'energy_consumption_kwh'],
                     parse_dates=['timestamp'])
    s = (df[df.building_id == OUTAGE_EXAMPLE['building_id']]
         .drop_duplicates('timestamp').set_index('timestamp')['energy_consumption_kwh'].asfreq('H')
         .interpolate(limit_direction='both'))
    mask = flag_outages(s)
    repaired = repair_outages(s, mask)
    window = slice(OUTAGE_EXAMPLE['start'], OUTAGE_EXAMPLE['end'])
    return {
        'building': short_name(OUTAGE_EXAMPLE['building_id']),
        't': [t.strftime('%Y-%m-%d %H:00') for t in s.loc[window].index],
        'raw': s.loc[window].round(1).tolist(),
        'repaired': repaired.loc[window].round(1).tolist(),
        'outage_hours_total': int(mask.sum()),
    }


def main():
    backtest = json.load(open(REPORTS / 'backtest_metrics.json'))
    business = json.load(open(REPORTS / 'business_metrics.json'))
    results = pd.read_csv(REPORTS / 'backtest_predictions.csv', parse_dates=['origin'])
    shap = pd.read_csv(REPORTS / 'shap_importance.csv').head(12)

    # Open the viewer on the median forecast, neither a best case nor a worst case
    per_forecast = results.assign(err=(results.actual_kwh - results.forecast_kwh).abs()).groupby(
        ['building_id', 'origin']).apply(lambda g: g.err.sum() / g.actual_kwh.sum()).sort_values()
    median_building, median_origin = per_forecast.index[len(per_forecast) // 2]

    data = {
        'default_forecast': {'building': median_building, 'origin': str(median_origin)[:10]},
        'headline': {k: backtest[k] for k in (
            'ensemble_wape', 'naive_wape', 'ensemble_wape_ci95', 'naive_wape_ci95', 'improvement_points_ci95',
            'n_buildings', 'n_forecasts', 'horizon_hours', 'forecast_hours', 'buildings_beating_naive',
            'excluded_meter_outage_hours')},
        'before_fix': BEFORE_OUTAGE_FIX,
        'buildings': {b: short_name(b) for b in sorted(results.building_id.unique())},
        'forecasts': forecasts(results),
        'breakdowns': business['breakdowns'],
        'peak': business['peak_detection'],
        'cost': business['cost_scenario'],
        'shap': shap.round(4).to_dict(orient='records'),
        'outage': outage_example(),
    }
    SITE.mkdir(exist_ok=True)
    with open(SITE / 'data.json', 'w') as f:
        json.dump(data, f, separators=(',', ':'), default=str)
    print(f"Wrote {SITE / 'data.json'} ({(SITE / 'data.json').stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
