#!/usr/bin/env python
"""
Builds and executes notebooks/02_business_summary.ipynb from the files in reports/,
so every number in the notebook comes from the latest backtest run.

    python3 notebooks/build_business_summary.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("""# LoadLens: business summary

**Question.** A facility manager runs a portfolio of buildings. Can a forecast of tomorrow's hourly
electricity use beat the rule of thumb ("same as this hour last week") by enough to matter for
energy purchasing and peak-demand planning?

**Data.** 10 real buildings at one Washington, DC site from the Building Data Genome Project 2
(hourly meters, 2016-2017) with real hourly weather from Open-Meteo.

**Test.** The model is trained on the earlier data, then makes recursive 48-hour forecasts from
about 20 start dates per building across the held-out autumn-winter 2017 period (about 200 forecasts).
Each forecast only sees data from before its start date, as it would in production. Run
`make backtest report` to regenerate everything below."""),
    code("""import json
import pandas as pd
from IPython.display import Image, Markdown, display

bt = json.load(open('../reports/backtest_metrics.json'))
biz = json.load(open('../reports/business_metrics.json'))
pd.set_option('display.precision', 1)"""),
    md("## 1. Headline result"),
    code("""lo, hi = bt['improvement_points_ci95']
display(Markdown(
    f"- **Forecast error (WAPE): {bt['ensemble_wape']:.1f}% for the model vs {bt['naive_wape']:.1f}% "
    f"for the naive baseline**, over {bt['n_forecasts']} forecasts of {bt['horizon_hours']} h "
    f"across {bt['n_buildings']} buildings.\\n"
    f"- 95% bootstrap interval for the improvement: **{lo:.1f} to {hi:.1f} percentage points** "
    f"(resampling whole forecasts, {bt['n_boot']} draws).\\n"
    f"- The model beats the baseline on **{bt['buildings_beating_naive']} of {bt['n_buildings']}** buildings.\\n"
    f"- {bt['excluded_meter_outage_hours']} forecast hours were not scored because the meter itself had failed."
))
Image('../reports/figures/example_forecast.png')"""),
    md("""## 2. The data-quality finding

The first version of this backtest made the model look barely better than the rule of thumb
(12.9% vs 13.5% WAPE, interval crossing zero). Error analysis showed why: a few forecasts
were off by 50-90%, and they all started on the **same dates in different buildings**.

The cause was in the data, not the model. Four buildings' meters drop to near zero at the same
hour and recover together days later, 17 days in 2017: a shared data-logger outage. The model
trusted the broken recent readings and forecast an empty building. The naive baseline looks a
full week back and skipped the outage by luck.

**Fix** (`src/data/quality.py`): flag a reading as an outage when it is below 5% of the
building's trailing 4-week median, repair it from the same hour a week earlier before it reaches
the model, never train on it, and never score a forecast against it. That is the step a real
deployment needs anyway, since meter feeds fail."""),
    md("## 3. Where the model is strong and where it fails"),
    code("""by_building = pd.DataFrame(biz['breakdowns']['by_building']).set_index('building_id')
by_building['improvement_pts'] = by_building.naive_wape - by_building.ensemble_wape
by_building.sort_values('ensemble_wape')"""),
    code("Image('../reports/figures/wape_by_building.png')"),
    code("""for key in ['by_forecast_day', 'by_day_type', 'by_temperature']:
    if key in biz['breakdowns']:
        display(pd.DataFrame(biz['breakdowns'][key]).set_index(pd.DataFrame(biz['breakdowns'][key]).columns[0]))"""),
    code("Image('../reports/figures/wape_by_hour.png')"),
    md("## 4. Does it find the daily peak?"),
    code("""p = biz['peak_detection']
display(Markdown(
    f"Over {p['building_days']} building-days, the forecast puts the daily peak within one hour of the real "
    f"peak **{p['peak_hour_hit_rate_pct']['ensemble']}%** of the time (baseline "
    f"{p['peak_hour_hit_rate_pct']['naive']}%). Median error in the peak's size: "
    f"**{p['median_peak_size_error_pct']['ensemble']}%** (baseline {p['median_peak_size_error_pct']['naive']}%).\\n\\n"
    "Peak demand sets demand charges, so this is the number a facility manager would act on, "
    "for example to pre-cool before the peak or shift flexible load."
))"""),
    md("""## 5. What the error is worth (scenario, not a measurement)

If energy is bought ahead of time based on the forecast, every kWh of error is settled at a
worse price. The kWh of error below are **measured** in the backtest; the price per kWh of error
is an **assumption**, shown as a range."""),
    code("""c = biz['cost_scenario']
display(Markdown(
    f"Average absolute error per building-hour: **{c['mean_abs_error_kwh_per_hour']['ensemble']} kWh** "
    f"(model) vs **{c['mean_abs_error_kwh_per_hour']['naive']} kWh** (baseline), i.e. about "
    f"**{c['annual_error_kwh_avoided_per_building']:,} kWh** less error per building per year."
))
pd.DataFrame(c['scenarios']).rename(columns={
    'price_usd_per_kwh_error': 'Assumed $ per kWh of error',
    'annual_cost_per_building_ensemble': 'Annual error cost / building, model ($)',
    'annual_cost_per_building_naive': 'Annual error cost / building, baseline ($)',
    'annual_saving_per_building': 'Saving / building ($)',
    'annual_saving_portfolio': f"Saving, {c['n_buildings']}-building portfolio ($)",
})"""),
    md("## 6. What drives the forecast (SHAP)"),
    code("""display(pd.DataFrame(biz['top_drivers']))
Image('../reports/figures/shap_importance.png')"""),
    code("""import os
if os.path.exists('../reports/figures/shap_temperature.png'):
    display(Image('../reports/figures/shap_temperature.png'))"""),
    md("""SHAP values are in units of the building's typical hour, because the model predicts consumption
relative to each building's previous-week average. That is also why one model can serve buildings
whose sizes differ by 100x.

## 7. Recommendation

1. **Use the model for day-ahead purchasing and peak planning** in buildings where it clearly beats
   the baseline (table in section 3); keep the rule of thumb as a fallback where it does not.
2. **Put the meter-outage check in front of any forecast.** It was the single largest source of error,
   and it costs nothing to run.
3. **Before acting on the savings estimate**, replace the assumed price of error with the site's real
   tariff and balancing costs; the kWh figures carry over unchanged.
4. **Next model improvement:** holiday and academic-calendar features. Seven of the ten buildings
   are education buildings, whose use follows term dates the model currently has no feature for."""),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
ExecutePreprocessor(timeout=300, kernel_name='python3').preprocess(nb, {'metadata': {'path': str(ROOT / 'notebooks')}})
out = ROOT / 'notebooks' / '02_business_summary.ipynb'
nbformat.write(nb, out)
print(f"Wrote {out}")
