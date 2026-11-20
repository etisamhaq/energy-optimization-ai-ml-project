"""
Generates the MkDocs source folder (docs_site/) from the repository's own files, so the documentation site can
never drift from the README or the saved backtest:

    README.md                      -> index.md
    DEPLOYMENT.md                  -> deployment.md
    reports/*.json + figures/*.png -> results.md   (tables written from the saved metrics)
    images/*                       -> images/*

Relative links to other files become links to the file on GitHub.

    python tools/build_docs.py && mkdocs build --strict
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs_site"
GITHUB = "https://github.com/etisamhaq/energy-optimization-ai-ml-project"
RESULTS_PAGE = "https://energy-optimization-ai-ml-project.vercel.app/"
PAGES = {"README.md": "index.md", "DEPLOYMENT.md": "deployment.md"}

LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")


def rewrite_links(text: str, source: str) -> str:
    def fix(match: re.Match) -> str:
        label, href = match.groups()
        if href.startswith("#"):  # GitHub slugs "a & b" as "a--b", MkDocs as "a-b"
            return f"{label}({re.sub('-{2,}', '-', href)})"
        if re.match(r"^(https?:|mailto:)", href):
            return match.group(0)
        path, _, anchor = href.partition("#")
        suffix = f"#{anchor}" if anchor else ""
        if path in PAGES:
            return f"{label}({PAGES[path]}{suffix})"
        if path.startswith("images/") and (ROOT / path).is_file():
            return match.group(0)
        if (ROOT / path).exists():
            kind = "tree" if (ROOT / path).is_dir() else "blob"
            return f"{label}({GITHUB}/{kind}/master/{path}{suffix})"
        raise FileNotFoundError(f"Broken link in {source}: {href}")

    return LINK.sub(fix, text)


def table(rows: list[dict], key: str, label: str) -> str:
    lines = [f"| {label} | Ensemble WAPE | Naive WAPE | Hours scored |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[key]} | {r['ensemble_wape']:.1f}% | {r['naive_wape']:.1f}% | {int(r['hours']):,} |")
    return "\n".join(lines)


def results_page() -> str:
    bt = json.loads((ROOT / "reports" / "backtest_metrics.json").read_text())
    bm = json.loads((ROOT / "reports" / "business_metrics.json").read_text())
    br, peak, cost = bm["breakdowns"], bm["peak_detection"], bm["cost_scenario"]
    lo, hi = bt["ensemble_wape_ci95"], bt["improvement_points_ci95"]
    nlo = bt["naive_wape_ci95"]
    hours = [r for r in br["by_hour"]]
    worst_hour = max(hours, key=lambda r: r["ensemble_wape"])
    best_hour = min(hours, key=lambda r: r["ensemble_wape"])

    cost_rows = "\n".join(
        f"| ${s['price_usd_per_kwh_error']:.2f} | ${s['annual_cost_per_building_ensemble']:,} | "
        f"${s['annual_cost_per_building_naive']:,} | ${s['annual_saving_per_building']:,} | "
        f"${s['annual_saving_portfolio']:,} |"
        for s in cost["scenarios"]
    )
    drivers = "\n".join(
        f"| {i} | `{d['feature']}` | {d['mean_abs_shap']:.4f} |" for i, d in enumerate(bm["top_drivers"], 1)
    )

    return f"""# Results

Every number on this page is written by `tools/build_docs.py` from the saved backtest files
([`reports/backtest_metrics.json`]({GITHUB}/blob/master/reports/backtest_metrics.json) and
[`reports/business_metrics.json`]({GITHUB}/blob/master/reports/business_metrics.json)), so it always matches the
committed results. `make backtest report` regenerates them. For an interactive version, see the
[results page]({RESULTS_PAGE}).

## How the test was run

- **Data:** {bt['n_buildings']} real buildings in Washington, DC (Building Data Genome Project 2) with real hourly weather.
- **Split:** models trained on 2016 to mid-2017, then made **{bt['n_forecasts']} recursive {bt['horizon_hours']}-hour forecasts**
  ({len(bt['origins'])} start dates per building, {bt['origins'][0]} to {bt['origins'][-1]}) on held-out data. No forecast sees the future.
- **Metric:** WAPE, total absolute error divided by total consumption, over {bt['forecast_hours']:,} forecast hours.
  {bt['excluded_meter_outage_hours']} hours where the meter itself had failed are not scored.
- **Uncertainty:** 95% intervals from {bt['n_boot']:,} bootstrap resamples of whole forecasts, because hours inside
  one forecast are not independent.

## Headline

| Model | WAPE | 95% interval |
|---|---|---|
| **Ensemble** (XGBoost + LightGBM + Random Forest) | **{bt['ensemble_wape']:.1f}%** | {lo[0]:.1f}–{lo[1]:.1f}% |
| Naive baseline (same hour last week) | {bt['naive_wape']:.1f}% | {nlo[0]:.1f}–{nlo[1]:.1f}% |

The ensemble's error is lower by **{bt['naive_wape'] - bt['ensemble_wape']:.1f} points** (95% interval {hi[0]:.1f} to
{hi[1]:.1f}), and it beats the baseline on **{bt['buildings_beating_naive']} of {bt['n_buildings']} buildings**.

![Example 48-hour forecast](figures/example_forecast.png)

## Where the error is

### By building

{table(br['by_building'], 'building_id', 'Building')}

Best: `{bm['best_building']}`. Worst: `{bm['worst_building']}`.

![WAPE by building](figures/wape_by_building.png)

### By forecast horizon

{table(br['by_forecast_day'], 'forecast_day', 'Forecast day')}

Error grows on the second day because each forecast feeds on its own earlier predictions.

![WAPE by horizon](figures/wape_by_horizon.png)

### By hour of day

The ensemble is most accurate at hour {best_hour['hour']} ({best_hour['ensemble_wape']:.1f}%) and least accurate at
hour {worst_hour['hour']} ({worst_hour['ensemble_wape']:.1f}%).

![WAPE by hour](figures/wape_by_hour.png)

### By day type and temperature

{table(br['by_day_type'], 'day_type', 'Day type')}

{table(br['by_temperature'], 'temperature_band', 'Outdoor temperature')}

![WAPE by day type](figures/wape_by_day_type.png)

![WAPE by temperature](figures/wape_by_temperature.png)

## Daily peaks

Over {peak['building_days']} building-days. {peak['definition']}

| | Ensemble | Naive |
|---|---|---|
| Peak hour hit rate | {peak['peak_hour_hit_rate_pct']['ensemble']:.1f}% | {peak['peak_hour_hit_rate_pct']['naive']:.1f}% |
| Median error in peak size | {peak['median_peak_size_error_pct']['ensemble']:.1f}% | {peak['median_peak_size_error_pct']['naive']:.1f}% |

!!! warning "Limitation"
    Peak *size* is forecast well, but the exact peak *hour* is hit only about a third of the time. The model is
    not yet good enough to time demand-response actions to a specific hour.

## Cost of forecast error

The ensemble's mean absolute error is {cost['mean_abs_error_kwh_per_hour']['ensemble']:.1f} kWh per hour against
{cost['mean_abs_error_kwh_per_hour']['naive']:.1f} for the baseline, about
**{cost['annual_error_kwh_avoided_per_building']:,} kWh of error avoided per building per year**.
What a kWh of error costs depends on the tariff and how forecasts are used, so it is shown as a scenario:

| Cost per kWh of error | Annual cost per building (ensemble) | (naive) | Saving per building | Saving, {cost['n_buildings']} buildings |
|---|---|---|---|---|
{cost_rows}

*{cost['note']}*

## What drives the forecasts (SHAP)

| Rank | Feature | Mean \\|SHAP\\| |
|---|---|---|
{drivers}

Recent consumption (1 hour, 1 day and 1 week ago) dominates; time of day comes next.

![SHAP importance](figures/shap_importance.png)

![SHAP temperature dependence](figures/shap_temperature.png)
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for source, target in PAGES.items():
        (OUT / target).write_text(rewrite_links((ROOT / source).read_text(), source))
    (OUT / "results.md").write_text(results_page())
    shutil.copytree(ROOT / "images", OUT / "images")
    shutil.copytree(ROOT / "reports" / "figures", OUT / "figures")
    print(f"Generated {len(PAGES) + 1} pages in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
