#!/usr/bin/env python
"""
Business analysis of the backtest: where the forecast fails, whether it finds each
day's peak, what its error would cost, and what drives its predictions.

Reads reports/backtest_predictions.csv (from backtest.py) plus the trained model and
test split, and writes reports/business_metrics.json and reports/figures/*.png.

Everything here is computed from the backtest, with one exception: the cost scenario
multiplies measured forecast error by an *assumed* price per kWh of error. That price
is a parameter, reported with a sensitivity range, never a measured saving.

    python3 src/evaluation/business_impact.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.backtest import wape  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Scenario prices for one kWh of forecast error (imbalance / balancing cost when energy is
# bought ahead on the forecast). Assumptions for illustration, not a tariff.
ERROR_PRICES_USD_PER_KWH = [0.02, 0.05, 0.10]
HOURS_PER_YEAR = 8760

# Chart styling: reference data-viz palette (light mode), recessive axes
ENSEMBLE_COLOR = '#2a78d6'
NAIVE_COLOR = '#eb6834'
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'


# ---------------------------------------------------------------- error analysis

def wape_by(results: pd.DataFrame, key) -> pd.DataFrame:
    """Ensemble and naive WAPE for each group."""
    return results.groupby(key).apply(lambda g: pd.Series({
        'ensemble_wape': wape(g.actual_kwh, g.forecast_kwh),
        'naive_wape': wape(g.actual_kwh, g.naive_kwh),
        'hours': len(g),
    }))


def add_context(results: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_type'] = np.where(df['timestamp'].dt.dayofweek >= 5, 'weekend', 'weekday')
    df['forecast_day'] = np.where(df['horizon_hour'] <= 24, 'day 1 (0-24 h)', 'day 2 (25-48 h)')
    if weather is not None:
        df = df.merge(weather[['timestamp', 'temperature_c']], on='timestamp', how='left')
        df['temperature_band'] = pd.cut(df['temperature_c'], [-50, 5, 15, 25, 50],
                                        labels=['< 5 °C', '5-15 °C', '15-25 °C', '> 25 °C'])
    return df


# ---------------------------------------------------------------- peak analysis

def peak_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each building and forecast day: does the forecast find the hour of the daily
    peak (within one hour), and how far off is the peak's size? Peak demand drives
    demand charges, so this is the number a facility manager acts on.
    """
    df = results.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    rows = []
    for (building, date), g in df.groupby(['building_id', 'date']):
        if len(g) < 24:
            continue
        g = g.reset_index(drop=True)
        actual_hour = int(g.actual_kwh.idxmax())
        row = {'building_id': building, 'date': date, 'actual_peak_kwh': g.actual_kwh.max()}
        for name, col in [('ensemble', 'forecast_kwh'), ('naive', 'naive_kwh')]:
            row[f'{name}_hour_hit'] = abs(int(g[col].idxmax()) - actual_hour) <= 1
            row[f'{name}_peak_error_pct'] = abs(g[col].max() - g.actual_kwh.max()) / g.actual_kwh.max() * 100
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cost scenario

def cost_scenario(results: pd.DataFrame) -> dict:
    """
    Annual cost of forecast error per building at each scenario price, for the model
    and the naive baseline. Mean absolute hourly error is measured; the price is not.
    """
    n_buildings = results['building_id'].nunique()
    mae_model = (results.actual_kwh - results.forecast_kwh).abs().mean()
    mae_naive = (results.actual_kwh - results.naive_kwh).abs().mean()
    annual_kwh_model = mae_model * HOURS_PER_YEAR
    annual_kwh_naive = mae_naive * HOURS_PER_YEAR
    return {
        'n_buildings': int(n_buildings),
        'mean_abs_error_kwh_per_hour': {'ensemble': round(mae_model, 2), 'naive': round(mae_naive, 2)},
        'annual_error_kwh_per_building': {'ensemble': round(annual_kwh_model), 'naive': round(annual_kwh_naive)},
        'annual_error_kwh_avoided_per_building': round(annual_kwh_naive - annual_kwh_model),
        'scenarios': [
            {
                'price_usd_per_kwh_error': p,
                'annual_cost_per_building_ensemble': round(annual_kwh_model * p),
                'annual_cost_per_building_naive': round(annual_kwh_naive * p),
                'annual_saving_per_building': round((annual_kwh_naive - annual_kwh_model) * p),
                'annual_saving_portfolio': round((annual_kwh_naive - annual_kwh_model) * p * n_buildings),
            }
            for p in ERROR_PRICES_USD_PER_KWH
        ],
        'note': 'Error kWh are measured in the backtest; $/kWh of error is a scenario assumption.',
    }


# ---------------------------------------------------------------- explainability

def shap_importance(data_dir: str, models_dir: str, sample: int, seed: int = 42) -> pd.DataFrame:
    """
    Mean |SHAP| per feature for the ensemble on a sample of the test split.

    The ensemble is a weighted average of tree models, and SHAP values are additive,
    so the ensemble's SHAP values are the same weighted average of each model's.
    Units: the model predicts consumption relative to the building's previous-week
    mean, so a value of 0.10 moves the forecast by 10% of a typical hour.
    """
    import shap
    from src.models.train import ModelTrainer
    from src.models.predict import EnergyPredictor

    trainer = ModelTrainer(data_dir, models_dir)
    datasets = trainer.load_processed_data()
    prepared = trainer.prepare_data_for_training(datasets, trainer.load_feature_info())
    X = prepared['X_test'].sample(min(sample, len(prepared['X_test'])), random_state=seed)

    ensemble = EnergyPredictor(models_dir, data_dir).load_latest_model('ensemble')
    combined = np.zeros(X.shape)
    for name, weight in ensemble.weights.items():
        estimator = ensemble.models[name].model
        logger.info(f"SHAP for {name} (weight {weight:.3f}) on {len(X)} rows")
        combined += weight * shap.TreeExplainer(estimator).shap_values(X, check_additivity=False)
    importance = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': np.abs(combined).mean(axis=0)})
    return importance.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True), X, combined


# ---------------------------------------------------------------- figures

def _style(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _paired_bars(table: pd.DataFrame, title: str, path: Path, xlabel: str = ''):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, table.ensemble_wape, w, color=ENSEMBLE_COLOR, label='Ensemble model')
    ax.bar(x + w / 2 + 0.01, table.naive_wape, w, color=NAIVE_COLOR, label='Naive (same hour last week)')
    for xi, v in zip(x, table.ensemble_wape):
        ax.text(xi - w / 2, v + 0.4, f'{v:.1f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, [str(i) for i in table.index])
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    _style(ax, title, 'Forecast error, WAPE (%)')
    ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _hour_line(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.plot(table.index, table.ensemble_wape, color=ENSEMBLE_COLOR, linewidth=2, label='Ensemble model')
    ax.plot(table.index, table.naive_wape, color=NAIVE_COLOR, linewidth=2, label='Naive (same hour last week)')
    ax.set_xticks(range(0, 24, 3), [f'{h:02d}:00' for h in range(0, 24, 3)])
    _style(ax, 'Forecast error by hour of day', 'Forecast error, WAPE (%)')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _example_forecast(results: pd.DataFrame, building: str, path: Path):
    g = results[results.building_id == building]
    origin = sorted(g.origin.unique())[0]
    g = g[g.origin == origin]
    t = pd.to_datetime(g.timestamp)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.plot(t, g.actual_kwh, color=INK, linewidth=2, label='Actual')
    ax.plot(t, g.forecast_kwh, color=ENSEMBLE_COLOR, linewidth=2, label='Ensemble forecast')
    ax.plot(t, g.naive_kwh, color=NAIVE_COLOR, linewidth=2, linestyle=(0, (4, 3)), label='Naive baseline')
    _style(ax, f'48-hour forecast for {building}, from {str(origin)[:10]}', 'Electricity (kWh per hour)')
    ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0, 1.02), ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _shap_bar(importance: pd.DataFrame, path: Path, top: int = 12):
    top_features = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.barh(top_features.feature, top_features.mean_abs_shap, color=ENSEMBLE_COLOR, height=0.6)
    ax.set_facecolor(SURFACE)
    ax.set_title('What drives the forecast (mean |SHAP|)', loc='left', fontsize=12, color=INK, pad=12)
    ax.set_xlabel('Average effect on prediction, as a share of the building\'s typical hour',
                  color=INK_MUTED, fontsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _shap_dependence(X: pd.DataFrame, shap_values: np.ndarray, feature: str, path: Path):
    i = list(X.columns).index(feature)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.scatter(X[feature], shap_values[:, i], s=8, alpha=0.4, color=ENSEMBLE_COLOR, linewidths=0)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    _style(ax, f'How {feature} moves the forecast', 'SHAP value (share of typical hour)')
    ax.set_xlabel(feature, color=INK_MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description='Business analysis of the forecast backtest')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--shap-sample', type=int, default=2000)
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(reports / 'backtest_predictions.csv', parse_dates=['timestamp', 'origin'])

    from src.data.data_loader import EnergyDataLoader
    weather = EnergyDataLoader(args.data_dir).load_weather_data(
        results.timestamp.min().strftime('%Y-%m-%d'), results.timestamp.max().strftime('%Y-%m-%d'))
    df = add_context(results, weather)

    breakdowns = {
        'by_forecast_day': wape_by(df, 'forecast_day'),
        'by_day_type': wape_by(df, 'day_type'),
        'by_hour': wape_by(df, 'hour'),
        'by_building': wape_by(df, 'building_id'),
    }
    if 'temperature_band' in df:
        breakdowns['by_temperature'] = wape_by(df, 'temperature_band')

    peaks = peak_accuracy(results)
    peak_summary = {
        'building_days': int(len(peaks)),
        'peak_hour_hit_rate_pct': {
            'ensemble': round(peaks.ensemble_hour_hit.mean() * 100, 1),
            'naive': round(peaks.naive_hour_hit.mean() * 100, 1),
        },
        'median_peak_size_error_pct': {
            'ensemble': round(peaks.ensemble_peak_error_pct.median(), 1),
            'naive': round(peaks.naive_peak_error_pct.median(), 1),
        },
        'definition': 'Hit = forecast peak hour within ±1 h of the actual daily peak hour.',
    }

    cost = cost_scenario(results)

    importance, X_shap, shap_values = shap_importance(args.data_dir, args.models_dir, args.shap_sample)
    importance.to_csv(reports / 'shap_importance.csv', index=False)

    worst = breakdowns['by_building'].sort_values('ensemble_wape').index[-1]
    best = breakdowns['by_building'].sort_values('ensemble_wape').index[0]
    metrics = {
        'overall': {'ensemble_wape': round(wape(results.actual_kwh, results.forecast_kwh), 2),
                    'naive_wape': round(wape(results.actual_kwh, results.naive_kwh), 2)},
        'breakdowns': {k: v.round(2).reset_index().astype({v.index.name or 'index': str})
                       .to_dict(orient='records') for k, v in breakdowns.items()},
        'peak_detection': peak_summary,
        'cost_scenario': cost,
        'top_drivers': importance.head(10).round(4).to_dict(orient='records'),
        'best_building': best,
        'worst_building': worst,
    }
    with open(reports / 'business_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    _paired_bars(breakdowns['by_building'], 'Forecast error by building', figures / 'wape_by_building.png')
    _paired_bars(breakdowns['by_forecast_day'], 'Forecast error by forecast horizon', figures / 'wape_by_horizon.png')
    _paired_bars(breakdowns['by_day_type'], 'Forecast error, weekdays vs weekends', figures / 'wape_by_day_type.png')
    if 'by_temperature' in breakdowns:
        _paired_bars(breakdowns['by_temperature'], 'Forecast error by outdoor temperature',
                     figures / 'wape_by_temperature.png')
    _hour_line(breakdowns['by_hour'], figures / 'wape_by_hour.png')
    _example_forecast(results, best, figures / 'example_forecast.png')
    _shap_bar(importance, figures / 'shap_importance.png')
    weather_features = [f for f in importance.feature if 'temperature' in f]
    if weather_features:
        _shap_dependence(X_shap, shap_values, weather_features[0], figures / 'shap_temperature.png')

    print(json.dumps({k: metrics[k] for k in ('overall', 'peak_detection')}, indent=2))
    print(json.dumps(cost, indent=2))
    print(importance.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
