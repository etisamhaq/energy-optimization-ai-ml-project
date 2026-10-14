# Energy Optimization ML: Intelligent Building Energy Prediction & Optimization

## Overview

This project implements an advanced machine learning system for predicting and optimizing energy consumption in buildings. It trains on real hourly meter data (or your own) with real weather, forecasts each building's consumption, and recommends measures to cut energy use, costs and carbon emissions.

## Problem Statement

Buildings account for approximately 40% of global energy consumption and 36% of CO2 emissions. This project addresses the critical need for intelligent energy management systems that can:

- Accurately predict energy consumption patterns
- Identify optimization opportunities
- Provide actionable recommendations for energy savings
- Scale from individual buildings to city-wide implementations

## Features

- **Multi-source Data Integration**: Combines energy consumption data, weather patterns, and building characteristics
- **ML Models**: Ensemble of XGBoost, LightGBM and Random Forest, forecasting each building relative to its own recent consumption
- **Real-time Predictions**: API-based system for continuous energy monitoring
- **Optimization Engine**: Provides specific recommendations for reducing energy consumption
- **Interactive Dashboard**: Visualizes predictions and optimization opportunities

![Alt Text](images/overview.png)


![Alt Text](images/overview1.png)


![Alt Text](images/predictions.png)


![Alt Text](images/optimization.png)


![Alt Text](images/building_information.png)


![Alt Text](images/report.png)

## Project Structure

```
energy-optimization-ml/
├── data/               # Data storage
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned and transformed data
│   └── external/      # External data sources (weather, etc.)
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data/          # Data processing modules
│   ├── features/      # Feature engineering
│   ├── models/        # ML model implementations
│   ├── visualization/ # Plotting and dashboard code
│   └── utils/         # Helper functions
├── models/            # Saved model artifacts
├── reports/           # Generated analysis reports
├── tests/             # Unit and integration tests
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/etisamhaq/energy-optimization-ai-ml-project.git
cd energy-optimization-ai-ml-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt -c constraints.txt
pip install -e .

# Optional: TensorFlow/PyTorch for the LSTM model
pip install -r requirements-dl.txt
```

4. Run the tests:
```bash
pytest
```

## Data Sources

The pipeline supports three sources (`src/data/collect_data.py --source ...`):

| Source | Energy data | Weather | Use for |
|--------|-------------|---------|---------|
| `bdg2` | **Real** hourly electricity meter readings, 2016–2017, from the [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) (CC BY 4.0) | **Real**, from [Open-Meteo](https://open-meteo.com) for the site's coordinates | Evaluating the models on real buildings |
| `file` | **Your own** meter data and building metadata (CSV) | **Real**, from Open-Meteo for coordinates you give | Production |
| `synthetic` (default) | Generated | Generated | Demos and tests (no network needed) |

Building metadata (type, floor area, year built) comes with the BDG2 data or your metadata file.
BDG2 has no Energy Star scores for most buildings, so a neutral 50 is used.

Open-Meteo is free and needs no key. Weather forecasts reach 15 days ahead, which limits how far ahead predictions can go.

### Using your own data
- `energy.csv`: `building_id, timestamp, energy_consumption_kwh` (hourly, local time)
- `metadata.csv`: `building_id, building_type, floor_area_sqm, year_built, occupancy_type, hvac_type, energy_star_rating`

```bash
python src/data/collect_data.py --source file \
    --energy-file energy.csv --metadata-file metadata.csv \
    --latitude 38.90 --longitude -77.01 --timezone America/New_York
```

## Methodology

### 1. Data Pipeline
- Automated data collection from multiple sources
- Data quality checks and validation
- Feature engineering for temporal and contextual patterns

### 2. Model Architecture
- **Machine Learning**: XGBoost, LightGBM and Random Forest on time, weather, building and lag features
- **Ensemble**: Combination weighted by validation error
- **Per-building scaling**: Models predict consumption relative to each building's previous-week average, so one model serves buildings of very different sizes
- **Recursive forecasting**: Each hour's prediction feeds the next hour's lag features

`src/models` also contains Prophet, ARIMA and LSTM/GRU/Transformer classes. They are experimental and not
served by the training or prediction pipeline: they are fitted on all buildings at once and cannot
yet forecast an individual building.

### 3. Optimization Algorithm
- Peak load detection and shifting recommendations
- HVAC optimization based on occupancy patterns
- Cost-benefit analysis for energy-saving measures

## Usage

### Running the Data Pipeline
```bash
# Real data: 10 buildings in Washington, DC (downloads ~175 MB once)
python src/data/collect_data.py --source bdg2 --site Rat --n-buildings 10

# Or synthetic demo data
python src/data/collect_data.py --source synthetic

python src/data/preprocess.py
```

### Training Models
```bash
python src/models/train.py --model ensemble
```

### Making Predictions
```bash
# Forecasts start after the building's observed history (BDG2 data ends 2017-12-31)
python src/models/predict.py --building-id Rat_office_Colby --start-date 2018-01-01 --end-date 2018-01-07
```

### Running the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive docs are served at http://localhost:8000/docs.

### Security and monitoring

| Setting | Purpose |
|---------|---------|
| `API_KEYS` | Comma-separated keys; clients send one in the `X-API-Key` header. **Unset disables authentication** (local development only). Docker Compose refuses to start without it. |
| `RATE_LIMIT_PER_MINUTE` | Requests per client per minute (default 60). Kept in memory per API process; with several replicas, also limit at the gateway. |
| `CORS_ORIGINS` | Comma-separated allowed browser origins. |
| `DATA_DIR`, `MODELS_DIR` | Where the API and dashboard read data and models. |

Endpoints without a key: `/health` (liveness), `/ready` (a model can be loaded, otherwise 503), `/metrics` (Prometheus: request counts, latency, predictions served) and `/docs`.

The dashboard and MLflow UI have no login. Docker Compose binds them to localhost; put them behind an authenticating reverse proxy before exposing them.

### Launching the Dashboard
```bash
streamlit run src/visualization/dashboard.py
```


## Performance Metrics

Measured on **real data**: 10 BDG2 buildings in Washington, DC, with real weather. The models were
trained on 2016 to mid-2017, then made 40 recursive 48-hour forecasts in the held-out autumn 2017
period. The error is WAPE: total absolute error divided by total consumption.

| Model | WAPE |
|-------|------|
| Ensemble (XGBoost + LightGBM + Random Forest) | **17.4%** |
| Naive baseline (same hour last week) | 21.7% |

The ensemble beats the baseline on all 10 buildings (per-building WAPE 5.5%–21.4%).
Accuracy on your buildings will differ, so re-run this backtest on your own data before relying on the forecasts.

Predictions need at least one week of hourly history before the start date and must start within
14 days of the last reading; otherwise the API returns HTTP 400 with the reason.

### Optimization estimates
Savings use rule-of-thumb factors (HVAC share of load, LED savings, installation costs, ...) collected in
`DEFAULT_ASSUMPTIONS` in `src/optimization/optimizer.py`; override them with `EnergyOptimizer(assumptions={...})`
when you know a site's figures. Payback and ROI are calculated from those inputs, and every report includes
the assumptions it used. The confidence score is a heuristic rating, not a probability.

## Real-World Impact

- **Environmental**: Reduces carbon emissions by optimizing energy usage
- **Economic**: Lowers operational costs for building managers
- **Social**: Contributes to sustainable urban development
- **Scalability**: Applicable from individual buildings to entire cities

## Future Enhancements

- Integration with IoT sensors for real-time data collection
- Reinforcement learning for dynamic optimization
- Multi-building coordination for grid-level optimization
- Mobile application for building managers

## Contributing

We welcome contributions! Please open an issue or pull request.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

## Acknowledgments

- Building Data Genome Project 2: Miller, C. et al. *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition.* Scientific Data 7, 368 (2020). Licensed CC BY 4.0.
- Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)
- Open-source ML community for tools and frameworks
