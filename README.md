# Energy Optimization ML: Intelligent Building Energy Prediction & Optimization

## Overview

This project implements an advanced machine learning system for predicting and optimizing energy consumption in buildings. By leveraging real-world data from the World Energy Outlook 2024 and building-specific datasets, our system aims to reduce energy consumption by 15-30% while lowering carbon emissions and operational costs.

## Problem Statement

Buildings account for approximately 40% of global energy consumption and 36% of CO2 emissions. This project addresses the critical need for intelligent energy management systems that can:

- Accurately predict energy consumption patterns
- Identify optimization opportunities
- Provide actionable recommendations for energy savings
- Scale from individual buildings to city-wide implementations

## Features

- **Multi-source Data Integration**: Combines energy consumption data, weather patterns, and building characteristics
- **Advanced ML Models**: Implements ensemble methods combining traditional time series models with deep learning
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
pip install -r requirements.txt
pip install -e .

# Optional: TensorFlow/PyTorch for the LSTM model
pip install -r requirements-dl.txt
```

4. Run the tests:
```bash
pytest
```

## Data Sources

### Primary Dataset
- **World Energy Outlook 2024 (IEA)**: Historical and projected energy consumption data
- **Building Energy Data**: Real-world consumption patterns from commercial and residential buildings

### Supplementary Data
- Weather data from NOAA Climate API
- Building metadata (size, type, occupancy)
- Energy pricing data
- Carbon emission factors

## Methodology

### 1. Data Pipeline
- Automated data collection from multiple sources
- Data quality checks and validation
- Feature engineering for temporal and contextual patterns

### 2. Model Architecture
- **Baseline Models**: ARIMA, Prophet for time series forecasting
- **Machine Learning**: XGBoost, LightGBM for feature-rich predictions
- **Deep Learning**: LSTM/GRU networks for sequential pattern learning
- **Ensemble**: Weighted combination of models for robust predictions

### 3. Optimization Algorithm
- Peak load detection and shifting recommendations
- HVAC optimization based on occupancy patterns
- Cost-benefit analysis for energy-saving measures

## Usage

### Running the Data Pipeline
```bash
python src/data/collect_data.py --source all
python src/data/preprocess.py
```

### Training Models
```bash
python src/models/train.py --model ensemble
```

### Making Predictions
```bash
python src/models/predict.py --building-id B001 --start-date 2024-03-01 --end-date 2024-03-07
```

### Running the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive docs are served at http://localhost:8000/docs. Set `CORS_ORIGINS` to a comma-separated list of allowed origins in production.

### Launching the Dashboard
```bash
streamlit run src/visualization/dashboard.py
```

> **Note:** The data pipeline currently generates synthetic energy, weather and building data (`src/data/data_loader.py`). Connect real data sources before using predictions for real decisions.

## Performance Metrics

Measured on the bundled synthetic dataset (3 buildings, 60 days, ~10% random noise per hour),
forecasting 48 held-out hours recursively from history:

| Model | MAPE |
|-------|------|
| XGBoost | 8–12% |
| Ensemble (XGBoost + LightGBM + Random Forest) | 8–12% |
| Naive baseline (same hour last week) | 10–13% |

These numbers describe the synthetic data only. Re-measure on real meter data before quoting accuracy.

Predictions need at least one week of hourly consumption history before the start date
(lag and rolling features); the API returns HTTP 400 when it is missing.

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

This project is licensed under the MIT License.

## Acknowledgments

- International Energy Agency for the World Energy Outlook dataset
- NOAA for climate data access
- Building operators who provided consumption data
- Open-source ML community for tools and frameworks
