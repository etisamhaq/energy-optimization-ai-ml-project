# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Energy Optimization ML system for predicting and optimizing building energy consumption. The project combines multiple ML approaches (time series, gradient boosting, deep learning) to achieve 15-30% energy reduction through intelligent predictions and actionable recommendations.

## Common Development Commands

### Quick Start
```bash
# Environment setup
make setup
make install

# Run complete pipeline
make pipeline  # Runs data collection, preprocessing, and training

# Run individual components
make collect-data   # Generate synthetic data
make preprocess     # Feature engineering
make train          # Train ensemble model
make run-api        # Start FastAPI (port 8000)
make run-dashboard  # Start Streamlit (port 8501)
```

### Testing
```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_models.py -v

# Run tests by marker
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Only unit tests
```

### Docker Deployment
```bash
make docker-build   # Build images
make docker-up      # Start all services
make docker-logs    # View logs
make docker-down    # Stop services
```

### Model Training Options
```bash
# Train with hyperparameter optimization
python src/models/train.py --model xgboost --optimize --n-trials 50

# Train specific model
python src/models/train.py --model lightgbm

# Train all models
python src/models/train.py --model all
```

### Making Predictions
```bash
# Single building prediction
python src/models/predict.py --building-id B001 --start-date 2024-03-01 --end-date 2024-03-07

# Compare models
python src/models/predict.py --building-id B001 --compare-models

# Multiple buildings
python src/models/predict.py --multiple-buildings B001,B002,B003
```

## High-Level Architecture

### Core Data Flow
```
Raw Data → Data Loader → Feature Engineering → Model Training → Predictions → Optimization → API/Dashboard
```

### Key Architectural Decisions

1. **Modular Model System**: Base model class (`BaseEnergyModel`) with consistent interface for all model types. Each model implements `fit()`, `predict()`, and `evaluate()` methods.

2. **Ensemble Architecture**: The `EnsembleModel` dynamically weights models based on validation performance. Models are trained independently and combined using weighted averaging.

3. **Feature Engineering Pipeline**: `EnergyFeatureEngineer` creates 40+ features including:
   - Temporal features with cyclical encoding
   - Lag features (1h, 24h, 168h)
   - Weather interactions (heating/cooling degree days)
   - Building efficiency metrics

4. **Optimization Engine**: `EnergyOptimizer` generates prioritized recommendations across 6 categories:
   - Peak load shifting
   - HVAC optimization
   - Schedule optimization
   - Equipment upgrades
   - Behavioral changes
   - Renewable integration

### Model Architecture Details

**Time Series Models** (`time_series_models.py`):
- Prophet: Handles seasonality and holidays
- ARIMA/SARIMAX: Captures autoregressive patterns
- Exponential Smoothing: Simple baseline

**ML Models** (`ml_models.py`):
- XGBoost/LightGBM: Primary workhorses with early stopping
- Random Forest: Robust to outliers
- Linear/Ridge: Interpretable baselines

**Deep Learning** (`deep_learning_models.py`):
- LSTM: Sequential pattern learning
- GRU: Lighter alternative to LSTM
- Transformer: Attention-based architecture

**Ensemble** (`ensemble_model.py`):
- Calculates weights based on validation RMSE
- Supports model-specific parameters
- Includes feature importance aggregation

### API Design

FastAPI endpoints (`api/main.py`):
- `/predict`: Single building predictions with optional optimization
- `/predict/batch`: Multiple buildings
- `/compare`: Model comparison
- `/optimize`: Standalone optimization analysis

Response includes predictions, summary statistics, and optimization recommendations with ROI calculations.

### Data Processing Pipeline

1. **Collection** (`collect_data.py`): 
   - Supports synthetic and real data sources
   - Integrates energy, weather, and building metadata
   - Saves raw and integrated datasets

2. **Preprocessing** (`preprocess.py`):
   - Applies feature engineering
   - Creates train/val/test splits (70/10/20)
   - Saves feature metadata for consistency

### Configuration System

Hierarchical configuration (`utils/config.py`):
- `ModelConfig`: Model hyperparameters
- `DataConfig`: Data paths and split ratios
- `OptimizationConfig`: Business rules and thresholds
- `APIConfig`: Service settings

Load from YAML/JSON or environment variables.

## Important Implementation Notes

1. **Import Path Fix**: The data collection scripts have been modified to use relative imports:
   ```python
   from data_loader import EnergyDataLoader  # Instead of src.data.data_loader
   ```

2. **Memory Considerations**: Deep learning models use simplified implementations for demonstration. Production deployment should implement proper sequence handling for LSTM/GRU.

3. **Model Persistence**: Models are saved with timestamps in the `models/` directory. Ensemble models have their own subdirectory structure.

4. **Feature Consistency**: The preprocessing script saves `feature_info.json` to ensure consistent feature ordering between training and inference.

5. **Optimization Calculations**: ROI calculations assume monthly cost savings. Payback period is capped at 60 months for practical recommendations.

## Performance Targets

- Prediction MAPE: < 5.2%
- API response time: < 500ms for single building
- Dashboard load time: < 3s
- Model training: < 30 min for ensemble on 1 year of data