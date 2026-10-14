"""
Energy optimization models package.

Deep-learning models (LSTM/GRU/Transformer) need TensorFlow and PyTorch, which are
optional. They are imported lazily so the API and tree-based models work without them.
"""

from .base_model import BaseEnergyModel
from .time_series_models import ProphetModel, ARIMAModel, ExponentialSmoothingModel
from .ml_models import (
    XGBoostModel, 
    LightGBMModel, 
    RandomForestModel, 
    LinearRegressionModel,
    RidgeModel
)
from .ensemble_model import EnsembleModel, lstm_model

_DEEP_LEARNING_MODELS = ('LSTMModel', 'GRUModel', 'TransformerModel')


def __getattr__(name):
    if name in _DEEP_LEARNING_MODELS:
        from . import deep_learning_models
        return getattr(deep_learning_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'BaseEnergyModel',
    'ProphetModel',
    'ARIMAModel',
    'ExponentialSmoothingModel',
    'XGBoostModel',
    'LightGBMModel',
    'RandomForestModel',
    'LinearRegressionModel',
    'RidgeModel',
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    'EnsembleModel',
    'lstm_model'
]
