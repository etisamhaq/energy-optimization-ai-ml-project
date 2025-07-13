"""
Configuration management for Energy Optimization ML project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configuration settings."""
    ensemble_models: list = None
    xgboost_params: dict = None
    lightgbm_params: dict = None
    random_forest_params: dict = None
    prophet_params: dict = None
    lstm_params: dict = None
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['xgboost', 'lightgbm', 'random_forest', 'prophet']
        
        if self.xgboost_params is None:
            self.xgboost_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        if self.lightgbm_params is None:
            self.lightgbm_params = {
                'n_estimators': 300,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        
        if self.prophet_params is None:
            self.prophet_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'multiplicative'
            }
        
        if self.lstm_params is None:
            self.lstm_params = {
                'sequence_length': 24,
                'lstm_units': [64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }


@dataclass
class DataConfig:
    """Data configuration settings."""
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    models_dir: str = "models"
    
    # Data generation parameters
    n_buildings: int = 10
    n_days_history: int = 365
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    
    # Feature engineering
    lag_periods: list = None
    rolling_windows: list = None
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 24, 168]  # 1 hour, 1 day, 1 week
        
        if self.rolling_windows is None:
            self.rolling_windows = [24, 168]  # 1 day, 1 week


@dataclass
class OptimizationConfig:
    """Optimization configuration settings."""
    electricity_rate: float = 0.12  # $/kWh
    peak_hours: list = None
    off_peak_hours: list = None
    
    # Optimization thresholds
    min_savings_kwh: float = 100
    min_roi_percentage: float = 10
    max_payback_months: float = 60
    
    # Carbon emissions
    carbon_intensity: float = 0.4  # kg CO2/kWh
    
    def __post_init__(self):
        if self.peak_hours is None:
            self.peak_hours = list(range(7, 19))  # 7 AM to 7 PM
        
        if self.off_peak_hours is None:
            self.off_peak_hours = list(range(0, 7)) + list(range(19, 24))


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    
    # API limits
    max_prediction_days: int = 365
    max_buildings_per_request: int = 50
    request_timeout: int = 300  # seconds
    
    # Authentication (placeholder for future implementation)
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    data: DataConfig = None
    optimization: OptimizationConfig = None
    api: APIConfig = None
    
    # General settings
    log_level: str = "INFO"
    random_seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.api is None:
            self.api = APIConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'optimization': asdict(self.optimization),
            'api': asdict(self.api),
            'log_level': self.log_level,
            'random_seed': self.random_seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            optimization=OptimizationConfig(**config_dict.get('optimization', {})),
            api=APIConfig(**config_dict.get('api', {})),
            log_level=config_dict.get('log_level', 'INFO'),
            random_seed=config_dict.get('random_seed', 42)
        )
    
    def save(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    
    if _config is None:
        # Try to load from environment variable
        config_path = os.environ.get('ENERGY_ML_CONFIG')
        
        if config_path and Path(config_path).exists():
            _config = Config.load(config_path)
        else:
            # Use default configuration
            _config = Config()
    
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config


def reset_config():
    """Reset configuration to default."""
    global _config
    _config = None