#!/usr/bin/env python
"""
System test script to verify all components are working.
"""

import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    modules = [
        'src.data.data_loader',
        'src.features.feature_engineering',
        'src.models.base_model',
        'src.models.ml_models',
        'src.models.time_series_models',
        'src.models.ensemble_model',
        'src.optimization.optimizer',
        'src.api.main',
        'src.visualization.dashboard',
        'src.utils.helpers',
        'src.utils.config'
    ]
    
    print("Testing imports...")
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    try:
        from src.data.data_loader import EnergyDataLoader
        
        loader = EnergyDataLoader()
        df = loader._generate_synthetic_energy_data(n_buildings=2, n_days=7)
        
        assert len(df) == 2 * 7 * 24  # 2 buildings, 7 days, 24 hours
        assert 'energy_consumption_kwh' in df.columns
        print("✓ Data generation working")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\nTesting feature engineering...")
    try:
        from src.data.data_loader import EnergyDataLoader
        from src.features.feature_engineering import EnergyFeatureEngineer
        
        loader = EnergyDataLoader()
        engineer = EnergyFeatureEngineer()
        
        # Generate sample data
        df = loader._generate_synthetic_energy_data(n_buildings=1, n_days=7)
        
        # Apply feature engineering
        df_featured = engineer.create_temporal_features(df)
        
        assert 'hour' in df_featured.columns
        assert 'is_weekend' in df_featured.columns
        print("✓ Feature engineering working")
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    try:
        from src.utils.config import Config, get_config
        
        config = get_config()
        assert config.model is not None
        assert config.data is not None
        assert config.optimization is not None
        assert config.api is not None
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def main():
    """Run all system tests."""
    print("Energy Optimization ML - System Test")
    print("====================================\n")
    
    tests = [
        test_imports,
        test_data_generation,
        test_feature_engineering,
        test_configuration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n====================================")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"====================================")
    
    if failed == 0:
        print("\n✓ All systems operational!")
        print("\nNext steps:")
        print("1. Run 'python src/data/collect_data.py' to generate data")
        print("2. Run 'python src/data/preprocess.py' to preprocess data")
        print("3. Run 'python src/models/train.py' to train models")
        print("4. Run 'uvicorn src.api.main:app' to start the API")
        print("5. Run 'streamlit run src/visualization/dashboard.py' to start the dashboard")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()