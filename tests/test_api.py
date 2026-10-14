"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app


@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory, monkeypatch_module):
    """Run the real pipeline on a small synthetic dataset and serve its models."""
    from src.data.collect_data import DataCollector
    from src.data.preprocess import DataPreprocessor
    from src.models.predict import EnergyPredictor
    from src.models.train import ModelTrainer

    root = tmp_path_factory.mktemp("pipeline")
    data_dir, models_dir = root / "data", root / "models"
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", (root / "mlruns").as_uri())

    DataCollector(str(data_dir)).collect_all_data(n_buildings=3, n_days=30)
    DataPreprocessor(str(data_dir)).run_preprocessing_pipeline()

    trainer = ModelTrainer(str(data_dir), str(models_dir))
    for model_name in ["xgboost", "lightgbm"]:
        trainer.save_model(trainer.train_single_model(model_name))
    trainer.save_model(trainer.train_ensemble())

    predictor = EnergyPredictor(str(models_dir), str(data_dir))
    monkeypatch_module.setattr(api_main, "predictor", predictor)
    return predictor


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


class TestAPI:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_predict_endpoint_basic(self, client, trained_predictor):
        """Test basic prediction endpoint."""
        request_data = {
            "building_id": "B001",
            "start_date": "2024-03-01",
            "end_date": "2024-03-02",
            "model_name": "ensemble",
            "include_optimization": False
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["building_id"] == "B001"
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 48  # every hour of both days (end date is inclusive)
    
    def test_predict_endpoint_with_optimization(self, client, trained_predictor):
        """Test prediction endpoint with optimization."""
        request_data = {
            "building_id": "B001",
            "start_date": "2024-03-01",
            "end_date": "2024-03-02",
            "model_name": "ensemble",
            "include_optimization": True
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "optimization_report" in data
        assert data["optimization_report"] is not None
    
    def test_predict_endpoint_invalid_dates(self, client):
        """Test prediction endpoint with invalid dates."""
        request_data = {
            "building_id": "B001",
            "start_date": "2024-03-02",
            "end_date": "2024-03-01",  # End before start
            "model_name": "ensemble"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        assert "End date must be after start date" in response.json()["detail"]
    
    def test_predict_endpoint_without_trained_model(self, client, monkeypatch, tmp_path):
        """Missing models should be reported as service unavailable, not a server crash."""
        from src.models.predict import EnergyPredictor
        monkeypatch.setattr(api_main, "predictor", EnergyPredictor(str(tmp_path), str(tmp_path)))
        
        response = client.post("/predict", json={
            "building_id": "B001",
            "start_date": "2024-03-01",
            "end_date": "2024-03-02"
        })
        assert response.status_code == 503
    
    def test_batch_predict_rejects_too_many_buildings(self, client):
        """Batch requests are capped to protect the server."""
        response = client.post("/predict/batch", json={
            "building_ids": [f"B{i:03d}" for i in range(100)],
            "start_date": "2024-03-01",
            "end_date": "2024-03-02"
        })
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self, client, trained_predictor):
        """Test batch prediction endpoint."""
        request_data = {
            "building_ids": ["B001", "B002", "B003"],
            "start_date": "2024-03-01",
            "end_date": "2024-03-02",
            "model_name": "ensemble"
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "buildings" in data
        assert len(data["buildings"]) == 3
        assert all("total_predicted_kwh" in b for b in data["buildings"])
    
    def test_compare_models_endpoint(self, client, trained_predictor):
        """Test model comparison endpoint."""
        request_data = {
            "building_id": "B001",
            "start_date": "2024-03-01",
            "end_date": "2024-03-02",
            "models": ["xgboost", "lightgbm"]
        }
        
        response = client.post("/compare", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "model_statistics" in data
        assert "ensemble_stats" in data
    
    def test_optimize_endpoint(self, client, trained_predictor):
        """Test optimization endpoint."""
        request_data = {
            "building_id": "B001",
            "analysis_period_days": 7,
            "electricity_rate": 0.15
        }
        
        response = client.post("/optimize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "optimization_report" in data
        assert "analysis_period" in data
    
    def test_buildings_endpoint(self, client):
        """Test buildings list endpoint."""
        response = client.get("/buildings")
        assert response.status_code == 200
        
        data = response.json()
        assert "buildings" in data
        assert isinstance(data["buildings"], list)
        assert len(data["buildings"]) > 0
    
    def test_models_endpoint(self, client):
        """Test models list endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        
        # Check model structure
        model = data["models"][0]
        assert "name" in model
        assert "description" in model
        assert "type" in model