"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

import sys
sys.path.append('..')
from src.api.main import app


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
    
    def test_predict_endpoint_basic(self, client):
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
        assert len(data["predictions"]) == 24  # 24 hours for one day
    
    def test_predict_endpoint_with_optimization(self, client):
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
    
    def test_batch_predict_endpoint(self, client):
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
    
    def test_compare_models_endpoint(self, client):
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
    
    def test_optimize_endpoint(self, client):
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