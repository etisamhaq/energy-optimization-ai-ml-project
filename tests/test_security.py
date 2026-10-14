"""
Tests for API authentication, rate limiting and monitoring endpoints.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app
from src.api.security import ObservabilityMiddleware, RateLimiter
from src.models.predict import EnergyPredictor


class TestAuthentication:
    def test_requests_need_key_when_keys_configured(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-1,secret-2")
        client = TestClient(app)

        assert client.get("/models").status_code == 401
        assert client.get("/models", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/models", headers={"X-API-Key": "secret-2"}).status_code == 200

    def test_public_endpoints_need_no_key(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "secret-1")
        client = TestClient(app)

        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_auth_disabled_without_configured_keys(self, monkeypatch):
        monkeypatch.delenv("API_KEYS", raising=False)
        assert TestClient(app).get("/models").status_code == 200


class TestRateLimiting:
    def test_rate_limiter_blocks_after_limit(self):
        limiter = RateLimiter(limit_per_minute=2)
        assert limiter.check("client") is None
        assert limiter.check("client") is None
        assert limiter.check("client") > 0
        # Other clients have their own budget
        assert limiter.check("other") is None

    def test_rate_limiter_window_expires(self):
        limiter = RateLimiter(limit_per_minute=1, window_seconds=0.05)
        assert limiter.check("client") is None
        assert limiter.check("client") is not None
        import time
        time.sleep(0.06)
        assert limiter.check("client") is None

    def test_middleware_returns_429_with_retry_after(self):
        small_app = FastAPI()
        small_app.add_middleware(ObservabilityMiddleware, limit_per_minute=2)

        @small_app.get("/thing")
        def thing():
            return {"ok": True}

        client = TestClient(small_app)
        assert client.get("/thing").status_code == 200
        assert client.get("/thing").status_code == 200
        response = client.get("/thing")
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1


class TestMonitoring:
    def test_metrics_exposes_request_counts(self):
        client = TestClient(app)
        client.get("/models")
        body = client.get("/metrics").text
        assert "energy_api_requests_total" in body
        assert 'path="/models"' in body

    def test_ready_reports_missing_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(api_main, "predictor", EnergyPredictor(str(tmp_path), str(tmp_path)))
        response = TestClient(app).get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not ready"
