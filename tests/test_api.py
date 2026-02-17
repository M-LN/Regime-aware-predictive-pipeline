"""
Integration tests for the API
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from src.inference.api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIEndpoints:
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "endpoints" in response.json()

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_prediction_endpoint(self, client):
        """Test single prediction endpoint"""
        payload = {
            "wind_speed": 8.5,
            "energy_production": 400.0,
            "temperature": 12.0,
            "price": 150.0,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "regime" in data
        assert "regime_confidence" in data
        assert "inference_latency_ms" in data

    def test_prediction_with_timestamp(self, client):
        """Test prediction with explicit timestamp"""
        payload = {
            "wind_speed": 8.5,
            "energy_production": 400.0,
            "temperature": 12.0,
            "price": 150.0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_prediction_invalid_input(self, client):
        """Test prediction with invalid input"""
        payload = {
            "wind_speed": -5.0,  # Invalid: negative
            "energy_production": 400.0,
            "temperature": 12.0,
            "price": 150.0,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self, client):
        """Test batch prediction endpoint"""
        payload = {
            "data": [
                {
                    "wind_speed": 8.5,
                    "energy_production": 400.0,
                    "temperature": 12.0,
                    "price": 150.0,
                },
                {
                    "wind_speed": 12.0,
                    "energy_production": 600.0,
                    "temperature": 10.0,
                    "price": 120.0,
                },
            ]
        }

        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["n_succeeded"] == 2
        assert len(data["predictions"]) == 2

    def test_status_endpoint(self, client):
        """Test status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data
        assert "config" in data
