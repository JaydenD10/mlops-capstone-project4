"""
Unit tests for FastAPI Serving API (Task 7).
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_predict_single_success(client):
    payload = {
        "Area_sqft": 1350.0,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Age_Years": 5.0,
        "Distance_to_Center_km": 8.0,
        "Neighborhood_Score": 7.0,
        "Has_Parking": 1,
        "Property_Condition": "Good",
        "City_Zone": "Central"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_lakhs" in data
    assert data["predicted_price_lakhs"] > 0
    assert data["confidence_interval_low_lakhs"] <= data["predicted_price_lakhs"]
    assert data["confidence_interval_high_lakhs"] >= data["predicted_price_lakhs"]
    assert data["latency_ms"] >= 0


def test_predict_invalid_input_validation(client):
    payload = {
        "Area_sqft": 50.0,  # Below min 100
        "Bedrooms": 0,      # Below min 1
        "Bathrooms": 1,
        "Has_Parking": 1,
        "Property_Condition": "Good",
        "City_Zone": "Central"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_prometheus_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_api_requests_total" in response.text


def test_lineage_endpoint(client):
    response = client.get("/lineage")
    assert response.status_code == 200
    data = response.json()
    assert "registered_model_name" in data
