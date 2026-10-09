"""
Health Endpoint Tests

Verifies the /health endpoint returns the expected response.
"""
from fastapi.testclient import TestClient

from services.api.app import app


client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_expected_json():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "jarvis"