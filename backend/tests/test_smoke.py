"""
RAGScope — Smoke Tests (Phase 0)

Basic tests to verify the scaffold is wired correctly.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthz():
    """Liveness probe should return alive."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readyz():
    """Readiness probe should return ready."""
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_api_v1_root():
    """API v1 root should list available endpoints."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()
    assert data["api"] == "RAGScope"
    assert data["version"] == "v1"
    assert "ingest" in data["endpoints"]
    assert "query" in data["endpoints"]
    assert "evaluate" in data["endpoints"]


def test_docs_available():
    """OpenAPI docs should be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """OpenAPI schema should be generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "RAGScope"
