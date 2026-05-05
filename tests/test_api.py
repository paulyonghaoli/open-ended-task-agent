"""Tests for the FastAPI /run endpoint."""
from __future__ import annotations

import pytest


def test_api_imports_if_available():
    try:
        from task_agent.api import app  # noqa: F401
    except ImportError:
        pytest.skip("FastAPI is not installed; API support is optional.")
    assert app is not None


def test_api_health_endpoint():
    try:
        from fastapi.testclient import TestClient
        from task_agent.api import app
    except ImportError:
        pytest.skip("FastAPI or httpx is not installed.")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_run_returns_final_answer():
    try:
        from fastapi.testclient import TestClient
        from task_agent.api import app
    except ImportError:
        pytest.skip("FastAPI or httpx is not installed.")
    client = TestClient(app)
    response = client.post("/run", json={"goal": "Compare DuckDB and SQLite", "max_steps": 5})
    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert len(data["final_answer"]) > 0


def test_api_run_returns_evidence():
    try:
        from fastapi.testclient import TestClient
        from task_agent.api import app
    except ImportError:
        pytest.skip("FastAPI or httpx is not installed.")
    client = TestClient(app)
    response = client.post("/run", json={"goal": "Compare DuckDB and SQLite", "max_steps": 5})
    assert response.status_code == 200
    data = response.json()
    assert "evidence" in data
    assert isinstance(data["evidence"], list)


def test_api_run_returns_confidence():
    try:
        from fastapi.testclient import TestClient
        from task_agent.api import app
    except ImportError:
        pytest.skip("FastAPI or httpx is not installed.")
    client = TestClient(app)
    response = client.post("/run", json={"goal": "Compare DuckDB and SQLite", "max_steps": 5})
    data = response.json()
    assert data["confidence"] in ("high", "medium", "low", "unknown")


def test_api_rejects_empty_goal():
    try:
        from fastapi.testclient import TestClient
        from task_agent.api import app
    except ImportError:
        pytest.skip("FastAPI or httpx is not installed.")
    client = TestClient(app)
    response = client.post("/run", json={"goal": "   "})
    assert response.status_code == 422
