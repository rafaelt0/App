"""Integration checks confirming all Wave 1 routers are mounted on backend.app with CSRF and session guards."""

from fastapi.testclient import TestClient

from backend.app import app

HEADERS = {"origin": "http://localhost:3000", "x-b3lab-request": "1"}


def test_wave1_presets_public_no_cookie(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.db._DB", str(tmp_path / "wave1.db"))
    client = TestClient(app)
    response = client.get("/api/stress/presets")
    assert response.status_code == 200
    presets = response.json()
    assert isinstance(presets, list) and len(presets) >= 5
    assert "b3lab_session" not in response.cookies


def test_wave1_mutations_enforce_csrf(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.db._DB", str(tmp_path / "wave1.db"))
    client = TestClient(app)
    for path, payload in (
        ("/api/portfolio/hurdle", {"revision": 0, "portfolio_id": "x"}),
        ("/api/rebalance/import", {"content": "ticker,qty\nPETR4,10"}),
        ("/api/rebalance/plan", {"revision": 0, "portfolio_id": "x", "contribution": 1000}),
        ("/api/income/calendar", {"revision": 0, "portfolio_id": "x"}),
        ("/api/stress/macro", {"revision": 0, "portfolio_id": "x"}),
    ):
        response = client.post(path, json=payload)
        assert response.status_code == 403, f"Expected 403 CSRF block on {path}, got {response.status_code}"


def test_wave1_owned_endpoints_require_loaded_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.db._DB", str(tmp_path / "wave1.db"))
    client = TestClient(app)
    for path, payload in (
        ("/api/portfolio/hurdle", {"revision": 0, "portfolio_id": "missing"}),
        ("/api/rebalance/plan", {"revision": 0, "portfolio_id": "missing", "contribution": 1000}),
        ("/api/income/calendar", {"revision": 0, "portfolio_id": "missing"}),
        ("/api/stress/macro", {"revision": 0, "portfolio_id": "missing"}),
    ):
        response = client.post(path, json=payload, headers=HEADERS)
        assert response.status_code == 409, f"Expected 409 without portfolio on {path}, got {response.status_code}"
