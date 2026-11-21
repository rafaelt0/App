from fastapi.testclient import TestClient
import pytest
import sqlite3

from backend import app as api_module, state
from utils import db


HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    with TestClient(api_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_reports_api_and_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_reports_degraded_when_the_database_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(db, "ready", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}


def test_readiness_rejects_a_database_that_cannot_be_written(client, tmp_path, monkeypatch):
    path = tmp_path / "read-only.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE seed (value INTEGER)")
    connection.commit()
    connection.close()
    path.chmod(0o444)
    monkeypatch.setattr(db, "_DB", str(path))
    monkeypatch.setattr(db, "_ready_key", None)

    assert db.ready() is False
    assert client.get("/health").status_code == 503


def test_readiness_probe_leaves_no_table_behind(tmp_path, monkeypatch):
    path = tmp_path / "probe.sqlite"
    monkeypatch.setattr(db, "_DB", str(path))
    monkeypatch.setattr(db, "_ready_key", None)

    assert db.ready() is True
    connection = sqlite3.connect(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        connection.close()
    assert "__b3lab_ready" not in tables


def test_an_unexpected_failure_keeps_the_portuguese_error_shape(client, monkeypatch):
    def boom():
        raise RuntimeError("provider blew up")

    monkeypatch.setattr(api_module, "get_listed_stocks", boom)
    response = client.put("/api/session/selection", json={"tickers": ["PETR4"], "revision": 0}, headers=HEADERS)
    assert response.status_code == 500
    assert response.json() == {"detail": "Falha interna ao processar a solicitação. Tente novamente."}
