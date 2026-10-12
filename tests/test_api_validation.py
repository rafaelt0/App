from fastapi.testclient import TestClient
import pytest

from backend import app as api_module, state
from utils import db


HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_missing_field_returns_portuguese_detail(client):
    response = client.put("/api/session/selection", json={"tickers": []}, headers=HEADERS)
    assert response.status_code == 422
    assert response.json() == {"detail": "Campo obrigatório ausente."}


def test_list_limit_returns_portuguese_detail(client):
    tickers = [f"T{index}" for index in range(21)]
    response = client.put("/api/session/selection", json={"tickers": tickers, "revision": 0}, headers=HEADERS)
    assert response.status_code == 422
    assert response.json() == {"detail": "A lista excede o limite de itens permitido."}


def test_validator_message_and_unmapped_types_stay_in_portuguese(client):
    duplicated = client.put("/api/session/selection", json={"tickers": ["WEGE3", "WEGE3"], "revision": 0}, headers=HEADERS)
    assert duplicated.json() == {"detail": "Selecione cada ativo apenas uma vez."}
    wrong_shape = client.put("/api/session/selection", json={"tickers": "WEGE3"}, headers=HEADERS)
    assert wrong_shape.json() == {"detail": "Envie uma lista de ativos. Campo obrigatório ausente."}


def test_current_home_analysis_is_empty_without_a_published_result(client):
    response = client.get("/api/home/current")
    assert response.status_code == 200
    assert response.json() is None
