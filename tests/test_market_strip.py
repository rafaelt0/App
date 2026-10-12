import datetime as dt

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import market
from utils import market_strip


@pytest.fixture(autouse=True)
def isolated_caches():
    for cached in (market_strip.get_market_quotes, market_strip.get_macro_series):
        cached.clear()
    yield
    for cached in (market_strip.get_market_quotes, market_strip.get_macro_series):
        cached.clear()


def series(values, end="2026-10-09"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)), dtype=float)


def test_quote_uses_raw_close_for_value_and_adjusted_close_for_change():
    close = series([float("nan"), 10.0, 9.0])
    adjusted = series([float("nan"), 8.0, 9.0])
    item = market_strip.quote_item("PETR4", "PETR4", "PETR4.SA", "money", "BRL", 2, close, adjusted)
    assert item["value"] == 9.0
    assert item["change"] == pytest.approx(9 / 8 - 1)
    assert item["group"] == "stock"
    assert item["as_of"] == "2026-10-09"


def test_quotes_cover_only_b3_and_brazilian_market():
    symbols = [quote[2] for quote in market_strip.QUOTES]
    assert all(symbol.endswith(".SA") for symbol in symbols if not symbol.startswith("^") and symbol != "BRL=X")
    assert set(symbol for symbol in symbols if symbol.startswith("^")) == {"^BVSP", "^IBX50"}


def test_quote_without_previous_close_has_no_change_and_empty_quote_is_missing():
    one = series([float("nan"), 5.0])
    assert market_strip.quote_item("usd", "Dólar", "BRL=X", "money", "BRL", 3, one, one)["change"] is None
    empty = series([float("nan"), float("nan")])
    assert market_strip.quote_item("usd", "Dólar", "BRL=X", "money", "BRL", 3, empty, empty) is None


def test_quote_change_ignores_adjusted_series_from_another_date():
    close = series([10.0, 11.0], end="2026-10-12")
    adjusted = series([10.0, 11.0], end="2026-10-09")
    assert market_strip.quote_item("ibov", "Ibovespa", "^BVSP", "number", None, 0, close, adjusted)["change"] is None


def test_macro_ignores_future_dated_and_invalid_rows():
    rows = [
        {"data": "09/10/2026", "valor": "15.00"},
        {"data": "12/10/2026", "valor": "13.75"},
        {"data": "04/11/2026", "valor": "12.00"},
        {"data": "10/10/2026", "valor": "n/d"},
        {"valor": "1"},
    ]
    item = market_strip.macro_item("selic", "Selic meta", 432, "a.a.", rows, dt.date(2026, 10, 12))
    assert item["value"] == 13.75
    assert item["as_of"] == "2026-10-12"
    assert item["unit"] == "percent"
    assert item["change"] is None
    assert market_strip.macro_item("selic", "Selic meta", 432, "a.a.", rows[2:], dt.date(2026, 10, 12)) is None


def test_strip_reports_failed_sources_as_missing_without_placeholders(monkeypatch):
    def macro(series, today):
        if series == 13522:
            raise TimeoutError("BCB offline")
        return [{"data": "01/10/2026", "valor": "10"}]

    monkeypatch.setattr(market_strip, "get_macro_series", macro)
    monkeypatch.setattr(market_strip, "get_market_quotes", lambda: (_ for _ in ()).throw(ValueError("Yahoo offline")))
    result = market_strip.market_strip()
    assert [item["key"] for item in result["items"]] == ["selic", "cdi", "unemployment"]
    assert result["missing"][-1] == "IPCA"
    assert "Ibovespa" in result["missing"] and "PETR4" in result["missing"]


def test_market_strip_route_returns_clean_json(monkeypatch):
    monkeypatch.setattr(market, "market_strip", lambda: {"items": [{"key": "x", "value": float("nan")}], "missing": [], "updated_at": "2026-10-12T20:00:00+00:00"})
    app = FastAPI()
    app.include_router(market.router)
    with TestClient(app) as client:
        response = client.get("/api/market/strip")
    assert response.status_code == 200
    assert response.json()["items"][0]["value"] is None
