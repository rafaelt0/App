import datetime as dt

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import market
from utils import market_strip


@pytest.fixture(autouse=True)
def isolated_caches():
    cached = (market_strip.get_market_quotes, market_strip.get_macro_series, market_strip.get_focus_annual,
              market_strip.get_focus_inflation_12m, market_strip.get_tesouro_rates)
    for function in cached:
        function.clear()
    yield
    for function in cached:
        function.clear()


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


def test_quotes_are_b3_tickers_plus_dollar_sp500_and_brent():
    symbols = [quote[2] for quote in market_strip.QUOTES]
    assert all(symbol.endswith(".SA") for symbol in symbols if symbol not in {"^BVSP", "^GSPC", "BRL=X", "BZ=F"})
    assert set(symbol for symbol in symbols if symbol.startswith("^")) == {"^BVSP", "^GSPC"}
    assert "BZ=F" in symbols and "^IBX50" not in symbols


def test_quote_without_previous_close_has_no_change_and_empty_quote_is_missing():
    one = series([float("nan"), 5.0])
    assert market_strip.quote_item("usd", "Dólar", "BRL=X", "money", "BRL", 3, one, one)["change"] is None
    empty = series([float("nan"), float("nan")])
    assert market_strip.quote_item("usd", "Dólar", "BRL=X", "money", "BRL", 3, empty, empty) is None


def test_quote_change_ignores_adjusted_series_from_another_date():
    close = series([10.0, 11.0], end="2026-10-12")
    adjusted = series([10.0, 11.0], end="2026-10-09")
    assert market_strip.quote_item("ibov", "Ibovespa", "^BVSP", "number", None, 0, close, adjusted)["change"] is None


TODAY = dt.date(2026, 11, 7)
SELIC_ROWS = [
    {"data": "16/09/2026", "valor": "14.00"},
    {"data": "17/09/2026", "valor": "13.75"},
    {"data": "06/11/2026", "valor": "13.75"},
    {"data": "07/11/2026", "valor": "13.75"},
    {"data": "10/12/2026", "valor": "13.50"},
    {"data": "08/11/2026", "valor": "n/d"},
    {"valor": "1"},
]
IPCA_ROWS = [{"data": "01/08/2026", "valor": "4.22"}, {"data": "01/09/2026", "valor": "4.58"}]
FOCUS_IPCA_ROWS = [
    {"Indicador": "IPCA", "Data": "2026-10-30", "DataReferencia": "2026", "Mediana": 5.2166, "baseCalculo": 0},
    {"Indicador": "IPCA", "Data": "2026-10-29", "DataReferencia": "2026", "Mediana": 5.1974, "baseCalculo": 0},
    {"Indicador": "IPCA", "Data": "2026-10-23", "DataReferencia": "2026", "Mediana": 5.1891, "baseCalculo": 0},
    {"Indicador": "IPCA", "Data": "2026-10-22", "DataReferencia": "2026", "Mediana": None, "baseCalculo": 0},
]
FOCUS_12M_ROWS = [{"Indicador": "IPCA", "Data": "2026-10-30", "Suavizada": "S", "Mediana": 5.0, "baseCalculo": 0}]
TESOURO_CSV = (
    "Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;PU Compra Manha;PU Venda Manha;PU Base Manha\n"
    "Tesouro Selic;01/03/2027;06/11/2026;0,02;0,03;20251,05;20240,05;20240,05\n"
    "Tesouro IPCA+;15/05/2035;06/11/2026;7,87;7,99;2515,93;2491,12;2491,12\n"
    "Tesouro IPCA+;15/05/2045;06/11/2026;7,43;7,55;1276,97;1250,39;1250,39\n"
    "Tesouro IPCA+ com Juros Semestrais;15/05/2035;06/11/2026;7,60;7,72;4000,00;3990,00;3990,00\n"
    "Tesouro IPCA+;15/05/2035;05/11/2026;7,94;8,06;2500,90;2476,57;2476,57\n"
    "Tesouro IPCA+;15/05/2045;05/11/2026;7,48;7,60;1265,47;1239,30;1239,30\n"
    "Tesouro IPCA+;15/05/2035;04/11/2026;7,9"
)


class Response:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code, self.payload, self.content = status_code, payload, content
        self.read = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            self.read += 1
            yield self.content[start:start + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_selic_reports_the_last_copom_move_in_percentage_points_and_ignores_future_rows():
    item = market_strip.selic_item(SELIC_ROWS, TODAY)
    assert item["value"] == 13.75
    assert item["unit"] == "percent"
    assert item["as_of"] == "2026-11-07"
    assert item["change"] == pytest.approx(-0.25)
    assert item["change_unit"] == "pp"
    assert item["change_note"] == "desde 17/09/2026 (Copom)"


def test_selic_without_a_move_in_the_window_has_no_change_and_no_rows_is_missing():
    held = market_strip.selic_item(SELIC_ROWS[1:4], TODAY)
    assert held["change"] is None and held["change_unit"] is None
    assert market_strip.selic_item(SELIC_ROWS[4:], TODAY) is None


def test_ipca_change_is_against_the_previous_month_in_percentage_points():
    item = market_strip.ipca_item(IPCA_ROWS, TODAY)
    assert item["value"] == 4.58
    assert item["change"] == pytest.approx(0.36)
    assert item["change_unit"] == "pp"
    assert item["change_note"] == "ante 08/2026"


def test_focus_uses_latest_median_and_compares_with_the_survey_a_week_earlier():
    item = market_strip.focus_item("focus_ipca", "IPCA Focus 2026", "esperado no ano", FOCUS_IPCA_ROWS)
    assert item["value"] == 5.2166
    assert item["unit"] == "percent"
    assert item["as_of"] == "2026-10-30"
    assert item["change"] == pytest.approx(5.2166 - 5.1891)
    assert item["change_unit"] == "pp"
    assert item["change_note"] == "ante Focus de 23/10/2026"
    assert market_strip.focus_item("focus_ipca", "IPCA Focus 2026", "", FOCUS_IPCA_ROWS[:2])["change"] is None
    assert market_strip.focus_item("focus_ipca", "IPCA Focus 2026", "", [{"Data": "2026-10-30", "Mediana": None}]) is None


def test_focus_request_filters_the_published_report_window(monkeypatch):
    calls = []

    def get(url, timeout):
        calls.append(url)
        return Response(payload={"value": FOCUS_IPCA_ROWS})

    monkeypatch.setattr(market_strip.requests, "get", get)
    assert market_strip.get_focus_annual("IPCA", 2026) == FOCUS_IPCA_ROWS
    assert "ExpectativasMercadoAnuais" in calls[0]
    assert "DataReferencia%20eq%20%272026%27" in calls[0] and "baseCalculo%20eq%200" in calls[0]


def test_fisher_real_rate_and_ex_ante_item():
    assert market_strip.fisher_real_rate(13.75, 5.0) == pytest.approx((1.1375 / 1.05 - 1) * 100)
    item = market_strip.real_rate_item(SELIC_ROWS, FOCUS_12M_ROWS, TODAY)
    assert item["value"] == pytest.approx(8.3333, abs=1e-4)
    assert item["unit"] == "percent"
    assert item["as_of"] == "2026-10-30"
    assert item["change"] is None
    assert "5,00%" in item["source"] and "Fisher" in item["note"]
    assert market_strip.real_rate_item([], FOCUS_12M_ROWS, TODAY) is None
    assert market_strip.real_rate_item(SELIC_ROWS, [], TODAY) is None


def test_tesouro_reads_ipca_plus_buy_rate_and_drops_a_cut_last_line(monkeypatch):
    monkeypatch.setattr(market_strip.requests, "get", lambda url, stream, timeout: Response(content=TESOURO_CSV.encode()))
    lines = market_strip.get_tesouro_rates()
    assert lines[-1].endswith("1239,30")
    item = market_strip.tesouro_item("ntnb_2035", "Tesouro IPCA+ 2035", "15/05/2035", lines)
    assert item["value"] == 7.87
    assert item["unit"] == "percent"
    assert item["as_of"] == "2026-11-06"
    assert item["change"] == pytest.approx(-0.07)
    assert item["change_unit"] == "pp"
    assert item["source"].startswith("Tesouro Transparente")
    assert market_strip.tesouro_item("x", "Tesouro IPCA+ 2099", "15/05/2099", lines) is None


def test_tesouro_stops_reading_after_the_head_of_a_large_file(monkeypatch):
    response = Response(content=(TESOURO_CSV.rsplit("\n", 1)[0] + "\n").encode() * 20000)
    monkeypatch.setattr(market_strip.requests, "get", lambda url, stream, timeout: response)
    lines = market_strip.get_tesouro_rates()
    assert response.read * 16384 < len(response.content) / 10
    assert lines[1].startswith("Tesouro Selic;")


def test_tesouro_rejects_an_unexpected_file(monkeypatch):
    monkeypatch.setattr(market_strip.requests, "get", lambda url, stream, timeout: Response(content=b"<html>erro</html>\n"))
    with pytest.raises(ValueError):
        market_strip.get_tesouro_rates()


def test_strip_has_no_removed_series_and_lists_new_items(monkeypatch):
    monkeypatch.setattr(market_strip, "get_macro_series", lambda series, today, window: SELIC_ROWS if series == 432 else IPCA_ROWS)
    monkeypatch.setattr(market_strip, "get_focus_annual", lambda indicator, year: FOCUS_IPCA_ROWS)
    monkeypatch.setattr(market_strip, "get_focus_inflation_12m", lambda: FOCUS_12M_ROWS)
    monkeypatch.setattr(market_strip, "get_tesouro_rates", lambda: TESOURO_CSV.splitlines()[:-1])
    monkeypatch.setattr(market_strip, "get_market_quotes", lambda: ([], []))
    result = market_strip.market_strip()
    keys = [item["key"] for item in result["items"]]
    assert keys == ["selic", "focus_ipca", "focus_selic", "real_rate", "ipca", "ntnb_2035", "ntnb_2045"]
    assert not {"cdi", "unemployment", "ibx50"} & set(keys)
    assert result["missing"] == []


def test_strip_reports_each_failed_source_as_missing_without_placeholders(monkeypatch):
    def macro(series, today, window):
        if series == 13522:
            raise TimeoutError("BCB offline")
        return SELIC_ROWS

    def offline(*args):
        raise TimeoutError("offline")

    monkeypatch.setattr(market_strip, "get_macro_series", macro)
    monkeypatch.setattr(market_strip, "get_focus_annual", lambda indicator, year: offline() if indicator == "IPCA" else FOCUS_IPCA_ROWS)
    monkeypatch.setattr(market_strip, "get_focus_inflation_12m", offline)
    monkeypatch.setattr(market_strip, "get_tesouro_rates", offline)
    monkeypatch.setattr(market_strip, "get_market_quotes", lambda: (_ for _ in ()).throw(ValueError("Yahoo offline")))
    result = market_strip.market_strip()
    assert [item["key"] for item in result["items"]] == ["selic", "focus_selic"]
    assert "Selic Focus" not in result["missing"]
    for label in ("IPCA", "IPCA Focus", "Juro real ex-ante", "IPCA+ 2035", "IPCA+ 2045",
                  "Ibovespa", "S&P 500", "Brent", "PETR4"):
        assert label in result["missing"]


def test_market_strip_route_returns_clean_json(monkeypatch):
    monkeypatch.setattr(market, "market_strip", lambda: {"items": [{"key": "x", "value": float("nan")}], "missing": [], "updated_at": "2026-10-12T20:00:00+00:00"})
    app = FastAPI()
    app.include_router(market.router)
    with TestClient(app) as client:
        response = client.get("/api/market/strip")
    assert response.status_code == 200
    assert response.json()["items"][0]["value"] is None
