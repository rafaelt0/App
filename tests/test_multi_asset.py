from datetime import date, timedelta
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import app
from backend import portfolio, simulation, state
from utils import assets, db, portfolio_data, treasury_data


def test_catalog_selection_favorites_and_saved_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", tmp_path / "multi-asset.sqlite3")
    monkeypatch.setattr(state, "_states", {})
    tickers = ["AAPL", "HGLG11", "VOO", "TD-SELIC-2031", "O"]
    headers = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}
    with TestClient(app, base_url=headers["Origin"], headers=headers) as client:
        catalog = {row["ticker"]: row for row in client.get("/api/universe").json()["stocks"]}
        assert [catalog[ticker]["asset_class"] for ticker in tickers] == [
            "acao_us", "fii", "etf", "titulo", "reit",
        ]
        assert catalog["TD-SELIC-2031"]["share_step"] == .01
        assert catalog["TD-SELIC-2031"]["supports_shares"]
        assert not catalog["TESOURO-SELIC"]["supports_shares"]
        revision = client.get("/api/session").json()["revision"]
        response = client.put("/api/session/selection", json={"tickers": tickers, "revision": revision})
        assert response.status_code == 200, response.text
        assert client.put("/api/watchlist/TD-SELIC-2031").json() == {"watchlist": ["TD-SELIC-2031"]}
        assert client.put("/api/session/selection", json={"tickers": ["INVALID/TICKER"], "revision": revision}).status_code == 422
        uid, = state._states
        assert db.portfolio_save(uid, tickers, dict.fromkeys(tickers, .2))
        monkeypatch.delitem(state._states, uid)
        assert client.get("/api/session").json()["selection"] == tickers


def test_mixed_prices_use_same_day_fx_and_distinct_treasury_buy_price(monkeypatch):
    start = date.today() - timedelta(days=8)
    dates = pd.bdate_range(start, periods=3)
    source = pd.DataFrame({
        "PETR4.SA": [30., 31., 32.],
        "AAPL": [20., float("nan"), 22.],
        "BRL=X": [5., 5.2, 5.5],
    }, index=dates)
    csv = "Tipo Titulo;Data Vencimento;Data Base;PU Compra Manha;PU Base Manha\n"
    for day, buy, mark in zip(dates, (104, 105, 106), (100, 102, 103)):
        csv += f"Tesouro Selic;01/03/2031;{day:%d/%m/%Y};{buy},50;{mark},25\n"
        csv += f"Tesouro Selic;01/03/2029;{day:%d/%m/%Y};1,00;1,00\n"
    monkeypatch.setattr(treasury_data, "_download_treasury_csv", lambda: StringIO(csv))
    def download(symbols, start, auto_adjust=True):
        return source[symbols] if isinstance(symbols, str) else source.loc[:, list(symbols)]

    monkeypatch.setattr(portfolio_data, "_download_close", download)
    tickers = ("PETR4.SA", "AAPL", "TD-SELIC-2031")
    try:
        adjusted = portfolio_data.get_portfolio_prices(tickers, start)
        raw = portfolio_data.get_portfolio_trade_prices(tickers)
        assert adjusted.columns.tolist() == list(tickers)
        assert adjusted["AAPL"].tolist() == pytest.approx([100., 104., 121.])
        assert adjusted["TD-SELIC-2031"].tolist() == pytest.approx([100.25, 102.25, 103.25])
        assert raw["TD-SELIC-2031"].tolist() == pytest.approx([104.50, 105.50, 106.50])
        assert pd.isna(raw["AAPL"].iloc[1])
        assert raw["AAPL"].iloc[-1] == pytest.approx(121.)
        detail = assets.get_extra_asset_fundamentals("TD-SELIC-2031")
        assert detail["available"] and detail["price"] == pytest.approx(106.50)
        assert detail["source"] == "Tesouro Transparente · PU Compra Manhã"
    finally:
        for fn in (treasury_data.get_treasury_history, portfolio_data.get_portfolio_prices, portfolio_data.get_portfolio_trade_prices):
            fn.clear()


def test_treasury_units_and_mixed_simulation_columns(monkeypatch):
    tickers = ["PETR4", "AAPL", "TD-SELIC-2031"]
    request = portfolio.AnalyzeRequest(tickers=tickers, revision=0, manual_shares={"TD-SELIC-2031": .32})
    assert request.manual_shares["TD-SELIC-2031"] == .32
    for shares in ({"TD-SELIC-2031": .321}, {"AAPL": .32}, {"TD-SELIC-2031": True}):
        with pytest.raises(ValidationError):
            portfolio.AnalyzeRequest(tickers=tickers, revision=0, manual_shares=shares)
    quotes = {"TD-SELIC-2031": (2000., pd.Timestamp("2026-10-10"))}
    rows, totals, missing = portfolio.purchase_suggestion({"TD-SELIC-2031": 1.}, 650., quotes)
    assert rows[0]["shares"] == .32
    assert totals["cash"] == 10.
    assert missing == []

    expected = ("PETR4.SA", "AAPL", "TD-SELIC-2031")
    prices = pd.DataFrame({ticker: range(100, 140) for ticker in expected}, index=pd.bdate_range("2026-01-01", periods=40))

    def history(symbols, start):
        assert symbols == expected
        return prices

    monkeypatch.setattr(simulation, "get_portfolio_prices", history)
    saved = SimpleNamespace(tickers=tickers, weights=dict.fromkeys(tickers, 1 / 3), returns=prices.pct_change().dropna())
    for window in ("portfolio", 1):
        aligned, weights, _ = simulation.calibration_sample(saved, window, date.today())
        assert tuple(aligned.columns) == expected
        assert weights == pytest.approx([1 / 3] * 3)


def test_mixed_portfolio_calculates_and_exports_in_reais(monkeypatch):
    tickers = ["HGLG11", "VOO", "TD-SELIC-2031", "O"]
    columns = ["HGLG11.SA", "VOO", "TD-SELIC-2031", "O"]
    dates = pd.bdate_range(end=date.today() - timedelta(days=1), periods=65)
    prices = pd.DataFrame({
        ticker: [100 + position * day / 10 + day % (position + 2) for day in range(len(dates))]
        for position, ticker in enumerate(columns, start=1)
    }, index=dates)
    benchmark_holiday = dates[31]
    benchmark = prices["HGLG11.SA"].drop([benchmark_holiday, dates[-1]])
    monkeypatch.setattr(portfolio, "get_portfolio_prices", lambda symbols, start: prices.loc[:, list(symbols)])
    monkeypatch.setattr(portfolio, "get_portfolio_trade_prices", lambda symbols: prices.loc[dates[-1:], list(symbols)])
    monkeypatch.setattr(portfolio, "get_benchmark_prices", lambda start: benchmark)
    monkeypatch.setattr(portfolio, "get_selic_rate", lambda: .0001)
    monkeypatch.setattr(portfolio, "get_cdi_daily_rates", lambda start: pd.Series(.0001, index=dates))
    monkeypatch.setattr(db, "portfolio_save", lambda *args: True)
    ctx = state.SessionState(uid="portfolio-test", selection=tickers)
    request = SimpleNamespace(state=SimpleNamespace(session=ctx))
    payload = portfolio.AnalyzeRequest(
        tickers=tickers, revision=0, strategy="manual",
        manual_shares={"HGLG11": 1, "VOO": 1, "TD-SELIC-2031": .25, "O": 1},
    )
    result = portfolio.analyze(payload, request)
    assert result["saved"]
    assert [row["ticker"] for row in result["weights"]] == tickers
    position_values = prices.iloc[-1].mul([1, 1, .25, 1])
    assert ctx.portfolio.weights == pytest.approx(dict(zip(tickers, position_values / position_values.sum())))
    expected_daily = (prices.pct_change().dropna()).dot(position_values / position_values.sum())
    assert result["metrics"]["total_return"] == pytest.approx(((1 + expected_daily).prod() - 1) * 100)
    assert result["comparison_sample"]["count"] == len(dates) - 2
    assert "USD/BRL" in result["source"]["adjusted_prices"]
    assert "PU Compra Manhã" in result["source"]["trade_quotes"]
    assert "Preço de referência (R$)" in portfolio.trades_csv(result["id"], request).body.decode("utf-8")


def test_fx_carries_recent_quote_across_calendar_gaps(monkeypatch):
    dates = pd.DatetimeIndex(["2025-04-17", "2025-04-21", "2025-04-22"])
    closes = pd.DataFrame({
        "AAPL": [20., 21., 22.], "PETR4.SA": [30., float("nan"), 31.],
        "BRL=X": [5., float("nan"), 5.5],
    }, index=dates)

    def download(symbols, start, auto_adjust=True):
        if isinstance(symbols, str):
            if auto_adjust:
                assert start == dates[0].date()
            return closes[symbols]
        data = closes.loc[:, list(symbols)]
        return data.loc[data.index.date >= start] if auto_adjust else data

    monkeypatch.setattr(portfolio_data, "_download_close", download)
    tickers = ("AAPL", "PETR4.SA")
    try:
        adjusted = portfolio_data.get_portfolio_prices(tickers, dates[1].date())
        assert adjusted.index.tolist() == dates[1:].tolist()
        assert adjusted["AAPL"].tolist() == pytest.approx([105., 121.])
        raw = portfolio_data.get_portfolio_trade_prices(tickers)
        assert pd.isna(raw.loc[dates[1], "AAPL"])
    finally:
        portfolio_data.get_portfolio_prices.clear()
        portfolio_data.get_portfolio_trade_prices.clear()


def test_history_rejects_fx_gap_when_previous_rate_is_too_old(monkeypatch):
    dates = pd.DatetimeIndex(["2025-05-05", "2025-05-12", "2025-05-13"])
    closes = pd.DataFrame({
        "AAPL": [20., float("nan"), 22.], "PETR4.SA": [30., 31., 32.],
        "BRL=X": [5., float("nan"), 5.5],
    }, index=dates)
    def download(symbols, *args, **kwargs):
        return closes[symbols] if isinstance(symbols, str) else closes.loc[:, list(symbols)]

    monkeypatch.setattr(portfolio_data, "_download_close", download)
    with pytest.raises(ValueError, match=r"USD/BRL.*AAPL.*12/05/2025"):
        portfolio_data.get_portfolio_prices(("AAPL", "PETR4.SA"), dates[0].date())


def test_raw_quote_keeps_exchange_prices_when_treasury_unavailable(monkeypatch):
    def unavailable(tickers, start_date, adjusted):
        raise ValueError("Tesouro Transparente indisponível")

    monkeypatch.setattr(portfolio_data, "treasury_prices", unavailable)
    dates = pd.bdate_range(end=date.today() - timedelta(days=1), periods=2)
    yahoo = pd.DataFrame({"PETR4.SA": [35., 36.]}, index=dates)
    monkeypatch.setattr(portfolio_data, "_download_close", lambda *args, **kwargs: yahoo)
    tickers = ("TD-PREFIXADO-2029", "PETR4.SA")
    try:
        quotes, failed = portfolio.trade_quotes(tickers)
        assert not failed
        assert quotes == {"PETR4.SA": (36., dates[-1])}
    finally:
        portfolio_data.get_portfolio_trade_prices.clear()


def test_simulation_restore_retries_with_positive_weights_after_quote_failure(monkeypatch):
    tickers = ["AAPL", "TD-SELIC-2031"]
    ctx = state.SessionState(uid="restore-test", selection=tickers, restored=True)
    monkeypatch.setattr(db, "portfolio_get", lambda uid: (tickers, {"AAPL": 1., "TD-SELIC-2031": 0.}))
    dates = pd.bdate_range(end=date.today() - timedelta(days=1), periods=40)
    prices = pd.DataFrame({"AAPL": range(100, 140)}, index=dates)

    def unavailable(symbols, start):
        raise ValueError("Fonte indisponível")

    monkeypatch.setattr(portfolio, "get_portfolio_prices", unavailable)
    news = portfolio.restore_portfolio(ctx, "news", 0)
    with pytest.raises(HTTPException, match="Fonte indisponível"):
        portfolio.restore_portfolio(ctx, "simulation", 0)
    assert state.require_portfolio(ctx) is news

    def download(symbols, start):
        assert symbols == ("AAPL",)
        return prices

    monkeypatch.setattr(portfolio, "get_portfolio_prices", download)
    restored = portfolio.restore_portfolio(ctx, "simulation", 0)
    assert restored.weights == {"AAPL": 1., "TD-SELIC-2031": 0.}
    assert state.require_portfolio(ctx, returns=True) is restored


def test_stress_compares_treasury_with_benchmark_over_same_valuation_window():
    dates = pd.bdate_range("2025-04-01", periods=15)
    treasury = pd.DataFrame({"TD-IPCA-2035": 100.}, index=dates[5:12])
    benchmark = pd.Series([100.] * 3 + [120.] * 10 + [130.] * 2, index=dates)
    result = portfolio_data.calculate_historical_stress(
        treasury, benchmark, {"TD-IPCA-2035": 1.}, {"Crise": (dates[0], dates[-1])},
    )
    assert result[0]["Portfólio"] == result[0]["IBOV"] == 0
