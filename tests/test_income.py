"""Deterministic regressions for the unified proventos calendar (WP-1.3)."""

import datetime as dt

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import income as income_api
from backend.state import PortfolioSnapshot, SessionState
from utils import income as engine

TODAY = dt.date(2026, 6, 15)
# The trailing window for TODAY is 2025-06-01 .. 2026-05-31 (completed months only).
CDI_ANNUAL = 0.1375


def dividend_series(pairs) -> pd.Series:
    return pd.Series(
        {pd.Timestamp(day): float(value) for day, value in pairs},
        dtype=float,
    )


def asset_history(ticker, pairs, *, currency="BRL", status="ok", note=None) -> dict:
    return {
        "ticker": ticker,
        "symbol": ticker,
        "currency": currency,
        "status": status,
        "income_mode": "dividends" if status == "ok" else "not_applicable",
        "dividends_brl": dividend_series(pairs) if pairs else pd.Series(dtype=float),
        "source": "Yahoo Finance",
        "note": note,
    }


def patch_histories(monkeypatch, mapping: dict[str, object]) -> None:
    def load(ticker, start_date):
        entry = mapping[ticker]
        if isinstance(entry, Exception):
            raise entry
        return entry

    monkeypatch.setattr(engine, "get_asset_dividend_history", load)


@pytest.fixture(autouse=True)
def market_stubs(monkeypatch):
    for cached in (engine.get_asset_dividend_history, engine.get_asset_price_brl, engine.get_fx_brl_history):
        cached.clear()
    monkeypatch.setattr(engine, "get_selic_rate", lambda: (1 + CDI_ANNUAL) ** (1 / 252) - 1)
    prices: dict[str, float] = {}

    def price(ticker):
        return {
            "ticker": ticker,
            "status": "ok",
            "price_brl": prices.get(ticker, 100.0),
            "price_date": "2026-06-12",
            "currency": "BRL",
            "source": "stub",
            "note": None,
        }

    monkeypatch.setattr(engine, "get_asset_price_brl", price)
    yield prices


def test_tax_math_fii_brazil_jcp_and_us_withholding(monkeypatch):
    patch_histories(monkeypatch, {
        "HGLG11": asset_history("HGLG11", [
            ("2025-08-15", 3), ("2025-11-15", 3), ("2026-02-15", 3), ("2026-05-15", 3),
        ]),
        "PETR4": asset_history("PETR4", [
            ("2025-09-15", 1), ("2025-12-15", 1), ("2026-03-15", 1), ("2026-05-15", 1),
        ]),
        "AAPL": asset_history("AAPL", [
            ("2025-09-15", .5), ("2025-12-15", .5), ("2026-03-15", .5), ("2026-05-15", .5),
        ], currency="USD"),
    })
    result = engine.build_income_calendar(
        ["HGLG11", "PETR4", "AAPL"],
        {"HGLG11": .5, "PETR4": .3, "AAPL": .2},
        100_000,
        jcp_share=.5,
        today=TODAY,
    )
    assets = {asset["ticker"]: asset for asset in result["assets"]}

    # 500 cotas x R$ 12 = R$ 6.000; FII distribution is exempt.
    assert assets["HGLG11"]["tax_rate"] == 0
    assert assets["HGLG11"]["annual_gross_brl"] == pytest.approx(6000)
    assert assets["HGLG11"]["annual_net_brl"] == pytest.approx(6000)
    # 300 cotas x R$ 4 = R$ 1.200; 50% estimated as JCP at 15% => 7,5% blended IR.
    assert assets["PETR4"]["tax_rate"] == pytest.approx(.075)
    assert assets["PETR4"]["annual_gross_brl"] == pytest.approx(1200)
    assert assets["PETR4"]["annual_net_brl"] == pytest.approx(1110)
    # 200 cotas x R$ 2 = R$ 400; 30% US withholding.
    assert assets["AAPL"]["tax_rate"] == pytest.approx(.30)
    assert assets["AAPL"]["annual_gross_brl"] == pytest.approx(400)
    assert assets["AAPL"]["annual_net_brl"] == pytest.approx(280)
    assert result["annual_gross_brl"] == pytest.approx(7600)
    assert result["annual_net_brl"] == pytest.approx(7390)
    assert result["portfolio_yield_net"] == pytest.approx(.0739)


def test_monthly_aggregation_and_capital_scaling(monkeypatch, market_stubs):
    patch_histories(monkeypatch, {
        "HGLG11": asset_history("HGLG11", [("2025-07-10", 1), ("2026-01-10", 2)]),
    })
    market_stubs["HGLG11"] = 10.0
    small = engine.build_income_calendar(
        ["HGLG11"], {"HGLG11": 1.0}, 10_000, today=TODAY,
    )
    large = engine.build_income_calendar(
        ["HGLG11"], {"HGLG11": 1.0}, 100_000, today=TODAY,
    )

    # 1.000 cotas x R$ 3 por cota = R$ 3.000; capital x10 scales linearly.
    assert small["annual_gross_brl"] == pytest.approx(3000)
    assert small["annual_net_brl"] == pytest.approx(3000)
    assert small["monthly_average_net_brl"] == pytest.approx(250)
    assert large["annual_gross_brl"] == pytest.approx(30_000)
    assert large["annual_net_brl"] == pytest.approx(30_000)

    calendar = {item["month_index"]: item for item in small["calendar"]}
    assert len(small["calendar"]) == 12
    assert calendar[7]["month_label"] == "Jul/2026"
    assert calendar[7]["gross_brl"] == pytest.approx(1000)
    assert calendar[7]["net_brl"] == pytest.approx(1000)
    assert calendar[7]["contributors"][0]["ticker"] == "HGLG11"
    assert calendar[1]["month_label"] == "Jan/2027"
    assert calendar[1]["gross_brl"] == pytest.approx(2000)
    assert calendar[6]["gross_brl"] == 0

    trailing = {item["month_label"]: item for item in small["trailing_calendar"]}
    assert len(small["trailing_calendar"]) == 12
    assert trailing["Jul/2025"]["gross_brl"] == pytest.approx(1000)
    assert trailing["Jan/2026"]["gross_brl"] == pytest.approx(2000)


def test_yield_on_cost_and_cdi_ratio(monkeypatch, market_stubs):
    patch_histories(monkeypatch, {
        "HGLG11": asset_history("HGLG11", [("2025-08-15", 1), ("2026-02-15", 2)]),
    })
    market_stubs["HGLG11"] = 10.0
    with_cost = engine.build_income_calendar(
        ["HGLG11"], {"HGLG11": 1.0}, 100_000, cost_basis_brl=50_000, today=TODAY,
    )
    without_cost = engine.build_income_calendar(
        ["HGLG11"], {"HGLG11": 1.0}, 100_000, today=TODAY,
    )

    annual_net = 10_000 * 3  # 10.000 cotas x R$ 3 por cota
    assert with_cost["annual_net_brl"] == pytest.approx(annual_net)
    assert with_cost["yield_on_cost_net"] == pytest.approx(annual_net / 50_000)
    assert without_cost["yield_on_cost_net"] is None
    assert with_cost["portfolio_yield_net"] == pytest.approx(annual_net / 100_000)
    assert with_cost["cdi_annual_rate"] == pytest.approx(CDI_ANNUAL)
    assert with_cost["cdi_net_rate"] == pytest.approx(CDI_ANNUAL * .85)
    assert with_cost["yield_vs_cdi_ratio"] == pytest.approx((annual_net / 100_000) / (CDI_ANNUAL * .85))


def test_treasury_non_coupon_title_reports_zero_without_source(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tesouro Direto must not query dividend history")

    monkeypatch.setattr(engine, "get_asset_dividend_history", forbidden)
    result = engine.build_income_calendar(
        ["TD-SELIC-2029"], {"TD-SELIC-2029": 1.0}, 10_000, today=TODAY,
    )
    asset = result["assets"][0]

    assert asset["income_mode"] == "no_intermediate_coupons"
    assert asset["status"] == "no_intermediate_coupons"
    assert asset["currency"] == "BRL"
    assert asset["annual_gross_brl"] == 0
    assert asset["annual_net_brl"] == 0
    assert asset["note"] == engine.TREASURY_NO_COUPON_NOTE
    assert "TD-SELIC-2029" not in result["missing_tickers"]
    assert "TD-SELIC-2029" in result["not_applicable_tickers"]
    assert result["annual_net_brl"] == 0
    assert all(item["net_brl"] == 0 for item in result["calendar"])


def test_missing_source_and_missing_price_warn_without_fabricating(monkeypatch, market_stubs):
    patch_histories(monkeypatch, {
        "PETR4": asset_history("PETR4", [("2025-12-15", 1.0)]),
        "AAPL": RuntimeError("provider down"),
        "VALE3": asset_history("VALE3", [("2025-12-15", 2.0)]),
    })
    market_stubs["VALE3"] = 0.0  # zero price is treated as unavailable, never divided by zero.
    result = engine.build_income_calendar(
        ["PETR4", "AAPL", "VALE3"],
        {"PETR4": .5, "AAPL": .3, "VALE3": .2},
        100_000,
        today=TODAY,
    )
    assets = {asset["ticker"]: asset for asset in result["assets"]}

    assert result["missing_tickers"] == ["AAPL", "VALE3"]
    assert assets["AAPL"]["status"] == "missing"
    assert assets["AAPL"]["annual_gross_brl"] == 0
    assert assets["VALE3"]["status"] == "missing"
    assert assets["VALE3"]["annual_gross_brl"] == 0
    assert assets["PETR4"]["status"] == "ok"
    # Only PETR4 contributes: 500 cotas x R$ 1 = R$ 500.
    assert result["annual_gross_brl"] == pytest.approx(500)
    assert result["annual_net_brl"] == pytest.approx(500)


def test_usd_dividends_are_converted_with_brl_fx(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.dividends = pd.Series({
                pd.Timestamp("2025-09-15"): 1.0,
                pd.Timestamp("2026-03-15"): .5,
            })

    monkeypatch.setattr(engine.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(engine, "get_fx_brl_history", lambda start_date: pd.Series({
        pd.Timestamp("2025-09-01"): 5.0,
        pd.Timestamp("2026-03-01"): 6.0,
    }))

    history = engine.get_asset_dividend_history("AAPL", dt.date(2025, 6, 1))

    assert history["status"] == "ok"
    assert history["currency"] == "USD"
    assert history["dividends_brl"].loc[pd.Timestamp("2025-09-15")] == pytest.approx(5.0)
    assert history["dividends_brl"].loc[pd.Timestamp("2026-03-15")] == pytest.approx(3.0)
    assert "BRL=X" in history["source"]


@pytest.fixture
def context():
    tickers = ["HGLG11", "PETR4"]
    portfolio = PortfolioSnapshot(
        "portfolio-one", 3, tickers, {"HGLG11": .5, "PETR4": .5},
        "manual", pd.DataFrame(), pd.DataFrame(), 365, {},
    )
    return SessionState("owner-one", selection=tickers, revision=3, portfolio=portfolio)


@pytest.fixture
def client(context):
    app = FastAPI()

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = context
        return await call_next(request)

    app.include_router(income_api.router)
    with TestClient(app) as test_client:
        yield test_client


def endpoint_histories(monkeypatch):
    patch_histories(monkeypatch, {
        "HGLG11": asset_history("HGLG11", [
            ("2025-08-15", 3), ("2025-11-15", 3), ("2026-02-15", 3), ("2026-05-15", 3),
        ]),
        "PETR4": asset_history("PETR4", [
            ("2025-09-15", 1), ("2025-12-15", 1), ("2026-03-15", 1), ("2026-05-15", 1),
        ]),
    })


def test_endpoint_returns_calendar_and_figure(client, monkeypatch):
    endpoint_histories(monkeypatch)
    response = client.post("/api/income/calendar", json={
        "revision": 3, "portfolio_id": "portfolio-one", "capital": 100_000,
        "cost_basis_brl": 80_000, "jcp_share": .2,
    })
    data = response.json()

    assert response.status_code == 200
    assert data["portfolio_id"] == "portfolio-one"
    assert data["revision"] == 3
    assert data["annual_gross_brl"] == pytest.approx(8000)
    assert data["annual_net_brl"] == pytest.approx(7940)  # PETR4 JCP 20% => 3% IR
    assert data["yield_on_cost_net"] == pytest.approx(7940 / 80_000)
    assert len(data["calendar"]) == 12
    assert data["chart"]["data"][0]["type"] == "bar"
    assert data["missing_tickers"] == []


def test_endpoint_rejects_stale_or_foreign_portfolio(client, monkeypatch):
    endpoint_histories(monkeypatch)
    body = {"revision": 3, "portfolio_id": "portfolio-one", "capital": 100_000}

    assert client.post("/api/income/calendar", json={**body, "portfolio_id": "other"}).status_code == 409
    assert client.post("/api/income/calendar", json={**body, "revision": 2}).status_code == 409


def test_endpoint_validates_request_bounds(client):
    assert client.post("/api/income/calendar", json={
        "revision": 3, "portfolio_id": "portfolio-one", "capital": 50,
    }).status_code == 422
    assert client.post("/api/income/calendar", json={
        "revision": 3, "portfolio_id": "portfolio-one", "capital": 100_000, "jcp_share": 1.5,
    }).status_code == 422
    assert client.post("/api/income/calendar", json={
        "revision": 3, "portfolio_id": "portfolio-one", "capital": 100_000, "cost_basis_brl": 0,
    }).status_code == 422
