import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import app as api_module, home, portfolio, screener, simulation, state
from utils import assets, db, portfolio_data

HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "multi_assets.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_universe_includes_all_required_asset_classes(client):
    response = client.get("/api/universe")
    assert response.status_code == 200
    body = response.json()
    stocks = body["stocks"]
    by_ticker = {item["ticker"]: item for item in stocks}
    classes = {item["asset_class"] for item in stocks}

    assert {"acao_br", "fii", "acao_us", "reit", "titulo", "etf", "outros"} <= classes
    # B3 equities
    assert by_ticker["PETR4"]["asset_class"] == "acao_br" and by_ticker["PETR4"]["currency"] == "BRL"
    # FIIs
    assert by_ticker["HGLG11"]["asset_class"] == "fii" and by_ticker["HGLG11"]["currency"] == "BRL"
    assert by_ticker["KNCR11"]["asset_class"] == "fii"
    # US Equities
    assert by_ticker["AAPL"]["asset_class"] == "acao_us" and by_ticker["AAPL"]["currency"] == "USD"
    assert by_ticker["NVDA"]["asset_class"] == "acao_us"
    # REITs
    assert by_ticker["O"]["asset_class"] == "reit" and by_ticker["O"]["currency"] == "USD"
    assert by_ticker["PLD"]["asset_class"] == "reit"
    # Títulos & Renda Fixa (BR & US)
    assert by_ticker["CDI"]["asset_class"] == "titulo" and by_ticker["CDI"]["currency"] == "BRL"
    assert by_ticker["TESOURO-SELIC"]["asset_class"] == "titulo"
    assert by_ticker["TESOURO-IPCA-2035"]["asset_class"] == "titulo"
    assert by_ticker["IMAB11"]["asset_class"] == "titulo"
    assert by_ticker["TLT"]["asset_class"] == "titulo" and by_ticker["TLT"]["currency"] == "USD"


def test_universe_search_resolves_catalog_custom_fixed_income_and_custom_tickers(client, monkeypatch):
    monkeypatch.setattr(assets, "search_yahoo_assets", lambda q: [
        {"ticker": "CROX", "company": "Crocs, Inc.", "legal_name": "Crocs, Inc.",
         "sector": "Consumo (EUA)", "asset_class": "acao_us", "currency": "USD", "yahoo_symbol": "CROX"}
    ] if "crox" in q.lower() else [])

    # Search for custom fixed income expression
    fi_res = client.get("/api/universe/search?q=CDB-112CDI").json()
    assert fi_res["items"][0]["ticker"] == "CDB-112CDI"
    assert fi_res["items"][0]["asset_class"] == "titulo"

    ipca_res = client.get("/api/universe/search?q=IPCA%2B7.5").json()
    assert ipca_res["items"][0]["ticker"] == "IPCA+7.5"
    assert ipca_res["items"][0]["asset_class"] == "titulo"

    # Search for remote Yahoo Finance asset
    remote_res = client.get("/api/universe/search?q=CROX").json()
    assert any(item["ticker"] == "CROX" and item["company"] == "Crocs, Inc." for item in remote_res["items"])


def test_session_selection_and_watchlist_accept_all_asset_classes(client):
    client.get("/api/session")
    mixed = ["PETR4", "HGLG11", "AAPL", "O", "TLT", "CDI", "TESOURO-IPCA-2035", "CDB-112CDI", "SPY", "BTC-USD"]
    selected = client.put("/api/session/selection", headers=HEADERS, json={"tickers": mixed, "revision": 0})
    assert selected.status_code == 200
    assert selected.json()["selection"] == mixed

    for ticker in ("HGLG11", "AAPL", "O", "TLT", "CDI", "CDB-112CDI"):
        fav = client.put(f"/api/watchlist/{ticker}", headers=HEADERS)
        assert fav.status_code == 200
        assert ticker in fav.json()["watchlist"]


def test_home_analysis_supports_mixed_b3_us_fii_reit_and_fixed_income(client, monkeypatch):
    client.get("/api/session")
    tickers = ["AAPL", "HGLG11", "O", "TLT", "CDI", "TESOURO-IPCA-2035"]
    sel = client.put("/api/session/selection", headers=HEADERS, json={"tickers": tickers, "revision": 0}).json()

    monkeypatch.setattr(home, "get_market_target_data", lambda t: {
        "company_name": t,
        "currency": "USD" if t in {"AAPL", "O", "TLT"} else "BRL",
        "price": 220.0 if t == "AAPL" else 58.0 if t == "O" else None,
        "target_low": 200.0 if t == "AAPL" else None,
        "target_mean": 250.0 if t == "AAPL" else None,
        "target_median": 248.0 if t == "AAPL" else None,
        "target_high": 270.0 if t == "AAPL" else None,
        "analyst_count": 35 if t == "AAPL" else None,
        "recommendation": "buy" if t == "AAPL" else None,
    })

    response = client.post("/api/home/analyze", headers=HEADERS, json={"tickers": tickers, "revision": sel["revision"]})
    assert response.status_code == 200
    data = response.json()
    by_ticker = {c["ticker"]: c for c in data["companies"]}

    assert set(by_ticker) == set(tickers)
    assert by_ticker["AAPL"]["market"]["currency"] == "USD"
    assert by_ticker["O"]["market"]["currency"] == "USD"
    assert by_ticker["HGLG11"]["market"]["currency"] == "BRL"
    assert by_ticker["CDI"]["market"]["currency"] == "BRL"
    assert by_ticker["CDI"]["indicators"]["dy"] == pytest.approx(13.75, rel=0.1)
    assert by_ticker["TESOURO-IPCA-2035"]["indicators"]["dy"] > 10.0


def test_portfolio_and_simulation_with_multi_asset_classes(monkeypatch):
    dates = pd.bdate_range(end=pd.Timestamp(dt.date.today()) - pd.Timedelta(days=1), periods=120)
    x = np.arange(len(dates))
    yahoo_prices = pd.DataFrame({
        "PETR4.SA": 35.0 * np.exp(0.0004 * x + 0.015 * np.sin(x * 0.2)),
        "HGLG11.SA": 155.0 * np.exp(0.0003 * x + 0.006 * np.cos(x * 0.17)),
        "AAPL": 200.0 * np.exp(0.0006 * x + 0.012 * np.sin(x * 0.25)),
        "O": 55.0 * np.exp(0.0003 * x + 0.009 * np.cos(x * 0.21)),
        "TLT": 92.0 * np.exp(0.0002 * x + 0.007 * np.sin(x * 0.15)),
        "BRL=X": 5.20 + 0.30 * np.sin(x * 0.05),
    }, index=dates)

    def fake_download(symbols, start_date, auto_adjust=True):
        if symbols == "BRL=X":
            return yahoo_prices["BRL=X"]
        cols = [symbols] if isinstance(symbols, str) else list(symbols)
        return yahoo_prices.loc[:, cols]

    monkeypatch.setattr(portfolio_data, "_download_close", fake_download)
    monkeypatch.setattr(portfolio_data, "get_cdi_daily_rates", lambda start: pd.Series(0.0005, index=dates))
    monkeypatch.setattr(portfolio, "get_cdi_daily_rates", lambda start: pd.Series(0.0005, index=dates))
    monkeypatch.setattr(portfolio, "get_selic_rate", lambda: 0.0005)
    monkeypatch.setattr(portfolio, "get_benchmark_prices", lambda start: pd.Series(120000.0 * np.exp(0.0003 * x + 0.01 * np.sin(x * 0.18)), index=dates))
    monkeypatch.setattr(portfolio.db, "portfolio_save", lambda uid, tickers, weights: True)

    portfolio_data.get_portfolio_prices.clear()
    portfolio_data.get_portfolio_trade_prices.clear()

    tickers = ["PETR4", "HGLG11", "AAPL", "O", "TLT", "CDI", "TESOURO-IPCA-2035"]
    ctx = state.SessionState(uid="multi-owner", selection=tickers)
    req = SimpleNamespace(state=SimpleNamespace(session=ctx))

    for strategy in ("min_volatility", "markowitz", "hrp"):
        res = portfolio.analyze(
            portfolio.AnalyzeRequest(tickers=tickers, revision=ctx.revision, strategy=strategy, capital=50000, lookback_days=365),
            req,
        )
        assert len(res["weights"]) == len(tickers)
        assert sum(w["weight"] for w in res["weights"]) == pytest.approx(1.0, abs=1e-4)
        assert res["trade_totals"]["allocated"] <= 50000

    # Verify Monte Carlo Simulation runs on the resulting multi-asset portfolio snapshot
    sim = simulation.compute_simulation(
        ctx.portfolio,
        simulation.SimulationRequest(revision=ctx.revision, portfolio_id=ctx.portfolio.id, simulations=100, initial_capital=50000, years=1),
    )
    assert sim.result["stats"]["mean_terminal"] > 0
    assert len(sim.result["allocation"]) == len(tickers)


def test_screener_supports_all_multi_asset_classes(client):
    client.get("/api/session")
    cfg = client.get("/api/screener/config").json()
    class_keys = {item["key"] for item in cfg["asset_classes"]}
    assert {"b3", "all", "fii", "acao_us", "reit", "titulo", "etf"} <= class_keys

    for cls in ("fii", "acao_us", "reit", "titulo", "etf"):
        res = client.post("/api/screener/query", headers=HEADERS, json={
            "asset_class": cls,
            "criteria": {"liq2m_min": 1_000_000},
            "sort": {"field": "liq2m", "direction": "desc"},
        })
        assert res.status_code == 200
        body = res.json()
        assert body["matched_count"] > 0
        assert len(body["rows"]) == body["matched_count"]
