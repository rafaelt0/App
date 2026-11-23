import datetime as dt
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import app as api_module, home, news, portfolio, screener, simulation, state
from utils import assets, db, portfolio_data
from utils.news import is_relevant_company_news

HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "multi_assets.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    monkeypatch.setattr(home, "search_yahoo_assets", lambda query: [])
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
    monkeypatch.setattr(home, "search_yahoo_assets", lambda q: [
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
    tickers = ["PETR4", "AAPL", "HGLG11", "O", "TLT", "CDI", "TESOURO-IPCA-2035", "SPY", "AAPL34", "BTC-USD"]
    sel = client.put("/api/session/selection", headers=HEADERS, json={"tickers": tickers, "revision": 0}).json()
    monkeypatch.setattr(home, "get_fundamentus_data", lambda tickers: pd.DataFrame(
        {"Empresa": ["Petrobras"], "Setor": ["Energia"], "Cotacao": ["35,00"], "PL": [600], "ROIC": ["12,00%"]}, index=tickers))
    monkeypatch.setattr(home, "get_sector_peers", pd.DataFrame)
    assets.get_extra_asset_fundamentals.clear()

    def yahoo_ticker(symbol):
        quote_type = {"TLT": "ETF", "SPY": "ETF", "BTC-USD": "CRYPTOCURRENCY"}.get(symbol, "EQUITY")
        return SimpleNamespace(info={"quoteType": quote_type, "regularMarketPrice": 90, "totalAssets": 3000000,
                                     "trailingPE": 19, "priceToBook": 1.25, "returnOnEquity": .15, "returnOnAssets": .12,
                                     "revenueGrowth": .25, "trailingAnnualDividendYield": 0.0, "yield": .035,
                                     "debtToEquity": 200, "currentRatio": 2, "enterpriseValue": 100, "ebitda": 10})

    monkeypatch.setattr(assets.yf, "Ticker", yahoo_ticker)

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
    assets.get_extra_asset_fundamentals.clear()
    assert response.status_code == 200
    data = response.json()
    by_ticker = {c["ticker"]: c for c in data["companies"]}

    assert set(by_ticker) == set(tickers)
    assert by_ticker["AAPL"]["market"]["currency"] == "USD"
    assert by_ticker["O"]["market"]["currency"] == "USD"
    assert by_ticker["HGLG11"]["market"]["currency"] == "BRL"
    assert by_ticker["CDI"]["market"]["currency"] == "BRL"
    assert by_ticker["PETR4"]["indicators"]["roic"] == 12
    assert by_ticker["AAPL"]["indicators"]["roe"] == 15
    assert by_ticker["AAPL"]["indicators"]["pl"] == 19
    assert "roic" not in by_ticker["AAPL"]["indicators"] and "growth" not in by_ticker["AAPL"]["indicators"]
    assert [metric["label"] for metric in by_ticker["AAPL"]["health"]["metrics"]] == ["Liquidez Corrente"]
    for ticker in ("HGLG11", "O"):
        assert by_ticker[ticker]["indicators"] == {"pvp": 1.25, "dy": 0.0}
        assert by_ticker[ticker]["health"]["metrics"] == []
    assert by_ticker["SPY"]["indicators"] == {"dy": pytest.approx(3.5)}
    assert by_ticker["AAPL34"]["indicators"] == {"dy": 0.0}
    for ticker in ("TLT", "CDI", "TESOURO-IPCA-2035", "BTC-USD"):
        assert by_ticker[ticker]["indicators"] == by_ticker[ticker]["metric_groups"] == {}
        assert by_ticker[ticker]["indicators_available"] is False
        assert by_ticker[ticker]["health"]["metrics"] == []
    assert by_ticker["CDI"]["fixed_income"] == {"label": "Percentual do CDI", "rate": 100, "unit": "% do CDI"}
    assert by_ticker["TESOURO-IPCA-2035"]["fixed_income"]["rate"] == 6.5
    assert by_ticker["CDI"]["market"]["price"] is None
    assert by_ticker["TLT"]["market"]["price"] == 90
    for key in ("market_cap", "low_52w", "high_52w", "volume_2m", "last_quote_date"):
        assert by_ticker["TLT"]["market"][key] is None
    assert {row["ticker"] for row in data["consensus"]["rows"]} == {"PETR4", "AAPL", "O"}
    exported = pd.read_csv(io.BytesIO(client.get(f"/api/home/{data['id']}/fundamentals.csv").content), index_col=0)
    assert pd.isna(exported.loc["TLT", "Div_Yield"])
    assert exported.loc["CDI", "Unidade_taxa_modelo"] == "% do CDI"


@pytest.mark.parametrize(("symbol", "quote_type", "name", "industry", "expected"), [
    ("UNIT11.SA", "EQUITY", "Banco Unit", "Banks", "acao_br"),
    ("FUND11.SA", "EQUITY", "FII Modelo", "", "fii"),
    ("AGRO11.SA", "MUTUALFUND", "Fiagro Crédito Renda Fixa", "", "fii"),
    ("IMOB11.SA", "ETF", "ETF Imobiliário", "", "etf"),
    ("NEWB11.SA", "ETF", "ETF Renda Fixa", "", "titulo"),
    ("USBD", "ETF", "Treasury Bond Fund", "", "titulo"),
    ("NEWRE", "ETF", "Real Estate REIT ETF", "REIT", "etf"),
    ("SAMPLE", "EQUITY", "Real Estate Brokerage", "Real Estate Services", "acao_us"),
    ("SAMPLE", "EQUITY", "Real Estate Trust", "REIT - Retail", "reit"),
    ("XRP-USD", "CRYPTOCURRENCY", "XRP", "", "outros"),
])
def test_yahoo_classification_uses_instrument_type_before_theme(symbol, quote_type, name, industry, expected):
    entry = assets._classify_yahoo_quote({"symbol": symbol, "quoteType": quote_type, "longname": name, "industry": industry})
    assert entry["asset_class"] == expected


def test_home_uses_nasdaq_quote_when_yahoo_is_unavailable(client, monkeypatch):
    assets.get_extra_asset_fundamentals.clear()
    assets._get_nasdaq_quote.clear()
    home.get_market_target_data.clear()
    yahoo_info = {}
    nasdaq_calls = []
    monkeypatch.setattr(assets.yf, "Ticker", lambda symbol: SimpleNamespace(info=yahoo_info))

    def nasdaq_quote(url, *, params, **kwargs):
        nasdaq_calls.append(url)
        assert url.endswith("/AAPL/info")
        assert params == {"assetclass": "stocks"}
        if len(nasdaq_calls) == 1:
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": None})
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {
            "symbol": "AAPL", "assetClass": "STOCKS",
            "primaryData": {"lastSalePrice": "$237.51", "lastTradeTimestamp": "Nov 20, 2026 4:00 PM ET"},
            "keyStats": {"fiftyTwoWeekHighLow": {"value": "175.00 - 260.25"}},
        }})

    monkeypatch.setattr(assets.requests, "get", nasdaq_quote)
    client.get("/api/session")
    selected = client.put("/api/session/selection", headers=HEADERS, json={"tickers": ["AAPL"], "revision": 0})
    payload = {"tickers": ["AAPL"], "revision": selected.json()["revision"]}
    try:
        unavailable = client.post("/api/home/analyze", headers=HEADERS, json=payload)
        assert unavailable.status_code == 503
        response = client.post("/api/home/analyze", headers=HEADERS, json=payload)
        assert response.status_code == 200
        result = response.json()
        company = result["companies"][0]
        assert result["missing_tickers"] == []
        assert company["source"] == "Nasdaq"
        assert company["market"] == {
            "price": 237.51, "currency": "USD", "last_quote_date": "2026-11-20",
            "low_52w": 175.0, "high_52w": 260.25, "volume_2m": None, "market_cap": None,
        }
        assert company["indicators_available"] is False
        assert result["consensus"]["failed_tickers"] == ["AAPL"]
        assert result["consensus"]["rows"][0]["currency"] == "USD"
        export = client.get(f"/api/home/{result['id']}/fundamentals.csv")
        row = pd.read_csv(io.BytesIO(export.content)).iloc[0]
        assert row["Fonte"] == "Nasdaq" and row["Cotacao"] == 237.51

        yahoo_info.update(regularMarketPrice=238.0, trailingPE=19, targetMeanPrice=250)
        recovered = client.post("/api/home/analyze", headers=HEADERS, json=payload).json()
        assert recovered["companies"][0]["source"] == "Yahoo Finance"
        assert recovered["companies"][0]["indicators"]["pl"] == 19
        assert recovered["consensus"]["has_targets"] is True
        assert len(nasdaq_calls) == 2
    finally:
        assets.get_extra_asset_fundamentals.clear()
        assets._get_nasdaq_quote.clear()
        home.get_market_target_data.clear()


def test_unavailable_yahoo_quote_does_not_use_catalog_values(monkeypatch):
    def unavailable(symbol):
        raise RuntimeError("Yahoo unavailable")

    monkeypatch.setattr(assets.yf, "Ticker", unavailable)
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data": None}))
    assert assets.get_extra_asset_fundamentals.__wrapped__("AAPL")["available"] is False
    assert assets.get_extra_asset_fundamentals.__wrapped__("TLT")["price"] is None


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
        for row in res["trade_suggestions"]:
            if row["ticker"] in {"CDI", "TESOURO-IPCA-2035"}:
                assert row["price"] is None
                if row["suggested_weight"] > 0:
                    assert row["shares"] is None and res["trade_totals"] is None
        assert any("não por cotações históricas" in warning for warning in res["warnings"])

    # Verify Monte Carlo Simulation runs on the resulting multi-asset portfolio snapshot
    sim = simulation.compute_simulation(
        ctx.portfolio,
        simulation.SimulationRequest(revision=ctx.revision, portfolio_id=ctx.portfolio.id, simulations=100, initial_capital=50000, years=1),
    )
    assert sim.result["stats"]["mean_terminal"] > 0
    assert len(sim.result["allocation"]) == len(tickers)


def test_screener_supports_all_multi_asset_classes(client, monkeypatch):
    monkeypatch.setattr(screener, "get_full_market_data", lambda: pd.DataFrame(
        {"P/L": [10, 20], "Liq.2meses": [2_000_000, 8_000_000]}, index=["PETR4", "TLT"]))
    client.get("/api/session")
    cfg = client.get("/api/screener/config").json()
    class_keys = {item["key"] for item in cfg["asset_classes"]}
    assert {"b3", "all", "fii", "acao_us", "reit", "titulo", "etf"} <= class_keys

    for cls in ("b3", "fii", "acao_us", "reit", "titulo", "etf", "outros", "all"):
        res = client.post("/api/screener/query", headers=HEADERS, json={
            "asset_class": cls,
            "criteria": {"liq2m_min": 1_000_000},
            "sort": {"field": "liq2m", "direction": "desc"},
        })
        assert res.status_code == 200
        body = res.json()
        assert body["matched_count"] > 0
        assert len(body["rows"]) == body["matched_count"]
        if cls == "b3":
            assert [row["ticker"] for row in body["rows"]] == ["PETR4"]
        if cls != "b3":
            assert body["fetched_at"] is None and "referência" in body["source"]
        if cls in {"titulo", "outros"}:
            assert {column["key"] for column in body["available_columns"]}.isdisjoint({"dy", "pvp", "pl", "roe"})
            invalid = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": cls, "criteria": {"dy_min": 0}})
            assert invalid.status_code == 422
        if cls == "all":
            listed = next(row for row in body["rows"] if row["ticker"] == "TLT")
            assert listed["asset_class"] == "titulo" and listed["currency"] == "USD"
            assert listed["pl"] is None and listed["pvp"] is None and listed["dy"] is None
    unfiltered = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": "titulo"}).json()
    modeled = next(row for row in unfiltered["rows"] if row["ticker"] == "CDI")
    assert modeled["cotacao"] is None and modeled["liq2m"] is None


def test_usd_fixed_income_history_includes_fx_but_does_not_quote_units(monkeypatch):
    dates = pd.bdate_range("2026-01-05", periods=4)
    fx = pd.Series([5.0, 5.1, 5.2, 5.3], index=dates)
    monkeypatch.setattr(portfolio_data, "get_cdi_daily_rates", lambda start: pd.Series(dtype=float))
    monkeypatch.setattr(portfolio_data, "_download_close", lambda *args, **kwargs: fx)
    modeled = "TREASURY-4.5"
    start = dates[0].date()
    native = assets.build_synthetic_fixed_income_prices([modeled], start)
    history = portfolio_data.get_portfolio_prices.__wrapped__((modeled,), start)
    pd.testing.assert_series_equal(history[modeled].loc[dates], native[modeled].loc[dates] * fx.rename(modeled))
    assert portfolio_data.get_portfolio_trade_prices.__wrapped__((modeled, "BTC-USD", "GC=F", "BRL=X")).empty


def test_missing_fx_does_not_return_usd_positions_as_reais(monkeypatch):
    prices = pd.DataFrame({"TLT": [91.0], "IMAB11.SA": [100.0]})
    monkeypatch.setattr(portfolio_data, "_download_close", lambda symbols, *args, **kwargs: pd.Series(dtype=float) if symbols == "BRL=X" else prices)
    assert list(portfolio_data.get_portfolio_trade_prices.__wrapped__(tuple(prices))) == ["IMAB11.SA"]


def test_fx_alias_preserves_catalog_identity_and_saved_weight():
    assert assets.provider_symbol("USDBRL=X") == "BRL=X"
    assert assets.asset_metadata("BRL=X")["asset_class"] == "outros"
    assert portfolio.saved_weights({"BRL=X": .4, "PETR4.SA": .6}) == {"USDBRL=X": .4, "PETR4": .6}


def test_only_real_instrument_mentions_drive_portfolio_news():
    assert news.get_brazilian_news.__wrapped__("TESOURO-IPCA-2035") == {"ok": True, "items": [], "modeled": True}
    assert not is_relevant_company_news("O", "O mercado acompanha a Selic hoje")
    assert is_relevant_company_news("O", "Realty Income anuncia dividendos")


def test_fii_falls_back_to_yahoo_dividend_yield_when_trailing_is_zero(monkeypatch):
    assets.get_extra_asset_fundamentals.clear()
    monkeypatch.setattr(assets.yf, "Ticker", lambda symbol: SimpleNamespace(info={
        "quoteType": "EQUITY",
        "regularMarketPrice": 140.5,
        "priceToBook": 0.89,
        "trailingAnnualDividendYield": 0.0,
        "dividendYield": 9.9,
        "currency": "BRL",
    }))
    try:
        data = assets.get_extra_asset_fundamentals("HGLG11")
    finally:
        assets.get_extra_asset_fundamentals.clear()
    assert data["indicators"]["dy"] == pytest.approx(9.9)
    assert data["indicators"]["pvp"] == pytest.approx(0.89)


def test_universe_exposes_supports_shares_and_manual_rejects_modeled_and_outros(client):
    universe = {item["ticker"]: item for item in client.get("/api/universe").json()["stocks"]}
    assert universe["PETR4"]["supports_shares"] is True
    assert universe["HGLG11"]["supports_shares"] is True
    assert universe["AAPL"]["supports_shares"] is True
    assert universe["CDI"]["supports_shares"] is False
    assert universe["BTC-USD"]["supports_shares"] is False

    client.get("/api/session")
    sel = client.put("/api/session/selection", headers=HEADERS, json={"tickers": ["PETR4", "CDI", "BTC-USD"], "revision": 0}).json()
    rejected = client.post("/api/portfolio/analyze", headers=HEADERS, json={
        "tickers": sel["selection"],
        "revision": sel["revision"],
        "strategy": "manual",
        "capital": 10000,
        "lookback_days": 365,
        "manual_shares": {"PETR4": 10, "CDI": 5, "BTC-USD": 1},
        "gamma": 1,
    })
    assert rejected.status_code == 422
    detail = rejected.json()["detail"]
    assert "CDI" in detail and "BTC-USD" in detail and "não se aplica" in detail


def test_historical_stress_uses_portuguese_month_names():
    dates = pd.bdate_range("2024-11-25", "2024-12-30")
    prices = pd.DataFrame({"PETR4.SA": [40.0 + i * 0.1 for i in range(len(dates))]}, index=dates)
    bench = pd.Series([120000.0 - i * 100.0 for i in range(len(dates))], index=dates)
    rows = portfolio_data.calculate_historical_stress(
        prices, bench, {"PETR4.SA": 1.0}, {"Crise fiscal brasileira (2024)": ("2024-11-26", "2024-12-30")}
    )
    assert len(rows) == 1
    assert rows[0]["Período"] == "Nov/2024 → Dez/2024"
