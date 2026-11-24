import datetime as dt
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import app as api_module, home, news, portfolio, screener, simulation, state
from utils import assets, db, fund_data, market_data, portfolio_data
from utils.news import is_relevant_company_news

HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "multi_assets.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    monkeypatch.setattr(home, "search_yahoo_assets", lambda query: [])
    monkeypatch.setattr(home, "load_fii_market_data", pd.DataFrame)
    monkeypatch.setattr(home, "get_yahoo_quotes", lambda symbols: {})
    monkeypatch.setattr(fund_data, "get_reference_rates", lambda: {
        "cdi": {"percent": 13.65, "date": "2026-11-19", "source": "BCB SGS 4389"},
        "ipca": {"percent": 4.63, "date": "2026-10-01", "source": "BCB SGS 13522"},
    })
    monkeypatch.setattr(fund_data, "get_fii_detail", lambda ticker: None)
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
    market_data.get_yahoo_info.clear()
    info_calls = []
    quote_batches = []

    def yahoo_ticker(symbol):
        info_calls.append(symbol)
        quote_type = "ETF" if symbol == "SPY" else "EQUITY"
        return SimpleNamespace(info={"quoteType": quote_type, "regularMarketPrice": 90, "totalAssets": 3000000,
                                     "trailingPE": 19, "priceToBook": 1.25, "returnOnEquity": .15, "returnOnAssets": .12,
                                     "revenueGrowth": .25, "trailingAnnualDividendYield": 0.0, "yield": .035,
                                     "debtToEquity": 200, "currentRatio": 2, "enterpriseValue": 100, "ebitda": 10})

    monkeypatch.setattr(market_data.yf, "Ticker", yahoo_ticker)

    def market_quotes(symbols):
        quote_batches.append(symbols)
        return {symbol: {"price": 90, "currency": "USD", "source": "Yahoo Finance"} for symbol in symbols}

    monkeypatch.setattr(home, "get_yahoo_quotes", market_quotes)

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
    market_data.get_yahoo_info.clear()
    assert response.status_code == 200
    data = response.json()
    by_ticker = {c["ticker"]: c for c in data["companies"]}

    assert set(by_ticker) == set(tickers)
    assert quote_batches == [("BTC-USD", "TLT")]
    assert sorted(info_calls) == ["AAPL", "AAPL34.SA", "BTC-USD", "HGLG11.SA", "O", "SPY", "TLT"]
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
    assert by_ticker["CDI"]["fixed_income"]["label"] == "Percentual do CDI"
    assert by_ticker["CDI"]["fixed_income"]["rate"] == 100
    assert by_ticker["CDI"]["fixed_income"]["unit"] == "% do CDI"
    assert by_ticker["CDI"]["fixed_income"]["indexer"] == "CDI"
    assert by_ticker["CDI"]["fixed_income"]["index_rate"] == 13.65
    assert by_ticker["CDI"]["fixed_income"]["index_rate_fallback"] is False
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
    market_data.get_yahoo_info.clear()
    assets._get_nasdaq_quote.clear()
    assets._get_nasdaq_fundamentals.clear()
    home.get_market_target_data.clear()
    yahoo_info = {}
    info_calls = []
    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: SimpleNamespace(info=yahoo_info))

    def nasdaq_http(url, *, params, **kwargs):
        if url.endswith("/AAPL/info"):
            info_calls.append(url)
            assert params == {"assetclass": "stocks"}
            if len(info_calls) == 1:
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": None})
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {
                "symbol": "AAPL", "assetClass": "STOCKS",
                "primaryData": {"lastSalePrice": "$237.51", "lastTradeTimestamp": "Nov 20, 2026 4:00 PM ET"},
                "keyStats": {"fiftyTwoWeekHighLow": {"value": "175.00 - 260.25"}},
            }})
        if url.endswith("/AAPL/summary"):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {
                "summaryData": {
                    "Sector": {"value": "Technology"},
                    "Industry": {"value": "Computer Manufacturing"},
                    "MarketCap": {"value": "4,000,000,000,000"},
                    "ShareVolume": {"value": "10,000,000"},
                    "Yield": {"value": "0.50%"},
                },
            }})
        if url.endswith("/AAPL/financials"):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {
                "financialRatiosTable": {"rows": [
                    {"value1": "Current Ratio", "value2": "125.0%"},
                    {"value1": "Operating Margin", "value2": "30.0%"},
                    {"value1": "Profit Margin", "value2": "25.0%"},
                    {"value1": "After Tax ROE", "value2": "100.0%"},
                ]},
                "incomeStatementTable": {"rows": [
                    {"value1": "Operating Income", "value2": "$120,000,000"},
                    {"value1": "Net Income", "value2": "$100,000,000"},
                ]},
                "balanceSheetTable": {"rows": [
                    {"value1": "Cash and Cash Equivalents", "value2": "$40,000,000"},
                    {"value1": "Short-Term Investments", "value2": "$10,000,000"},
                    {"value1": "Short-Term Debt / Current Portion of Long-Term Debt", "value2": "$20,000,000"},
                    {"value1": "Long-Term Debt", "value2": "$80,000,000"},
                    {"value1": "Total Equity", "value2": "$100,000,000"},
                ]},
            }})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(assets.requests, "get", nasdaq_http)
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
            "low_52w": 175.0, "high_52w": 260.25, "volume_2m": pytest.approx(2_375_100_000.0),
            "market_cap": 4_000_000_000_000.0, "price_unit": "money", "quote_pair": None,
            "volume_24h": None, "session_volume": None, "volume_window": None, "contract": None,
        }
        assert company["indicators_available"] is True
        assert company["indicators"]["pl"] == pytest.approx(40.0)
        assert company["indicators"]["pvp"] == pytest.approx(40.0)
        assert company["indicators"]["roe"] == pytest.approx(100.0)
        assert company["indicators"]["net_margin"] == pytest.approx(25.0)
        assert company["indicators"]["ebit_margin"] == pytest.approx(30.0)
        assert company["indicators"]["dy"] == pytest.approx(0.5)
        health_by_label = {metric["label"]: metric["value"] for metric in company["health"]["metrics"]}
        assert health_by_label["Liquidez Corrente"] == pytest.approx(1.25)
        assert health_by_label["Dív. Líquida/PL"] == pytest.approx(0.5)
        assert health_by_label["EV / EBIT"] == pytest.approx(4_050 / 120)
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
        assert len(info_calls) == 2
    finally:
        market_data.get_yahoo_info.clear()
        assets._get_nasdaq_quote.clear()
        assets._get_nasdaq_fundamentals.clear()
        home.get_market_target_data.clear()


def test_home_and_consensus_share_yahoo_info_until_explicit_refresh(client, monkeypatch):
    market_data.get_yahoo_info.clear()
    home.get_market_target_data.clear()
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return SimpleNamespace(info={"regularMarketPrice": 100, "targetMeanPrice": 110 + len(calls)})

    monkeypatch.setattr(market_data.yf, "Ticker", fetch)
    client.get("/api/session")
    tickers = ["AAPL"]
    selection = client.put("/api/session/selection", headers=HEADERS, json={"tickers": tickers, "revision": 0}).json()
    payload = {"tickers": tickers, "revision": selection["revision"]}
    try:
        initial = client.post("/api/home/analyze", headers=HEADERS, json=payload).json()
        assert initial["consensus"]["rows"][0]["target_mean"] == 111
        assert calls == ["AAPL"]
        refreshed = client.post("/api/home/consensus/refresh", headers=HEADERS, json=payload).json()
        assert refreshed["rows"][0]["target_mean"] == 112
        assert calls == ["AAPL", "AAPL"]
    finally:
        market_data.get_yahoo_info.clear()
        home.get_market_target_data.clear()


def test_home_uses_validated_fii_snapshot_without_yahoo_info(client, monkeypatch):
    def unexpected_info(symbol):
        pytest.fail(f"unexpected Yahoo Finance fundamentals for {symbol}")

    monkeypatch.setattr(assets, "get_yahoo_info", unexpected_info)
    monkeypatch.setattr(home, "load_fii_market_data", lambda: pd.DataFrame(
        {"price": [140.5], "pvp": [.89], "dy": [9.9], "liq2m": [715_027]}, index=["HGLG11"]))
    client.get("/api/session")
    tickers = ["HGLG11"]
    selection = client.put("/api/session/selection", headers=HEADERS, json={"tickers": tickers, "revision": 0}).json()
    response = client.post("/api/home/analyze", headers=HEADERS, json={
        "tickers": tickers, "revision": selection["revision"]})

    assert response.status_code == 200
    company = response.json()["companies"][0]
    assert company["source"] == "Fundamentus · FIIs"
    assert company["indicators"] == {"pvp": .89, "dy": 9.9}
    assert company["market"]["volume_2m"] == 715_027


def test_unavailable_yahoo_quote_does_not_use_catalog_values(monkeypatch):
    def unavailable(symbol):
        raise RuntimeError("Yahoo unavailable")

    monkeypatch.setattr(assets, "get_yahoo_info", unavailable)
    monkeypatch.setattr(assets, "get_yahoo_quotes", lambda symbols: {})
    assets._get_nasdaq_quote.clear()
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"data": None}))
    assert assets.get_extra_asset_fundamentals("AAPL")["available"] is False
    assert assets.get_extra_asset_fundamentals("TLT")["price"] is None


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
    fii = pd.DataFrame({"price": [42.5], "pvp": [0.9], "dy": [12.5], "liq2m": [2_000_000],
                        "setor": ["Logística"]}, index=["HGLG11"])
    fii.attrs["fetched_at"] = "2026-01-02T18:00:00+00:00"
    monkeypatch.setattr(screener, "get_fii_market_data", lambda: fii.copy())
    us = pd.DataFrame({
        "currency": ["USD"], "asset_class": ["acao_us"], "setor": ["Tecnologia (EUA)"],
        "cotacao": [349.0], "pl": [38.5], "pvp": [45.0], "dy": [0.0032], "roe": [1.42],
        "evebitda": [29.8], "mrgliq": [0.276], "mrgebit": [0.33], "liq2m": [15_000_000_000.0],
    }, index=["AAPL"])
    us.attrs["fetched_at"] = "2026-01-02T18:05:00+00:00"
    monkeypatch.setattr(screener, "get_us_market_data", lambda: us.copy())
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
        if cls == "fii":
            # The live Fundamentus table is merged with the reference catalog and each row keeps its own source.
            assert body["source"].startswith("Fundamentus · FIIs")
            assert body["fetched_at"] == fii.attrs["fetched_at"]
            live = next(row for row in body["rows"] if row["ticker"] == "HGLG11")
            assert live["source"] == "Fundamentus · FIIs"
            assert live["dy"] == .125
            assert live["cotacao"] == 42.5
        elif cls == "acao_us":
            assert body["source"] == "Yahoo Finance · Ações EUA"
            assert body["fetched_at"] == us.attrs["fetched_at"]
            assert body["rows"][0]["ticker"] == "AAPL"
            assert body["rows"][0]["mrgliq"] == pytest.approx(0.276)
            assert body["rows"][0]["mrgebit"] == pytest.approx(0.33)
        elif cls != "b3":
            assert body["fetched_at"] is None
            # Reference classes name their source; "all" mixes live and reference rows per row.
            assert cls == "all" or "referência" in body["source"]
        if cls in {"titulo", "outros"}:
            assert {column["key"] for column in body["available_columns"]}.isdisjoint({"dy", "pvp", "pl", "roe"})
            invalid = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": cls, "criteria": {"dy_min": 0}})
            assert invalid.status_code == 422
        if cls == "all":
            listed = next(row for row in body["rows"] if row["ticker"] == "TLT")
            assert listed["asset_class"] == "titulo" and listed["currency"] == "USD"
            assert listed["pl"] is None and listed["pvp"] is None and listed["dy"] is None
            aapl = next(row for row in body["rows"] if row["ticker"] == "AAPL")
            assert aapl["cotacao"] == 349.0 and aapl["mrgliq"] == pytest.approx(0.276)
    unfiltered = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": "titulo"}).json()
    modeled = next(row for row in unfiltered["rows"] if row["ticker"] == "CDI")
    assert modeled["cotacao"] is None and modeled["liq2m"] is None


def test_screener_does_not_substitute_fii_reference_values_during_outage(client, monkeypatch):
    def offline():
        raise RuntimeError("provider offline")

    monkeypatch.setattr(screener, "get_fii_market_data", offline)
    monkeypatch.setattr(screener, "get_us_market_data", offline)
    monkeypatch.setattr(screener, "get_full_market_data", lambda: pd.DataFrame({"P/L": [10]}, index=["PETR4"]))
    client.get("/api/session")

    funds = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": "fii"})
    us = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": "acao_us"})
    combined = client.post("/api/screener/query", headers=HEADERS, json={"asset_class": "all"})
    assert funds.status_code == 503
    assert "Fundamentus" in funds.json()["detail"]
    assert us.status_code == 503
    assert "Yahoo Finance" in us.json()["detail"]
    assert combined.status_code == 200
    assert "FIIs indisponíveis" in combined.json()["source"]
    assert "ações EUA indisponíveis" in combined.json()["source"]
    assert all(row["asset_class"] not in {"fii", "acao_us"} for row in combined.json()["rows"])


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
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: {
        "quoteType": "EQUITY",
        "regularMarketPrice": 140.5,
        "priceToBook": 0.89,
        "trailingAnnualDividendYield": 0.0,
        "dividendYield": 9.9,
        "currency": "BRL",
    })
    data = assets.get_extra_asset_fundamentals("HGLG11", fii_table=pd.DataFrame())
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


def test_catalog_assets_have_bundled_logos():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "frontend/lib/stock-logo-manifest.json").read_text(encoding="utf-8"))
    by_ticker = {ticker: entry["asset"] for entry in manifest for ticker in entry["tickers"]}
    tickers = assets.EXTRA_ASSETS_BY_TICKER.keys()
    assert not (tickers - by_ticker.keys())
    for ticker in tickers:
        asset_path = root / "frontend/public/stock-logos" / by_ticker[ticker]
        assert asset_path.is_file() and asset_path.stat().st_size > 100


def test_us_equity_extracts_health_metrics_volume_and_guards_dual_class_pvp(monkeypatch):
    payloads = {
        "AAPL": {
            "quoteType": "EQUITY", "currency": "USD", "sector": "Technology", "industry": "Consumer Electronics",
            "regularMarketPrice": 350.0, "marketCap": 5_000_000_000_000.0, "priceToBook": 50.0,
            "trailingPE": 35.0, "returnOnEquity": 1.4, "profitMargins": 0.28, "operatingMargins": 0.30,
            "totalRevenue": 400_000_000_000.0, "enterpriseValue": 5_040_000_000_000.0, "ebitda": 150_000_000_000.0,
            "totalDebt": 100_000_000_000.0, "totalCash": 60_000_000_000.0, "currentRatio": 1.05,
            "averageDailyVolume3Month": 40_000_000, "trailingAnnualDividendYield": 0.003,
        },
        "BRK-B": {
            "quoteType": "EQUITY", "currency": "USD", "sector": "Financial Services", "industry": "Insurance - Diversified",
            "regularMarketPrice": 572.36, "bookValue": 536413.0, "priceToBook": 0.001067,
            "marketCap": 1_215_989_809_152.0, "totalDebt": 130_783_002_624.0, "debtToEquity": 17.127,
            "enterpriseValue": -243_581_337_600.0, "enterpriseToEbitda": -2.04, "trailingPE": 16.0,
        },
        "JPM": {
            "quoteType": "EQUITY", "currency": "USD", "sector": "Financial Services", "industry": "Banks - Diversified",
            "regularMarketPrice": 316.0, "marketCap": 840_000_000_000.0, "priceToBook": 2.4,
            "totalDebt": 1_300_000_000_000.0, "totalCash": 1_500_000_000_000.0, "totalRevenue": 170_000_000_000.0,
            "operatingMargins": 0.44, "enterpriseValue": 600_000_000_000.0,
        },
    }
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: payloads[symbol])

    aapl = assets.get_extra_asset_fundamentals("AAPL")
    assert aapl["volume_2m"] == pytest.approx(14_000_000_000.0)
    assert aapl["indicators"]["ev_ebitda"] == pytest.approx(5_040 / 150)
    assert aapl["health"]["current_ratio"] == pytest.approx(1.05)
    assert aapl["health"]["debt_equity"] == pytest.approx(0.4)
    assert aapl["health"]["ev_ebit"] == pytest.approx(5_040 / 120)

    brk = assets.get_extra_asset_fundamentals("BRK-B")
    assert brk["indicators"]["pvp"] == pytest.approx(1.5924, rel=1e-3)
    assert brk["indicators"]["ev_ebitda"] is None
    assert "debt_equity" not in brk["health"] and "ev_ebit" not in brk["health"]

    jpm = assets.get_extra_asset_fundamentals("JPM")
    assert jpm["health"] == {}
