import io
import threading

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import home
from backend.state import SessionState, replace_selection
from utils import market_data, snapshot


@pytest.fixture
def home_api(monkeypatch):
    ctx = SessionState(uid="home-test", selection=["AAA3", "BBB4"])
    universe = pd.DataFrame({"Ticker": ["AAA3", "BBB4", "CCC3", "DDD3"], "Setor": ["Banco"] * 4})
    monkeypatch.setattr(home, "get_listed_stocks", lambda: universe)
    home.changes_snapshot.clear()
    app = FastAPI()

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(home.router)
    app.include_router(home.catalog_router)
    return TestClient(app), ctx, universe


def detail_frame():
    return pd.DataFrame({"Empresa": ["A antiga", "A atual"], "Setor": ["Banco"] * 2,
                         "Subsetor": ["Crédito"] * 2, "Cotacao": ["10,00", "20,00"],
                         "PL": ["900", "1250"], "PVP": ["100", "150"], "EV_EBITDA": ["0", "0"],
                         "ROE": ["5,00%", "15,00%"], "ROIC": ["12,00%", "12,00%"],
                         "Div_Yield": ["4,00%", "4,01%"], "Div_Liq_Patrim": ["150", "287"],
                         "Liquidez_Corr": ["150", "100"], "EV_EBIT": ["1200", "2000"],
                         "Marg_Liquida": ["0,0%", "0,0%"]}, index=["AAA3.SA", "AAA3"])


def test_partial_analysis_preserves_order_nulls_raw_export_and_corrected_units(home_api, monkeypatch):
    client, _, _ = home_api
    cleared = []
    monkeypatch.setattr(home, "get_fundamentus_data", lambda tickers: detail_frame())
    monkeypatch.setattr(home, "get_sector_peers", lambda: pd.DataFrame())
    monkeypatch.setattr(home, "get_market_target_data", lambda ticker: {"_error": "offline"})
    monkeypatch.setattr(home, "clear_fundamentus_cache", lambda: cleared.append(True))
    response = client.post("/api/home/analyze", json={"tickers": ["AAA3", "BBB4"], "revision": 0})
    assert response.status_code == 200
    result = response.json()
    first, missing = result["companies"]
    assert result["missing_tickers"] == ["BBB4"]
    assert cleared == [True]
    assert first["identity"]["company"] == "A atual"
    assert first["market"]["price"] == 20
    assert first["indicators"]["pl"] == 12.5
    assert first["indicators"]["roe"] == 15
    assert first["indicators"]["net_margin"] is None  # Fundamentus 0,0% placeholder for banks
    assert missing["indicators"]["pl"] is None
    assert missing["health"]["metrics"] == []
    assert [(m["label"], m["value"]) for m in first["health"]["metrics"]] == [
        ("Dív. Líquida/PL", 2.87), ("Liquidez Corrente", 1.0), ("EV / EBIT", 20.0)]
    assert [m["status"] for m in first["health"]["metrics"]] == ["warning", "warning", "warning"]
    missing_metrics = [m for group in missing["metric_groups"].values() for m in group]
    assert all(m["value"] is None and m["status"] == "neutral" and m["classification"] is None for m in missing_metrics)
    raw = client.get(f"/api/home/{result['id']}/fundamentals.csv")
    assert raw.content.startswith(b"\xef\xbb\xbf")
    csv = pd.read_csv(io.BytesIO(raw.content), index_col=0)
    assert csv["PL"].tolist() == [900, 1250]
    assert csv["ROE"].tolist() == ["5,00%", "15,00%"]
    assert "fundamentus_AAA3+BBB4_" in raw.headers["content-disposition"]


def test_missing_indicator_stays_null_without_classification():
    for key in ("roic", "dy", "roe", "pl", "ev_ebitda"):
        metric = home.indicator_metric(key, "X", "percent", "", None, "Intermediários Financeiros")
        assert metric["value"] is None and metric["status"] == "neutral" and metric["classification"] is None
    roic = home.indicator_metric("roic", "ROIC", "percent", "", 0, "Intermediários Financeiros")
    assert roic["value"] == 0 and roic["classification"] == "Fraco" and roic["status"] == "negative"


def test_only_missing_roe_is_not_classified_as_weak_while_other_fields_stay_present():
    raw = pd.DataFrame({"Empresa": ["A"], "Setor": ["Banco"], "Subsetor": ["Crédito"], "Cotacao": ["10,00"],
                        "PL": ["1000"], "ROE": [None], "ROIC": ["12,00%"], "Div_Yield": [None],
                        "Marg_Liquida": ["10,00%"], "Marg_EBIT": ["12,00%"], "Cres_Rec_5a": [None]}, index=["AAA3"])
    universe = pd.DataFrame({"Ticker": ["AAA3"], "Setor": ["Banco"]})
    result = home.build_analysis(raw, pd.DataFrame(), ["AAA3"], universe)
    metrics = {metric["label"]: metric for group in result["companies"][0]["metric_groups"].values() for metric in group}
    assert metrics["P/L"]["value"] == 10
    for label in ("ROE", "Dividend Yield", "Crescimento Receita (5 anos)"):
        assert metrics[label]["value"] is None and metrics[label]["classification"] is None and metrics[label]["status"] == "neutral"
    assert result["companies"][0]["indicator_available"]["roe"] is False
    assert "Dívida Líquida/PL acima de 3×" in result["health_summary"]


def test_negative_profit_multiples_are_kept_but_not_classified_as_cheap():
    pl = home.indicator_metric("pl", "P/L", "multiple", "", -4.72, "Serviços")
    assert pl["value"] == -4.72 and pl["status"] == "neutral"
    assert pl["classification"] == "Prejuízo — múltiplo não interpretável"
    ev = home.indicator_metric("ev_ebitda", "EV/EBITDA", "multiple", "", -0.5, "Serviços")
    assert ev["value"] == -0.5 and ev["status"] == "neutral"
    assert ev["classification"] == "EBITDA negativo — múltiplo não interpretável"
    assert home.indicator_metric("pl", "P/L", "multiple", "", 8, "Serviços")["classification"] is None


def test_negative_equity_marks_pvp_and_roe_as_not_interpretable():
    for key, value in (("pvp", -0.4), ("roe", 69.0)):
        metric = home.indicator_metric(key, key.upper(), "percent", "", value, "Telecomunicações", negative_equity=True)
        assert metric["value"] == value and metric["status"] == "neutral"
        assert metric["classification"] == "Patrimônio líquido negativo — indicador não interpretável"
    assert home.indicator_metric("roe", "ROE", "percent", "", 69.0, "")["classification"] == "Forte"


def test_financial_intermediaries_get_bank_ev_ebitda_explanation():
    bank = home.indicator_metric("ev_ebitda", "EV/EBITDA", "multiple", "", .009, "Intermediários Financeiros")
    assert bank["value"] is None and bank["classification"] == "P/L · P/VP"
    tech = home.indicator_metric("ev_ebitda", "EV/EBITDA", "multiple", "", .009, "Programas e Serviços")
    assert tech["classification"] == "EV/Sales"


def test_net_debt_and_liquidity_use_fundamentus_keys_scaled_from_hundredths():
    net_cash = home.health(pd.Series({"Div_Liq_Patrim": "-020", "Liquidez_Corr": "250"}))
    assert net_cash["metrics"][0] == {"label": "Dív. Líquida/PL", "value": pytest.approx(-0.2), "unit": "multiple", "status": "positive"}
    assert net_cash["metrics"][1]["value"] == pytest.approx(2.5)
    assert net_cash["diagnostics"][0]["status"] == "positive" and "caixa líquido" in net_cash["diagnostics"][0]["text"]
    stale_keys = home.health(pd.Series({"Dív.Brut/Patrim.": "300", "Liq. Corr.": "100"}))
    assert stale_keys["metrics"] == []


def test_single_metric_exact_classification_and_health_boundaries():
    assert home.indicator_metric("roe", "ROE", "percent", "", 15, "")["classification"] == "Moderado"
    assert home.indicator_metric("roe", "ROE", "percent", "", 15.01, "")["status"] == "positive"
    assert home.indicator_metric("roic", "ROIC", "percent", "", 5, "")["classification"] == "Moderado"
    dy = home.indicator_metric("dy", "DY", "percent", "", 4.01, "")
    assert dy["classification"] == "Moderado" and dy["status"] == "positive"
    ev = home.indicator_metric("ev_ebitda", "EV/EBITDA", "ratio", "", .009, "Banco")
    assert ev["value"] is None and ev["classification"] == "P/L · P/VP"
    result = home.health(pd.Series({"Div_Liq_Patrim": "310", "Liquidez_Corr": "100", "EV/EBIT": "2000"}))
    assert [metric["status"] for metric in result["metrics"]] == ["negative", "warning", "warning"]
    assert len(result["diagnostics"]) == 2
    healthy = home.health(pd.Series({"Div_Liq_Patrim": "150", "Liquidez_Corr": "150", "EV_EBIT": "1200"}))
    assert [metric["status"] for metric in healthy["metrics"]] == ["positive"] * 3
    negative_ebit = home.health(pd.Series({"EV_EBIT": "-300"}))
    assert negative_ebit["metrics"][0]["status"] == "neutral" and "não interpretável" in negative_ebit["metrics"][0]["help"]


def test_sector_percentiles_ties_and_minimum_observations():
    universe = pd.DataFrame({"Ticker": ["AAA3", "BBB3", "CCC3", "DDD3"], "Setor": ["Máquinas"] * 4})
    peers = pd.DataFrame({"PL": [1, 2, 3, 4], "ROE": [.4, .3, .2, .1], "Div_Yield": [.1, .1, .05, .02]}, index=universe.Ticker)
    result = home.build_analysis(pd.DataFrame({"Setor": ["Outro setor da fonte"]}, index=["AAA3"]), peers, ["AAA3"], universe)["companies"][0]["synthesis"]
    assert result["sector"] == "Máquinas"
    # AAA3 is best on P/L and ROE; its DY ties BBB3 and beats CCC3 and DDD3: (2 + 0.5) / 3.
    assert result["favorable_categories"] == 3
    assert result["categorias"]["Valuation"]["percentil"] == 100
    assert result["categorias"]["Yield"]["percentil"] == 83.3
    yield_detail = next(detail for detail in result["details"] if detail["category"] == "Yield")
    assert yield_detail["value"] == 10 and yield_detail["observations"] == 4
    insufficient = home.build_analysis(pd.DataFrame({"Setor": ["Máquinas"]}, index=["AAA3"]), peers.iloc[:2], ["AAA3"], universe)["companies"][0]["synthesis"]
    assert insufficient["categorias_validas"] == 0 and insufficient["details"] == []


def test_comparison_availability_preserves_near_zero_boundary_and_missing_values():
    tickers = ["AAA3", "BBB4", "CCC3"]
    raw = pd.DataFrame({
        "Empresa": ["A", "B", "C"], "Setor": ["Banco"] * 3,
        "EV_EBITDA": [.99, 1.0, None],
    }, index=tickers)
    universe = pd.DataFrame({"Ticker": tickers, "Setor": ["Banco"] * 3})
    companies = home.build_analysis(raw, pd.DataFrame(), tickers, universe)["companies"]
    assert companies[0]["indicators"]["ev_ebitda"] == pytest.approx(.0099, abs=1e-12)
    assert companies[1]["indicators"]["ev_ebitda"] == pytest.approx(.01, abs=1e-12)
    assert companies[2]["indicators"]["ev_ebitda"] is None
    assert [company["indicator_available"]["ev_ebitda"] for company in companies] == [False, True, False]


def test_consensus_fallback_currency_invalid_numbers_and_failure(monkeypatch):
    sources = {"AAA3": {"price": -1, "regular_price": 10, "target_low": 0, "target_mean": 8,
                        "target_high": float("inf"), "analyst_count": 3.9, "currency": "usd", "recommendation_mean": 2.5},
               "BBB4": {"_error": "offline"}}
    monkeypatch.setattr(home, "get_market_target_data", lambda ticker: sources[ticker])
    result = home.consensus(["AAA3", "BBB4"])
    row = result["rows"][0]
    assert row["currency"] == "USD" and row["price"] == 10
    assert row["upside"] == pytest.approx(-20)
    assert row["target_low"] is None and row["target_high"] is None
    assert row["analyst_count"] == 3 and row["recommendation"] == "Compra"
    assert result["failed_tickers"] == ["BBB4"] and result["has_targets"]
    assert result["rows"][1]["price"] is None


def test_explicit_gate_revision_and_owner_bound_artifact(home_api, monkeypatch):
    client, ctx, _ = home_api
    assert client.post("/api/home/consensus/refresh", json={"tickers": ctx.selection, "revision": 0}).status_code == 409
    assert client.post("/api/home/analyze", json={"tickers": ["AAA3"], "revision": 0}).status_code == 409
    monkeypatch.setattr(home, "get_fundamentus_data", lambda tickers: detail_frame())
    monkeypatch.setattr(home, "get_sector_peers", lambda: pd.DataFrame())
    monkeypatch.setattr(home, "get_market_target_data", lambda ticker: {})
    monkeypatch.setattr(home, "clear_fundamentus_cache", lambda: 0)
    result = client.post("/api/home/analyze", json={"tickers": ctx.selection, "revision": 0}).json()
    replace_selection(ctx, ["AAA3"], 0)
    assert client.get(f"/api/home/{result['id']}/fundamentals.csv").status_code == 404
    assert client.post("/api/home/analyze", json={"tickers": ["AAA3"], "revision": 0}).status_code == 409


def test_selection_changed_during_fetch_cannot_publish(home_api, monkeypatch):
    client, ctx, _ = home_api

    def changed(tickers):
        replace_selection(ctx, ["AAA3"], 0)
        return detail_frame()

    monkeypatch.setattr(home, "get_fundamentus_data", changed)
    monkeypatch.setattr(home, "get_sector_peers", lambda: pd.DataFrame())
    monkeypatch.setattr(home, "get_market_target_data", lambda ticker: {})
    monkeypatch.setattr(home, "clear_fundamentus_cache", lambda: 0)
    response = client.post("/api/home/analyze", json={"tickers": ["AAA3", "BBB4"], "revision": 0})
    assert response.status_code == 409 and "home" not in ctx.artifacts


def test_analysis_io_failure_clears_caches_and_reports_retry(home_api, monkeypatch):
    from utils import cache
    client, ctx, _ = home_api
    cleared = []
    monkeypatch.setattr(cache, "clear_all", lambda: cleared.append(True))

    def failed(tickers):
        raise OSError("fixture IO failure")

    monkeypatch.setattr(home, "get_fundamentus_data", failed)
    response = client.post("/api/home/analyze", json={"tickers": ctx.selection, "revision": 0})
    assert response.status_code == 503
    assert "cache foi limpo" in response.json()["detail"] and cleared == [True]
    assert "home" not in ctx.artifacts


def test_consensus_refresh_only_evicts_analyzed_selection(home_api, monkeypatch):
    client, ctx, _ = home_api

    class Source:
        def __init__(self):
            self.values = {"AAA3": {"price": 10, "target_mean": 12}, "BBB4": {}, "UNSELECTED": {"price": 1, "target_mean": 9}}
            self.evicted = []

        def __call__(self, ticker):
            return self.values[ticker]

        def clear(self, ticker):
            self.evicted.append(ticker)
            if ticker == "AAA3":
                self.values[ticker] = {"price": 10, "target_mean": 8}

    source = Source()
    monkeypatch.setattr(home, "get_market_target_data", source)
    monkeypatch.setattr(home, "get_fundamentus_data", lambda tickers: detail_frame())
    monkeypatch.setattr(home, "get_sector_peers", lambda: pd.DataFrame())
    monkeypatch.setattr(home, "clear_fundamentus_cache", lambda: 0)
    initial = client.post("/api/home/analyze", json={"tickers": ctx.selection, "revision": 0}).json()
    assert initial["consensus"]["rows"][0]["upside"] == pytest.approx(20)
    refreshed = client.post("/api/home/consensus/refresh", json={"tickers": ctx.selection, "revision": 0}).json()
    assert refreshed["rows"][0]["upside"] == pytest.approx(-20)
    assert source.evicted == ["AAA3", "BBB4"]
    assert source.values["UNSELECTED"]["target_mean"] == 9


def test_empty_detail_source_reports_failure_not_zero_cards(home_api, monkeypatch):
    client, ctx, _ = home_api
    monkeypatch.setattr(home, "get_fundamentus_data", lambda tickers: pd.DataFrame())
    response = client.post("/api/home/analyze", json={"tickers": ctx.selection, "revision": 0})
    assert response.status_code == 503 and "não retornou dados" in response.json()["detail"]
    assert "home" not in ctx.artifacts


def test_universe_liquidity_maps_brazilian_values_and_nulls_missing_tickers(home_api, monkeypatch):
    client, _, _ = home_api
    frame = pd.DataFrame({"Liq.2meses": ["1.234.567,89", "", "0,00"]}, index=["AAA3", "BBB4", "DDD3"])
    frame.attrs["fetched_at"] = "2026-10-12T12:00:00+00:00"
    monkeypatch.setattr(market_data, "get_full_market_data", lambda: frame)
    response = client.get("/api/universe/liquidity")
    assert response.status_code == 200
    assert response.json() == {"source": "Fundamentus · liquidez média diária em 2 meses (R$)", "fetched_at": "2026-10-12T12:00:00+00:00",
                               "liquidity": {"AAA3": 1234567.89, "BBB4": None, "CCC3": None, "DDD3": 0.0}}


def test_universe_liquidity_fetch_failure_is_503_not_empty_success(home_api, monkeypatch):
    client, _, _ = home_api

    def unavailable():
        raise RuntimeError("fonte indisponível")

    monkeypatch.setattr(market_data, "get_full_market_data", unavailable)
    response = client.get("/api/universe/liquidity")
    assert response.status_code == 503
    assert response.json()["detail"] == "Não foi possível carregar a liquidez da Fundamentus: fonte indisponível"


def adjusted_closes():
    sessions = pd.to_datetime([f"2026-11-0{day}" for day in range(1, 8)])
    nan = float("nan")
    return pd.DataFrame({"AAA3.SA": [100, 101, 102, 103, 104, 110, 99],
                         "BBB4.SA": [10, 10, 10, 10, 10, nan, 10],
                         "CCC3.SA": [5, 5, 5, 5, 5, 5, nan]}, index=sessions)


def test_universe_changes_use_the_latest_common_sessions_and_never_span_a_gap():
    result = market_data.universe_changes(adjusted_closes(), ["AAA3", "BBB4", "CCC3", "DDD3"])
    assert result["session"] == "2026-11-07"
    assert result["changes"]["AAA3"] == {"day": pytest.approx(99 / 110 - 1), "week": pytest.approx(99 / 101 - 1)}
    # BBB4 missed the previous session, CCC3 the latest one, DDD3 is absent from the download.
    assert result["changes"]["BBB4"] == {"day": None, "week": pytest.approx(0.0)}
    assert result["changes"]["CCC3"] == {"day": None, "week": None}
    assert result["changes"]["DDD3"] == {"day": None, "week": None}


def test_universe_changes_ignore_a_partial_row_that_is_not_a_session():
    # Yahoo can return a Saturday row quoted for a single symbol; it must not become the latest session.
    adjusted = pd.concat([adjusted_closes().iloc[:-1],
                          pd.DataFrame({"AAA3.SA": [98.0]}, index=pd.to_datetime(["2026-11-07"]))])
    result = market_data.universe_changes(adjusted, ["AAA3", "BBB4", "CCC3"])
    assert result["session"] == "2026-11-06"
    assert result["changes"]["AAA3"]["day"] == pytest.approx(110 / 104 - 1)
    assert result["changes"]["CCC3"] == {"day": pytest.approx(0.0), "week": pytest.approx(0.0)}


def test_universe_changes_fail_when_most_symbols_came_back_empty():
    # A rate-limited download keeps the failed symbols as all-NaN columns instead of raising.
    nan = float("nan")
    adjusted = adjusted_closes().assign(**{"BBB4.SA": nan, "CCC3.SA": nan})
    with pytest.raises(ValueError, match="poucas ações"):
        market_data.universe_changes(adjusted, ["AAA3", "BBB4", "CCC3", "DDD3"])


def changes_at(fetched_at, day_close=99):
    closes = adjusted_closes()
    closes.iloc[-1, 0] = day_close
    return {**market_data.universe_changes(closes, ("AAA3", "BBB4", "CCC3", "DDD3")), "fetched_at": fetched_at}


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(snapshot.time, "monotonic", lambda: now[0])
    return now


def test_universe_changes_route_reports_fractions_with_their_fetch_time(home_api, monkeypatch):
    client, _, _ = home_api
    seen = []

    def fetch(tickers):
        seen.append(tickers)
        return changes_at("2026-11-07T18:00:00+00:00")

    monkeypatch.setattr(home, "get_universe_changes", fetch)
    body = client.get("/api/universe/changes").json()
    assert seen == [("AAA3", "BBB4", "CCC3", "DDD3")]
    assert body["unit"] == "fraction" and body["session"] == "2026-11-07" and body["changes"]["DDD3"] == {"day": None, "week": None}
    assert (body["fetched_at"], body["stale"], body["refreshing"], body["refresh_error"]) == ("2026-11-07T18:00:00+00:00", False, False, None)
    assert client.get("/api/universe/changes").json()["fetched_at"] == "2026-11-07T18:00:00+00:00"
    assert len(seen) == 1


def test_universe_changes_without_any_snapshot_fail_with_503_instead_of_inventing_data(home_api, monkeypatch):
    client, _, _ = home_api

    def unavailable(tickers):
        raise ValueError("Yahoo Finance não retornou cotações.")

    monkeypatch.setattr(home, "get_universe_changes", unavailable)
    response = client.get("/api/universe/changes")
    assert response.status_code == 503
    assert response.json()["detail"] == "Não foi possível carregar as variações do Yahoo Finance: Yahoo Finance não retornou cotações."


def test_stale_universe_changes_are_served_at_once_and_revalidated_in_the_background(home_api, monkeypatch, clock):
    client, _, _ = home_api
    release = threading.Event()
    results = iter([changes_at("2026-11-06T20:10:00+00:00", day_close=110), changes_at("2026-11-07T14:00:00+00:00", day_close=99)])

    def fetch(tickers):
        value = next(results)
        if value["fetched_at"].startswith("2026-11-07"):
            assert release.wait(5), "background refresh never released"
        return value

    monkeypatch.setattr(home, "get_universe_changes", fetch)
    assert client.get("/api/universe/changes").json()["fetched_at"] == "2026-11-06T20:10:00+00:00"
    clock[0] += home.UNIVERSE_CHANGES_TTL_SECONDS

    stale = client.get("/api/universe/changes").json()
    assert (stale["fetched_at"], stale["stale"], stale["refreshing"]) == ("2026-11-06T20:10:00+00:00", True, True)
    assert stale["changes"]["AAA3"]["day"] == pytest.approx(0.0)

    release.set()
    fresh = client.get("/api/universe/changes", params={"wait": "true"}).json()
    assert (fresh["fetched_at"], fresh["stale"], fresh["refreshing"], fresh["refresh_error"]) == ("2026-11-07T14:00:00+00:00", False, False, None)
    assert fresh["changes"]["AAA3"]["day"] == pytest.approx(99 / 110 - 1)


def test_a_rejected_partial_download_keeps_the_last_snapshot_and_reports_the_failure(home_api, monkeypatch, clock):
    client, _, _ = home_api
    monkeypatch.setattr(home, "get_universe_changes", lambda tickers: changes_at("2026-11-06T20:10:00+00:00"))
    client.get("/api/universe/changes")
    clock[0] += home.UNIVERSE_CHANGES_TTL_SECONDS
    nan = float("nan")

    def partial(tickers):
        return market_data.universe_changes(adjusted_closes().assign(**{"BBB4.SA": nan, "CCC3.SA": nan}), tickers)

    monkeypatch.setattr(home, "get_universe_changes", partial)
    body = client.get("/api/universe/changes", params={"wait": "true"}).json()
    assert (body["fetched_at"], body["stale"], body["refreshing"]) == ("2026-11-06T20:10:00+00:00", True, False)
    assert body["refresh_error"] == "Yahoo Finance devolveu cotações para poucas ações."


def test_scheduled_refresh_requires_the_configured_token(home_api, monkeypatch):
    client, _, _ = home_api
    calls = []
    monkeypatch.setattr(home, "get_universe_changes", lambda tickers: calls.append(tickers) or changes_at("2026-11-07T14:00:00+00:00"))

    monkeypatch.delenv(home.REFRESH_TOKEN_ENV, raising=False)
    assert client.post("/api/universe/changes/refresh", headers={"Authorization": "Bearer "}).status_code == 503

    monkeypatch.setenv(home.REFRESH_TOKEN_ENV, "s3cret")
    assert client.post("/api/universe/changes/refresh").status_code == 401
    assert client.post("/api/universe/changes/refresh", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert calls == []

    response = client.post("/api/universe/changes/refresh", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200
    assert response.json() == {"session": "2026-11-07", "fetched_at": "2026-11-07T14:00:00+00:00"}
    assert len(calls) == 1
    assert client.get("/api/universe/changes").json()["stale"] is False and len(calls) == 1


def test_scheduled_refresh_reports_a_failed_first_fetch_as_503(home_api, monkeypatch):
    client, _, _ = home_api
    monkeypatch.setenv(home.REFRESH_TOKEN_ENV, "s3cret")

    def unavailable(tickers):
        raise ValueError("Yahoo Finance não retornou cotações.")

    monkeypatch.setattr(home, "get_universe_changes", unavailable)
    response = client.post("/api/universe/changes/refresh", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Não foi possível atualizar as variações do Yahoo Finance: Yahoo Finance não retornou cotações."
