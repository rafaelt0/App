import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import stress as stress_backend
from backend.state import PortfolioSnapshot, SessionState
from utils import macro_shock
from utils.macro_shock import (
    FACTOR_SYMBOLS,
    MACRO_PRESETS,
    MacroShockInput,
    simulate_macro_shock,
)

AS_OF = dt.date(2026, 1, 1)


def factor_levels(periods=180, seed=7):
    """Random but reproducible factor levels and their daily moves."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-01", periods=periods)
    moves = pd.DataFrame({
        "ibov": rng.normal(0, .011, periods),
        "usdbrl": rng.normal(0, .008, periods),
        "sp500": rng.normal(0, .009, periods),
        "us10y": rng.normal(0, .04, periods),
        "dxy": rng.normal(0, .005, periods),
    }, index=index)
    levels = pd.DataFrame(index=index)
    for key in ("ibov", "usdbrl", "sp500", "dxy"):
        levels[key] = 100 * np.cumprod(1 + moves[key])
    levels["us10y"] = 4.0 + np.cumsum(moves["us10y"])
    return levels, moves


def asset_returns(moves, sensitivity=1.5, seed=3):
    rng = np.random.default_rng(seed)
    returns = sensitivity * moves["sp500"] + rng.normal(0, .001, len(moves))
    returns.iloc[0] = np.nan  # the first factor return has no prior close
    return pd.DataFrame({"PETR4.SA": returns}, index=moves.index)


def td_returns(periods=120):
    return pd.DataFrame(index=pd.bdate_range("2025-01-01", periods=periods))


def test_treasury_duration_is_monotonic_and_selic_is_near_zero():
    weights = {"TD-IPCA-2040": .4, "TD-PREFIXADO-2029": .3, "TD-SELIC-2029": .3}
    levels, _ = factor_levels()
    result = simulate_macro_shock(
        td_returns(), weights, 100000, MacroShockInput(selic_bps=200), as_of=AS_OF, factor_series=levels,
    )
    impacts = {row["ticker"]: row for row in result["asset_impacts"]}
    assert impacts["TD-IPCA-2040"]["method"] == "analytical_duration"
    assert impacts["TD-IPCA-2040"]["impact_pct"] < impacts["TD-PREFIXADO-2029"]["impact_pct"] < 0
    assert abs(impacts["TD-SELIC-2029"]["impact_pct"]) < .01
    assert result["portfolio_impact_pct"] < 0
    assert result["factor_contributions"]["selic_bps"]["impact_brl"] < 0
    assert result["factor_contributions"]["ipca_bps"]["impact_brl"] == 0.0


def test_zero_shock_is_exactly_zero_everywhere():
    levels, moves = factor_levels()
    returns = asset_returns(moves)
    result = simulate_macro_shock(returns, {"PETR4": 1.0}, 50000, MacroShockInput(), as_of=AS_OF, factor_series=levels)
    assert result["portfolio_impact_pct"] == 0.0
    assert result["portfolio_impact_brl"] == 0.0
    assert result["post_shock_capital_brl"] == 50000.0
    assert all(row["impact_pct"] == 0.0 and row["impact_brl"] == 0.0 for row in result["asset_impacts"])
    assert all(value["impact_brl"] == 0.0 and value["impact_pct"] == 0.0 for value in result["factor_contributions"].values())


def test_factor_regression_recovers_synthetic_sp500_beta():
    levels, moves = factor_levels()
    returns = asset_returns(moves, sensitivity=1.5)
    result = simulate_macro_shock(
        returns, {"PETR4": 1.0}, 100000, MacroShockInput(sp500_pct=10), as_of=AS_OF, factor_series=levels,
    )
    asset = result["asset_impacts"][0]
    assert asset["method"] == "factor_regression"
    assert asset["impact_pct"] == pytest.approx(.15, abs=.02)
    assert asset["r2_or_duration"] is not None and asset["r2_or_duration"] > .8
    assert asset["betas"]["sp500"] == pytest.approx(1.5, abs=.15)
    assert result["factor_contributions"]["sp500_pct"]["impact_brl"] == pytest.approx(15000, rel=.12)


def test_usd_assets_get_direct_fx_while_brl_assets_do_not():
    levels, _ = factor_levels()
    index = levels.index
    returns = pd.DataFrame({"AAPL": 0.0, "PETR4.SA": 0.0}, index=index)
    assert macro_shock.asset_currency("AAPL") == "USD"
    assert macro_shock.asset_currency("PETR4") == "BRL"
    result = simulate_macro_shock(
        returns, {"AAPL": .5, "PETR4": .5}, 100000, MacroShockInput(usd_brl_pct=10),
        as_of=AS_OF, factor_series=levels,
    )
    impacts = {row["ticker"]: row for row in result["asset_impacts"]}
    assert impacts["AAPL"]["method"] == "fx_direct"
    assert impacts["AAPL"]["impact_pct"] == pytest.approx(.10, abs=1e-9)
    assert impacts["PETR4"]["impact_pct"] == pytest.approx(0.0, abs=1e-12)
    assert result["factor_contributions"]["usd_brl_pct"]["impact_brl"] == pytest.approx(5000, abs=1e-6)


def test_missing_factor_series_degrades_without_fabricating_data(monkeypatch):
    levels, moves = factor_levels()
    partial = levels.drop(columns=["dxy", "us10y"])
    monkeypatch.setattr(macro_shock, "get_macro_factor_series", lambda start: (partial, ["us10y", "dxy"]))
    returns = asset_returns(moves)
    result = simulate_macro_shock(returns, {"PETR4": 1.0}, 10000, MacroShockInput(sp500_pct=5), as_of=AS_OF)
    assert set(result["missing_factors"]) == {"us10y", "dxy"}
    assert result["asset_impacts"][0]["method"] == "factor_regression"
    assert np.isfinite(result["portfolio_impact_pct"])
    assert any("US 10Y" in warning and "DXY" in warning for warning in result["warnings"])


def test_all_factors_missing_keeps_treasury_and_usd_fx_working(monkeypatch):
    monkeypatch.setattr(macro_shock, "get_macro_factor_series", lambda start: (pd.DataFrame(), list(FACTOR_SYMBOLS)))
    returns = pd.DataFrame({"AAPL": 0.0}, index=pd.bdate_range("2025-01-01", periods=60))
    weights = {"TD-PREFIXADO-2029": .5, "AAPL": .5}
    result = simulate_macro_shock(returns, weights, 10000, MacroShockInput(selic_bps=200, usd_brl_pct=10), as_of=AS_OF)
    impacts = {row["ticker"]: row for row in result["asset_impacts"]}
    assert set(result["missing_factors"]) == set(FACTOR_SYMBOLS)
    assert impacts["TD-PREFIXADO-2029"]["impact_pct"] < 0
    assert impacts["TD-PREFIXADO-2029"]["method"] == "analytical_duration"
    assert impacts["AAPL"]["method"] == "fx_direct"
    assert impacts["AAPL"]["impact_pct"] == pytest.approx(.10)
    assert np.isfinite(result["portfolio_impact_pct"])


def snapshot(returns=None, weights=None):
    index = pd.bdate_range("2025-01-01", periods=120)
    if returns is None:
        returns = pd.DataFrame({"PETR4.SA": np.zeros(120), "TD-PREFIXADO-2029": np.zeros(120)}, index=index)
    return PortfolioSnapshot(
        id="pf-macro", revision=0, tickers=["PETR4", "TD-PREFIXADO-2029"],
        weights=weights or {"PETR4": .5, "TD-PREFIXADO-2029": .5}, mode="manual",
        returns=returns, prices=pd.DataFrame(), lookback_days=730, result={},
    )


def client_for(ctx):
    app = FastAPI()

    @app.middleware("http")
    async def owned_session(request: Request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(stress_backend.router)
    return TestClient(app)


def macro_payload(**changes):
    return {**{"revision": 0, "portfolio_id": "pf-macro", "capital": 10000}, **changes}


def test_presets_endpoint_needs_no_portfolio():
    response = client_for(SessionState(uid="presets-only")).get("/api/stress/presets")
    assert response.status_code == 200
    presets = response.json()
    assert [preset["key"] for preset in presets] == [preset["key"] for preset in MACRO_PRESETS]
    assert presets[0]["shock"]["selic_bps"] == 200


def test_macro_endpoint_uses_owned_portfolio_and_returns_serialized_chart(monkeypatch):
    levels, _ = factor_levels()
    monkeypatch.setattr(macro_shock, "get_macro_factor_series", lambda start: (levels, []))
    portfolio = snapshot()
    ctx = SessionState(uid="macro-owner", selection=portfolio.tickers, portfolio=portfolio)
    client = client_for(ctx)
    response = client.post("/api/stress/macro", json=macro_payload(selic_bps=200))
    assert response.status_code == 200
    result = response.json()
    assert result["portfolio_id"] == "pf-macro"
    assert result["revision"] == 0
    assert isinstance(result["portfolio_impact_pct"], float)
    assert result["post_shock_capital_brl"] == pytest.approx(result["capital"] * (1 + result["portfolio_impact_pct"]))
    assert result["chart"]["data"] and result["chart"]["layout"]
    assert {row["ticker"] for row in result["asset_impacts"]} == {"PETR4", "TD-PREFIXADO-2029"}
    assert result["factor_contributions"]["selic_bps"]["impact_brl"] == pytest.approx(
        next(row["impact_brl"] for row in result["asset_impacts"] if row["ticker"] == "TD-PREFIXADO-2029")
    )


def test_macro_endpoint_rejects_stale_portfolio_and_revision(monkeypatch):
    levels, _ = factor_levels()
    monkeypatch.setattr(macro_shock, "get_macro_factor_series", lambda start: (levels, []))
    portfolio = snapshot()
    ctx = SessionState(uid="macro-stale", selection=portfolio.tickers, portfolio=portfolio)
    client = client_for(ctx)
    assert client.post("/api/stress/macro", json=macro_payload(portfolio_id="old")).status_code == 409
    assert client.post("/api/stress/macro", json=macro_payload(revision=1)).status_code == 409
    missing = client_for(SessionState(uid="macro-empty", selection=portfolio.tickers))
    assert missing.post("/api/stress/macro", json=macro_payload()).status_code == 409


@pytest.mark.parametrize("payload", [
    macro_payload(selic_bps=601),
    macro_payload(capital=99),
    macro_payload(sp500_pct=-51),
    macro_payload(extra="x"),
])
def test_macro_request_boundaries(payload):
    with pytest.raises(ValidationError):
        stress_backend.MacroShockRequest(**payload)
