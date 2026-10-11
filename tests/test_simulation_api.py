import datetime as dt
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import simulation
from backend.simulation import SimulationRequest, calibration_sample, compute_simulation
from backend.state import PortfolioSnapshot, SessionState, replace_selection


TODAY = dt.date(2026, 1, 1)
NORMAL_ORACLE = np.array([
    11915.150621784478, 12393.159988941678, 12084.363174553902, 11743.227652788053,
    10698.309407143934, 12639.900700570535, 10995.02706276426, 11805.8137941963,
    11596.93116975135, 11141.930249895007,
])
BOOTSTRAP_ORACLE = np.array([
    11573.034793340546, 12036.809293761171, 10199.677087299502, 12983.279160305223,
    11767.434066859352, 11681.461686764571, 11219.082991082658, 10941.525838483429,
    10480.698714452135, 11033.73453752845,
])


def snapshot(returns=None, weights=None):
    if returns is None:
        x = np.arange(80)
        returns = pd.DataFrame({"AAA3.SA": .0005 + .01 * np.sin(x / 3), "BBB4.SA": .0002 + .008 * np.cos(x / 5)},
                               index=pd.bdate_range("2025-01-01", periods=80))
    return PortfolioSnapshot(id="portfolio-1", revision=0, tickers=["AAA3", "BBB4"], weights=weights or {"AAA3": .6, "BBB4": .4},
                             mode="markowitz", returns=returns, prices=pd.DataFrame(), lookback_days=730, result={})


def params(**changes):
    return SimulationRequest(**{"revision": 0, "portfolio_id": "portfolio-1", "simulations": 10, **changes})


def client_for(ctx):
    app = FastAPI()

    @app.middleware("http")
    async def owned_session(request: Request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(simulation.router)
    return TestClient(app)


def test_captured_float64_oracle_and_all_nine_statistics():
    calculated = compute_simulation(snapshot(), params(), today=TODAY)
    result = calculated.result
    final = np.array([trace["y"][-1] for trace in result["charts"]["individual"]["data"]])
    np.testing.assert_allclose(final, NORMAL_ORACLE, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose([result["calibration"]["annual_log_mean"], result["calibration"]["iid_standard_error"]],
                               [.13253027861055938, .1441703471399038], rtol=1e-12, atol=1e-10)
    expected = {
        "mean_terminal": NORMAL_ORACLE.mean(), "gain_probability": (NORMAL_ORACLE > 10000).mean(),
        "mean_implied_cagr": NORMAL_ORACLE.mean() / 10000 - 1,
        "p75_implied_cagr": np.percentile(NORMAL_ORACLE, 75) / 10000 - 1,
        "p5_terminal": np.percentile(NORMAL_ORACLE, 5), "p5_return": np.percentile(NORMAL_ORACLE, 5) / 10000 - 1,
        "lower_tail_mean": NORMAL_ORACLE[NORMAL_ORACLE <= np.percentile(NORMAL_ORACLE, 5)].mean(),
        "minimum": NORMAL_ORACLE.min(), "maximum": NORMAL_ORACLE.max(),
    }
    assert set(result["stats"]) == set(expected)
    for key, value in expected.items():
        assert result["stats"][key] == pytest.approx(value, rel=1e-12, abs=1e-10)
    bootstrap = result["comparison"][1]
    np.testing.assert_allclose([bootstrap["p5"], bootstrap["p95"], bootstrap["gain_probability"]],
                               [np.percentile(BOOTSTRAP_ORACLE, 5), np.percentile(BOOTSTRAP_ORACLE, 95), (BOOTSTRAP_ORACLE > 10000).mean()],
                               rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(calculated.percentiles.iloc[-1], np.percentile(NORMAL_ORACLE, [5, 25, 50, 75, 95]), rtol=1e-12, atol=1e-10)
    assert result["distribution"]["standard_deviation"] == pytest.approx(NORMAL_ORACLE.std(ddof=1))
    assert result["monte_carlo_error_pp"] == pytest.approx(98 / np.sqrt(10))


def test_multiyear_cagr_is_implied_by_terminal_statistic_not_mean_path_cagr():
    result = compute_simulation(snapshot(), params(years=3, simulations=100), today=TODAY).result
    final = np.asarray(result["charts"]["histogram"]["data"][0]["x"])
    expected = (final.mean() / 10000) ** (1 / 3) - 1
    assert result["stats"]["mean_implied_cagr"] == pytest.approx(expected)
    assert not np.isclose(expected, ((final / 10000) ** (1 / 3) - 1).mean(), rtol=1e-5)
    assert result["stats"]["p75_implied_cagr"] == pytest.approx((np.percentile(final, 75) / 10000) ** (1 / 3) - 1)


def test_sample_horizon_and_lower_tail_warning_boundaries():
    returns = pd.DataFrame(np.zeros((252, 2)), columns=["AAA3.SA", "BBB4.SA"],
                           index=pd.bdate_range("2024-01-01", periods=252))
    adequate = compute_simulation(snapshot(returns), params(simulations=1000), today=TODAY).result
    assert adequate["warnings"] == []
    longer = compute_simulation(snapshot(returns), params(simulations=999, years=2), today=TODAY).result
    assert len(longer["warnings"]) == 2


def test_chunk64_and_full_matrix_fan_not_only_displayed_paths():
    portfolio = snapshot()
    count = 130
    calculated = compute_simulation(portfolio, params(simulations=count), today=TODAY)
    logs = np.log1p(portfolio.returns)
    rng = np.random.default_rng(42)
    expected = np.empty((253, count))
    expected[0] = 10000
    for first in range(0, count, 64):
        n = min(64, count - first)
        shocks = rng.multivariate_normal(logs.mean(), logs.cov(), size=(252, n))
        expected[1:, first:first + n] = 10000 * np.cumprod(1 + np.expm1(shocks) @ [.6, .4], axis=0)
    np.testing.assert_allclose(calculated.percentiles, np.quantile(expected, [.05, .25, .5, .75, .95], axis=1).T)
    assert not np.allclose(calculated.percentiles.iloc[-1], np.quantile(expected[-1, :50], [.05, .25, .5, .75, .95]))
    np.testing.assert_allclose(calculated.result["charts"]["individual"]["data"][49]["y"], expected[:, 49])
    assert len(calculated.result["charts"]["individual"]["data"]) == 50
    assert calculated.percentiles.index[0] == "2026-01-01"
    assert calculated.percentiles.index[2] == "2026-01-05"
    np.testing.assert_array_equal(calculated.percentiles.iloc[0].to_numpy(), [10000, 10000, 10000, 10000, 10000])


def test_zero_weights_do_not_collapse_sample_and_tiny_positive_is_retained():
    returns = snapshot().returns
    returns["BBB4.SA"] = np.nan
    portfolio = snapshot(returns, {"AAA3": 2., "BBB4": 0.})
    aligned, weights, allocation = calibration_sample(portfolio, "portfolio", TODAY)
    assert len(aligned) == 80
    assert weights.tolist() == [1.]
    assert allocation == [{"ticker": "AAA3", "weight": 1.}, {"ticker": "BBB4", "weight": 0.}]
    portfolio.weights = {"AAA3": 1., "BBB4": 1e-10}
    with pytest.raises(HTTPException) as error:
        calibration_sample(portfolio, "portfolio", TODAY)
    assert error.value.status_code == 422
    portfolio.returns = snapshot().returns[["BBB4.SA", "AAA3.SA"]]
    aligned, weights, _ = calibration_sample(portfolio, "portfolio", TODAY)
    assert aligned.columns.tolist() == ["AAA3.SA", "BBB4.SA"]
    assert weights[1] > 0
    assert weights[1] == pytest.approx(1e-10 / (1 + 1e-10))


@pytest.mark.parametrize("weights", [{"AAA3": -1., "BBB4": 2.}, {"AAA3": np.inf, "BBB4": 1.}, {"AAA3": 0., "BBB4": 0.}])
def test_invalid_canonical_weights_rejected(weights):
    with pytest.raises(HTTPException) as error:
        calibration_sample(snapshot(weights=weights), "portfolio", TODAY)
    assert error.value.status_code == 422


@pytest.mark.parametrize("invalid", [-1., -1.1, np.inf, -np.inf])
def test_invalid_historical_returns_rejected(invalid):
    portfolio = snapshot()
    portfolio.returns.iloc[4, 1] = invalid
    with pytest.raises(HTTPException) as error:
        calibration_sample(portfolio, "portfolio", TODAY)
    assert error.value.status_code == 422


def test_complete_row_gate_and_missing_positive_symbol():
    portfolio = snapshot()
    portfolio.returns.iloc[3, 0] = np.nan
    aligned, _, _ = calibration_sample(portfolio, "portfolio", TODAY)
    assert len(aligned) == 79
    portfolio.returns = portfolio.returns.iloc[:30]
    with pytest.raises(HTTPException) as error:
        calibration_sample(portfolio, "portfolio", TODAY)
    assert error.value.status_code == 422
    portfolio.returns = snapshot().returns.drop(columns="BBB4.SA")
    with pytest.raises(HTTPException, match="BBB4"):
        calibration_sample(portfolio, "portfolio", TODAY)


def test_joint_day_bootstrap_daily_rebalance_strict_gain_and_tail_ties():
    returns = pd.DataFrame(np.tile([[.2, -.2], [-.2, .2]], (20, 1)),
                           columns=["AAA3.SA", "BBB4.SA"], index=pd.bdate_range("2025-01-01", periods=40))
    calculated = compute_simulation(snapshot(returns, {"AAA3": .5, "BBB4": .5}), params(simulations=100), today=TODAY)
    bootstrap = calculated.result["comparison"][1]
    assert bootstrap["p5"] == 10000
    assert bootstrap["p95"] == 10000
    assert bootstrap["gain_probability"] == 0
    flat = returns * 0
    tied = compute_simulation(snapshot(flat), params(), today=TODAY).result
    assert tied["stats"]["lower_tail_mean"] == 10000
    assert tied["stats"]["gain_probability"] == 0
    assert tied["stats"]["p5_return"] == 0


def test_fresh_calibration_sensitivity_same_endpoint_and_independent_horizon(monkeypatch):
    x = np.arange(1301)
    prices = pd.DataFrame({"AAA3.SA": 100 * np.exp(.0003 * x + .01 * np.sin(x / 20)), "BBB4.SA": np.nan},
                          index=pd.bdate_range(end="2025-12-31", periods=1301))
    monkeypatch.setattr(simulation, "get_portfolio_prices", lambda *args: prices)
    portfolio = snapshot(weights={"AAA3": 1., "BBB4": 0.})
    calculated = compute_simulation(portfolio, params(calibration=5, years=1), today=TODAY)
    calibration = calculated.result["calibration"]
    assert calibration["count"] == 1300
    windows = calibration["sensitivity"]
    assert windows[0]["count"] < windows[1]["count"] < windows[2]["count"]
    assert {row["end"] for row in windows} == {"2025-12-31"}
    assert len(calculated.percentiles) == 253
    for row in windows:
        returns = prices["AAA3.SA"].pct_change(fill_method=None).dropna()
        returns = returns.loc[returns.index >= returns.index.max() - pd.DateOffset(years=row["years"])]
        daily_log = np.log1p(returns)
        assert row["annual_log_mean"] == pytest.approx(252 * daily_log.mean())
        assert row["iid_standard_error"] == pytest.approx(252 * daily_log.std(ddof=1) / np.sqrt(len(returns)))


@pytest.mark.parametrize("changes", [{"simulations": 9}, {"simulations": 3001}, {"simulations": 10.5}, {"years": 0}, {"years": 11}, {"years": 1.5}, {"initial_capital": 99}, {"initial_capital": np.inf}, {"calibration": 4}, {"display_count": 10}])
def test_request_boundaries_and_display_not_a_randomness_input(changes):
    with pytest.raises(ValidationError):
        params(**changes)


def test_owned_api_csv_numeric_units_and_stale_result_invalidation():
    portfolio = snapshot()
    ctx = SessionState(uid="simulation-test", selection=portfolio.tickers, portfolio=portfolio)
    client = client_for(ctx)
    response = client.post("/api/simulation/run", json=params().model_dump())
    assert response.status_code == 200
    result = response.json()
    summary = client.get(f"/api/simulation/{result['id']}/summary.csv")
    assert summary.content.startswith(b"\xef\xbb\xbf")
    assert "monte_carlo_resumo_" in summary.headers["content-disposition"]
    rows = pd.read_csv(StringIO(summary.content.decode("utf-8-sig"))).set_index("Métrica")["Valor"]
    money = float(rows["Valor final P5"].removeprefix("R$ ").replace(",", ""))
    assert money == pytest.approx(result["stats"]["p5_terminal"], abs=.005)
    assert float(rows["Retorno final P5"].removesuffix("%")) == pytest.approx(result["stats"]["p5_return"] * 100, abs=.05)
    assert float(rows["Reamostragem histórica — Ganho sobre o capital inicial"].removesuffix("%")) == pytest.approx(result["comparison"][1]["gain_probability"] * 100, abs=.05)
    exported = client.get(f"/api/simulation/{result['id']}/percentiles.csv")
    assert exported.content.startswith(b"\xef\xbb\xbf")
    assert "monte_carlo_percentis_" in exported.headers["content-disposition"]
    fan = pd.read_csv(StringIO(exported.content.decode("utf-8-sig")), index_col=0)
    assert fan.columns.tolist() == ["P5", "P25", "P50", "P75", "P95"]
    assert fan.iloc[-1]["P5"] == pytest.approx(result["stats"]["p5_terminal"])
    other = client_for(SessionState(uid="other-owner", selection=portfolio.tickers))
    assert other.get(f"/api/simulation/{result['id']}/summary.csv").status_code == 404
    replace_selection(ctx, ["CCC3", "BBB4"], 0)
    assert client.get(f"/api/simulation/{result['id']}/summary.csv").status_code == 404
    assert client.post("/api/simulation/run", json=params().model_dump()).status_code == 409


def test_wrong_portfolio_and_weight_only_context_are_not_simulatable():
    portfolio = snapshot()
    ctx = SessionState(uid="gates", selection=portfolio.tickers, portfolio=portfolio)
    client = client_for(ctx)
    assert client.post("/api/simulation/run", json=params(portfolio_id="old").model_dump()).status_code == 409
    portfolio.returns = pd.DataFrame()
    assert client.post("/api/simulation/run", json=params().model_dump()).status_code == 409
    assert "simulation" not in ctx.artifacts


def test_selection_change_during_compute_cannot_publish(monkeypatch):
    portfolio = snapshot()
    ctx = SessionState(uid="race", selection=portfolio.tickers, portfolio=portfolio)
    original = simulation.compute_simulation

    def changed_during_compute(*args, **kwargs):
        calculated = original(*args, **kwargs)
        replace_selection(ctx, ["CCC3", "BBB4"], 0)
        return calculated

    monkeypatch.setattr(simulation, "compute_simulation", changed_during_compute)
    response = client_for(ctx).post("/api/simulation/run", json=params().model_dump())
    assert response.status_code == 409
    assert "simulation" not in ctx.artifacts
