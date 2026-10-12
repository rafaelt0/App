from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend import portfolio as api
from backend.state import SessionState, begin, replace_selection, require_portfolio


def request(ctx):
    return SimpleNamespace(state=SimpleNamespace(session=ctx))


@pytest.fixture
def market(monkeypatch):
    dates = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=1), periods=181)
    x = np.arange(180)
    daily = pd.DataFrame({"AAA3.SA": .0009 + .002 * np.sin(x * .31), "BBB4.SA": .0005 + .018 * np.cos(x * .23)}, index=dates[1:])
    prices = pd.concat([pd.DataFrame([[100., 80.]], columns=daily.columns, index=dates[:1]), (1 + daily).cumprod().mul([100., 80.])])
    benchmark_returns = pd.Series(.0006 + .006 * np.sin(x * .19), index=dates[1:])
    benchmark = pd.concat([pd.Series([1000.], index=dates[:1]), (1 + benchmark_returns).cumprod() * 1000])
    raw_quotes = pd.DataFrame({"AAA3.SA": [37.], "BBB4.SA": [23.]}, index=dates[-1:])
    monkeypatch.setattr(api, "get_listed_stocks", lambda: pd.DataFrame({"Ticker": ["AAA3", "BBB4"]}))
    monkeypatch.setattr(api, "get_portfolio_prices", lambda tickers, start: prices.loc[:, list(tickers)])
    monkeypatch.setattr(api, "get_portfolio_trade_prices", lambda tickers: raw_quotes.loc[:, list(tickers)])
    monkeypatch.setattr(api, "get_benchmark_prices", lambda start: benchmark)
    monkeypatch.setattr(api, "get_selic_rate", lambda: .0001)
    monkeypatch.setattr(api, "get_cdi_daily_rates", lambda start: pd.Series(.0001, index=prices.index))
    saved = []
    monkeypatch.setattr(api.db, "portfolio_save", lambda uid, tickers, weights: saved.append((uid, tickers, weights)) or True)
    ctx = SessionState(uid="owner", selection=["AAA3", "BBB4"])
    return SimpleNamespace(ctx=ctx, prices=prices, returns=prices.pct_change(fill_method=None).dropna(), benchmark=benchmark, benchmark_returns=benchmark_returns, quotes=raw_quotes, saved=saved)


def payload(market, **kwargs):
    return api.AnalyzeRequest(tickers=market.ctx.selection, revision=market.ctx.revision, **kwargs)


def test_manual_zero_asset_does_not_collapse_history_and_uses_raw_quotes(market, monkeypatch):
    fetched = []
    def download(tickers, start):
        fetched.append(tuple(tickers))
        return market.prices.loc[:, list(tickers)]
    monkeypatch.setattr(api, "get_portfolio_prices", download)
    result = api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 3, "BBB4": 0}, capital=10000), request(market.ctx))
    assert fetched == [("AAA3.SA",)]
    assert market.ctx.portfolio.weights == {"AAA3": 1., "BBB4": 0.}
    assert list(market.ctx.portfolio.returns) == ["AAA3.SA"]
    assert result["sample"]["zero_excluded"] is True
    assert result["trade_suggestions"][0]["shares"] == 270
    assert result["trade_suggestions"][0]["effective_value"] == 9990
    assert result["trade_suggestions"][1]["price"] is None
    assert result["trade_suggestions"][1]["shares"] == 0
    assert result["trade_totals"] == {"allocated": 9990., "cash": 10., "efficiency": .999}
    assert result["capital"] == 10000  # historical base, not 3*37 holding value
    assert result["metrics"]["total_return"] == pytest.approx(((1 + market.returns["AAA3.SA"]).prod() - 1) * 100)
    assert market.saved[0][2] == {"AAA3.SA": 1., "BBB4.SA": 0.}


def test_manual_missing_positive_quote_is_fatal_before_history(market, monkeypatch):
    monkeypatch.setattr(api, "get_portfolio_trade_prices", lambda tickers: market.quotes[["AAA3.SA"]])
    monkeypatch.setattr(api, "get_portfolio_prices", lambda *args: pytest.fail("history must not load without required raw quote"))
    with pytest.raises(HTTPException, match="BBB4") as error:
        api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 1, "BBB4": 1}), request(market.ctx))
    assert error.value.status_code == 422
    assert market.ctx.portfolio is None
    assert market.saved == []


def test_partial_quotes_suppress_totals_without_losing_allocation():
    rows, totals, missing = api.purchase_suggestion({"AAA3.SA": .5, "BBB4.SA": .5, "CCC3.SA": 0.}, 1000., {"AAA3.SA": (37., pd.Timestamp("2025-01-02"))})
    assert totals is None
    assert missing == ["BBB4"]
    assert rows[0]["shares"] == 13 and rows[0]["effective_value"] == 481
    assert rows[0]["effective_weight"] is None
    assert rows[1]["shares"] is None and rows[1]["effective_value"] is None
    assert rows[2]["shares"] == 0 and rows[2]["effective_weight"] == 0


def test_min_volatility_solves_risk_not_equal_weight(market):
    result = api.analyze(payload(market), request(market.ctx))
    weights = np.array(list(market.ctx.portfolio.weights.values()))
    covariance = api.estimate_markowitz_inputs(market.returns)[1].values
    assert weights @ covariance @ weights < np.array([.5, .5]) @ covariance @ np.array([.5, .5])
    assert weights[0] > .95
    assert "frontier" not in result["charts"]
    assert require_portfolio(market.ctx, returns=True).id == result["id"]


def test_hrp_preserves_library_allocation(market):
    expected = dict(api.HRPOpt(market.returns).optimize())
    api.analyze(payload(market, strategy="hrp"), request(market.ctx))
    actual = market.ctx.portfolio.weights
    assert actual["AAA3"] == pytest.approx(expected["AAA3.SA"])
    assert actual["BBB4"] == pytest.approx(expected["BBB4.SA"])
    assert 0 < actual["BBB4"] < .1


def test_markowitz_l2_target_return_and_full_frontier(market):
    result = api.analyze(payload(market, strategy="markowitz", gamma=1), request(market.ctx))
    assert result["allocation_label"] == "Retorno-alvo com regularização L2"
    frontier = result["charts"]["frontier"]["data"]
    random = next(trace for trace in frontier if trace["name"] == "Portfólios Aleatórios")
    assert len(random["x"]) == 5000
    star = next(trace for trace in frontier if trace["name"] == "Retorno-alvo com regularização L2")
    mu, covariance = api.estimate_markowitz_inputs(market.returns)
    weights = np.array(list(market.ctx.portfolio.weights.values()))
    assert star["y"][0] == pytest.approx(weights @ mu.values)
    assert star["x"][0] == pytest.approx(np.sqrt(weights @ covariance.values @ weights))
    assert any(trace["name"] == "Linha de alocação (carteira selecionada)" for trace in frontier)


def test_diversification_level_strictly_lowers_hhi_when_max_sharpe_is_a_corner():
    mu = np.array([.60, .03, .05, .00, .04])
    vols = np.sqrt([.09, .04, .05, .03, .04])
    correlation = np.eye(5)
    correlation[0, 1:] = correlation[1:, 0] = .1
    covariance = np.diag(vols) @ correlation @ np.diag(vols)
    daily = np.random.default_rng(3).multivariate_normal(mu / 252, covariance / 252, size=2000)
    returns = pd.DataFrame(daily, columns=[f"A{i}.SA" for i in range(5)], index=pd.bdate_range("2018-01-01", periods=2000))
    hhi = []
    for gamma in (0, 1, 2, 3):
        weights, _, _, _ = api.optimize("markowitz", returns, {}, .10, gamma, [])
        hhi.append(float(np.square(list(weights.values())).sum()))
    assert hhi[0] > .95  # Max Sharpe is a corner: almost all capital in one asset.
    assert hhi[0] > hhi[1] > hhi[2] > hhi[3]
    assert hhi[3] >= 1 / 5 - 1e-9


def test_markowitz_negative_excess_returns_falls_back_with_warning(market, monkeypatch):
    monkeypatch.setattr(api, "get_selic_rate", lambda: .01)
    result = api.analyze(payload(market, strategy="markowitz", gamma=1), request(market.ctx))
    assert result["allocation_label"] == "Mínima Volatilidade"
    assert any("Max Sharpe falhou" in warning for warning in result["warnings"])
    assert all(row["weight"] >= 0 for row in result["weights"])
    assert sum(row["weight"] for row in result["weights"]) == pytest.approx(1, abs=2e-5)


def test_benchmark_outage_preserves_core_and_simulation_context(market, monkeypatch):
    def unavailable(start):
        raise OSError("provider outage")
    monkeypatch.setattr(api, "get_benchmark_prices", unavailable)
    result = api.analyze(payload(market), request(market.ctx))
    assert result["benchmark_available"] is False
    assert set(result["metrics"]) == {"total_return", "volatility", "sharpe", "sortino", "max_drawdown"}
    assert result["health"] is None and result["stress"] is None and result["regime"] is None and result["concentration"] is None
    assert set(result["charts"]) == {"value"}
    assert require_portfolio(market.ctx, returns=True).returns.equals(market.returns)
    weights = list(market.ctx.portfolio.weights.values())
    assert result["metrics"]["total_return"] == pytest.approx(((1 + market.returns.dot(weights)).prod() - 1) * 100)


def test_benchmark_alignment_changes_metric_window_not_calibration(market, monkeypatch):
    shortened = market.benchmark.iloc[61:]
    monkeypatch.setattr(api, "get_benchmark_prices", lambda start: shortened)
    result = api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 1, "BBB4": 1}), request(market.ctx))
    aligned_dates = shortened.pct_change(fill_method=None).dropna().index
    daily = market.returns.dot(list(market.ctx.portfolio.weights.values())).loc[aligned_dates]
    assert result["sample"]["count"] == 180
    assert result["comparison_sample"]["count"] == len(aligned_dates)
    assert market.ctx.portfolio.returns.equals(market.returns)
    assert result["metrics"]["total_return"] == pytest.approx(((1 + daily).prod() - 1) * 100)
    expected_beta = np.cov(daily, shortened.pct_change(fill_method=None).dropna())[0, 1] / shortened.pct_change(fill_method=None).dropna().var()
    assert result["metrics"]["beta"] == pytest.approx(expected_beta)
    assert result["metrics"]["annual_alpha"] == pytest.approx(((daily.mean() - .0001) - expected_beta * (shortened.pct_change(fill_method=None).dropna().mean() - .0001)) * 252 * 100)
    for row in result["monthly_returns"]:
        assert row["ytd"] == pytest.approx(((1 + daily[daily.index.year == row["year"]]).prod() - 1) * 100)
    asset_dd = (1 + market.returns).cumprod()
    asset_dd = asset_dd / asset_dd.cummax() - 1
    for row in result["asset_drawdowns"]:
        assert row["max_drawdown"] == pytest.approx(asset_dd[row["ticker"] + ".SA"].min() * 100)


def test_degenerate_benchmark_does_not_invent_relative_metrics(market, monkeypatch):
    monkeypatch.setattr(api, "get_benchmark_prices", lambda start: pd.Series(100., index=market.prices.index))
    result = api.analyze(payload(market), request(market.ctx))
    assert result["benchmark_available"] is True
    for key in ("beta", "annual_alpha", "r_squared", "information_ratio"):
        assert result["metrics"][key] is None
    assert result["health"]["coverage"] == 75


@pytest.mark.parametrize("invalidation", ["generation", "selection"])
def test_stale_analysis_never_publishes_or_saves(market, monkeypatch, invalidation):
    def invalidate(start):
        if invalidation == "generation":
            begin(market.ctx, "portfolio", market.ctx.revision)
        else:
            replace_selection(market.ctx, ["BBB4", "AAA3"], market.ctx.revision)
        return market.benchmark
    monkeypatch.setattr(api, "get_benchmark_prices", invalidate)
    with pytest.raises(HTTPException) as error:
        api.analyze(payload(market), request(market.ctx))
    assert error.value.status_code == 409
    assert market.saved == []
    assert market.ctx.portfolio is None


def test_persistence_failure_does_not_erase_successful_analysis(market, monkeypatch):
    monkeypatch.setattr(api.db, "portfolio_save", lambda *args: False)
    result = api.analyze(payload(market), request(market.ctx))
    assert result["saved"] is False
    assert any("salvar" in warning for warning in result["warnings"])
    assert require_portfolio(market.ctx, returns=True).id == result["id"]


def test_restore_targets_have_distinct_missing_weight_defaults():
    saved = {"AAA3.SA": .8, "BBB4.SA": "nan", "OTHER.SA": 10}
    assert api.restored_weights(["AAA3", "BBB4"], saved, "news") == {"AAA3": 1., "BBB4": 0.}
    assert api.restored_weights(["AAA3", "BBB4"], saved, "simulation") == pytest.approx({"AAA3": .8 / 1.3, "BBB4": .5 / 1.3})
    assert api.restored_weights(["AAA3", "BBB4"], {"AAA3": 0, "BBB4": -1}, "news") == {"AAA3": .5, "BBB4": .5}


def test_news_weight_restore_needs_no_quotes_and_can_upgrade_to_simulation(market, monkeypatch):
    market.ctx.restored = True
    monkeypatch.setattr(api.db, "portfolio_get", lambda uid: (["AAA3", "BBB4"], {"AAA3.SA": .8}))
    original_history = api.get_portfolio_prices
    monkeypatch.setattr(api, "get_portfolio_prices", lambda *args: pytest.fail("news restore must not fetch history"))
    news = api.restore_portfolio(market.ctx, "news", 0)
    assert news.returns.empty
    assert news.weights == {"AAA3": 1., "BBB4": 0.}
    with pytest.raises(HTTPException):
        require_portfolio(market.ctx, returns=True)
    monkeypatch.setattr(api, "get_portfolio_prices", original_history)
    simulation = api.restore_portfolio(market.ctx, "simulation", 0)
    assert simulation.id != news.id
    assert simulation.weights == pytest.approx({"AAA3": .8 / 1.3, "BBB4": .5 / 1.3})
    assert simulation.returns.equals(market.returns)
    assert market.saved == []


def test_restore_cannot_override_edited_selection(market, monkeypatch):
    market.ctx.restored = True
    monkeypatch.setattr(api.db, "portfolio_get", lambda uid: (["AAA3", "BBB4"], {}))
    replace_selection(market.ctx, ["BBB4", "AAA3"], 0)
    with pytest.raises(HTTPException) as error:
        api.restore_portfolio(market.ctx, "simulation", 1)
    assert error.value.status_code == 409
    assert market.ctx.portfolio is None


def test_history_reports_short_symbol_and_bridges_interior_gap(market, monkeypatch):
    prices = market.prices.copy()
    prices.loc[prices.index[:-20], "BBB4.SA"] = np.nan
    monkeypatch.setattr(api, "get_portfolio_prices", lambda *args: prices)
    with pytest.raises(HTTPException, match="BBB4") as error:
        api.history(["AAA3.SA", "BBB4.SA"], date.today() - timedelta(days=730))
    assert error.value.status_code == 422
    original = market.prices.copy()
    prices = original.copy()
    prices.iloc[90, 0] = np.nan
    monkeypatch.setattr(api, "get_portfolio_prices", lambda *args: prices)
    _, returns = api.history(["AAA3.SA", "BBB4.SA"], date.today() - timedelta(days=730))
    # The missing close no longer erases the days around it: the move is carried by the day the price returns.
    assert len(returns) == 180
    assert returns.loc[prices.index[90], "AAA3.SA"] == 0.0
    assert returns.loc[prices.index[91], "AAA3.SA"] == pytest.approx(original.iloc[91, 0] / original.iloc[89, 0] - 1)
    assert returns.loc[prices.index[91], "BBB4.SA"] == pytest.approx(original.iloc[91, 1] / original.iloc[90, 1] - 1)


def test_history_one_missing_price_keeps_a_20_percent_drop(market, monkeypatch):
    dates = pd.bdate_range(end=pd.Timestamp(date.today()) - pd.Timedelta(days=1), periods=41)
    prices = pd.DataFrame({"AAA3.SA": np.full(41, 100.0), "BBB4.SA": np.linspace(100.0, 80.0, 41)}, index=dates)
    prices.iloc[20, 1] = np.nan
    monkeypatch.setattr(api, "get_portfolio_prices", lambda *args: prices)
    _, returns = api.history(["AAA3.SA", "BBB4.SA"], date.today() - timedelta(days=730))
    assert (1 + returns["BBB4.SA"]).prod() - 1 == pytest.approx(-0.20)
    assert returns["AAA3.SA"].abs().max() == 0.0


def test_csv_preserves_bare_tickers_percentage_units_bom_and_owner(market):
    result = api.analyze(payload(market), request(market.ctx))
    response = api.allocation_csv(result["id"], request(market.ctx))
    assert response.body.startswith(b"\xef\xbb\xbf")
    frame = pd.read_csv(__import__("io").BytesIO(response.body), index_col=0)
    assert list(frame.index) == ["AAA3", "BBB4"]
    assert list(frame.columns) == ["Peso (%)"]
    for ticker in frame.index:
        assert frame.loc[ticker, "Peso (%)"] == round(market.ctx.portfolio.weights[ticker] * 100, 2)
    assert f"portfolio_alocacao_{date.today()}.csv" in response.headers["content-disposition"]
    with pytest.raises(HTTPException) as error:
        api.allocation_csv(result["id"], request(SessionState(uid="other")))
    assert error.value.status_code == 404


def test_long_stress_uses_active_assets_and_checks_current_snapshot(market, monkeypatch):
    result = api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 1, "BBB4": 0}), request(market.ctx))
    dates = pd.bdate_range("2024-11-25", "2024-12-30")
    prices = pd.DataFrame({"AAA3.SA": 100 * 1.01 ** np.arange(len(dates))}, index=dates)
    fetched = []
    def long_prices(tickers, start):
        fetched.append((tuple(tickers), start))
        return prices
    monkeypatch.setattr(api, "get_portfolio_prices", long_prices)
    monkeypatch.setattr(api, "get_benchmark_prices", lambda start: pd.Series(100 * 1.005 ** np.arange(len(dates)), index=dates))
    stress = api.stress(result["id"], api.StressRequest(long_history=True), request(market.ctx))
    assert fetched == [(("AAA3.SA",), date(2008, 8, 1))]
    fiscal = next(row for row in stress["rows"] if row["crisis"] == "Crise fiscal brasileira (2024)")
    # Both sides start from the last close before the crisis (2024-11-25), so the start day's return is included.
    within = prices.loc["2024-11-26":"2024-12-30"]
    assert fiscal["portfolio"] == pytest.approx(1.01 ** len(within) - 1)
    assert fiscal["benchmark"] == pytest.approx(1.005 ** len(within) - 1)
    assert fiscal["difference_pp"] == pytest.approx((fiscal["portfolio"] - fiscal["benchmark"]) * 100)
    def stale_prices(tickers, start):
        begin(market.ctx, "portfolio", market.ctx.revision)
        return prices
    monkeypatch.setattr(api, "get_portfolio_prices", stale_prices)
    with pytest.raises(HTTPException) as error:
        api.stress(result["id"], api.StressRequest(long_history=True), request(market.ctx))
    assert error.value.status_code == 409


def test_manual_shares_reject_fractional_and_boolean_values():
    for invalid in (.5, True):
        with pytest.raises(ValidationError):
            api.AnalyzeRequest(tickers=["AAA3", "BBB4"], revision=0, manual_shares={"AAA3": invalid})


def test_cleaned_weights_are_not_renormalized_for_historical_performance(market, monkeypatch):
    cleaned = {"AAA3.SA": .33333, "BBB4.SA": .66665}
    monkeypatch.setattr(api, "optimize", lambda *args: (cleaned, "Mínima Volatilidade", None, None))
    result = api.analyze(payload(market), request(market.ctx))
    raw_daily = market.returns.dot(list(cleaned.values()))
    normalized_daily = raw_daily / sum(cleaned.values())
    expected = ((1 + raw_daily).prod() - 1) * 100
    assert result["metrics"]["total_return"] == pytest.approx(expected, abs=1e-12)
    assert not np.isclose(expected, ((1 + normalized_daily).prod() - 1) * 100, atol=1e-8, rtol=0)
    assert market.ctx.portfolio.weights == {"AAA3": .33333, "BBB4": .66665}


def test_refresh_invalidates_all_cross_page_readiness_and_source_caches(market, monkeypatch):
    result = api.analyze(payload(market), request(market.ctx))
    market.ctx.artifacts["simulation"] = {"id": "simulation"}
    market.ctx.artifacts["news"] = {"id": "news"}
    cleared = []
    for name in ("get_portfolio_prices", "get_portfolio_trade_prices", "get_benchmark_prices", "get_selic_rate", "get_cdi_daily_rates"):
        monkeypatch.setattr(getattr(api, name), "clear", lambda name=name: cleared.append(name), raising=False)
    api.refresh(api.RefreshRequest(), request(market.ctx))
    assert set(cleared) == {"get_portfolio_prices", "get_portfolio_trade_prices", "get_benchmark_prices", "get_selic_rate", "get_cdi_daily_rates"}
    assert market.ctx.portfolio is None
    assert market.ctx.artifacts == {}
    assert market.ctx.restored is False
    with pytest.raises(HTTPException):
        api.allocation_csv(result["id"], request(market.ctx))
    with pytest.raises(HTTPException):
        require_portfolio(market.ctx, returns=True)


def realized_daily(market):
    weights = market.ctx.portfolio.weights
    return market.returns.dot([weights[column.removesuffix(".SA")] for column in market.returns.columns])


def test_historical_ratios_use_daily_cdi_excess_returns(market, monkeypatch):
    cdi = pd.Series(np.linspace(.0001, .0004, len(market.prices.index)), index=market.prices.index)
    monkeypatch.setattr(api, "get_cdi_daily_rates", lambda start: cdi)
    result = api.analyze(payload(market), request(market.ctx))
    daily = realized_daily(market)
    excess = daily - cdi.loc[daily.index]
    downside = np.sqrt((np.minimum(excess, 0) ** 2).sum() / len(excess))
    assert result["source"]["historical_rf"].startswith("BCB SGS 12")
    assert result["metrics"]["sharpe"] == pytest.approx(excess.mean() / excess.std() * np.sqrt(252))
    assert result["metrics"]["sortino"] == pytest.approx(excess.mean() / downside * np.sqrt(252))
    benchmark = market.benchmark_returns.loc[daily.index]
    bench_excess = benchmark - cdi.loc[daily.index]
    beta = np.cov(daily, benchmark)[0, 1] / np.cov(daily, benchmark)[1, 1]
    assert result["metrics"]["annual_alpha"] == pytest.approx((excess.mean() - beta * bench_excess.mean()) * 252 * 100)
    tail = excess.tail(60)
    rolling = result["charts"]["rolling_sharpe"]["data"][0]["y"]
    assert rolling[-1] == pytest.approx(tail.mean() / tail.std() * np.sqrt(252))


def test_cdi_outage_flags_constant_selic_fallback(market, monkeypatch):
    def unavailable(start):
        raise OSError("BCB outage")
    monkeypatch.setattr(api, "get_cdi_daily_rates", unavailable)
    result = api.analyze(payload(market), request(market.ctx))
    daily = realized_daily(market)
    excess = daily - .0001
    assert result["source"]["historical_rf"].startswith("Selic atual constante")
    assert any("CDI" in warning and "BCB outage" in warning for warning in result["warnings"])
    assert result["metrics"]["sharpe"] == pytest.approx(excess.mean() / excess.std() * np.sqrt(252))


def test_information_ratio_is_annualized(market):
    result = api.analyze(payload(market), request(market.ctx))
    daily = realized_daily(market)
    difference = daily - market.benchmark_returns.loc[daily.index]
    assert result["metrics"]["information_ratio"] == pytest.approx(difference.mean() / difference.std() * np.sqrt(252))


def test_max_drawdown_counts_first_day_loss_against_initial_capital(market, monkeypatch):
    prices = market.prices.copy()
    prices.iloc[1] = prices.iloc[0] * .9
    monkeypatch.setattr(api, "get_portfolio_prices", lambda tickers, start: prices.loc[:, list(tickers)])
    result = api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 1, "BBB4": 1}), request(market.ctx))
    returns = prices.pct_change(fill_method=None).dropna()
    weights = market.ctx.portfolio.weights
    daily = returns.dot([weights[column.removesuffix(".SA")] for column in returns.columns])
    wealth = (1 + daily).cumprod()
    expected = float((wealth / np.maximum(wealth.cummax(), 1.0) - 1).min())
    assert daily.iloc[0] == pytest.approx(-.1)
    assert result["metrics"]["max_drawdown"] == pytest.approx(expected * 100)
    assert result["metrics"]["max_drawdown"] == pytest.approx(-10)


def test_var_metrics_separate_parametric_historical_and_expected_shortfall(market):
    from statistics import NormalDist

    result = api.analyze(payload(market), request(market.ctx))
    daily = realized_daily(market)
    cutoff = np.percentile(daily, 5)
    assert result["metrics"]["daily_var"] == pytest.approx(NormalDist(daily.mean(), daily.std()).inv_cdf(.05) * 100)
    assert result["metrics"]["historical_var_95"] == pytest.approx(cutoff * 100)
    assert result["metrics"]["cvar_95"] == pytest.approx(daily[daily <= cutoff].mean() * 100)
    assert result["metrics"]["cvar_95"] <= result["metrics"]["historical_var_95"]


def test_effective_assets_keep_one_decimal(market):
    result = api.analyze(payload(market, strategy="manual", manual_shares={"AAA3": 69, "BBB4": 74}), request(market.ctx))
    assert result["concentration"]["hhi"] == pytest.approx(5200)
    assert result["concentration"]["effective_assets"] == pytest.approx(1.9)


@pytest.mark.parametrize(("saved", "target", "mode"), [
    ({"AAA3.SA": .8, "BBB4.SA": .2}, "news", "Carteira restaurada"),
    ({"AAA3.SA": .8, "BBB4.SA": .2}, "simulation", "Pesos salvos (restaurados)"),
    ({"AAA3.SA": .8}, "simulation", "Pesos salvos com complemento igual (restaurados)"),
    ({}, "simulation", "Pesos iguais (sem otimização)"),
])
def test_restored_label_names_the_weight_source_not_an_optimizer(market, monkeypatch, saved, target, mode):
    market.ctx.restored = True
    monkeypatch.setattr(api.db, "portfolio_get", lambda uid: (["AAA3", "BBB4"], saved))
    monkeypatch.setattr(api, "get_portfolio_prices", lambda tickers, start: market.prices.loc[:, list(tickers)])
    restored = api.restore_portfolio(market.ctx, target, 0)
    assert restored.result["mode"] == mode
    assert "Markowitz" not in restored.result["mode"]
    assert "730" not in restored.result["mode"]


def test_current_returns_only_the_session_analysis_for_the_current_selection(market):
    assert api.current(request(market.ctx)) is None
    result = api.analyze(payload(market, strategy="hrp", capital=5000), request(market.ctx))
    current = api.current(request(market.ctx))
    assert current["id"] == result["id"]
    assert current["inputs"] == {"strategy": "hrp", "capital": 5000, "lookback_days": 730, "gamma": 1, "manual_shares": {}}
    assert api.current(request(SessionState(uid="other", selection=["AAA3", "BBB4"]))) is None
    replace_selection(market.ctx, ["AAA3"], 0)
    assert api.current(request(market.ctx)) is None


def test_trade_suggestions_export_keeps_raw_price_units_and_owner(market):
    result = api.analyze(payload(market), request(market.ctx))
    response = api.trades_csv(result["id"], request(market.ctx))
    frame = pd.read_csv(__import__("io").BytesIO(response.body), encoding="utf-8-sig")
    assert sorted(frame["Ativo"]) == ["AAA3", "BBB4"]
    assert "Preço bruto de fechamento (R$)" in frame.columns
    assert frame["Peso sugerido (%)"].sum() == pytest.approx(100, abs=.1)
    assert f"portfolio_compras_{date.today()}.csv" in response.headers["content-disposition"]
    with pytest.raises(HTTPException) as error:
        api.trades_csv(result["id"], request(SessionState(uid="other")))
    assert error.value.status_code == 404


def test_metric_signal_and_help_are_owned_by_python(market):
    result = api.analyze(payload(market), request(market.ctx))
    assert set(result["metric_signal"]) == set(result["metric_status"])
    for key, label in result["metric_status"].items():
        assert result["metric_signal"][key] == api.signal_for(label)
    assert api.signal_for("Sem dados") == "neutral"
    assert api.signal_for("Baixa") == "positive" and api.signal_for("Alta") == "negative"
    assert api.signal_for("Moderada") == "warning"
    assert result["metric_help"]["sharpe"].endswith("Acima de 1,0 é excelente.")


def test_markowitz_missing_target_return_warns_and_keeps_min_volatility(market, monkeypatch):
    monkeypatch.setattr(api, "bound_efficient_return", lambda *args: None)
    result = api.analyze(payload(market, strategy="markowitz", gamma=1), request(market.ctx))
    assert result["allocation_label"] == "Mínima Volatilidade"
    assert any("retorno-alvo" in warning for warning in result["warnings"])
