"""Deterministic regressions for the after-tax and real-return hurdle engine (WP-1.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import hurdle as hurdle_api
from backend.state import PortfolioSnapshot, SessionState
from utils import tax_hurdle
from utils.regime import RegimeConfig, currency_metadata, default_hurdle_label


def daily_frame(days: int = 252, columns: tuple[str, ...] = ("PETR4.SA",), drift: float = 0.0006) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=days)
    steps = np.arange(days)
    return pd.DataFrame(
        {column: drift + 0.004 * np.sin(steps * 0.37 + offset) for offset, column in enumerate(columns)},
        index=index,
    )


def tax_result(weights, returns, **kwargs):
    defaults = {"tax_profile": "BR_PF", "holding_days": 400}
    defaults.update(kwargs)
    return tax_hurdle.portfolio_tax_analysis(weights, returns, **defaults)


# ---------------------------------------------------------------------------
# Real return: exact Fisher relation
# ---------------------------------------------------------------------------


def test_fisher_equation_is_exact_not_subtraction():
    assert tax_hurdle.deflate_return(0.10, 0.04) == pytest.approx(1.10 / 1.04 - 1)
    assert tax_hurdle.deflate_return(0.10, 0.04) != pytest.approx(0.06)
    assert tax_hurdle.deflate_return(None, 0.04) is None
    assert tax_hurdle.deflate_return(0.10, -1.0) is None

    index = pd.bdate_range("2024-01-02", periods=3)
    nominal = pd.Series([0.01, -0.02, 0.03], index=index)
    inflation = pd.Series([0.001, 0.0015, 0.002], index=index)
    expected = (1 + nominal) / (1 + inflation) - 1
    pd.testing.assert_series_equal(tax_hurdle.deflate_series(nominal, inflation), expected, check_names=False)


# ---------------------------------------------------------------------------
# Regressive IR tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "rate"),
    [(0, 0.225), (180, 0.225), (181, 0.20), (360, 0.20), (361, 0.175), (720, 0.175), (721, 0.15), (3000, 0.15)],
)
def test_regressive_ir_tiers(days, rate):
    assert tax_hurdle.regressive_ir_rate(days) == pytest.approx(rate)


# ---------------------------------------------------------------------------
# Asset classification and per-asset tax rules
# ---------------------------------------------------------------------------


def test_asset_classification_uses_shared_catalog_and_fixed_income_prefixes():
    assert tax_hurdle.classify_asset("PETR4").category == "equity_br"
    assert tax_hurdle.classify_asset("HGLG11").category == "fii"
    assert tax_hurdle.classify_asset("BOVA11").category == "etf"
    assert tax_hurdle.classify_asset("AAPL").category == "offshore"
    assert tax_hurdle.classify_asset("LCI-95CDI").category == "fixed_income_exempt"
    assert tax_hurdle.classify_asset("LCA-95CDI").category == "fixed_income_exempt"
    assert tax_hurdle.classify_asset("CRI-CDI+2").category == "fixed_income_exempt"
    assert tax_hurdle.classify_asset("CRA-CDI+2").category == "fixed_income_exempt"
    assert tax_hurdle.classify_asset("POUPANCA").category == "fixed_income_exempt"
    assert tax_hurdle.classify_asset("CDB-100CDI").category == "fixed_income_taxable"
    assert tax_hurdle.classify_asset("TESOURO-IPCA-2035").category == "fixed_income_taxable"
    assert tax_hurdle.classify_asset("CDI").category == "fixed_income_taxable"
    assert tax_hurdle.classify_asset("SELIC").category == "fixed_income_taxable"
    assert tax_hurdle.classify_asset("IPCA+6").category == "fixed_income_taxable"
    assert tax_hurdle.classify_asset("PRE-14").category == "fixed_income_taxable"


def test_tax_exempt_lci_has_zero_ir_and_taxable_cdb_uses_regressive_table():
    exempt = tax_result({"LCI-95CDI": 1.0}, {"LCI-95CDI": 0.10}, holding_days=200)
    assert exempt.tax_amount == pytest.approx(0.0)
    assert exempt.net_return == pytest.approx(0.10)
    assert exempt.breakdown[0]["tax_rule"] == "br_fi_exempt"
    assert exempt.breakdown[0]["applied_rate"] == pytest.approx(0.0)

    taxable = tax_result({"CDB-100CDI": 1.0}, {"CDB-100CDI": 0.10}, holding_days=200)
    assert taxable.breakdown[0]["applied_rate"] == pytest.approx(0.20)
    assert taxable.tax_amount == pytest.approx(0.02)
    assert taxable.net_return == pytest.approx(0.08)
    assert taxable.effective_tax_rate == pytest.approx(0.20)

    treasury = tax_result({"TESOURO-IPCA-2035": 1.0}, {"TESOURO-IPCA-2035": 0.10}, holding_days=800)
    assert treasury.tax_amount == pytest.approx(0.015)


def test_fii_twenty_percent_equity_fifteen_percent_and_small_sale_exemption():
    fii = tax_result({"HGLG11": 1.0}, {"HGLG11": 0.10}, holding_days=300)
    assert fii.tax_amount == pytest.approx(0.02)

    equity = tax_result({"PETR4": 1.0}, {"PETR4": 0.10}, holding_days=30)
    assert equity.tax_amount == pytest.approx(0.015)
    assert "15%" in equity.breakdown[0]["tax_rule_label"]

    exempt = tax_result({"PETR4": 1.0}, {"PETR4": 0.10}, holding_days=30, small_sale_exempt=True)
    assert exempt.tax_amount == pytest.approx(0.0)
    assert "R$ 20.000,00" in exempt.breakdown[0]["tax_rule_label"]
    assert any("small_sale_exempt" in line for line in exempt.assumptions)


def test_offshore_flat_tax_plus_fx_drag_and_losses_have_no_ir():
    offshore = tax_result({"AAPL": 1.0}, {"AAPL": 0.10})
    assert offshore.tax_amount == pytest.approx(0.015)
    assert offshore.fx_drag == pytest.approx(0.10 * tax_hurdle.DEFAULT_FX_DRAG)
    assert offshore.total_drag == pytest.approx(0.015 + 0.00038)
    assert offshore.effective_tax_rate == pytest.approx(0.15)

    loss = tax_result({"AAPL": 1.0}, {"AAPL": -0.05})
    assert loss.tax_amount == pytest.approx(0.0)
    assert loss.fx_drag == pytest.approx(0.0)
    assert loss.total_drag == pytest.approx(0.0)
    assert loss.net_return == pytest.approx(-0.05)


def test_tax_profiles_none_and_us_taxable():
    untaxed = tax_result({"PETR4": 1.0}, {"PETR4": 0.10}, tax_profile="NONE")
    assert untaxed.tax_amount == pytest.approx(0.0)
    assert untaxed.breakdown[0]["tax_rule"] == "exempt"

    long_term = tax_result({"PETR4": 1.0}, {"PETR4": 0.10}, tax_profile="US_TAXABLE", holding_days=400)
    assert long_term.breakdown[0]["applied_rate"] == pytest.approx(0.15)
    shortest = tax_result({"PETR4": 1.0}, {"PETR4": 0.10}, tax_profile="US_TAXABLE", holding_days=300)
    assert shortest.breakdown[0]["applied_rate"] == pytest.approx(0.24)


def test_weighted_portfolio_tax_breakdown_is_proportional():
    result = tax_result(
        {"PETR4": 0.5, "LCI-95CDI": 0.5},
        {"PETR4": 0.20, "LCI-95CDI": 0.10},
        holding_days=100,
    )
    assert result.gross_return == pytest.approx(0.15)
    assert result.tax_amount == pytest.approx(0.5 * 0.20 * 0.15)
    assert result.net_return == pytest.approx(0.15 - 0.015)
    assert [row["ticker"] for row in result.breakdown] == ["LCI-95CDI", "PETR4"]
    assert sum(row["weight"] for row in result.breakdown) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hurdle, excess return and Sharpe math
# ---------------------------------------------------------------------------


def test_hurdle_excess_return_and_sharpe_math():
    frame = daily_frame()
    cdi = pd.Series(0.0002, index=frame.index)
    result = tax_hurdle.analyze_hurdle(
        frame,
        {"PETR4": 1.0},
        hurdle="CDI",
        tax_profile="NONE",
        net_of_tax=False,
        cdi_rates=cdi,
    )
    daily = frame["PETR4.SA"]
    portfolio_total = float((1 + daily).prod() - 1)
    hurdle_total = float((1 + cdi).prod() - 1)
    cagr = (1 + portfolio_total) ** (252 / len(daily)) - 1
    hurdle_cagr = (1 + hurdle_total) ** (252 / len(daily)) - 1
    volatility = float(daily.std(ddof=1) * np.sqrt(252))

    assert result["metrics"]["total_return"] == {"value": pytest.approx(portfolio_total), "unit": "fraction"}
    assert result["metrics"]["hurdle_total_return"]["value"] == pytest.approx(hurdle_total)
    assert result["metrics"]["excess_return"]["value"] == pytest.approx(portfolio_total - hurdle_total)
    assert result["metrics"]["cagr"]["value"] == pytest.approx(cagr)
    assert result["metrics"]["hurdle_cagr"]["value"] == pytest.approx(hurdle_cagr)
    assert result["metrics"]["excess_cagr"]["value"] == pytest.approx(cagr - hurdle_cagr)
    assert result["metrics"]["volatility"]["value"] == pytest.approx(volatility)
    assert result["metrics"]["excess_sharpe"]["value"] == pytest.approx((cagr - hurdle_cagr) / volatility)
    assert result["metrics"]["excess_sharpe"]["unit"] == "ratio"
    assert result["regime"]["hurdle_label"] == "CDI"
    assert result["hurdle"]["source"] == "BCB SGS 12 (CDI diário)"


def test_net_of_tax_reduces_portfolio_and_investable_hurdle():
    frame = daily_frame()
    cdi = pd.Series(0.0002, index=frame.index)
    gross = tax_hurdle.analyze_hurdle(
        frame, {"PETR4": 1.0}, hurdle="CDI", tax_profile="BR_PF", net_of_tax=False, cdi_rates=cdi,
    )
    net = tax_hurdle.analyze_hurdle(
        frame, {"PETR4": 1.0}, hurdle="CDI", tax_profile="BR_PF", net_of_tax=True, cdi_rates=cdi,
    )
    gross_total = gross["metrics"]["total_return"]["value"]
    net_total = net["metrics"]["total_return"]["value"]
    assert net_total is not None and gross_total is not None
    assert net_total < gross_total
    assert net["metrics"]["tax_drag_pp"]["value"] > 0
    assert net["metrics"]["effective_tax_rate"]["value"] == pytest.approx(0.15)
    assert net["metrics"]["hurdle_total_return"]["value"] < gross["metrics"]["hurdle_total_return"]["value"]


def test_real_return_uses_exact_fisher_on_portfolio_and_cdi_hurdle():
    frame = daily_frame()
    cdi = pd.Series(0.00025, index=frame.index)
    inflation = pd.Series(0.0001, index=frame.index)
    result = tax_hurdle.analyze_hurdle(
        frame,
        {"PETR4": 1.0},
        hurdle="CDI",
        tax_profile="NONE",
        net_of_tax=False,
        real_return=True,
        cdi_rates=cdi,
        inflation_daily=inflation,
    )
    nominal_total = float((1 + frame["PETR4.SA"]).prod() - 1)
    inflation_total = float((1 + inflation).prod() - 1)
    cdi_total = float((1 + cdi).prod() - 1)
    assert result["metrics"]["total_return"]["value"] == pytest.approx((1 + nominal_total) / (1 + inflation_total) - 1)
    assert result["metrics"]["hurdle_total_return"]["value"] == pytest.approx((1 + cdi_total) / (1 + inflation_total) - 1)
    assert result["regime"]["real_return"] is True
    assert result["regime"]["inflation_source"] is not None


def test_ipca_plus_six_real_hurdle_is_six_percent_per_year():
    frame = daily_frame(days=252)
    result = tax_hurdle.analyze_hurdle(
        frame,
        {"PETR4": 1.0},
        hurdle="IPCA_PLUS_6",
        tax_profile="NONE",
        net_of_tax=False,
        real_return=True,
        inflation_daily=pd.Series(0.0, index=frame.index),
    )
    expected = 1.06 ** (252 / len(frame)) - 1
    assert result["metrics"]["hurdle_total_return"]["value"] == pytest.approx(expected)


def test_missing_provider_series_stays_none_with_warning(monkeypatch):
    index = pd.bdate_range("2024-01-02", periods=30)
    warnings: list[str] = []
    monkeypatch.setattr(tax_hurdle, "get_bcb_monthly_inflation", lambda *args, **kwargs: None)
    assert tax_hurdle.inflation_daily_series(index, warnings=warnings) is None
    assert warnings and "inflação" in warnings[0]

    monkeypatch.setattr(tax_hurdle, "get_fred_series", lambda *args, **kwargs: None)
    warnings = []
    series, source = tax_hurdle.build_hurdle_series("SOFR", index, warnings=warnings)
    assert series is None
    assert source.startswith("FRED SOFR")
    assert warnings and "SOFR" in warnings[0]


def test_bcb_fetch_failure_returns_none_without_fabricating(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(tax_hurdle.requests, "get", explode)
    assert tax_hurdle.get_bcb_monthly_inflation(13522, pd.Timestamp("2001-01-01").date()) is None


def test_chart_compares_portfolio_and_hurdle_normalized_to_100():
    frame = daily_frame(days=60)
    result = tax_hurdle.analyze_hurdle(
        frame,
        {"PETR4": 1.0},
        hurdle="CDI",
        tax_profile="NONE",
        net_of_tax=False,
        cdi_rates=pd.Series(0.0002, index=frame.index),
    )
    figure = result["chart"]
    assert len(figure.data) == 2
    assert figure.data[0].name == "Portfólio"
    assert figure.data[1].name == "Hurdle · CDI"
    assert float(figure.data[0].y[0]) == pytest.approx(100.0)
    assert figure.data[1].line.dash == "dash"


# ---------------------------------------------------------------------------
# Regime metadata
# ---------------------------------------------------------------------------


def test_regime_config_labels_and_defaults():
    config = RegimeConfig()
    assert config.base_currency == "BRL"
    assert config.hurdle == "CDI"
    assert config.tax_profile == "BR_PF"
    assert config.real_return is False and config.net_of_tax is False
    assert config.hurdle_label == "CDI"
    assert currency_metadata("USD")["symbol"] == "US$"
    assert default_hurdle_label("IPCA_PLUS_6") == "IPCA + 6%"
    assert config.describe()["annualization_days"] == 252
    with pytest.raises(ValueError):
        RegimeConfig(hurdle="BOGUS")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def snapshot(revision: int = 1, identifier: str = "pf-1") -> PortfolioSnapshot:
    frame = daily_frame()
    return PortfolioSnapshot(
        id=identifier,
        revision=revision,
        tickers=["PETR4"],
        weights={"PETR4": 1.0},
        mode="Alocação Manual",
        returns=frame,
        prices=pd.DataFrame(),
        lookback_days=252,
        result={},
    )


@pytest.fixture
def client(monkeypatch):
    ctx = SessionState(uid="owner", selection=["PETR4"], revision=1)
    ctx.portfolio = snapshot()
    monkeypatch.setattr(
        "utils.portfolio_data.get_cdi_daily_rates",
        lambda start: pd.Series(0.0002, index=ctx.portfolio.returns.index),
    )
    app = FastAPI()

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(hurdle_api.router)
    with TestClient(app) as test_client:
        yield test_client, ctx


def payload(**overrides):
    body = {"revision": 1, "portfolio_id": "pf-1", "hurdle": "CDI", "net_of_tax": False}
    body.update(overrides)
    return body


def test_endpoint_returns_metrics_breakdown_and_chart(client):
    test_client, _ = client
    response = test_client.post("/api/portfolio/hurdle", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["total_return"]["unit"] == "fraction"
    assert body["metrics"]["excess_sharpe"]["unit"] == "ratio"
    assert body["asset_breakdown"][0]["ticker"] == "PETR4"
    assert body["asset_breakdown"][0]["asset_class"] == "acao_br"
    assert body["assumptions"] and isinstance(body["warnings"], list)
    assert body["chart"]["data"][0]["name"] == "Portfólio"
    assert body["labels"]["hurdles"]["CDI"] == "CDI"
    assert body["regime"]["tax_profile"] == "BR_PF"


def test_endpoint_rejects_stale_revision_and_foreign_portfolio(client):
    test_client, _ = client
    assert test_client.post("/api/portfolio/hurdle", json=payload(revision=0)).status_code == 409
    assert test_client.post("/api/portfolio/hurdle", json=payload(portfolio_id="outro")).status_code == 409


def test_endpoint_rejects_invalid_payloads(client):
    test_client, _ = client
    assert test_client.post("/api/portfolio/hurdle", json=payload(hurdle="BOGUS")).status_code == 422
    assert test_client.post("/api/portfolio/hurdle", json=payload(tax_profile="BOGUS")).status_code == 422
    assert test_client.post("/api/portfolio/hurdle", json=payload(revision=-1)).status_code == 422
    assert test_client.post("/api/portfolio/hurdle", json=payload(extra="x")).status_code == 422
    assert test_client.post("/api/portfolio/hurdle", json={"revision": 1}).status_code == 422
