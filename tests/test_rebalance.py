from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import rebalance as api
from backend.state import PortfolioSnapshot, SessionState
from utils.importers import MAX_IMPORT_BYTES, MAX_IMPORT_ROWS, parse_positions
from utils.rebalance import plan_buy_only_rebalance, share_step


def positions_by_ticker(result):
    return {position["ticker"]: position for position in result["positions"]}


# ── Importer ─────────────────────────────────────────────────────────────────


def test_b3_area_investidor_semicolon_bom_brazilian_numbers_and_duplicates():
    content = (
        "\ufeffCódigo de Negociação;Produto;Quantidade;Preço de Fechamento;Valor Atualizado\n"
        "PETR4;PETROLEO BRASILEIRO S.A. - PETR4;1.000;38,50;38.500,00\n"
        "PETR4;PETROLEO BRASILEIRO S.A. - PETR4;500;39,00;19.500,00\n"
        "VALE3;VALE S.A. - VALE3;200;61,20;12.240,00\n"
    )
    result = parse_positions(content)
    assert result["format_detected"] == "b3_area_investidor"
    assert result["skipped"] == []
    positions = positions_by_ticker(result)
    assert positions["PETR4"]["quantity"] == 1500
    assert positions["PETR4"]["price"] == pytest.approx((1000 * 38.5 + 500 * 39.0) / 1500)
    assert positions["PETR4"]["value"] == pytest.approx(58_000)
    assert positions["VALE3"]["quantity"] == 200
    assert positions["VALE3"]["price"] == pytest.approx(61.2)
    assert any("2 linhas consolidadas" in warning for warning in result["warnings"])


def test_b3_area_investidor_comma_delimiter_with_quoted_decimal_comma():
    content = (
        "Código de Negociação,Quantidade,Preço de Fechamento\n"
        'PETR4,100,"38,50"\n'
        'VALE3,200,"61,20"\n'
    )
    result = parse_positions(content)
    assert result["format_detected"] == "b3_area_investidor"
    positions = positions_by_ticker(result)
    assert positions["PETR4"]["quantity"] == 100
    assert positions["PETR4"]["price"] == pytest.approx(38.5)
    assert positions["VALE3"]["price"] == pytest.approx(61.2)


def test_generic_broker_csv_aliases_and_invalid_ticker_handling():
    content = (
        "symbol,shares,avg_price\n"
        "AAPL,10,180.50\n"
        "MSFT,5,410.25\n"
        "ATIVO-INVALIDO###,3,10.00\n"
    )
    result = parse_positions(content)
    assert result["format_detected"] == "generic_csv"
    positions = positions_by_ticker(result)
    assert positions["AAPL"]["quantity"] == 10
    assert positions["AAPL"]["price"] == pytest.approx(180.5)
    assert positions["MSFT"]["quantity"] == 5
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "Ativo não reconhecido."


def test_generic_value_column_derives_unit_price():
    result = parse_positions("ticker,quantity,value\nPETR4,100,3850.00\n")
    position = positions_by_ticker(result)["PETR4"]
    assert position["quantity"] == 100
    assert position["price"] == pytest.approx(38.5)
    assert position["value"] == pytest.approx(3850)


def test_importer_rejects_oversized_row_and_missing_header_inputs():
    with pytest.raises(ValueError, match="200 KB"):
        parse_positions("ticker,quantity\n" + "X" * (MAX_IMPORT_BYTES + 1))

    rows = "\n".join(f"PETR{index % 10},1" for index in range(MAX_IMPORT_ROWS + 1))
    with pytest.raises(ValueError, match="500 linhas"):
        parse_positions("ticker,quantity\n" + rows)

    exactly_limit = "\n".join("PETR4,1" for _ in range(MAX_IMPORT_ROWS))
    limited = parse_positions("ticker,quantity\n" + exactly_limit + "\n")
    assert positions_by_ticker(limited)["PETR4"]["quantity"] == MAX_IMPORT_ROWS

    with pytest.raises(ValueError, match="colunas"):
        parse_positions("nome;descricao\nPETR4;Petrobras\n")

    with pytest.raises(ValueError, match="posições"):
        parse_positions("   \n")


def test_importer_rejects_invalid_quantity_rows_without_failing_the_whole_table():
    result = parse_positions("ticker,quantity,price\nPETR4,abc,38.5\nVALE3,10,61.2\n")
    assert [position["ticker"] for position in result["positions"]] == ["VALE3"]
    assert result["skipped"][0]["reason"] == "Quantidade ausente ou inválida."


# ── Rebalancer ───────────────────────────────────────────────────────────────


def test_buy_only_never_emits_negative_quantity_for_overweight_asset():
    result = plan_buy_only_rebalance(
        {"WEGE3": 0.5, "PETR4": 0.5},
        {"WEGE3": 1000, "PETR4": 1},
        {"WEGE3": 10.0, "PETR4": 10.0},
        1000.0,
    )
    assert result["summary"]["zero_sell_guaranteed"] is True
    assert all(row["buy_qty"] >= 0 for row in result["rows"])
    rows = {row["ticker"]: row for row in result["rows"]}
    assert rows["WEGE3"]["buy_qty"] == 0
    assert rows["WEGE3"]["post_qty"] == 1000
    assert rows["PETR4"]["buy_qty"] == 100


def test_integer_b3_lots_and_fractional_treasury_titles():
    result = plan_buy_only_rebalance(
        {"PETR4": 0.5, "TD-IPCA-2035": 0.5},
        {},
        {"PETR4": 38.5, "TD-IPCA-2035": 3000.0},
        4000.0,
    )
    rows = {row["ticker"]: row for row in result["rows"]}
    assert share_step("PETR4") == Decimal("1")
    assert share_step("TD-IPCA-2035") == Decimal("0.01")
    assert rows["PETR4"]["share_step"] == 1
    assert rows["TD-IPCA-2035"]["share_step"] == pytest.approx(0.01)
    assert rows["PETR4"]["buy_qty"] == int(rows["PETR4"]["buy_qty"])
    assert rows["TD-IPCA-2035"]["buy_qty"] * 100 == pytest.approx(
        round(rows["TD-IPCA-2035"]["buy_qty"] * 100)
    )
    assert rows["TD-IPCA-2035"]["buy_qty"] > 0


def test_us_fractional_step_only_when_enabled():
    prices = {"AAPL": 100.0, "PETR4": 10.0}
    fractional = plan_buy_only_rebalance(
        {"AAPL": 0.5, "PETR4": 0.5}, {}, prices, 1000.0, allow_fractional_us=True
    )
    whole = plan_buy_only_rebalance(
        {"AAPL": 0.5, "PETR4": 0.5}, {}, prices, 1000.0, allow_fractional_us=False
    )
    fractional_rows = {row["ticker"]: row for row in fractional["rows"]}
    whole_rows = {row["ticker"]: row for row in whole["rows"]}
    assert fractional_rows["AAPL"]["share_step"] == pytest.approx(0.0001)
    assert whole_rows["AAPL"]["share_step"] == 1
    assert fractional_rows["AAPL"]["buy_qty"] == pytest.approx(5.0)


def test_l2_drift_improves_and_cash_is_conserved_with_leftover():
    result = plan_buy_only_rebalance(
        {"PETR4": 0.5, "VALE3": 0.5},
        {},
        {"PETR4": 30.0, "VALE3": 70.0},
        50.0,
    )
    summary = result["summary"]
    assert summary["allocated_cash"] + summary["leftover_cash"] == pytest.approx(50.0)
    assert summary["leftover_cash"] > 0
    assert summary["l2_drift_after"] <= summary["l2_drift_before"]
    assert summary["total_value_after"] == pytest.approx(
        summary["total_value_before"] + summary["contribution"]
    )


def test_target_weights_are_fully_reached_when_lots_fit():
    result = plan_buy_only_rebalance(
        {"PETR4": 1, "VALE3": 1},
        {},
        {"PETR4": 10.0, "VALE3": 15.0},
        300.0,
    )
    summary = result["summary"]
    assert summary["allocated_cash"] == pytest.approx(300)
    assert summary["leftover_cash"] == pytest.approx(0)
    assert summary["l2_drift_after"] == pytest.approx(0, abs=1e-12)


def test_modeled_fixed_income_is_skipped_with_warning():
    result = plan_buy_only_rebalance(
        {"CDI": 0.5, "PETR4": 0.5},
        {},
        {"PETR4": 10.0},
        1000.0,
    )
    rows = {row["ticker"]: row for row in result["rows"]}
    assert rows["CDI"]["buy_qty"] == 0
    assert any("sem cota negociável" in warning for warning in result["warnings"])
    assert rows["PETR4"]["buy_qty"] > 0


def test_missing_price_is_warned_and_does_not_invent_a_quote():
    result = plan_buy_only_rebalance(
        {"PETR4": 0.5, "VALE3": 0.5},
        {"PETR4": 10},
        {"PETR4": 10.0},
        1000.0,
    )
    rows = {row["ticker"]: row for row in result["rows"]}
    assert rows["VALE3"]["buy_qty"] == 0
    assert rows["VALE3"]["unit_price"] is None
    assert any("VALE3" in warning and "sem preço" in warning for warning in result["warnings"])


def test_rebalance_is_deterministic_for_the_same_inputs():
    arguments = (
        {"PETR4": 0.35, "VALE3": 0.35, "TD-IPCA-2035": 0.30},
        {"PETR4": 7, "VALE3": 3},
        {"PETR4": 38.5, "VALE3": 61.2, "TD-IPCA-2035": 2515.0},
        2500.0,
    )
    first = plan_buy_only_rebalance(*arguments)
    second = plan_buy_only_rebalance(*arguments)
    assert first == second


def test_invalid_rebalance_inputs_raise_portuguese_errors():
    with pytest.raises(ValueError, match="soma positiva"):
        plan_buy_only_rebalance({}, {}, {}, 100.0)
    with pytest.raises(ValueError, match="aporte"):
        plan_buy_only_rebalance({"PETR4": 1.0}, {}, {"PETR4": 10.0}, 0.0)
    with pytest.raises(ValueError, match="não negativas"):
        plan_buy_only_rebalance({"PETR4": 1.0}, {"PETR4": -1}, {"PETR4": 10.0}, 100.0)


# ── API ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def rebalance_api(monkeypatch):
    ctx = SessionState(uid="rebalance-test", selection=["PETR4", "VALE3"])
    ctx.portfolio = PortfolioSnapshot(
        id="portfolio-1",
        revision=ctx.revision,
        tickers=["PETR4", "VALE3"],
        weights={"PETR4": 0.5, "VALE3": 0.5},
        mode="Otimização de Mínima Volatilidade",
        returns=pd.DataFrame(),
        prices=pd.DataFrame(),
        lookback_days=730,
        result={},
    )
    quotes = pd.DataFrame({"PETR4.SA": [30.0], "VALE3.SA": [70.0]})
    monkeypatch.setattr(api, "get_listed_stocks", lambda: pd.DataFrame({"Ticker": ["PETR4", "VALE3"]}))
    monkeypatch.setattr(
        api,
        "get_portfolio_trade_prices",
        lambda tickers: quotes.loc[:, [ticker for ticker in tickers if ticker in quotes]],
    )
    app = FastAPI()

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(api.router)
    return TestClient(app), ctx


def plan_payload(**overrides):
    payload = {
        "revision": 0,
        "portfolio_id": "portfolio-1",
        "contribution": 1000.0,
        "current_quantities": {},
        "allow_fractional_us": False,
    }
    payload.update(overrides)
    return payload


def test_plan_endpoint_returns_buy_ticket(rebalance_api):
    client, _ = rebalance_api
    response = client.post("/api/rebalance/plan", json=plan_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["portfolio_id"] == "portfolio-1"
    assert body["revision"] == 0
    assert body["summary"]["zero_sell_guaranteed"] is True
    assert body["summary"]["allocated_cash"] + body["summary"]["leftover_cash"] == pytest.approx(1000.0)
    assert all(row["buy_qty"] >= 0 for row in body["rows"])
    assert body["price_source"]


def test_plan_endpoint_conflicts_on_stale_portfolio_or_revision(rebalance_api):
    client, _ = rebalance_api
    stale_id = client.post("/api/rebalance/plan", json=plan_payload(portfolio_id="other"))
    assert stale_id.status_code == 409
    stale_revision = client.post("/api/rebalance/plan", json=plan_payload(revision=1))
    assert stale_revision.status_code == 409


def test_plan_endpoint_validates_contribution_and_quantities(rebalance_api):
    client, _ = rebalance_api
    assert client.post("/api/rebalance/plan", json=plan_payload(contribution=0)).status_code == 422
    assert client.post("/api/rebalance/plan", json=plan_payload(contribution=10_000_001)).status_code == 422
    negative = client.post(
        "/api/rebalance/plan",
        json=plan_payload(current_quantities={"PETR4": -1}),
    )
    assert negative.status_code == 422
    assert "não negativas" in negative.json()["detail"]


def test_import_endpoint_returns_positions_and_validation_errors(rebalance_api):
    client, _ = rebalance_api
    content = "Código de Negociação;Quantidade;Preço de Fechamento\nPETR4;100;38,50\n"
    response = client.post("/api/rebalance/import", json={"content": content})
    assert response.status_code == 200
    body = response.json()
    assert body["positions"][0]["ticker"] == "PETR4"
    assert body["positions"][0]["quantity"] == 100
    assert client.post("/api/rebalance/import", json={"content": ""}).status_code == 422
    invalid = client.post("/api/rebalance/import", json={"content": "nome;descricao\nPETR4;Petrobras"})
    assert invalid.status_code == 422
    assert "colunas" in invalid.json()["detail"]
