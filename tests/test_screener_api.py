import io

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import screener
from backend.state import SessionState, replace_selection


@pytest.fixture
def screener_api(monkeypatch):
    ctx = SessionState(uid="screener-test")
    app = FastAPI()

    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = ctx
        return await call_next(request)

    app.include_router(screener.router)
    return TestClient(app), ctx


def market_frame():
    frame = pd.DataFrame({"Cotação": ["10,50", "20,00", "3,00", "4,00"], "P/L": [0, 15, 10, None],
                          "Div.Yield": [.05, .08, .02, None], "ROE": [.12, .18, .20, None],
                          "Liq.2meses": [1_000_000, 2_000_000, 2_000_000, 3_000_000],
                          "Dív. Líq./Patrim.": [1, 2, 3, None], "Cresc. Rec.5a": [.1, .2, .3, None]},
                         index=["ZERO3", "BBB4", "AAA3", "NULL3"])
    frame.attrs["fetched_at"] = "2025-01-01T12:00:00+00:00"
    return frame


def test_presets_exclusive_boundaries_and_stable_ties():
    raw = market_frame()
    body = screener.Query(criteria=screener.Criteria(**screener.PRESET_FILTERS["Lucro a preço moderado"]))
    result, exported = screener.screen(raw, body)
    assert [row["ticker"] for row in result["rows"]] == ["BBB4", "AAA3"]
    assert result["metrics"]["mean_dy"] == pytest.approx(5)
    assert result["metrics"]["mean_roe"] == pytest.approx(19)
    assert result["active_filters"] == ["Liquidez 2m ≥ R$ 1.000.000", "P/L > 0 e ≤ 15", "ROE ≥ 12%"]
    assert result["rows"][0]["pl"] == 15  # no detail scraper /100 correction
    assert exported["Div. Yield (%)"].tolist() == [8, 2]
    assert exported["Dív. Líq./Patrim."].tolist() == [2, 3]
    assert raw["P/L"].iloc[1] == 15 and raw["Div.Yield"].iloc[1] == .08


def test_disabled_missing_fields_do_not_filter_and_active_missing_fields_fail():
    raw = market_frame().drop(columns=["ROE", "P/L"])
    result, _ = screener.screen(raw, screener.Query(criteria=screener.Criteria(liq2m_min=1_000_000)))
    assert result["matched_count"] == 4
    assert result["metrics"]["mean_roe"] is None
    assert [row["ticker"] for row in result["rows"]] == ["NULL3", "BBB4", "AAA3", "ZERO3"]
    with pytest.raises(Exception) as exc:
        screener.screen(raw, screener.Query(criteria=screener.Criteria(roe_min=.12)))
    assert exc.value.status_code == 422 and "roe" in exc.value.detail
    with pytest.raises(Exception) as exc:
        screener.screen(raw, screener.Query(sort=screener.Sort(field="pl", direction="asc")))
    assert exc.value.status_code == 422 and "pl" in exc.value.detail


def test_valid_zero_matches_are_not_source_failure():
    result, exported = screener.screen(market_frame(), screener.Query(criteria=screener.Criteria(liq2m_min=99_000_000)))
    assert result["matched_count"] == 0 and result["total_count"] == 4
    assert result["metrics"] == {"mean_dy": None, "mean_roe": None}
    assert exported.empty
    with pytest.raises(Exception) as exc:
        screener.screen(pd.DataFrame(), screener.Query())
    assert exc.value.status_code == 503


def test_unknown_filters_ranges_and_invalid_sort_rejected(screener_api):
    client, _ = screener_api
    for criteria in [{"made_up": 3}, {"pl_min": 20, "pl_max": 10}, {"roe_min": .81},
                     {"dy_min": .31}, {"liq2m_min": 100_000_001}, {"pl_min_exclusive": True}]:
        assert client.post("/api/screener/query", json={"criteria": criteria}).status_code == 422
    with pytest.raises(ValidationError):
        screener.Sort(field="roe", direction="asc")


def test_sort_nulls_last_and_exact_lower_boundary():
    body = screener.Query(criteria=screener.Criteria(pl_min=0, pl_max=15), sort=screener.Sort(field="pl", direction="asc"))
    result, _ = screener.screen(market_frame(), body)
    assert [row["ticker"] for row in result["rows"]] == ["ZERO3", "AAA3", "BBB4"]
    all_rows, _ = screener.screen(market_frame(), screener.Query(sort=screener.Sort(field="pl", direction="asc")))
    assert [row["ticker"] for row in all_rows["rows"]] == ["ZERO3", "AAA3", "BBB4", "NULL3"]


def test_csv_retains_all_matches_columns_and_units_independent_of_display(screener_api, monkeypatch):
    client, ctx = screener_api
    monkeypatch.setattr(screener, "get_full_market_data", market_frame)
    response = client.post("/api/screener/query", json={"criteria": {"liq2m_min": 1_000_000}})
    assert response.status_code == 200
    result = response.json()
    assert [row["ticker"] for row in result["rows"]] == ["NULL3", "BBB4", "AAA3", "ZERO3"]
    assert result["metrics"]["mean_dy"] == pytest.approx(5)
    csv_response = client.get(f"/api/screener/{result['id']}/results.csv")
    assert csv_response.content.startswith(b"\xef\xbb\xbf")
    exported = pd.read_csv(io.BytesIO(csv_response.content), index_col=0)
    assert exported.index.name == "Papel"
    assert exported.index.tolist() == ["NULL3", "BBB4", "AAA3", "ZERO3"]
    assert exported.loc["BBB4", "ROE (%)"] == 18
    assert exported.loc["ZERO3", "Cresc. Rec. 5a (%)"] == 10
    assert exported.loc["BBB4", "Cotação (R$)"] == 20
    assert exported.loc["BBB4", "Dív. Líq./Patrim."] == 2
    assert "screener_b3_" in csv_response.headers["content-disposition"]
    assert client.get("/api/screener/not-owned/results.csv").status_code == 404
    replace_selection(ctx, ["AAA3"], 0)
    assert client.get(f"/api/screener/{result['id']}/results.csv").status_code == 404


def test_source_timestamp_age_never_negative_and_invalid_is_explicit():
    assert screener.source_time("2999-01-01T12:00:00+00:00")["age_minutes"] == 0
    assert screener.source_time("bad") == {"fetched_at": None, "age_minutes": None}
    assert screener.source_time(None) == {"fetched_at": None, "age_minutes": None}
    assert screener.source_time("2025-01-01T12:00:00") == {"fetched_at": None, "age_minutes": None}


def test_source_error_and_changed_selection_do_not_publish(screener_api, monkeypatch):
    client, ctx = screener_api

    def failed():
        raise OSError("offline fixture")

    monkeypatch.setattr(screener, "get_full_market_data", failed)
    response = client.post("/api/screener/query", json={})
    assert response.status_code == 503 and "screener" not in ctx.artifacts

    def changed():
        replace_selection(ctx, ["AAA3"], 0)
        return market_frame()

    monkeypatch.setattr(screener, "get_full_market_data", changed)
    assert client.post("/api/screener/query", json={}).status_code == 409
    assert "screener" not in ctx.artifacts


def fundamentus_frame():
    frame = pd.DataFrame({
        "Cotação": ["30,00", "15,00", "5,00"],
        "P/L": [9.5, 12, -3], "P/VP": [1.9, 2.0, -1.2], "PSR": [0, 1.5, .4],
        "Div.Yield": [0.0, .07, 0.0], "EV/EBIT": [0, 6, 0], "EV/EBITDA": [0, 5, 7],
        "Mrg Ebit": [0, .2, -.1], "Mrg. Líq.": [0, .1, -.3], "ROIC": [0, .15, -.05],
        "ROE": [.2, .18, .69], "Liq. Corr.": [0, 1.3, .8], "Liq.2meses": [5_000_000, 4_000_000, 3_000_000],
        "Dív. Líq./Patrim.": [0, .5, 4], "Cresc. Rec.5a": [.1, .1, -.2],
    }, index=["ITUB4", "BBSE3", "OIBR3"])
    frame.attrs["fetched_at"] = "2025-01-01T12:00:00+00:00"
    return frame


BANK_ZERO_COLUMNS = ("psr", "evebit", "evebitda", "mrgebit", "mrgliq", "roic", "liqc", "divbpatr")


def test_zero_placeholders_for_banks_are_missing_while_real_zeros_stay():
    frame = screener.normalize_market(fundamentus_frame())
    for column in BANK_ZERO_COLUMNS:
        assert pd.isna(frame.loc["ITUB4", column]), column
    assert frame.loc["ITUB4", "dy"] == 0.0  # a real zero DY is valid
    assert frame.loc["ITUB4", "pl"] == 9.5
    assert frame.loc["BBSE3", "evebitda"] == 5
    assert frame.loc["BBSE3", "roic"] == .15


def test_placeholder_missing_values_serialize_as_null_in_results():
    result, exported = screener.screen(fundamentus_frame(), screener.Query())
    rows = {row["ticker"]: row for row in result["rows"]}
    assert rows["ITUB4"]["roic"] is None and rows["ITUB4"]["evebitda"] is None
    assert rows["ITUB4"]["liqc"] is None and rows["ITUB4"]["divbpatr"] is None
    assert rows["ITUB4"]["dy"] == 0.0
    assert rows["BBSE3"]["evebitda"] == 5
    assert "roic" in result["default_columns"] and "evebitda" in result["default_columns"]
    assert pd.isna(exported.loc["ITUB4", "ROIC (%)"])


def test_negative_equity_drops_roe_and_pvp_so_roe_filters_exclude_it():
    frame = screener.normalize_market(fundamentus_frame())
    assert pd.isna(frame.loc["OIBR3", "pvp"]) and pd.isna(frame.loc["OIBR3", "roe"])
    assert frame.loc["OIBR3", "pl"] == -3  # negative P/L is a real reported value
    assert frame.loc["BBSE3", "roe"] == .18 and frame.loc["BBSE3", "pvp"] == 2.0

    result, _ = screener.screen(fundamentus_frame(), screener.Query(criteria=screener.Criteria(roe_min=.12)))
    assert [row["ticker"] for row in result["rows"]] == ["ITUB4", "BBSE3"]

    everything, _ = screener.screen(fundamentus_frame(), screener.Query())
    oibr = next(row for row in everything["rows"] if row["ticker"] == "OIBR3")
    assert oibr["roe"] is None and oibr["pvp"] is None
    assert everything["metrics"]["mean_roe"] == pytest.approx(19)


def test_dividend_preset_is_labelled_as_trailing_twelve_month_yield():
    presets = {preset["name"]: preset for preset in screener.config()["presets"]}
    assert "Dividendos sustentáveis" not in presets
    preset = presets["Renda atual (12m)"]
    assert preset["criteria"]["dy_min"] == 0.06
    assert "12 meses" in preset["description"] and "payout" in preset["description"]
