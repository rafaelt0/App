from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.screener import PRESET_FILTERS, filter_stocks, prepare_export


@pytest.fixture

def stocks():
    return pd.DataFrame(
        {
            "pl": [10.0, 0.0, 16.0, 8.0, 12.0],
            "roe": [0.15, 0.20, 0.12, None, 0.10],
            "dy": [0.06, 0.05, 0.04, 0.08, 0.07],
            "liq2m": [2_000_000, 3_000_000, 4_000_000, 2_000_000, 500_000],
        },
        index=["AAA3", "BBB3", "CCC3", "DDD3", "EEE3"],
    )


def test_presets_include_only_stocks_meeting_their_criteria(stocks):
    assert set(filter_stocks(stocks, PRESET_FILTERS["Explorar B3"]).index) == {
        "AAA3", "BBB3", "CCC3", "DDD3"
    }
    assert set(filter_stocks(stocks, PRESET_FILTERS["Lucro a preço moderado"]).index) == {
        "AAA3"
    }
    assert set(filter_stocks(stocks, PRESET_FILTERS["Renda atual"]).index) == {
        "AAA3", "BBB3"
    }


def test_missing_active_values_fail_but_inactive_missing_columns_do_not(stocks):
    assert "DDD3" not in filter_stocks(
        stocks, {"roe_min": 0.12}
    ).index  # active but missing ROE
    assert len(filter_stocks(stocks[["liq2m"]], {"liq2m_min": 1_000_000})) == 4


def test_missing_active_column_is_an_error(stocks):
    with pytest.raises(ValueError, match="Coluna necessária"):
        filter_stocks(stocks.drop(columns="dy"), {"dy_min": 0.05})


def test_sorting_does_not_change_which_stocks_match(stocks):
    criteria = PRESET_FILTERS["Renda atual"]
    matched = filter_stocks(stocks, criteria)
    ascending = matched.sort_values("liq2m").index
    descending = matched.sort_values("liq2m", ascending=False).index
    assert set(ascending) == set(descending) == set(matched.index)


def test_export_uses_percentage_points_and_keeps_every_row(stocks):
    exported = prepare_export(
        stocks,
        {"dy": "Div. Yield (%)", "roe": "ROE (%)", "pl": "P/L"},
    )

    assert len(exported) == len(stocks)
    assert exported.loc["AAA3", "Div. Yield (%)"] == 6.0
    assert exported.loc["AAA3", "ROE (%)"] == 15.0
    assert exported.index.name == "Papel"


def test_screener_preset_change_and_manual_edit_update_ui_state():
    raw = pd.DataFrame(
        {
            "Cotação": [12.5, 15.0, 17.0],
            "P/L": [10.0, 0.0, 16.0],
            "P/VP": [1.2, 0.8, 1.0],
            "Div.Yield": [0.06, 0.07, 0.04],
            "ROE": [0.15, 0.12, 0.12],
            "ROIC": [0.1, 0.11, 0.1],
            "EV/EBITDA": [5.0, 6.0, 7.0],
            "Liq.2meses": [2_000_000, 3_000_000, 4_000_000],
        },
        index=["AAA3", "BBB3", "CCC3"],
    )
    raw.attrs["fetched_at"] = "2026-09-24T12:00:00+00:00"
    with patch("utils.market_data.get_full_market_data", return_value=raw):
        app = AppTest.from_file("pages/5_Screener.py").run()
        assert not app.exception
        assert any("3 de 3 ativos" in item.value for item in app.caption)

        app.selectbox(key="preset_select").select("Renda atual").run()
        assert app.selectbox(key="preset_select").value == "Renda atual"
        assert any("DY ≥ 5%" in item.value and "ROE ≥ 10%" in item.value for item in app.caption)

        app.number_input(key="roe_min").set_value(20).run()
        assert app.selectbox(key="preset_select").value == "Personalizado"
