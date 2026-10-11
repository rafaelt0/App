import pandas as pd
import pytest

from utils.screener import PRESET_FILTERS, filter_stocks, prepare_export


@pytest.fixture

def stocks():
    return pd.DataFrame(
        {
            "pl": [10.0, 0.0, 16.0, 8.0, 12.0],
            "roe": [0.15, 0.20, 0.12, None, 0.10],
            "dy": [0.06, 0.05, 0.04, 0.08, 0.07],
            "liq2m": [2_000_000, 3_000_000, 4_000_000, 2_000_000, 500_000],
            "c5y": [0.10, 0.12, 0.15, None, 0.09],
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
    assert set(filter_stocks(stocks, PRESET_FILTERS["Crescimento com lucro"]).index) == {
        "AAA3", "BBB3", "CCC3"
    }


def test_missing_active_values_fail_but_inactive_missing_columns_do_not(stocks):
    assert "DDD3" not in filter_stocks(
        stocks, {"roe_min": 0.12}
    ).index  # active but missing ROE
    assert len(filter_stocks(stocks[["liq2m"]], {"liq2m_min": 1_000_000})) == 4


def test_missing_active_column_is_an_error(stocks):
    with pytest.raises(ValueError, match="Coluna necessária"):
        filter_stocks(stocks.drop(columns="dy"), {"dy_min": 0.05})
    with pytest.raises(ValueError, match="Coluna necessária"):
        filter_stocks(stocks.drop(columns="c5y"), {"c5y_min": 0.10})


def test_new_presets_apply_only_their_stated_thresholds():
    candidates = pd.DataFrame(
        {
            "pl": [10, 0, 21, 10, None, None, None, 10, 21, 10, 10, 10],
            "roe": [
                0.20, 0.30, 0.30, 0.17, None, None, None,
                0.12, 0.20, 0.119, 0.20, None,
            ],
            "liq2m": [
                2_000_000, 2_000_000, 2_000_000, 2_000_000,
                10_000_000, 9_999_999, 10_000_000, 1_000_000,
                10_000_000, 10_000_000, 999_999, 10_000_000,
            ],
            "c5y": [None, None, None, None, None, None, None, 0.10, 0.099, 0.20, 0.20, 0.20],
        },
        index=[
            "QUALITY", "ZERO_PL", "HIGH_PL", "LOW_ROE", "LIQUID", "LOW_LIQUID",
            "LIQUID_2", "GROWTH", "LOW_GROWTH", "LOW_GROWTH_ROE",
            "LOW_GROWTH_LIQ", "MISSING_GROWTH_ROE",
        ],
    )

    assert list(
        filter_stocks(candidates, PRESET_FILTERS["Qualidade rentável"]).index
    ) == ["QUALITY"]
    assert set(filter_stocks(candidates, PRESET_FILTERS["Alta liquidez"]).index) == {
        "LIQUID", "LIQUID_2", "LOW_GROWTH", "LOW_GROWTH_ROE", "MISSING_GROWTH_ROE"
    }
    assert list(
        filter_stocks(candidates, PRESET_FILTERS["Crescimento com lucro"]).index
    ) == ["GROWTH"]


def test_export_uses_percentage_points_and_keeps_every_row(stocks):
    exported = prepare_export(
        stocks,
        {"dy": "Div. Yield (%)", "roe": "ROE (%)", "pl": "P/L"},
    )

    assert list(exported.index) == ["AAA3", "BBB3", "CCC3", "DDD3", "EEE3"]
    assert exported.loc["AAA3", "Div. Yield (%)"] == 6.0
    assert exported.loc["AAA3", "ROE (%)"] == 15.0
    assert exported.index.name == "Papel"


