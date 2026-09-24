import pandas as pd

from utils.market_data import (
    _normalize_listed_stocks,
    clean_numeric_column,
    get_listed_stocks,
)


def test_listed_stock_universe_has_required_schema_and_unique_tickers():
    frame = get_listed_stocks()

    assert {"Ticker", "Setor"}.issubset(frame.columns)
    assert frame["Ticker"].is_unique
    assert frame["Ticker"].notna().all()
    assert (frame["Ticker"] == frame["Ticker"].str.upper()).all()


def test_normalize_listed_stocks_rejects_missing_schema():
    try:
        _normalize_listed_stocks(pd.DataFrame({"Ticker": ["PETR4"]}))
    except ValueError as exc:
        assert "Setor" in str(exc)
    else:
        raise AssertionError("missing Setor column should raise ValueError")


def test_normalize_listed_stocks_rejects_empty_frame():
    try:
        _normalize_listed_stocks(pd.DataFrame(columns=["Ticker", "Setor"]))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty stock universe should raise ValueError")


def test_normalize_listed_stocks_rejects_invalid_tickers():
    frame = pd.DataFrame({"Ticker": ["", "nan"], "Setor": ["A", "B"]})
    try:
        _normalize_listed_stocks(frame)
    except ValueError as exc:
        assert "valid tickers" in str(exc)
    else:
        raise AssertionError("invalid tickers should raise ValueError")


def test_clean_numeric_column_parses_brazilian_decimals():
    values = clean_numeric_column(pd.Series(["1,25%", "-3,5", "—"]))

    assert values.iloc[0] == 1.25
    assert values.iloc[1] == -3.5
    assert pd.isna(values.iloc[2])
