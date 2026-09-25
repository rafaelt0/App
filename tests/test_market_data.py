import pandas as pd

from utils.market_data import (
    _normalize_listed_stocks,
    clean_numeric_column,
    compute_target_upside,
    get_listed_stocks,
)


def test_listed_stock_universe_has_required_schema_and_unique_tickers():
    frame = get_listed_stocks()

    assert {"Ticker", "Setor", "Empresa", "RazaoSocial"}.issubset(frame.columns)
    assert len(frame) == 335
    assert frame["Empresa"].str.strip().ne("").sum() == 335
    assert frame["RazaoSocial"].str.strip().ne("").sum() == 334
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


def test_compute_target_upside_returns_percentage_gain_or_loss():
    assert compute_target_upside(100, 125) == 25
    assert round(compute_target_upside(100, 80), 8) == -20


def test_compute_target_upside_rejects_invalid_prices():
    assert compute_target_upside(0, 125) is None
    assert compute_target_upside(100, float("nan")) is None
    assert compute_target_upside(None, 125) is None


def test_full_market_fetch_records_successful_fetch_time(monkeypatch):
    import fundamentus.resultado as resultado
    import utils.market_data as market_data

    raw = pd.DataFrame({"P/L": [10]}, index=["TEST3"])
    monkeypatch.setattr(market_data, "_clear_fundamentus_http_cache", lambda: None)
    monkeypatch.setattr(resultado, "get_resultado_raw", lambda: raw.copy())
    market_data.get_full_market_data.clear()

    fetched = market_data.get_full_market_data()

    assert fetched.attrs["fetched_at"].endswith("+00:00")
    market_data.get_full_market_data.clear()


def test_http_cache_clear_evicts_fundamentus_non_expiring_result(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    from utils.market_data import _clear_fundamentus_http_cache

    deleted = []
    response = SimpleNamespace(url="http://www.fundamentus.com.br/resultado.php")
    cache = SimpleNamespace(
        responses={"stale-response": response},
        delete=deleted.append,
    )

    class Session:
        def __enter__(self):
            return SimpleNamespace(cache=cache)

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "requests_cache",
        types.SimpleNamespace(CachedSession=lambda name: Session()),
    )

    _clear_fundamentus_http_cache()

    assert deleted == ["stale-response"]
