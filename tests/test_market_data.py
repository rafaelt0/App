import contextlib
import threading
import time
from types import SimpleNamespace

import pandas as pd

from utils import market_data
from utils.market_data import (
    _normalize_listed_stocks,
    clean_numeric_column,
    compute_target_upside,
    get_listed_stocks,
)


def test_listed_stock_universe_has_required_schema_and_unique_tickers():
    frame = get_listed_stocks()

    assert {"Ticker", "Setor", "Empresa", "RazaoSocial"}.issubset(frame.columns)
    assert len(frame) == 332
    assert frame["Empresa"].str.strip().ne("").sum() == 332
    assert frame["RazaoSocial"].str.strip().ne("").sum() == 332
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


def test_yahoo_bulk_quotes_keep_correct_symbols_and_native_units(monkeypatch):
    def row(symbol, currency, stamp, price, provider_symbol=None):
        return {"symbol": symbol, "response": [{"meta": {"symbol": provider_symbol or symbol, "currency": currency,
                "regularMarketTime": stamp, "regularMarketPrice": price, "exchangeTimezoneName": "America/Sao_Paulo"}}]}

    def fetch(url, *, params, **kwargs):
        assert params["symbols"] == "BRL=X,BTC-USD,INVALID,STALE"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"spark": {"result": [
            row("BTC-USD", "USD", 1795120200, 100_000),
            row("BRL=X", "BRL", 1795120200, 5.5),
            row("INVALID", "USD", 1795120200, 42, provider_symbol="WRONG"),
            row("STALE", "USD", None, 20),
            row("UNREQUESTED", "BRL", 1795120200, 12),
        ]}})

    monkeypatch.setattr(market_data, "_fetch_public_market_data", fetch)
    quotes = market_data.get_yahoo_quotes.__wrapped__(("BTC-USD", "BRL=X", "STALE", "INVALID"))
    assert set(quotes) == {"BTC-USD", "BRL=X"}
    assert quotes["BRL=X"]["currency"] == "BRL" and quotes["BRL=X"]["price"] == 5.5
    assert quotes["BTC-USD"]["last_quote_date"] == "2026-11-19"


def test_fundamentus_fii_feed_keeps_yield_points_and_brl_turnover(monkeypatch):
    content = """<table id="tabelaResultado"><tr><th>Papel</th><th>Segmento</th><th>Cotação</th><th>P/VP</th>
      <th>Dividend Yield</th><th>Liquidez</th></tr><tr><td>HGLG11</td><td>Logística</td><td>140,50</td>
      <td>0,89</td><td>9,90%</td><td>715.027</td></tr></table>""".encode("iso-8859-1")
    monkeypatch.setattr(market_data, "_fetch_public_market_data", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, content=content))
    table = market_data.get_fii_market_data.__wrapped__()
    assert table.loc["HGLG11"].to_dict() == {"setor": "Logística", "price": 140.5, "pvp": .89, "dy": 9.9, "liq2m": 715_027}
    assert table.attrs["fetched_at"].endswith("+00:00")


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


def test_yahoo_downloads_never_run_at_the_same_time(monkeypatch):
    """The cache warm-up and a visitor can ask Yahoo together; yfinance shares one
    process-wide state and corrupts itself when two downloads overlap."""
    in_flight, overlaps = [], []

    def fake_download(symbols, **kwargs):
        in_flight.append(1)
        overlaps.append(len(in_flight))
        time.sleep(0.05)
        in_flight.remove(1)
        return pd.DataFrame()

    def run(ticker):
        with contextlib.suppress(ValueError):
            market_data.get_universe_changes((ticker,))

    market_data.get_universe_changes.clear()
    monkeypatch.setattr(market_data.yf, "download", fake_download)
    workers = [threading.Thread(target=run, args=(f"AAA{index}",)) for index in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)

    assert overlaps == [1, 1, 1]


def test_yahoo_info_does_not_block_behind_download_lock(monkeypatch):
    """Bulk yf.download warmups must never stall single-ticker Yahoo info lookups."""
    market_data.get_yahoo_info.clear()
    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: SimpleNamespace(info={"regularMarketPrice": 225.0}))
    done = threading.Event()
    result = {}

    def fetch():
        result["info"] = market_data.get_yahoo_info("AAPL")
        done.set()

    with market_data.YFINANCE_REQUEST_LOCK:
        worker = threading.Thread(target=fetch)
        worker.start()
        assert done.wait(timeout=2.0)
        worker.join(2.0)
    market_data.get_yahoo_info.clear()
    assert result["info"] == {"regularMarketPrice": 225.0}
