"""Shared cached access to B3 fundamentals and Yahoo Finance market targets.

Fetching the full Fundamentus table is expensive (scrapes & parses ~300+
tickers), so route callers through this module's shared cache.
"""

import datetime
import logging
import math
import threading
from pathlib import Path
from urllib.parse import urlsplit


import pandas as pd
from utils.cache import cache_data
import yfinance as yf
from utils.formatting import normalize_numeric_text

logger = logging.getLogger(__name__)
# ponytail: one global lock serializes Fundamentus' requests.Session patch;
# use private sessions instead if the shared fetch needs more concurrency.
FUNDAMENTUS_REQUEST_LOCK = threading.Lock()


def _clear_fundamentus_http_cache() -> None:
    """Remove non-expiring Fundamentus result responses before a fresh fetch."""
    import requests_cache

    with requests_cache.CachedSession("http_cache") as session:
        cache = session.cache
        for key in list(cache.responses.keys()):
            response = cache.responses.get(key)
            if response is None:
                continue
            url = urlsplit(response.url)
            if url.hostname == "www.fundamentus.com.br" and url.path == "/resultado.php":
                cache.delete(key)


def _normalize_listed_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the local stock-universe schema."""
    required = {"Ticker", "Setor"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B3 universe is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("B3 universe is empty")
    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["Ticker"].ne("") & frame["Ticker"].ne("NAN")]
    for column in ("Empresa", "RazaoSocial"):
        if column in frame:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame.empty:
        raise ValueError("B3 universe contains no valid tickers")
    return frame.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


@cache_data(ttl=3600)
def get_listed_stocks() -> pd.DataFrame:
    """Load the shared local B3 universe once per hour."""
    frame = pd.read_csv(Path(__file__).resolve().parent.parent / "acoes-listadas-b3.csv")
    return _normalize_listed_stocks(frame)


def clean_numeric_column(col):
    """Parse Brazilian decimals and thousands into numeric values."""
    return pd.to_numeric(col.map(normalize_numeric_text), errors="coerce")


def compute_target_upside(current_price, target_price):
    """Return target-price upside in percent, or None for invalid prices."""
    try:
        current, target = float(current_price), float(target_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current) or not math.isfinite(target) or current <= 0 or target <= 0:
        return None
    return (target / current - 1) * 100


@cache_data(ttl=3600)
def get_market_target_data(ticker_b3):
    """Fetch Yahoo Finance's target-price and recommendation data for a B3 ticker."""
    try:
        info = yf.Ticker(f"{ticker_b3}.SA").info or {}
        return {
            "company_name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency") or "BRL",
            "price": info.get("currentPrice"),
            "regular_price": info.get("regularMarketPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "target_high": info.get("targetHighPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
        }
    except Exception as exc:
        logger.warning("Market target fetch failed for %s: %s", ticker_b3, exc)
        logger.debug("Market target fetch details", exc_info=True)
        return {"_error": str(exc)}


@cache_data(ttl=3600)
def get_full_market_data():
    """Fetch an hourly Fundamentus snapshot and retain its successful fetch time."""
    import fundamentus.resultado as fzr

    # The installed Fundamentus scraper wraps requests in a non-expiring HTTP
    # cache. Evict its result URL whenever the hourly application cache misses.
    with FUNDAMENTUS_REQUEST_LOCK:
        _clear_fundamentus_http_cache()
        frame = fzr.get_resultado_raw()
    frame.attrs["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return frame


def get_liquidity_by_ticker() -> pd.Series:
    """Return Fundamentus' two-month average daily traded volume (BRL) indexed by ticker."""
    frame = get_full_market_data()
    liquidity = clean_numeric_column(frame["Liq.2meses"]).astype(float)
    liquidity.attrs["fetched_at"] = frame.attrs.get("fetched_at")
    return liquidity


@cache_data(ttl=86400)
def get_sorted_tickers_by_liquidity(tickers_list):
    try:
        df = get_full_market_data()
        df = df.copy()
        df["_liquidity_sort"] = clean_numeric_column(df["Liq.2meses"])
        df = df.sort_values("_liquidity_sort", ascending=False, na_position="last")
        sorted_all = df.index.tolist()
        sorted_filtered = [t for t in sorted_all if t in tickers_list]
        remaining = [t for t in tickers_list if t not in sorted_filtered]
        return sorted_filtered + remaining
    except Exception:
        logger.warning("get_sorted_tickers_by_liquidity failed, returning unsorted list", exc_info=True)
        return tickers_list


_YF_UNIVERSE_TIMEOUT_SECONDS = 30


def _change(last, previous):
    try:
        last, previous = float(last), float(previous)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(last) or not math.isfinite(previous) or previous <= 0:
        return None
    return last / previous - 1


def universe_changes(adjusted: pd.DataFrame, tickers) -> dict:
    """Daily and five-session changes, as fractions, from Yahoo adjusted closes.

    `adjusted` has one column per Yahoo symbol and one row per session. Every
    change is measured on the download's own latest sessions, so a stock that
    did not trade on one of them gets None instead of a change over a longer span.
    """
    adjusted = adjusted.dropna(how="all")
    if adjusted.empty:
        raise ValueError("Yahoo Finance não retornou cotações.")

    def between(column, back):
        return _change(column.iloc[-1], column.iloc[-1 - back]) if len(column) > back else None

    changes = {}
    for ticker in tickers:
        symbol = f"{ticker}.SA"
        column = adjusted[symbol] if symbol in adjusted else None
        changes[ticker] = {"day": None, "week": None} if column is None else {"day": between(column, 1), "week": between(column, 5)}
    return {"session": adjusted.index[-1].date().isoformat(), "changes": changes}


@cache_data(ttl=600)
def get_universe_changes(tickers: tuple) -> dict:
    """Fetch one Yahoo download for the whole universe and return its latest-session changes."""
    downloaded = yf.download([f"{ticker}.SA" for ticker in tickers], period="15d", interval="1d", auto_adjust=False,
                             progress=False, timeout=_YF_UNIVERSE_TIMEOUT_SECONDS)
    if downloaded.empty or "Adj Close" not in downloaded:
        raise ValueError("Yahoo Finance não retornou cotações.")
    result = universe_changes(downloaded["Adj Close"], tickers)
    result["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result
