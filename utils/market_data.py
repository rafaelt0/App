"""Shared cached access to B3 fundamentals and Yahoo Finance market targets.

Fetching the full Fundamentus table is expensive (scrapes & parses ~300+
tickers), so route callers through this module's shared cache.
"""

import datetime
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


import pandas as pd
import requests
from requests.sessions import Session as UncachedSession
from utils.cache import cache_data
import yfinance as yf
from utils.formatting import normalize_numeric_text

logger = logging.getLogger(__name__)
# ponytail: one global lock serializes Fundamentus' requests.Session patch;
# use private sessions instead if the shared fetch needs more concurrency.
FUNDAMENTUS_REQUEST_LOCK = threading.Lock()
# yfinance.download keeps state shared by the whole process (`shared._DFS` and
# the error table), and two downloads in flight corrupt it — "dictionary changed
# size during iteration" — or drop columns. Serialize every yf.download call on
# this lock, including the ones in market_strip and portfolio_data.
# yf.Ticker(symbol).info does not touch shared._DFS and uses YfData's own
# cookie lock, so it runs concurrently without blocking behind bulk downloads.
YFINANCE_REQUEST_LOCK = threading.Lock()
DAY_SECONDS = 86400
# Covers a weekend plus a holiday. Within it a stale quote snapshot is served at
# once, labelled with its fetch time, while a background refresh replaces it.
QUOTE_STALE_SECONDS = 4 * DAY_SECONDS


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


@cache_data(ttl=900, max_entries=512, cache_if=bool)
def get_yahoo_info(symbol):
    """Share the full Yahoo response between analysis, screener, and analyst consensus."""
    return yf.Ticker(symbol).info or {}


def _positive_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _fetch_public_market_data(url, **kwargs):
    """Bypass the global HTTP cache installed by the Fundamentus stock scraper."""
    with UncachedSession() as session:
        response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4, **kwargs)
        response.raise_for_status()
        return response


@cache_data(ttl=120, cache_if=bool)
def get_yahoo_quotes(symbols: tuple[str, ...]) -> dict[str, dict]:
    """Load delayed, unadjusted market quotes without requesting corporate fundamentals."""
    quotes = {}
    for offset in range(0, len(symbols), 20):
        batch = set(symbols[offset:offset + 20])
        try:
            response = _fetch_public_market_data(
                "https://query1.finance.yahoo.com/v7/finance/spark",
                params={"symbols": ",".join(sorted(batch)), "range": "5d", "interval": "1d"},
            )
            rows = response.json()["spark"]["result"]
            if not isinstance(rows, list):
                raise ValueError("Yahoo Finance não retornou uma lista de cotações.")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = row.get("symbol")
                items = row.get("response")
                if not isinstance(symbol, str) or symbol not in batch or not isinstance(items, list) or not items:
                    continue
                meta = items[0].get("meta") if isinstance(items[0], dict) else None
                if not isinstance(meta, dict):
                    continue
                price = _positive_number(meta.get("regularMarketPrice"))
                currency = meta.get("currency")
                stamp = _positive_number(meta.get("regularMarketTime"))
                if meta.get("symbol") != symbol or price is None or stamp is None:
                    continue
                if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isalpha():
                    continue
                try:
                    zone = ZoneInfo(meta.get("exchangeTimezoneName") or "UTC")
                except (TypeError, ValueError, ZoneInfoNotFoundError):
                    zone = datetime.timezone.utc
                try:
                    quote_date = datetime.datetime.fromtimestamp(stamp, zone).date().isoformat()
                except (OSError, OverflowError, ValueError):
                    continue
                quotes[symbol] = {
                    "price": price, "currency": currency, "last_quote_date": quote_date,
                    "low_52w": _positive_number(meta.get("fiftyTwoWeekLow")),
                    "high_52w": _positive_number(meta.get("fiftyTwoWeekHigh")),
                    "source": "Yahoo Finance",
                }
        except (requests.RequestException, AttributeError, KeyError, TypeError, ValueError):
            logger.warning("Yahoo Finance quotes unavailable for %s", ", ".join(sorted(batch)), exc_info=True)
    return quotes


@cache_data(ttl=900, stale=DAY_SECONDS)
def get_fii_market_data() -> pd.DataFrame:
    """Read the dedicated Fundamentus FII table in one request; yields are percentage points."""
    response = _fetch_public_market_data("https://www.fundamentus.com.br/fii_resultado.php")
    columns = {"Papel": "ticker", "Segmento": "setor", "Cotação": "price",
               "P/VP": "pvp", "Dividend Yield": "dy", "Liquidez": "liq2m"}
    raw, *_ = pd.read_html(StringIO(response.content.decode("iso-8859-1")),
                           attrs={"id": "tabelaResultado"}, decimal=",", thousands=".",
                           converters=dict.fromkeys(columns, str))
    if raw.empty or not columns.keys() <= set(raw.columns):
        raise ValueError("Fundamentus não retornou a tabela de FIIs esperada.")
    frame = raw[list(columns)].rename(columns=columns)
    frame["ticker"] = frame["ticker"].str.strip().str.upper()
    frame = frame.loc[frame["ticker"].str.fullmatch(r"[A-Z0-9]{4}11", na=False)]
    if frame.empty:
        raise ValueError("Fundamentus não retornou FIIs válidos.")
    frame = frame.drop_duplicates("ticker").set_index("ticker")
    for column in ("price", "pvp", "dy"):
        frame[column] = clean_numeric_column(frame[column]).replace([math.inf, -math.inf], float("nan"))
    frame["liq2m"] = clean_numeric_column(frame["liq2m"].str.replace(".", "", regex=False))
    frame["price"] = frame["price"].where(frame["price"].gt(0))
    frame["pvp"] = frame["pvp"].where(frame["pvp"].gt(0))
    frame.attrs["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return frame


@cache_data(ttl=900, stale=DAY_SECONDS, cache_if=lambda frame: not frame.empty)
def get_us_market_data() -> pd.DataFrame:
    """Load live Yahoo Finance fundamentals for catalog US equities; ratios/margins are fractions."""
    from utils.assets import EXTRA_ASSETS_BY_TICKER, yahoo_equity_screener_row

    entries = [
        (ticker, item)
        for ticker, item in EXTRA_ASSETS_BY_TICKER.items()
        if item["asset_class"] == "acao_us" and item.get("yahoo_symbol")
    ]
    if not entries:
        raise ValueError("Nenhuma ação americana configurada no catálogo.")

    yfd = yf.data.YfData()
    symbols = [item["yahoo_symbol"] for _, item in entries]
    quotes: dict[str, dict] = {}
    for offset in range(0, len(symbols), 50):
        batch = symbols[offset:offset + 50]
        try:
            payload = yfd.get_raw_json(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": ",".join(batch)},
                timeout=8,
            )
            for row in (payload.get("quoteResponse") or {}).get("result") or []:
                sym = row.get("symbol")
                if isinstance(sym, str):
                    quotes[sym] = row
        except Exception:
            logger.warning("Yahoo Finance US batch quote unavailable for offset=%d", offset, exc_info=True)

    def _fetch_summary(symbol: str) -> tuple[str, dict]:
        try:
            payload = yfd.get_raw_json(
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
                params={"modules": "financialData,defaultKeyStatistics", "formatted": "false"},
                timeout=6,
            )
            results = (payload.get("quoteSummary") or {}).get("result") or []
            if not results or not isinstance(results[0], dict):
                return symbol, {}
            merged = {}
            for module in ("defaultKeyStatistics", "financialData"):
                data = results[0].get(module)
                if isinstance(data, dict):
                    merged.update({k: v for k, v in data.items() if v is not None})
            return symbol, merged
        except Exception:
            logger.debug("Yahoo Finance US summary unavailable for %s", symbol, exc_info=True)
            return symbol, {}

    summary_symbols = [sym for sym in symbols if sym in quotes] or symbols
    with ThreadPoolExecutor(max_workers=min(10, max(1, len(summary_symbols)))) as pool:
        summaries = dict(pool.map(_fetch_summary, summary_symbols))

    rows = []
    for ticker, item in entries:
        sym = item["yahoo_symbol"]
        combined = {**quotes.get(sym, {}), **summaries.get(sym, {})}
        row = yahoo_equity_screener_row(ticker, item, combined)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("Yahoo Finance não retornou dados para as ações americanas.")
    frame = pd.DataFrame(rows).set_index("ticker")
    frame.attrs["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return frame


def compute_target_upside(current_price, target_price):
    """Return target-price upside in percent, or None for invalid prices."""
    try:
        current, target = float(current_price), float(target_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current) or not math.isfinite(target) or current <= 0 or target <= 0:
        return None
    return (target / current - 1) * 100


@cache_data(ttl=3600, cache_if=lambda data: "_error" not in data)
def get_market_target_data(ticker_b3):
    """Fetch Yahoo Finance's target-price and recommendation data for a ticker."""
    from utils.assets import asset_metadata, is_synthetic_fixed_income, provider_symbol

    if is_synthetic_fixed_income(ticker_b3):
        meta = asset_metadata(ticker_b3)
        return {
            "company_name": meta.get("company") or ticker_b3,
            "currency": meta.get("currency") or "BRL",
            "price": None,
            "regular_price": None,
            "target_low": None,
            "target_mean": None,
            "target_median": None,
            "target_high": None,
            "analyst_count": None,
            "recommendation": None,
            "recommendation_mean": None,
        }
    symbol = provider_symbol(ticker_b3)
    default_currency = "BRL" if symbol.endswith(".SA") else "USD"
    try:
        info = get_yahoo_info(symbol)
        if not info:
            return {"_error": "Yahoo Finance não retornou dados"}
        return {
            "company_name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency") or default_currency,
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


@cache_data(ttl=3600, stale=DAY_SECONDS)
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
_MIN_SESSION_COVERAGE = 0.5


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
    Rows quoted for under half the symbols (a stray weekend quote) are not sessions,
    and a latest session quoted for under half the tickers (a rate-limited download
    keeps failed symbols as empty columns) raises instead of being cached as a result.
    """
    adjusted = adjusted.dropna(how="all")
    if adjusted.empty:
        raise ValueError("Yahoo Finance não retornou cotações.")
    quoted = adjusted.notna().sum(axis=1)
    adjusted = adjusted[quoted >= _MIN_SESSION_COVERAGE * quoted.max()]
    if adjusted.iloc[-1].notna().sum() < _MIN_SESSION_COVERAGE * len(tickers):
        raise ValueError("Yahoo Finance devolveu cotações para poucas ações.")

    def between(column, back):
        return _change(column.iloc[-1], column.iloc[-1 - back]) if len(column) > back else None

    changes = {}
    for ticker in tickers:
        symbol = f"{ticker}.SA"
        column = adjusted[symbol] if symbol in adjusted else None
        changes[ticker] = {"day": None, "week": None} if column is None else {"day": between(column, 1), "week": between(column, 5)}
    return {"session": adjusted.index[-1].date().isoformat(), "changes": changes}


@cache_data(ttl=600, stale=QUOTE_STALE_SECONDS)
def get_universe_changes(tickers: tuple) -> dict:
    """Fetch one Yahoo download for the whole universe and return its latest-session changes."""
    with YFINANCE_REQUEST_LOCK:
        downloaded = yf.download([f"{ticker}.SA" for ticker in tickers], period="15d", interval="1d", auto_adjust=False,
                                 progress=False, timeout=_YF_UNIVERSE_TIMEOUT_SECONDS)
    if downloaded.empty or "Adj Close" not in downloaded:
        raise ValueError("Yahoo Finance não retornou cotações.")
    result = universe_changes(downloaded["Adj Close"], tickers)
    result["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return result
