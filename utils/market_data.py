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
from urllib.parse import quote as url_quote, urlsplit
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
YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
# Fundamentus' Liq.2meses is a two-month mean; 61 days covers the sessions.
LIQUIDITY_WINDOW_DAYS = 61
# Chart history is fetched one symbol at a time and only on the fallback path.
LIQUIDITY_WORKERS = 6


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


def _quote_date(stamp, zone_name):
    """Exchange-local calendar date for a Yahoo epoch quote stamp."""
    try:
        zone = ZoneInfo(zone_name or "UTC")
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        zone = datetime.timezone.utc
    try:
        return datetime.datetime.fromtimestamp(stamp, zone).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _average_daily_volume(result) -> float | None:
    """Mean daily volume over the last two months of real chart sessions."""
    timestamps = result.get("timestamp")
    blocks = (result.get("indicators") or {}).get("quote") or []
    volumes = blocks[0].get("volume") if blocks and isinstance(blocks[0], dict) else None
    if not isinstance(timestamps, list) or not isinstance(volumes, list):
        return None
    pairs = []
    for stamp, volume in zip(timestamps, volumes):
        stamp, volume = _positive_number(stamp), _positive_number(volume)
        if stamp is not None and volume is not None:
            pairs.append((stamp, volume))
    if not pairs:
        return None
    latest = max(stamp for stamp, _ in pairs)
    window = [volume for stamp, volume in pairs if stamp >= latest - LIQUIDITY_WINDOW_DAYS * DAY_SECONDS]
    return sum(window) / len(window) if len(window) > 1 else None


def _trailing_dividend_rate(result) -> float | None:
    """Cash dividends per share over the 365 days ending at the chart's last session.

    A missing events block is unknown. An empty dividend map means the window
    paid nothing, which is a real yield of zero rather than a blank.
    """
    events = result.get("events")
    if not isinstance(events, dict) or "dividends" not in events:
        return None
    dividends = events.get("dividends")
    if not isinstance(dividends, dict):
        return None
    latest = None
    for stamp in result.get("timestamp") or []:
        stamp = _positive_number(stamp)
        if stamp is not None and (latest is None or stamp > latest):
            latest = stamp
    if latest is None:
        return None
    cutoff = latest - 365 * DAY_SECONDS
    total = 0.0
    for item in dividends.values():
        if not isinstance(item, dict):
            continue
        stamp = _positive_number(item.get("date"))
        try:
            amount = float(item.get("amount"))
        except (TypeError, ValueError):
            continue
        if stamp is not None and stamp >= cutoff and math.isfinite(amount) and amount > 0:
            total += amount
    return total


def _fetch_public_chart_stats(symbol):
    """Two-month average daily volume and trailing dividends from Yahoo's public chart."""
    empty = {"volume": None, "dividend_rate": None}
    try:
        response = _fetch_public_market_data(
            YAHOO_CHART_URL.format(symbol=url_quote(symbol, safe="")),
            params={"range": "1y", "interval": "1d", "events": "div"},
        )
        results = (response.json().get("chart") or {}).get("result") or []
        result = results[0] if results and isinstance(results[0], dict) else None
        if not isinstance(result, dict) or (result.get("meta") or {}).get("symbol") != symbol:
            return symbol, empty
        return symbol, {"volume": _average_daily_volume(result), "dividend_rate": _trailing_dividend_rate(result)}
    except (requests.RequestException, AttributeError, KeyError, TypeError, ValueError):
        logger.debug("Yahoo Finance chart history unavailable for %s", symbol, exc_info=True)
        return symbol, empty


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
                quote_date = _quote_date(stamp, meta.get("exchangeTimezoneName"))
                if quote_date is None:
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


_DIVIDEND_KEYS = ("trailingAnnualDividendYield", "dividendYield", "trailingAnnualDividendRate")
_NASDAQ_INDICATOR_KEYS = {"trailingPE", "priceToBook", "returnOnEquity", "profitMargins", "operatingMargins", "enterpriseToEbitda"}


def _us_frame_has_fundamentals(frame: pd.DataFrame) -> bool:
    """Cache a US screen only when at least one valuation or return column arrived."""
    if frame is None or frame.empty:
        return False
    return any(column in frame.columns and frame[column].notna().any() for column in ("pl", "roe", "dy"))


def _missing_statement(info: dict) -> bool:
    has_earnings = info.get("trailingPE") is not None or info.get("trailingEps") is not None
    return not has_earnings or info.get("priceToBook") is None or info.get("returnOnEquity") is None


def _missing_dividend(info: dict) -> bool:
    return all(info.get(key) is None for key in _DIVIDEND_KEYS)


@cache_data(ttl=900, stale=DAY_SECONDS, cache_if=_us_frame_has_fundamentals)
def get_us_market_data() -> pd.DataFrame:
    """Load catalog US equities; ratios and margins stay fractions.

    Authenticated Yahoo quotes and summaries are preferred. When they are
    rejected, prices come from the public spark feed, trailing dividends from
    the public chart, and P/L, P/VP, ROE, margins and EV/EBITDA from Nasdaq's
    latest annual statements. A row says which of those filled it.
    """
    from utils.assets import EXTRA_ASSETS_BY_TICKER, nasdaq_equity_indicators, yahoo_equity_screener_row

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
                sym = row.get("symbol") if isinstance(row, dict) else None
                if isinstance(sym, str) and sym in batch:
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

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(symbols)))) as pool:
        summaries = dict(pool.map(_fetch_summary, symbols))

    def usable_price(info: dict) -> float | None:
        return (_positive_number(info.get("regularMarketPrice"))
                or _positive_number(info.get("currentPrice"))
                or _positive_number(info.get("previousClose")))

    missing_price = tuple(sym for sym in symbols if usable_price(quotes.get(sym, {})) is None)
    public_quotes = get_yahoo_quotes(missing_price) if missing_price else {}
    public_quotes = {sym: quote for sym, quote in public_quotes.items() if quote["currency"].upper() == "USD"}
    if public_quotes:
        logger.info("Yahoo Finance US quotes: %d of %d symbols via public spark", len(public_quotes), len(symbols))

    def base_info(sym: str) -> dict:
        return {**quotes.get(sym, {}), **summaries.get(sym, {})}

    priced = [sym for sym in symbols if usable_price(base_info(sym)) is not None or sym in public_quotes]
    chart_symbols = [sym for sym in priced if _missing_dividend(base_info(sym)) or (
        _positive_number(base_info(sym).get("averageDailyVolume3Month")) is None
        and _positive_number(base_info(sym).get("averageVolume")) is None)]
    statement_symbols = [sym for sym in priced if _missing_statement(base_info(sym))]
    chart_stats: dict[str, dict] = {}
    nasdaq_info: dict[str, dict] = {}
    jobs = len(chart_symbols) + len(statement_symbols)
    if jobs:
        with ThreadPoolExecutor(max_workers=min(LIQUIDITY_WORKERS, jobs)) as pool:
            chart_futures = {sym: pool.submit(_fetch_public_chart_stats, sym) for sym in chart_symbols}
            nasdaq_futures = {sym: pool.submit(nasdaq_equity_indicators, sym) for sym in statement_symbols}
            chart_stats = {sym: future.result()[1] for sym, future in chart_futures.items()}
            nasdaq_info = {sym: future.result() for sym, future in nasdaq_futures.items()}

    rows = []
    nasdaq_rows = 0
    for ticker, item in entries:
        sym = item["yahoo_symbol"]
        combined = base_info(sym)
        public_quote = public_quotes.get(sym)
        if public_quote:
            combined.update(regularMarketPrice=public_quote["price"], currency=public_quote["currency"])
            quote_date = public_quote.get("last_quote_date")
            source = "Yahoo Finance · cotação pública"
        else:
            combined["regularMarketPrice"] = usable_price(combined)
            stamp = _positive_number(quotes.get(sym, {}).get("regularMarketTime"))
            quote_date = _quote_date(stamp, quotes.get(sym, {}).get("exchangeTimezoneName")) if stamp is not None else None
            source = "Yahoo Finance · Ações EUA"
        supplemented = False
        for key, value in (nasdaq_info.get(sym) or {}).items():
            if key in _NASDAQ_INDICATOR_KEYS and combined.get(key) is None and value is not None:
                combined[key] = value
                supplemented = True
        stats = chart_stats.get(sym) or {}
        if stats.get("volume") is not None and _positive_number(combined.get("averageDailyVolume3Month")) is None and _positive_number(combined.get("averageVolume")) is None:
            combined["averageVolume"] = stats["volume"]
        if stats.get("dividend_rate") is not None and _missing_dividend(combined):
            combined["trailingAnnualDividendRate"] = stats["dividend_rate"]
        if supplemented:
            source += " + Nasdaq · indicadores anuais"
            nasdaq_rows += 1
        row = yahoo_equity_screener_row(ticker, item, combined)
        if row is not None:
            row.update(source=source, quote_date=quote_date)
            rows.append(row)
    if not rows:
        raise ValueError("Yahoo Finance não retornou dados para as ações americanas.")
    frame = pd.DataFrame(rows).set_index("ticker")
    frame.attrs["fetched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if nasdaq_rows:
        frame.attrs["source"] = "Yahoo Finance · cotação com atraso + Nasdaq · indicadores anuais"
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
