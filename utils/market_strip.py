"""Latest market quotes and BCB macro indicators for the shell's market strip.

Each item carries its own unit, reference date and source. A source that
fails or returns no valid value is reported as missing, never filled in.
"""

import datetime
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import requests
import yfinance as yf

from utils.cache import cache_data

logger = logging.getLogger(__name__)
_YF_TIMEOUT_SECONDS = 15
_BCB_TIMEOUT_SECONDS = 10
# Long enough to reach the latest monthly IPCA and the lagged PNAD quarter.
_BCB_WINDOW_DAYS = 200

# B3 and Brazilian-market quotes: key, label, Yahoo symbol, unit, currency,
# digits, note. Values are Yahoo's last close, or the latest delayed print for
# the current session. Yahoo has no usable Small Cap or IFIX index history, so
# their listed ETFs are shown under their own tickers.
INDICES = (
    ("ibov", "Ibovespa", "^BVSP", "number", None, 0, None),
    ("ibx50", "IBrX 50", "^IBX50", "number", None, 0, None),
    ("SMAL11", "SMAL11", "SMAL11.SA", "money", "BRL", 2, "ETF small caps"),
    ("XFIX11", "XFIX11", "XFIX11.SA", "money", "BRL", 2, "ETF de FIIs"),
    ("usd", "Dólar", "BRL=X", "money", "BRL", 3, None),
)
STOCKS = ("PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "B3SA3", "ITSA4", "BPAC11",
          "WEGE3", "ABEV3", "PRIO3", "SUZB3", "RENT3", "MGLU3")
QUOTES = INDICES + tuple((ticker, ticker, f"{ticker}.SA", "money", "BRL", 2, None) for ticker in STOCKS)
STOCK_KEYS = set(STOCKS)

# key, label, BCB SGS series, period note. All four series are published in
# percent (percentage points), so the strip shows them with unit "percent".
MACRO = (
    ("selic", "Selic meta", 432, "a.a."),
    ("cdi", "CDI", 4389, "a.a."),
    ("ipca", "IPCA", 13522, "12 meses"),
    ("unemployment", "Desemprego", 24369, "PNAD trimestral"),
)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quote_item(key, label, symbol, unit, currency, digits, close, adjusted, note=None):
    """Latest raw close, with the daily change taken from adjusted closes."""
    close = close.dropna()
    if close.empty or _finite(close.iloc[-1]) is None:
        return None
    adjusted = adjusted.dropna()
    change = None
    if len(adjusted) >= 2 and adjusted.index[-1] == close.index[-1]:
        previous, last = _finite(adjusted.iloc[-2]), _finite(adjusted.iloc[-1])
        if previous and last is not None and previous > 0:
            change = last / previous - 1
    return {
        "key": key, "group": "stock" if key in STOCK_KEYS else "market", "label": label, "symbol": symbol,
        "value": float(close.iloc[-1]), "unit": unit, "currency": currency, "digits": digits,
        "change": change, "note": note, "as_of": close.index[-1].date().isoformat(), "source": "Yahoo Finance",
    }


@cache_data(ttl=300)
def get_market_quotes():
    """Return (items, missing labels) for the configured Yahoo symbols."""
    symbols = [quote[2] for quote in QUOTES]
    downloaded = yf.download(symbols, period="10d", interval="1d", auto_adjust=False,
                             progress=False, timeout=_YF_TIMEOUT_SECONDS)
    if downloaded.empty or "Close" not in downloaded or "Adj Close" not in downloaded:
        raise ValueError("Yahoo Finance não retornou cotações.")
    items, missing = [], []
    for key, label, symbol, unit, currency, digits, note in QUOTES:
        if symbol not in downloaded["Close"]:
            missing.append(label)
            continue
        item = quote_item(key, label, symbol, unit, currency, digits,
                          downloaded["Close"][symbol], downloaded["Adj Close"][symbol], note)
        if item is None:
            missing.append(label)
        else:
            items.append(item)
    return items, missing


def macro_item(key, label, series, note, rows, today=None):
    """Latest published SGS value dated on or before today, in percent."""
    today = today or datetime.date.today()
    points = []
    for row in rows:
        try:
            date = datetime.datetime.strptime(row["data"], "%d/%m/%Y").date()
        except (KeyError, TypeError, ValueError):
            continue
        value = _finite(str(row.get("valor", "")).replace(",", "."))
        if value is not None and date <= today:
            points.append((date, value))
    if not points:
        return None
    date, value = max(points)
    return {
        "key": key, "group": "macro", "label": label, "symbol": None, "value": value, "unit": "percent",
        "currency": None, "digits": 2, "change": None, "note": note, "as_of": date.isoformat(),
        "source": f"BCB SGS {series}",
    }


@cache_data(ttl=3600)
def get_macro_series(series, today):
    """SGS observations up to today. Selic meta (432) is published ahead through
    the next Copom meeting, so a dated window is used instead of 'ultimos'."""
    start = today - datetime.timedelta(days=_BCB_WINDOW_DAYS)
    response = requests.get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados",
                            params={"formato": "json", "dataInicial": f"{start:%d/%m/%Y}", "dataFinal": f"{today:%d/%m/%Y}"},
                            timeout=_BCB_TIMEOUT_SECONDS)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError(f"BCB retornou formato inesperado para a série {series}.")
    return rows


def _macro(entry):
    key, label, series, note = entry
    try:
        today = datetime.date.today()
        return macro_item(key, label, series, note, get_macro_series(series, today), today)
    except Exception:
        logger.warning("BCB series %s unavailable for the market strip", series, exc_info=True)
        return None


def _quotes():
    try:
        return get_market_quotes()
    except Exception:
        logger.warning("Yahoo quotes unavailable for the market strip", exc_info=True)
        return [], [quote[1] for quote in QUOTES]


def market_strip():
    """B3 indices and liquid stocks first, then BCB macro; failed sources listed as missing."""
    with ThreadPoolExecutor(max_workers=len(MACRO) + 1) as pool:
        quotes = pool.submit(_quotes)
        macro = list(pool.map(_macro, MACRO))
        quote_items, missing = quotes.result()
    missing = missing + [entry[1] for entry, item in zip(MACRO, macro) if item is None]
    return {"items": quote_items + [item for item in macro if item is not None], "missing": missing,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
