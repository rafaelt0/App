"""Latest market quotes, BCB macro indicators, Focus expectations and Tesouro
IPCA+ rates for the shell's market strip.

Each item carries its own unit, reference date and source. A source that
fails or returns no valid value is reported as missing, never filled in.
Quote changes are daily fractions ("fraction"); macro changes are
percentage-point differences ("pp") against the base named in change_note.
"""

import datetime
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests
import yfinance as yf

from utils.cache import cache_data

logger = logging.getLogger(__name__)
_YF_TIMEOUT_SECONDS = 15
_BCB_TIMEOUT_SECONDS = 10
_TESOURO_TIMEOUT_SECONDS = 20
_FOCUS_URL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"
_TESOURO_URL = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
                "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv")
# The CSV (~14 MB) is ordered newest base date first, so its head covers several weeks.
_TESOURO_HEAD_BYTES = 131072

# Quotes: key, label, Yahoo symbol, unit, currency, digits, note. B3 plus the
# dollar, S&P 500 and Brent. Values are Yahoo's last close, or the latest
# delayed print for the current session. Yahoo has no usable Small Cap or IFIX index history, so
# their listed ETFs are shown under their own tickers.
INDICES = (
    ("ibov", "Ibovespa", "^BVSP", "number", None, 0, None),
    ("SMAL11", "SMAL11", "SMAL11.SA", "money", "BRL", 2, "ETF small caps"),
    ("XFIX11", "XFIX11", "XFIX11.SA", "money", "BRL", 2, "ETF de FIIs"),
    ("usd", "Dólar", "BRL=X", "money", "BRL", 3, None),
    ("spx", "S&P 500", "^GSPC", "number", None, 0, None),
    ("brent", "Brent", "BZ=F", "money", "USD", 2, "barril"),
)
STOCKS = ("PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "B3SA3", "ITSA4", "BPAC11",
          "WEGE3", "ABEV3", "PRIO3", "SUZB3", "RENT3", "MGLU3")
QUOTES = INDICES + tuple((ticker, ticker, f"{ticker}.SA", "money", "BRL", 2, None) for ticker in STOCKS)
STOCK_KEYS = set(STOCKS)

SELIC_SERIES, IPCA_SERIES = 432, 13522
# Selic meta can hold for well over a year; four years reaches the last Copom change.
_SELIC_WINDOW_DAYS = 1500
_IPCA_WINDOW_DAYS = 200
# key, label, Tesouro IPCA+ maturity. Long real rates for equity valuation.
TESOURO = (("ntnb_2035", "IPCA+ 2035", "15/05/2035"), ("ntnb_2045", "IPCA+ 2045", "15/05/2045"))


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
        "change": change, "change_unit": "fraction", "change_note": "dia", "note": note,
        "as_of": close.index[-1].date().isoformat(), "source": "Yahoo Finance",
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


def _br_date(date):
    return f"{date:%d/%m/%Y}"


def _pp(current, base):
    """Percentage-point difference, rounded past float noise."""
    return round(current - base, 4)


def macro_entry(key, label, value, as_of, source, note, change=None, change_note=None):
    """A strip item for a rate or expectation published in percent."""
    return {
        "key": key, "group": "macro", "label": label, "symbol": None, "value": value, "unit": "percent",
        "currency": None, "digits": 2, "change": change, "change_unit": "pp" if change is not None else None,
        "change_note": change_note if change is not None else None, "note": note, "as_of": as_of.isoformat(),
        "source": source,
    }


def sgs_points(rows, today):
    """(date, value) SGS observations dated on or before today, oldest first."""
    points = []
    for row in rows:
        try:
            date = datetime.datetime.strptime(row["data"], "%d/%m/%Y").date()
        except (KeyError, TypeError, ValueError):
            continue
        value = _finite(str(row.get("valor", "")).replace(",", "."))
        if value is not None and date <= today:
            points.append((date, value))
    return sorted(points)


def selic_item(rows, today):
    """Selic meta in % a.a., with the change set by the last Copom that moved it."""
    points = sgs_points(rows, today)
    if not points:
        return None
    date, value = points[-1]
    change = change_note = None
    for index in range(len(points) - 1, 0, -1):
        if points[index - 1][1] != value:
            change, change_note = _pp(value, points[index - 1][1]), f"desde {_br_date(points[index][0])} (Copom)"
            break
    return macro_entry("selic", "Selic meta", value, date, f"BCB SGS {SELIC_SERIES}", "a.a.", change, change_note)


def ipca_item(rows, today):
    """Trailing 12-month IPCA in %, with the change against the previous month."""
    points = sgs_points(rows, today)
    if not points:
        return None
    date, value = points[-1]
    change = change_note = None
    if len(points) >= 2:
        change, change_note = _pp(value, points[-2][1]), f"ante {points[-2][0]:%m/%Y}"
    return macro_entry("ipca", "IPCA", value, date, f"BCB SGS {IPCA_SERIES}", "12 meses", change, change_note)


@cache_data(ttl=3600)
def get_macro_series(series, today, window_days):
    """SGS observations up to today. Selic meta (432) is published ahead through
    the next Copom meeting, so a dated window is used instead of 'ultimos'."""
    start = today - datetime.timedelta(days=window_days)
    response = requests.get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados",
                            params={"formato": "json", "dataInicial": _br_date(start), "dataFinal": _br_date(today)},
                            timeout=_BCB_TIMEOUT_SECONDS)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError(f"BCB retornou formato inesperado para a série {series}.")
    return rows


def focus_medians(rows):
    """(date, median) Focus observations, newest first."""
    points = []
    for row in rows:
        try:
            date = datetime.date.fromisoformat(row["Data"])
        except (KeyError, TypeError, ValueError):
            continue
        value = _finite(row.get("Mediana"))
        if value is not None:
            points.append((date, value))
    return sorted(points, reverse=True)


def focus_item(key, label, note, rows):
    """Latest Focus median in %, with the change against the survey a week or more earlier."""
    points = focus_medians(rows)
    if not points:
        return None
    date, value = points[0]
    previous = next((point for point in points if point[0] <= date - datetime.timedelta(days=7)), None)
    change = change_note = None
    if previous:
        change, change_note = _pp(value, previous[1]), f"ante Focus de {_br_date(previous[0])}"
    return macro_entry(key, label, value, date, "BCB Focus", note, change, change_note)


def _focus_get(resource, filters, top):
    query = "&".join((f"$filter={quote(filters)}", f"$orderby={quote('Data desc')}", f"$top={top}", "$format=json"))
    response = requests.get(f"{_FOCUS_URL}/{resource}?{query}", timeout=_BCB_TIMEOUT_SECONDS)
    response.raise_for_status()
    rows = response.json().get("value")
    if not isinstance(rows, list):
        raise ValueError(f"Focus retornou formato inesperado para {resource}.")
    return rows


@cache_data(ttl=3600)
def get_focus_annual(indicator, year):
    """Daily Focus statistics for one indicator at the end of a calendar year,
    30-day respondent window (the published Focus report), about four weeks."""
    return _focus_get("ExpectativasMercadoAnuais",
                      f"Indicador eq '{indicator}' and DataReferencia eq '{year}' and baseCalculo eq 0", 20)


@cache_data(ttl=3600)
def get_focus_inflation_12m():
    """Smoothed 12-month-ahead IPCA expectations, the BCB's input for the ex-ante real rate."""
    return _focus_get("ExpectativasMercadoInflacao12Meses", "Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0", 5)


def fisher_real_rate(nominal, inflation):
    """Real rate in % from a nominal rate and an inflation rate, both in %: (1+i)/(1+pi) - 1.
    The same compounding as the Tesouro IPCA+ real rate, so both read alike."""
    return ((1 + nominal / 100) / (1 + inflation / 100) - 1) * 100


def real_rate_item(selic_rows, focus_rows, today):
    """Ex-ante real rate: Selic meta deflated by the Focus 12-month IPCA median."""
    selic = sgs_points(selic_rows, today)
    focus = focus_medians(focus_rows)
    if not selic or not focus:
        return None
    (selic_date, nominal), (focus_date, inflation) = selic[-1], focus[0]
    rate = f"{inflation:.2f}".replace(".", ",")
    source = f"Selic meta (BCB SGS {SELIC_SERIES}) e IPCA Focus 12 meses de {_br_date(focus_date)} ({rate}%)"
    return macro_entry("real_rate", "Juro real ex-ante", round(fisher_real_rate(nominal, inflation), 4),
                       min(selic_date, focus_date), source, "a.a., Fisher")


@cache_data(ttl=3600)
def get_tesouro_rates():
    """Head of the Tesouro Direto offered-rates CSV, newest base date first.
    The server answers a Range request with the whole file, so the body is
    streamed and closed after the head; a cut last line is dropped."""
    with requests.get(_TESOURO_URL, stream=True, timeout=_TESOURO_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        head = b""
        for chunk in response.iter_content(chunk_size=16384):
            head += chunk
            if len(head) >= _TESOURO_HEAD_BYTES:
                break
    lines = head.decode("utf-8", errors="replace").splitlines()
    if not head.endswith(b"\n"):
        lines = lines[:-1]
    if not lines or not lines[0].startswith("Tipo Titulo;"):
        raise ValueError("Tesouro Transparente retornou formato inesperado.")
    return lines


def tesouro_item(key, label, maturity, lines):
    """Morning buy rate (real % a.a.) of one Tesouro IPCA+ bond, with the change against the prior base date."""
    header = lines[0].split(";") if lines else []
    try:
        kind, due, base, buy = (header.index(column) for column in
                                ("Tipo Titulo", "Data Vencimento", "Data Base", "Taxa Compra Manha"))
    except ValueError:
        return None
    points = []
    for line in lines[1:]:
        fields = line.split(";")
        if len(fields) < len(header) or fields[kind] != "Tesouro IPCA+" or fields[due] != maturity:
            continue
        rate = _finite(fields[buy].replace(",", "."))
        try:
            date = datetime.datetime.strptime(fields[base], "%d/%m/%Y").date()
        except ValueError:
            continue
        if rate is not None:
            points.append((date, rate))
    if not points:
        return None
    points.sort(reverse=True)
    date, value = points[0]
    change = change_note = None
    if len(points) >= 2:
        change, change_note = _pp(value, points[1][1]), f"ante {_br_date(points[1][0])}"
    return macro_entry(key, label, value, date, "Tesouro Transparente, taxa de compra", "Tesouro, real a.a.", change, change_note)


def _quotes():
    try:
        return get_market_quotes()
    except Exception:
        logger.warning("Yahoo quotes unavailable for the market strip", exc_info=True)
        return [], [quote[1] for quote in QUOTES]


def _selic(today):
    return [selic_item(get_macro_series(SELIC_SERIES, today, _SELIC_WINDOW_DAYS), today)]


def _ipca(today):
    return [ipca_item(get_macro_series(IPCA_SERIES, today, _IPCA_WINDOW_DAYS), today)]


def _focus_ipca(today):
    return [focus_item("focus_ipca", f"IPCA Focus {today.year}", "mediana", get_focus_annual("IPCA", today.year))]


def _focus_selic(today):
    return [focus_item("focus_selic", f"Selic Focus {today.year}", "mediana", get_focus_annual("Selic", today.year))]


def _real_rate(today):
    return [real_rate_item(get_macro_series(SELIC_SERIES, today, _SELIC_WINDOW_DAYS), get_focus_inflation_12m(), today)]


def _tesouro(today):
    lines = get_tesouro_rates()
    return [tesouro_item(key, label, maturity, lines) for key, label, maturity in TESOURO]


# Builder, labels of the items it returns in order. A builder that raises
# reports all of its labels as missing; a None item reports its own label.
MACRO = (
    (_selic, ("Selic meta",)),
    (_focus_ipca, ("IPCA Focus",)),
    (_focus_selic, ("Selic Focus",)),
    (_real_rate, ("Juro real ex-ante",)),
    (_ipca, ("IPCA",)),
    (_tesouro, tuple(entry[1] for entry in TESOURO)),
)


def _macro(entry, today):
    builder, labels = entry
    try:
        items = builder(today)
    except Exception:
        logger.warning("%s unavailable for the market strip", ", ".join(labels), exc_info=True)
        items = [None] * len(labels)
    return items, [label for label, item in zip(labels, items) if item is None]


def market_strip():
    """Quotes first, then macro; failed sources listed as missing."""
    today = datetime.date.today()
    with ThreadPoolExecutor(max_workers=len(MACRO) + 1) as pool:
        quotes = pool.submit(_quotes)
        macro = list(pool.map(lambda entry: _macro(entry, today), MACRO))
        quote_items, missing = quotes.result()
    items = quote_items + [item for built, _ in macro for item in built if item is not None]
    missing = missing + [label for _, labels in macro for label in labels]
    return {"items": items, "missing": missing, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
