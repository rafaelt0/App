"""Tesouro Direto unit prices for catalog titles with no intermediate payments."""

import re
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import requests

from utils.cache import cache_data

# Stable catalog IDs for specific Tesouro Direto titles without intermediate payments.
TREASURY_TITLES = {
    "TD-SELIC-2029": ("Tesouro Selic", date(2029, 3, 1)),
    "TD-SELIC-2031": ("Tesouro Selic", date(2031, 3, 1)),
    "TD-PREFIXADO-2029": ("Tesouro Prefixado", date(2029, 1, 1)),
    "TD-PREFIXADO-2032": ("Tesouro Prefixado", date(2032, 1, 1)),
    "TD-IPCA-2035": ("Tesouro IPCA+", date(2035, 5, 15)),
    "TD-IPCA-2040": ("Tesouro IPCA+", date(2040, 8, 15)),
}
TREASURY_SHARE_STEP = 0.01

TREASURY_CSV = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3"
    "/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)
_TYPE = "Tipo Titulo"
_MATURITY = "Data Vencimento"
_DATE = "Data Base"
_MARK = "PU Base Manha"
_BUY = "PU Compra Manha"
# Published by Tesouro Transparente alongside the buy price. They are optional
# so a trimmed CSV still loads instead of failing the whole history.
_SELL = "PU Venda Manha"
_BUY_RATE = "Taxa Compra Manha"
_SELL_RATE = "Taxa Venda Manha"
_NUMERIC = (_MARK, _BUY, _SELL, _BUY_RATE, _SELL_RATE)


def _download_treasury_csv():
    response = requests.get(TREASURY_CSV, timeout=25)
    response.raise_for_status()
    return BytesIO(response.content)


@cache_data(ttl=3600, stale=86400)
def get_treasury_history():
    """Read the official CSV once and retain only the supported title/maturity pairs."""
    try:
        wanted = {_TYPE, _MATURITY, _DATE, _MARK, _BUY, _SELL, _BUY_RATE, _SELL_RATE}
        frame = pd.read_csv(
            _download_treasury_csv(), sep=";", decimal=",", encoding="utf-8-sig",
            usecols=lambda column: column in wanted,
        )
        for column in (_SELL, _BUY_RATE, _SELL_RATE):
            if column not in frame:
                frame[column] = float("nan")
        symbols = {
            (title, maturity.strftime("%d/%m/%Y")): ticker
            for ticker, (title, maturity) in TREASURY_TITLES.items()
        }
        frame["ticker"] = [symbols.get(pair) for pair in zip(frame[_TYPE], frame[_MATURITY])]
        frame = frame.dropna(subset=["ticker"]).copy()
        frame[_DATE] = pd.to_datetime(frame[_DATE], format="%d/%m/%Y", errors="coerce")
        for column in _NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
        frame = frame.dropna(subset=[_DATE]).drop_duplicates(subset=[_DATE, "ticker"])
        if frame.empty:
            raise ValueError("nenhum título do catálogo no CSV")
        return frame
    except (OSError, ValueError, requests.RequestException) as exc:
        raise ValueError("Tesouro Transparente não retornou um histórico válido") from exc


def treasury_prices(tickers, start_date, adjusted):
    frame = get_treasury_history()
    matching = frame.loc[frame["ticker"].isin(tickers) & frame[_DATE].ge(pd.Timestamp(start_date))]
    price = _MARK if adjusted else _BUY
    closes = matching.pivot(index=_DATE, columns="ticker", values=price).apply(pd.to_numeric, errors="coerce")
    return closes.where(closes.gt(0) & closes.lt(float("inf")))


def latest_treasury_quote(ticker):
    prices = treasury_prices((ticker,), date.today() - timedelta(days=14), adjusted=False)
    if ticker in prices:
        quotes = prices[ticker].dropna()
        if not quotes.empty:
            return float(quotes.iloc[-1]), quotes.index[-1].date().isoformat()
    return None, None


# Friday to Tuesday still covers a weekend plus one holiday. A wider hole is
# not a one-session move.
_SESSION_GAP_DAYS = 4
_FAMILY = {"SELIC": "Tesouro Selic", "IPCA": "Tesouro IPCA+", "PRE": "Tesouro Prefixado"}


def resolve_treasury_ticker(ticker, today=None):
    """Catalog id of a title this module prices, or None when none matches.

    A year in the ticker must match that maturity year. A family name without
    a year uses the soonest outstanding title of that family. The returned id
    is the one `treasury_prices` knows; callers show that maturity with the PU.
    """
    today = today or date.today()
    if ticker in TREASURY_TITLES:
        title, maturity = TREASURY_TITLES[ticker]
        return ticker if maturity >= today else None
    match = re.fullmatch(r"TESOURO-(SELIC|IPCA|PRE)(?:-(\d{4}))?", str(ticker or ""))
    if not match:
        return None
    family = _FAMILY[match.group(1)]
    year = int(match.group(2)) if match.group(2) else None
    open_titles = [
        (maturity, key)
        for key, (title, maturity) in TREASURY_TITLES.items()
        if title == family and maturity >= today and (year is None or maturity.year == year)
    ]
    if year is not None:
        return open_titles[0][1] if len(open_titles) == 1 else None
    return min(open_titles)[1] if open_titles else None


def _positive_price(value):
    number = _scalar(value)
    return number if number is not None and number > 0 else None


def _session_move(series):
    """Latest positive print and the fractional move against the adjacent prior row.

    A missing row between the last two prints is a longer span, so the change
    stays None while the latest price is still returned.
    """
    if series is None or len(series) == 0:
        return None
    positions = [index for index, value in enumerate(series.tolist()) if _positive_price(value) is not None]
    if not positions:
        return None
    last = positions[-1]
    price = float(series.iloc[last])
    as_of = pd.Timestamp(series.index[last]).date()
    change = None
    if len(positions) >= 2 and last - positions[-2] == 1:
        previous = float(series.iloc[positions[-2]])
        gap = (as_of - pd.Timestamp(series.index[positions[-2]]).date()).days
        if 0 < gap <= _SESSION_GAP_DAYS and previous > 0:
            change = price / previous - 1
    return price, change, as_of.isoformat()


def treasury_moves(tickers):
    """Buy PU and one-session change for official titles, keyed by title id.

    The price is PU Compra Manhã. A title the series does not print is absent.
    A title with one print, or a hole before it, has a price and change None.
    """
    wanted = tuple(dict.fromkeys(ticker for ticker in tickers if ticker in TREASURY_TITLES))
    if not wanted:
        return {}
    frame = treasury_prices(wanted, date.today() - timedelta(days=14), adjusted=False)
    moves = {}
    for ticker in wanted:
        if ticker not in getattr(frame, "columns", ()):
            continue
        move = _session_move(frame[ticker])
        if move is None:
            continue
        price, change, as_of = move
        title, maturity = TREASURY_TITLES[ticker]
        moves[ticker] = {
            "price": price,
            "change": change,
            "currency": "BRL",
            "digits": 2,
            "as_of": as_of,
            "unit": "money",
            "change_unit": "fraction" if change is not None else None,
            "source": f"Tesouro Transparente · PU Compra Manhã · {title} {maturity:%d/%m/%Y}",
        }
    return moves


def _scalar(value):
    """Finite float from a CSV cell, or None for the source's missing marker."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and abs(number) != float("inf") else None


def latest_treasury_detail(ticker):
    """Latest official buy/sell prices, buy/sell rates and maturity for one catalog title."""
    frame = get_treasury_history()
    rows = frame.loc[frame["ticker"] == ticker]
    if rows.empty:
        return None
    row = rows.sort_values(_DATE).iloc[-1]
    maturity = pd.to_datetime(str(row[_MATURITY]), format="%d/%m/%Y", errors="coerce")
    return {
        "buy_price": _scalar(row[_BUY]),
        "sell_price": _scalar(row[_SELL]),
        "buy_rate": _scalar(row[_BUY_RATE]),
        "sell_rate": _scalar(row[_SELL_RATE]),
        "rate_unit": "% a.a.",
        "maturity": None if pd.isna(maturity) else maturity.date().isoformat(),
        "quote_date": None if pd.isna(row[_DATE]) else row[_DATE].date().isoformat(),
        "source": "Tesouro Transparente · Taxa e PU Compra/Venda Manhã",
    }
