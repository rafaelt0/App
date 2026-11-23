"""Tesouro Direto unit prices for catalog titles with no intermediate payments."""

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


def _download_treasury_csv():
    response = requests.get(TREASURY_CSV, timeout=25)
    response.raise_for_status()
    return BytesIO(response.content)


@cache_data(ttl=3600, stale=86400)
def get_treasury_history():
    """Read the official CSV once and retain only the supported title/maturity pairs."""
    try:
        frame = pd.read_csv(
            _download_treasury_csv(), sep=";", decimal=",", encoding="utf-8-sig",
            usecols=[_TYPE, _MATURITY, _DATE, _MARK, _BUY],
        )
        symbols = {
            (title, maturity.strftime("%d/%m/%Y")): ticker
            for ticker, (title, maturity) in TREASURY_TITLES.items()
        }
        frame["ticker"] = [symbols.get(pair) for pair in zip(frame[_TYPE], frame[_MATURITY])]
        frame = frame.dropna(subset=["ticker"]).copy()
        frame[_DATE] = pd.to_datetime(frame[_DATE], format="%d/%m/%Y", errors="coerce")
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
