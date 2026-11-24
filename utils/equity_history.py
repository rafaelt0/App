"""Dividend and annual statement history for listed equities, read from Yahoo Finance.

Every field is optional: a source that does not publish dividends, statements or a
fiscal-year end returns null for that block instead of a zero or an estimate.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import pandas as pd
import yfinance as yf

from utils.cache import cache_data
from utils.market_data import get_yahoo_info

logger = logging.getLogger(__name__)

HISTORY_TTL = 6 * 3600
MAX_PERIODS = 5
MAX_DIVIDEND_YEARS = 5

_INCOME_LABELS = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "operating_income": ("Operating Income", "EBIT"),
}
_CASH_LABELS = {
    "operating_cash_flow": ("Operating Cash Flow", "Total Cash From Operating Activities"),
    "free_cash_flow": ("Free Cash Flow",),
}
_BALANCE_LABELS = {
    "total_assets": ("Total Assets",),
    "total_equity": ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _epoch_date(value: Any) -> str | None:
    epoch = _finite(value)
    if epoch is None or epoch <= 0:
        return None
    try:
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _ticker(symbol: str):
    """Yahoo handle; tests replace this single seam."""
    return yf.Ticker(symbol)


def _safe_frame(handle, name: str) -> pd.DataFrame | None:
    try:
        frame = getattr(handle, name, None)
    except Exception:
        logger.warning("Yahoo Finance %s unavailable for history", name, exc_info=True)
        return None
    return frame if isinstance(frame, pd.DataFrame) and not frame.empty else None


def _period(column: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(column)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def _periods(frame: pd.DataFrame | None) -> dict[pd.Timestamp, Any]:
    """Map each normalized period end to the frame's original column."""
    if frame is None:
        return {}
    result = {}
    for column in frame.columns:
        stamp = _period(column)
        if stamp is not None:
            result[stamp.normalize()] = column
    return result


def _cell(frame: pd.DataFrame | None, column: Any, labels: tuple[str, ...]) -> float | None:
    if frame is None or column is None:
        return None
    for label in labels:
        if label in frame.index:
            value = _finite(frame.at[label, column])
            if value is not None:
                return value
    return None


def _dividends(handle) -> dict[str, Any] | None:
    try:
        series = getattr(handle, "dividends", None)
    except Exception:
        logger.warning("Yahoo Finance dividends unavailable for history", exc_info=True)
        return None
    if not isinstance(series, pd.Series) or series.empty:
        return None
    payments = []
    for stamp, value in series.items():
        amount = _finite(value)
        date = _period(stamp)
        if amount is None or date is None:
            continue
        payments.append((date.date(), amount))
    if not payments:
        return None
    today = dt.date.today()
    window_start = today - dt.timedelta(days=365)
    recent = [(date, amount) for date, amount in payments if window_start <= date <= today]
    by_year: dict[int, float] = {}
    for date, amount in payments:
        by_year[date.year] = by_year.get(date.year, 0.0) + amount
    years = sorted(by_year)[-MAX_DIVIDEND_YEARS:]
    return {
        "total_12m": round(sum(amount for _, amount in recent), 6) if recent else 0.0,
        "last_payment": max(date for date, _ in payments).isoformat(),
        "payments_12m": len(recent),
        "annual": [{"year": year, "total": round(by_year[year], 6)} for year in years],
    }


def _statements(handle) -> list[dict[str, Any]]:
    income = _safe_frame(handle, "income_stmt")
    cash = _safe_frame(handle, "cashflow")
    balance = _safe_frame(handle, "balance_sheet")
    income_periods = _periods(income)
    cash_periods = _periods(cash)
    balance_periods = _periods(balance)
    periods = sorted(set(income_periods) | set(cash_periods) | set(balance_periods), reverse=True)[:MAX_PERIODS]
    rows = []
    for period in periods:
        row = {
            "period_end": period.date().isoformat(),
            "revenue": _cell(income, income_periods.get(period), _INCOME_LABELS["revenue"]),
            "net_income": _cell(income, income_periods.get(period), _INCOME_LABELS["net_income"]),
            "operating_income": _cell(income, income_periods.get(period), _INCOME_LABELS["operating_income"]),
            "operating_cash_flow": _cell(cash, cash_periods.get(period), _CASH_LABELS["operating_cash_flow"]),
            "free_cash_flow": _cell(cash, cash_periods.get(period), _CASH_LABELS["free_cash_flow"]),
            "total_assets": _cell(balance, balance_periods.get(period), _BALANCE_LABELS["total_assets"]),
            "total_equity": _cell(balance, balance_periods.get(period), _BALANCE_LABELS["total_equity"]),
        }
        if any(value is not None for key, value in row.items() if key != "period_end"):
            rows.append(row)
    return rows


def _empty(ticker: str, note: str | None = None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "available": False,
        "currency": None,
        "dividends": None,
        "statements": [],
        "fiscal_year_end": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "source": "Yahoo Finance",
        "note": note,
    }


@cache_data(ttl=HISTORY_TTL, stale=86400, max_entries=512, cache_if=lambda block: block.get("available"))
def get_equity_history(ticker: str, symbol: str | None = None) -> dict[str, Any]:
    """Dividends and annual statements for one listed equity; never raises."""
    canonical = str(ticker or "").strip().upper()
    yahoo_symbol = str(symbol or canonical).strip()
    if not canonical or not yahoo_symbol:
        return _empty(canonical, "Ticker inválido para histórico.")
    try:
        handle = _ticker(yahoo_symbol)
        info = get_yahoo_info(yahoo_symbol) or {}
        dividends = _dividends(handle)
        statements = _statements(handle)
    except Exception:
        logger.warning("Yahoo Finance history unavailable for %s", yahoo_symbol, exc_info=True)
        return _empty(canonical, "Yahoo Finance não retornou o histórico deste ativo.")
    currency = str(info.get("currency") or "").strip().upper() or None
    if currency is None:
        try:
            from utils.assets import asset_currency

            currency = asset_currency(yahoo_symbol)
        except Exception:
            currency = None
    growth = _finite(info.get("revenueGrowth"))
    earnings_growth = _finite(info.get("earningsGrowth"))
    available = bool(dividends or statements)
    return {
        "ticker": canonical,
        "available": available,
        "currency": currency,
        "dividends": dividends,
        "statements": statements,
        "fiscal_year_end": _epoch_date(info.get("lastFiscalYearEnd")),
        "revenue_growth": growth * 100.0 if growth is not None else None,
        "earnings_growth": earnings_growth * 100.0 if earnings_growth is not None else None,
        "source": "Yahoo Finance",
        "note": None if available else "Yahoo Finance não retornou proventos nem demonstrações para este ativo.",
    }
