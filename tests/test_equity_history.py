"""Deterministic checks for the Yahoo Finance dividend and statement history."""

import datetime as dt

import pandas as pd
import pytest

from utils import assets, equity_history


class FakeTicker:
    def __init__(self, *, dividends=None, income=None, cash=None, balance=None):
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)
        self.income_stmt = income if income is not None else pd.DataFrame()
        self.cashflow = cash if cash is not None else pd.DataFrame()
        self.balance_sheet = balance if balance is not None else pd.DataFrame()


@pytest.fixture(autouse=True)
def _clear_history_cache():
    equity_history.get_equity_history.clear()
    yield
    equity_history.get_equity_history.clear()


def _statement(columns, rows):
    return pd.DataFrame({column: list(values) for column, values in zip(columns, zip(*rows))},
                        index=["Total Revenue", "Net Income", "Operating Income"])


def test_history_aggregates_dividends_and_statements(monkeypatch):
    today = dt.date.today()
    recent = pd.Timestamp(today - dt.timedelta(days=30))
    older = pd.Timestamp(today - dt.timedelta(days=200))
    dividends = pd.Series([1.0, 2.0, 0.5, 0.25], index=pd.to_datetime(
        [recent, older, pd.Timestamp("2023-06-01"), pd.Timestamp("2022-06-01")]))
    income = _statement([pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")],
                        [(1000.0, 900.0), (120.0, 100.0), (180.0, 150.0)])
    cash = pd.DataFrame({
        pd.Timestamp("2025-12-31"): [200.0, 80.0],
        pd.Timestamp("2024-12-31"): [180.0, 70.0],
    }, index=["Operating Cash Flow", "Free Cash Flow"])
    balance = pd.DataFrame({
        pd.Timestamp("2025-12-31"): [3000.0, 1500.0],
        pd.Timestamp("2024-12-31"): [2800.0, 1400.0],
    }, index=["Total Assets", "Stockholders Equity"])
    monkeypatch.setattr(equity_history, "_ticker", lambda symbol: FakeTicker(
        dividends=dividends, income=income, cash=cash, balance=balance))
    monkeypatch.setattr(equity_history, "get_yahoo_info", lambda symbol: {
        "currency": "BRL", "lastFiscalYearEnd": 1767139200,
        "revenueGrowth": 0.1, "earningsGrowth": -0.05,
    })

    block = equity_history.get_equity_history("PETR4", "PETR4.SA")

    assert block["available"] is True
    assert block["currency"] == "BRL"
    assert block["dividends"]["total_12m"] == pytest.approx(3.0)
    assert block["dividends"]["payments_12m"] == 2
    assert block["dividends"]["last_payment"] == (today - dt.timedelta(days=30)).isoformat()
    assert {entry["year"] for entry in block["dividends"]["annual"]} == {2022, 2023, today.year}
    assert [row["period_end"] for row in block["statements"]] == ["2025-12-31", "2024-12-31"]
    assert block["statements"][0]["revenue"] == pytest.approx(1000.0)
    assert block["statements"][0]["free_cash_flow"] == pytest.approx(80.0)
    assert block["statements"][0]["total_equity"] == pytest.approx(1500.0)
    assert block["revenue_growth"] == pytest.approx(10.0)
    assert block["earnings_growth"] == pytest.approx(-5.0)
    assert block["fiscal_year_end"] == "2025-12-31"


def test_history_without_dividends_keeps_statements(monkeypatch):
    income = _statement([pd.Timestamp("2025-12-31")], [(500.0, 50.0, 70.0)])
    monkeypatch.setattr(equity_history, "_ticker", lambda symbol: FakeTicker(income=income))
    monkeypatch.setattr(equity_history, "get_yahoo_info", lambda symbol: {"currency": "USD"})

    block = equity_history.get_equity_history("XYZ", "XYZ")

    assert block["available"] is True
    assert block["dividends"] is None
    assert len(block["statements"]) == 1
    assert block["statements"][0]["operating_cash_flow"] is None


def test_history_failure_is_honest(monkeypatch):
    def fail(symbol):
        raise RuntimeError("source down")

    monkeypatch.setattr(equity_history, "_ticker", fail)

    block = equity_history.get_equity_history("PETR4", "PETR4.SA")

    assert block["available"] is False
    assert block["dividends"] is None
    assert block["statements"] == []
    assert "não retornou" in block["note"]


def test_extra_fundamentals_attaches_history(monkeypatch):
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: {
        "currency": "USD", "regularMarketPrice": 100.0, "marketCap": 1_000_000,
        "returnOnEquity": 0.2, "profitMargins": 0.1, "operatingMargins": 0.15,
    })
    monkeypatch.setattr(assets, "get_equity_history", lambda ticker, symbol=None: {
        "ticker": ticker, "available": True, "statements": [], "dividends": None,
    })

    detail = assets.get_extra_asset_fundamentals("AAPL")

    assert detail["history"] == {"ticker": "AAPL", "available": True, "statements": [], "dividends": None}
