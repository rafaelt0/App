"""Keep the network out of the test suite: history has its own unit tests."""

import pytest

from backend import home
from utils import assets


def _unavailable(ticker, symbol=None):
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
        "note": "stub",
    }


@pytest.fixture(autouse=True)
def stub_equity_history(monkeypatch):
    monkeypatch.setattr(home, "get_equity_history", _unavailable)
    monkeypatch.setattr(assets, "get_equity_history", _unavailable)
