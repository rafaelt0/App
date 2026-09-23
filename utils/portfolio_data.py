"""Cached data-fetching helpers for pages/1_Portfolio.py."""

import datetime

import streamlit as st
import yfinance as yf
from bcb import sgs


@st.cache_data(ttl=3600, show_spinner=False)
def get_selic_rate():
    taxa_selic = sgs.get(432, last=1)
    val = (taxa_selic.iloc[-1, 0]) / 100
    daily_val = (1 + val) ** (1 / 252) - 1
    return daily_val


@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio_prices(tickers_yf, start_date):
    today = datetime.date.today()
    return yf.download(tickers_yf, start=start_date, end=today, progress=False, auto_adjust=True)["Close"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_prices(start_date):
    return yf.download("^BVSP", start=start_date, progress=False, auto_adjust=True)["Close"].squeeze()

def align_benchmark_returns(portfolio_returns, benchmark_prices):
    """Align portfolio and benchmark returns when comparable data exists."""
    if benchmark_prices is None or benchmark_prices.empty:
        return portfolio_returns, None

    benchmark_returns = benchmark_prices.pct_change().dropna()
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if common_idx.empty:
        return portfolio_returns, None

    return portfolio_returns.loc[common_idx], benchmark_returns.loc[common_idx]
