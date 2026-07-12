"""Cached data-fetching helpers for pages/1_Portfolio.py."""

import datetime

import streamlit as st
import yfinance as yf
from bcb import sgs


@st.cache_data(ttl=3600, show_spinner=False)
def get_selic_rate(start_date):
    taxa_selic = sgs.get(432, start=start_date)
    val = (taxa_selic.iloc[-1, 0]) / 100
    daily_val = (1 + val) ** (1 / 252) - 1
    return daily_val


@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio_prices(tickers_yf, start_date):
    today = datetime.date.today()
    return yf.download(tickers_yf, start=start_date, end=today, progress=False)["Close"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_prices(start_date):
    return yf.download("^BVSP", start=start_date, progress=False)["Close"].squeeze()
