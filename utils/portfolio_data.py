"""Cached data-fetching helpers for pages/1_Portfolio.py."""

import datetime
import logging
import math
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from bcb import sgs
from pypfopt import expected_returns, risk_models

logger = logging.getLogger(__name__)
_YF_DOWNLOAD_ATTEMPTS = 3
_YF_DOWNLOAD_TIMEOUT_SECONDS = 15


def _download_close(tickers_yf, start_date):
    """Fetch adjusted closes with bounded retries for transient Yahoo failures."""
    last_exc = None
    for attempt in range(_YF_DOWNLOAD_ATTEMPTS):
        try:
            downloaded = yf.download(
                tickers_yf,
                start=start_date,
                end=datetime.date.today(),
                progress=False,
                auto_adjust=True,
                timeout=_YF_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if "Close" not in downloaded:
                raise ValueError("Yahoo Finance não retornou a coluna Close.")
            close = downloaded["Close"]
            if close.empty:
                raise ValueError("Yahoo Finance retornou histórico vazio.")
            return close
        except Exception as exc:
            last_exc = exc
            if attempt < _YF_DOWNLOAD_ATTEMPTS - 1:
                logger.warning(
                    "Yahoo Finance download attempt %d/%d failed: %s",
                    attempt + 1,
                    _YF_DOWNLOAD_ATTEMPTS,
                    exc,
                )
                time.sleep(1)
    raise last_exc


def bound_efficient_return(
    target_return: float,
    minimum_return: float,
    maximum_return: float,
    epsilon: float = 1e-6,
):
    """Keep an efficient-return target feasible, or return None if degenerate."""
    if not all(math.isfinite(value) for value in (target_return, minimum_return, maximum_return)):
        return None
    if maximum_return - minimum_return <= 2 * epsilon:
        return None
    return max(min(float(target_return), maximum_return - epsilon), minimum_return + epsilon)



@st.cache_data(ttl=3600, show_spinner=False)
def get_selic_rate():
    taxa_selic = sgs.get(432, last=1)
    val = float(taxa_selic.iloc[-1, 0]) / 100
    if not (val >= 0 and val < float("inf")):
        raise ValueError(f"BCB returned invalid Selic value: {val!r}")
    daily_val = (1 + val) ** (1 / 252) - 1
    return daily_val


@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio_prices(tickers_yf, start_date):
    return _download_close(tickers_yf, start_date)


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_prices(start_date):
    return _download_close("^BVSP", start_date).squeeze()

def estimate_markowitz_inputs(returns):
    """Estimate both optimizer inputs from the same gap-free return rows."""
    return (
        expected_returns.mean_historical_return(returns, returns_data=True, frequency=252),
        risk_models.sample_cov(returns, returns_data=True, frequency=252),
    )


def align_weights_to_columns(weights, columns):
    """Return weights in column order, failing when a required ticker is absent."""
    missing = [column for column in columns if column not in weights]
    if missing:
        tickers = ", ".join(map(str, missing))
        raise ValueError(f"Missing portfolio weights for: {tickers}")
    return [weights[column] for column in columns]


def calculate_historical_stress(portfolio_prices, benchmark_prices, weights, crises):
    """Calculate fixed-weight portfolio returns for crises with enough shared data."""
    if portfolio_prices is None or portfolio_prices.empty:
        return []

    benchmark_returns = (
        benchmark_prices.pct_change(fill_method=None).dropna()
        if benchmark_prices is not None and not benchmark_prices.empty
        else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    )
    results = []
    for name, (start, end) in crises.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        period_prices = portfolio_prices.loc[
            (portfolio_prices.index >= start) & (portfolio_prices.index <= end)
        ]
        period_returns = period_prices.pct_change(fill_method=None).dropna()
        if len(period_returns) < 5:
            continue

        aligned_weights = np.asarray(
            align_weights_to_columns(weights, period_returns.columns)
        )
        weight_sum = aligned_weights.sum()
        if weight_sum <= 0:
            continue
        portfolio_return = (
            1 + period_returns.dot(aligned_weights / weight_sum)
        ).prod() - 1
        period_benchmark = benchmark_returns.loc[
            (benchmark_returns.index >= start) & (benchmark_returns.index <= end)
        ]
        results.append(
            {
                "Crise": name,
                "Período": (
                    f"{period_returns.index.min().strftime('%b/%Y')} → "
                    f"{period_returns.index.max().strftime('%b/%Y')}"
                ),
                "Portfólio": portfolio_return,
                "IBOV": (1 + period_benchmark).prod() - 1
                if len(period_benchmark) >= 5
                else None,
            }
        )
    return results


def find_crisis_history_gaps(portfolio_prices, crises, history_start):
    """Flag tickers whose available price history does not span a crisis."""
    if portfolio_prices is None or portfolio_prices.empty:
        return {}

    history_start = pd.Timestamp(history_start)
    history_end = portfolio_prices.index.max()
    gaps = {}
    for name, (start, end) in crises.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if history_start > start or history_end < end:
            continue

        missing = {}
        for ticker in portfolio_prices.columns:
            prices = portfolio_prices[ticker]
            first, last = prices.first_valid_index(), prices.last_valid_index()
            if first is None:
                missing[ticker] = "sem cotações históricas"
            elif first > start:
                missing[ticker] = f"histórico começa em {first:%d/%m/%Y}"
            elif last < end:
                missing[ticker] = f"histórico termina em {last:%d/%m/%Y}"
        if missing:
            gaps[name] = missing
    return gaps


def align_benchmark_returns(portfolio_returns, benchmark_prices):
    """Align portfolio and benchmark returns when comparable data exists."""
    if benchmark_prices is None or benchmark_prices.empty:
        return portfolio_returns, None

    benchmark_returns = benchmark_prices.pct_change(fill_method=None).dropna()
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30:
        return portfolio_returns, None

    return portfolio_returns.loc[common_idx], benchmark_returns.loc[common_idx]
