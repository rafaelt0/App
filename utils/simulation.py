"""Session-cached Monte Carlo path generation for Streamlit reruns."""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st


def _normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
        or (weights < 0).any() or weights.sum() <= 0
    ):
        raise ValueError("Portfolio weights must be finite, nonnegative, and have positive total.")
    return weights / weights.sum()


def annualized_log_return_stats(asset_returns, weights, trading_days=252):
    """Return annualized log mean and IID standard error for a daily-rebalanced portfolio."""
    asset_returns = np.asarray(asset_returns, dtype=float)
    weights = _normalized_weights(weights)
    if (
        asset_returns.ndim != 2 or asset_returns.shape[0] < 2
        or asset_returns.shape[1] != len(weights)
        or not np.isfinite(asset_returns).all()
        or not np.isfinite(trading_days) or trading_days <= 0
    ):
        raise ValueError("Daily returns and annualization factor must be finite and aligned.")

    portfolio_returns = asset_returns @ weights
    if (portfolio_returns <= -1).any():
        raise ValueError("Portfolio daily returns must be greater than -100%.")
    daily_log_returns = np.log1p(portfolio_returns)
    return (
        float(trading_days * daily_log_returns.mean()),
        float(trading_days * daily_log_returns.std(ddof=1) / np.sqrt(len(daily_log_returns))),
    )


def simulate_portfolio(
    mu, covariance, weights, days, simulations, initial_value, start_date, chunk_size=64
):
    """Simulate daily-rebalanced portfolio paths with deterministic local randomness."""
    inputs = (mu, covariance, weights, days, simulations, initial_value, start_date, chunk_size)
    fingerprint = hashlib.sha256(repr(inputs).encode()).hexdigest()
    cached = st.session_state.get("_simulation_result")
    if cached and cached[0] == fingerprint:
        return cached[1]

    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    weights = _normalized_weights(weights)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    rng = np.random.default_rng(42)
    portfolio_values = np.empty((days, simulations), dtype=float)
    for first in range(0, simulations, chunk_size):
        count = min(chunk_size, simulations - first)
        log_returns = rng.multivariate_normal(mu, covariance, size=(days, count))
        daily_returns = np.expm1(log_returns) @ weights
        portfolio_values[:, first:first + count] = initial_value * np.cumprod(
            1 + daily_returns, axis=0
        )

    dates = pd.date_range(start=start_date, periods=days + 1, freq="B")
    result = pd.DataFrame(
        np.vstack([np.full(simulations, initial_value), portfolio_values]), index=dates
    )
    st.session_state["_simulation_result"] = (fingerprint, result)
    return result


def bootstrap_terminal_values(returns, weights, days, simulations, initial_value, chunk_size=64):
    """Resample whole historical days to preserve cross-asset co-movements."""
    returns = np.asarray(returns, dtype=float)
    weights = _normalized_weights(weights)
    if (
        returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] != len(weights)
        or not np.isfinite(returns).all() or (returns <= -1).any()
    ):
        raise ValueError("Historical returns must be complete daily rows above -100%.")
    if days < 1 or simulations < 1 or chunk_size < 1 or not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Days, simulations, chunk size and initial value must be positive.")

    inputs = (returns.shape, tuple(weights), days, simulations, initial_value, chunk_size)
    fingerprint = hashlib.sha256(repr(inputs).encode() + returns.tobytes()).hexdigest()
    cached = st.session_state.get("_bootstrap_terminal_result")
    if cached and cached[0] == fingerprint:
        return cached[1]

    rng = np.random.default_rng(43)
    final_values = np.empty(simulations, dtype=float)
    for first in range(0, simulations, chunk_size):
        count = min(chunk_size, simulations - first)
        # ponytail: iid daily blocks omit volatility clustering; use multi-day blocks if validated.
        indices = rng.integers(len(returns), size=(days, count))
        daily_returns = returns[indices] @ weights
        final_values[first:first + count] = initial_value * np.prod(1 + daily_returns, axis=0)

    st.session_state["_bootstrap_terminal_result"] = (fingerprint, final_values)
    return final_values
