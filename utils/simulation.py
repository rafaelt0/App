"""Session-cached Monte Carlo path generation for Streamlit reruns."""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st


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
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
        or (weights < 0).any() or weights.sum() <= 0
    ):
        raise ValueError("Portfolio weights must be finite, nonnegative, and have positive total.")
    weights = weights / weights.sum()
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
