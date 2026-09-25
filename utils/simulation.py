"""Session-cached Monte Carlo path generation for Streamlit reruns."""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st


def simulate_portfolio(mu, covariance, weights, days, simulations, initial_value, start_date):
    """Generate paths once per session/input set and reuse them on widget reruns."""
    inputs = (mu, covariance, weights, days, simulations, initial_value, start_date)
    fingerprint = hashlib.sha256(repr(inputs).encode()).hexdigest()
    cached = st.session_state.get("_simulation_result")
    if cached and cached[0] == fingerprint:
        return cached[1]

    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(
        np.asarray(mu), np.asarray(covariance), size=(days, simulations)
    )
    asset_prices = np.exp(simulated_returns.cumsum(axis=0))
    portfolio_values = (asset_prices * np.asarray(weights)).sum(axis=2) * initial_value
    dates = pd.date_range(start=start_date, periods=days + 1, freq="B")
    result = pd.DataFrame(
        np.vstack([np.ones(simulations) * initial_value, portfolio_values]),
        index=dates,
    )
    st.session_state["_simulation_result"] = (fingerprint, result)
    return result
