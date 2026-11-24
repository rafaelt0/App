"""Deterministic Monte Carlo calculations independent of the web framework."""

import numpy as np
import pandas as pd


_DAYS_PER_BLOCK = 256


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

    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    weights = _normalized_weights(weights)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    rng = np.random.default_rng(42)
    # Include the initial row in the allocation instead of copying the full matrix with vstack.
    portfolio_values = np.empty((days + 1, simulations), dtype=float)
    portfolio_values[0] = initial_value
    for first in range(0, simulations, chunk_size):
        count = min(chunk_size, simulations - first)
        daily_returns = np.empty((days, count), dtype=float)
        # Consecutive day blocks keep the same draw order within each group of paths.
        for start in range(0, days, _DAYS_PER_BLOCK):
            block = daily_returns[start:start + _DAYS_PER_BLOCK]
            log_returns = rng.multivariate_normal(mu, covariance, size=block.shape)
            np.expm1(log_returns, out=log_returns)
            np.matmul(log_returns, weights, out=block)
        daily_returns += 1
        np.cumprod(daily_returns, axis=0, out=daily_returns)
        np.multiply(initial_value, daily_returns, out=portfolio_values[1:, first:first + count])

    dates = pd.date_range(start=start_date, periods=days + 1, freq="B")
    result = pd.DataFrame(portfolio_values, index=dates)
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


    rng = np.random.default_rng(43)
    final_values = np.empty(simulations, dtype=float)
    for first in range(0, simulations, chunk_size):
        count = min(chunk_size, simulations - first)
        # ponytail: iid daily blocks omit volatility clustering; use multi-day blocks if validated.
        indices = rng.integers(len(returns), size=(days, count))
        daily_returns = np.empty((days, count), dtype=float)
        for start in range(0, days, _DAYS_PER_BLOCK):
            block = slice(start, start + _DAYS_PER_BLOCK)
            daily_returns[block] = returns[indices[block]] @ weights
        final_values[first:first + count] = initial_value * np.prod(1 + daily_returns, axis=0)

    return final_values
