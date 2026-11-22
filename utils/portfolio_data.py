"""Cached financial sources and portfolio calculation helpers."""

import datetime
import logging
import math
import time

import numpy as np
import pandas as pd
import requests
from utils.assets import is_treasury, is_us_asset
from utils.cache import cache_data
from utils.treasury_data import treasury_prices
from utils.market_data import YFINANCE_REQUEST_LOCK
import yfinance as yf
from bcb import sgs
from pypfopt import expected_returns, risk_models

logger = logging.getLogger(__name__)
_YF_DOWNLOAD_ATTEMPTS = 3
_YF_DOWNLOAD_TIMEOUT_SECONDS = 15
_BCB_CDI_DAILY_SERIES = 12
_BCB_TIMEOUT_SECONDS = 15
# The BCB daily SGS endpoint rejects long date ranges, so CDI is requested in chunks.
_BCB_MAX_WINDOW_DAYS = 1825


def _download_close(tickers_yf, start_date, auto_adjust=True):
    """Fetch close prices with bounded retries for transient Yahoo failures."""
    last_exc = None
    for attempt in range(_YF_DOWNLOAD_ATTEMPTS):
        try:
            with YFINANCE_REQUEST_LOCK:
                downloaded = yf.download(
                    tickers_yf,
                    start=start_date,
                    end=datetime.date.today(),
                    progress=False,
                    auto_adjust=auto_adjust,
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


def finite_or_none(value):
    """Return finite scalar metrics as floats; undefined ratios become None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def evaluate_portfolio_health(
    sharpe_ratio,
    sortino_ratio,
    max_drawdown_pct,
    annual_alpha_pct,
    max_position_pct,
):
    """Return an equal-weighted heuristic score and its data coverage."""
    dimensions = []
    risk_adjusted = [
        1.0 if value > 1.0 else 0.5 if value > 0.5 else 0.0
        for value in (finite_or_none(sharpe_ratio), finite_or_none(sortino_ratio))
        if value is not None
    ]
    if risk_adjusted:
        dimensions.append(sum(risk_adjusted) / len(risk_adjusted))

    drawdown = finite_or_none(max_drawdown_pct)
    if drawdown is not None:
        dimensions.append(1.0 if drawdown > -10 else 0.5 if drawdown > -20 else 0.0)

    alpha = finite_or_none(annual_alpha_pct)
    if alpha is not None:
        dimensions.append(1.0 if alpha > 5 else 0.5 if alpha > 0 else 0.0)

    concentration = finite_or_none(max_position_pct)
    if concentration is not None:
        dimensions.append(
            1.0 if concentration <= 30 else 0.5 if concentration <= 50 else 0.0
        )

    coverage = len(dimensions) * 25
    # ponytail: heuristic 75% coverage gate; validate before decision-grade use.
    if coverage < 75:
        return None, coverage
    return round(sum(dimensions) / len(dimensions) * 100), coverage


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



@cache_data(ttl=3600)
def get_selic_rate():
    taxa_selic = sgs.get(432, last=1)
    val = float(taxa_selic.iloc[-1, 0]) / 100
    if not (val >= 0 and val < float("inf")):
        raise ValueError(f"BCB returned invalid Selic value: {val!r}")
    daily_val = (1 + val) ** (1 / 252) - 1
    return daily_val


def _fetch_bcb_series(url):
    response = requests.get(url, timeout=_BCB_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


@cache_data(ttl=3600)
def get_cdi_daily_rates(start_date):
    """Return daily CDI as fractions per day, indexed by date, from before start_date."""
    # Margin so forward-filling can supply a rate for the first trading day.
    first = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    last = pd.Timestamp(datetime.date.today())
    rows = []
    window_start = first
    while window_start <= last:
        window_end = min(window_start + pd.Timedelta(days=_BCB_MAX_WINDOW_DAYS - 1), last)
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{_BCB_CDI_DAILY_SERIES}/dados"
            f"?formato=json&dataInicial={window_start:%d/%m/%Y}&dataFinal={window_end:%d/%m/%Y}"
        )
        rows.extend(_fetch_bcb_series(url))
        window_start = window_end + pd.Timedelta(days=1)
    if not rows:
        raise ValueError("BCB retornou série CDI vazia.")
    dates = pd.to_datetime([row["data"] for row in rows], format="%d/%m/%Y")
    # SGS 12 is published in percent per day.
    values = np.array([float(row["valor"]) for row in rows], dtype=float) / 100
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("BCB retornou valores de CDI inválidos.")
    rates = pd.Series(values, index=dates).sort_index()
    return rates[~rates.index.duplicated(keep="last")]


def align_daily_rates(rates, index):
    """Forward-fill daily rates onto portfolio dates; fail if any date stays uncovered."""
    aligned = rates.reindex(rates.index.union(index)).sort_index().ffill().reindex(index)
    if aligned.isna().any():
        raise ValueError("A série de CDI não cobre todas as datas da carteira.")
    return aligned


def drawdown_from_returns(returns):
    """Drawdown per date for a wealth curve that starts at 1.0 before the first return."""
    wealth = (1 + returns).cumprod()
    peaks = wealth.cummax().clip(lower=1.0)
    return wealth / peaks - 1


def historical_var_cvar(returns, confidence=0.95):
    """Empirical VaR and expected shortfall of daily returns, as fractions."""
    var = float(returns.quantile(1 - confidence))
    tail = returns[returns <= var]
    return var, float(tail.mean())


_USD_BRL = "BRL=X"
_FX_MAX_AGE = datetime.timedelta(days=4)


def _portfolio_closes(symbols, start_date, adjusted):
    treasury = tuple(symbol for symbol in symbols if is_treasury(symbol))
    listed = tuple(symbol for symbol in symbols if not is_treasury(symbol))
    american = [symbol for symbol in listed if is_us_asset(symbol)]
    if not treasury and not american:
        close = _download_close(listed, start_date, auto_adjust=adjusted)
        return close.to_frame(name=listed[0]) if not adjusted and isinstance(close, pd.Series) else close

    treasury_close = pd.DataFrame()
    if treasury:
        try:
            treasury_close = treasury_prices(treasury, start_date, adjusted)
        except Exception:
            if adjusted or not listed:
                raise
            logger.warning("Tesouro Transparente purchase prices unavailable", exc_info=True)
    if not listed:
        return treasury_close

    yahoo_symbols = (*listed, _USD_BRL) if american else listed
    download_start = pd.Timestamp(start_date).date() - _FX_MAX_AGE if american and adjusted else start_date
    try:
        close = _download_close(yahoo_symbols, download_start, auto_adjust=adjusted)
    except Exception:
        if adjusted or not treasury:
            raise
        logger.warning("Yahoo Finance purchase prices unavailable", exc_info=True)
        return treasury_close.reindex(columns=symbols)
    if isinstance(close, pd.Series):
        close = close.to_frame(name=listed[0])
    exchange = close.reindex(columns=listed).dropna(how="all")
    prices = pd.concat([exchange, treasury_close], axis=1).sort_index()
    if american:
        fx_source = close.reindex(columns=[_USD_BRL])[_USD_BRL]
        fx_source = fx_source.loc[fx_source.gt(0) & fx_source.lt(float("inf"))].sort_index()
        usd_prices = prices[american]
        if adjusted:
            # Mark the last USD close in reais on days when only the local market trades.
            usd_prices = usd_prices.ffill().where(usd_prices.bfill().notna())
            # The FX feed can be closed while US equities trade. Value with the last
            # published FX within a long holiday weekend, and surface longer gaps.
            fx = fx_source.reindex(prices.index, method="ffill", tolerance=_FX_MAX_AGE)
            coverage_gaps = usd_prices.loc[fx.isna() & (prices.index >= pd.Timestamp(start_date))].dropna(how="all")
            if not coverage_gaps.empty:
                affected = ", ".join(coverage_gaps.columns[coverage_gaps.notna().any()])
                dates = ", ".join(f"{day:%d/%m/%Y}" for day in coverage_gaps.index[:5])
                remainder = f" (+{len(coverage_gaps) - 5} datas)" if len(coverage_gaps) > 5 else ""
                raise ValueError(f"Câmbio USD/BRL indisponível para {affected} em {dates}{remainder}")
        else:
            fx = fx_source.reindex(prices.index)
        prices[american] = usd_prices.mul(fx, axis=0)
        prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
    return prices.reindex(columns=symbols)


@cache_data(ttl=3600)
def get_portfolio_prices(tickers_yf, start_date):
    return _portfolio_closes(tickers_yf, start_date, adjusted=True)


@cache_data(ttl=900)
def get_portfolio_trade_prices(tickers_yf):
    """Fetch recent purchase references in reais, independently of adjusted returns."""
    if not tickers_yf:
        return pd.DataFrame()
    start_date = datetime.date.today() - datetime.timedelta(days=14)
    return _portfolio_closes(tickers_yf, start_date, adjusted=False)


@cache_data(ttl=3600)
def get_benchmark_prices(start_date):
    return _download_close("^BVSP", start_date).squeeze()


def estimate_markowitz_inputs(returns):
    """Estimate both optimizer inputs from the same gap-free return rows."""
    return (
        expected_returns.mean_historical_return(returns, returns_data=True, frequency=252),
        risk_models.sample_cov(returns, returns_data=True, frequency=252),
    )


def daily_returns(prices):
    """Return simple daily returns without erasing other assets' moves over a missing price.

    Interior gaps are forward-filled per series, so the move across a gap is kept on the
    day the price reappears. Leading and trailing gaps stay missing, and only the rows
    left incomplete by them are dropped.
    """
    filled = prices.ffill().where(prices.bfill().notna())
    gaps = int((prices.isna() & filled.notna()).sum().sum())
    if gaps:
        logger.info("Filled %d interior missing price(s) before computing returns.", gaps)
    return filled.pct_change(fill_method=None).dropna()


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
    active_columns = [
        column for column in portfolio_prices.columns if weights.get(column, 0) > 0
    ]
    if not active_columns:
        return []
    portfolio_prices = portfolio_prices.loc[:, active_columns]

    benchmark_closes = (
        benchmark_prices.dropna()
        if benchmark_prices is not None
        else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    )
    benchmark_returns = daily_returns(benchmark_closes)
    results = []
    for name, (start, end) in crises.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # Start from the last close before the crisis so the start day's return is included,
        # matching the benchmark returns (which are computed from the full price series).
        prior_closes = portfolio_prices.index[portfolio_prices.index < start]
        window_start = prior_closes[-1] if len(prior_closes) else start
        period_prices = portfolio_prices.loc[
            (portfolio_prices.index >= window_start) & (portfolio_prices.index <= end)
        ]
        period_returns = daily_returns(period_prices)
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
        base = period_prices.index[period_prices.index.get_loc(period_returns.index[0]) - 1]
        last = period_returns.index[-1]
        period_benchmark = benchmark_returns.loc[
            (benchmark_returns.index > base) & (benchmark_returns.index <= last)
        ]
        benchmark_covers_window = (
            len(period_benchmark) >= 5
            and benchmark_closes.index[0] <= base
            and benchmark_closes.index[-1] >= last
        )
        results.append(
            {
                "Crise": name,
                "Período": (
                    f"{period_returns.index.min().strftime('%b/%Y')} → "
                    f"{period_returns.index.max().strftime('%b/%Y')}"
                ),
                "Portfólio": portfolio_return,
                "IBOV": (1 + period_benchmark).prod() - 1
                if benchmark_covers_window
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
