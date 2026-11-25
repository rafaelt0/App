"""Deterministic Brazil and global macro shock engine.

The engine combines two transparent methods:

* Tesouro Direto titles are repriced with an analytical modified-duration and
  convexity approximation from their remaining years to maturity.
* Market assets (B3 stocks, FIIs, US equities, REITs, ETFs and Global) are
  mapped to historical factor betas estimated with ridge regression. USD
  assets keep a separate direct FX translation so the BRL effect is not hidden
  or counted twice.

The historical factor series are fetched from Yahoo Finance when available.
Every factor that fails is reported in ``missing_factors`` instead of being
filled with invented data.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from utils.assets import asset_currency, normalize_ticker, provider_symbol
from utils.cache import cache_data
from utils.charts import CHART_DANGER, CHART_POSITIVE, apply_plotly_theme
from utils.market_data import YFINANCE_REQUEST_LOCK
from utils.treasury_data import TREASURY_TITLES

logger = logging.getLogger(__name__)

RIDGE_LAMBDA = 1.0
MIN_OBSERVATIONS = 30
PREFIXED_YIELD = 0.135
REAL_YIELD = 0.06
SELIC_MODIFIED_DURATION = 0.05
TD_IMPACT_FLOOR = -0.95
TD_IMPACT_CAP = 3.0
MARKET_IMPACT_FLOOR = -0.95
MARKET_IMPACT_CAP = 3.0
_YF_TIMEOUT_SECONDS = 15

# Stable factor ids used by the API and by the frontend labels.
FACTOR_SYMBOLS: dict[str, str] = {
    "ibov": "^BVSP",
    "usdbrl": "BRL=X",
    "sp500": "^GSPC",
    "us10y": "^TNX",
    "dxy": "DX-Y.NYB",
}
FACTOR_LABELS: dict[str, str] = {
    "ibov": "IBOVESPA",
    "usdbrl": "USD/BRL",
    "sp500": "S&P 500",
    "us10y": "US 10Y",
    "dxy": "DXY",
}
# Market factor id -> shock field used by the API and the attribution table.
FACTOR_TO_SHOCK: dict[str, str] = {
    "usdbrl": "usd_brl_pct",
    "sp500": "sp500_pct",
    "us10y": "us10y_bps",
    "dxy": "dxy_pct",
}

SHOCK_FIELDS: tuple[str, ...] = (
    "selic_bps",
    "ipca_bps",
    "usd_brl_pct",
    "us10y_bps",
    "sp500_pct",
    "dxy_pct",
)
SHOCK_LABELS: dict[str, str] = {
    "selic_bps": "Selic",
    "ipca_bps": "IPCA",
    "usd_brl_pct": "USD/BRL",
    "us10y_bps": "US 10Y",
    "sp500_pct": "S&P 500",
    "dxy_pct": "DXY",
}
SHOCK_LIMITS: dict[str, tuple[float, float]] = {
    "selic_bps": (-600.0, 600.0),
    "ipca_bps": (-500.0, 500.0),
    "usd_brl_pct": (-40.0, 60.0),
    "us10y_bps": (-300.0, 300.0),
    "sp500_pct": (-50.0, 50.0),
    "dxy_pct": (-30.0, 30.0),
}

METHOD_LABELS: dict[str, str] = {
    "analytical_duration": "Duração",
    "factor_regression": "Betas",
    "fx_direct": "Câmbio",
    "insufficient_history": "Sem histórico",
}


@dataclass(frozen=True)
class MacroShockInput:
    """A macro shock vector; finite values are clamped to the documented ranges."""

    selic_bps: float = 0.0
    ipca_bps: float = 0.0
    usd_brl_pct: float = 0.0
    us10y_bps: float = 0.0
    sp500_pct: float = 0.0
    dxy_pct: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Choque inválido para {SHOCK_LABELS[field.name]}.") from exc
            if not math.isfinite(number):
                raise ValueError(f"Choque não finito para {SHOCK_LABELS[field.name]}.")
            low, high = SHOCK_LIMITS[field.name]
            object.__setattr__(self, field.name, min(max(number, low), high))

    def as_dict(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


def _preset(key: str, label: str, **shock) -> dict:
    return {"key": key, "label": label, "shock": MacroShockInput(**shock).as_dict()}


# The presets are part of the API contract: the frontend only presents them.
MACRO_PRESETS: tuple[dict, ...] = (
    _preset("selic_alta_200", "Choque Selic +200 bps", selic_bps=200, usd_brl_pct=5),
    _preset("dolar_dispara", "Dólar +20% (Estresse Cambial)", usd_brl_pct=20, selic_bps=100, ipca_bps=100),
    _preset("us10y_spike", "Treasury 10Y EUA +100 bps", us10y_bps=100, sp500_pct=-10, dxy_pct=6, usd_brl_pct=8),
    _preset("risk_off_global", "Risk-Off Global", sp500_pct=-20, us10y_bps=-50, usd_brl_pct=15, dxy_pct=8),
    _preset("desinflacao_corte", "Desinflação e Corte de Juros", selic_bps=-200, ipca_bps=-100, usd_brl_pct=-8, sp500_pct=8),
)


def presets() -> list[dict]:
    """A JSON-safe copy of the built-in scenarios."""
    return [
        {"key": preset["key"], "label": preset["label"], "shock": dict(preset["shock"])}
        for preset in MACRO_PRESETS
    ]


def _close_series(raw, symbol: str) -> pd.Series | None:
    """Extract one close column from a Yahoo download, accepting Series or flat frames."""
    if raw is None or getattr(raw, "empty", True):
        return None
    frame = raw
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=symbol)
    if isinstance(frame.columns, pd.MultiIndex):
        levels = frame.columns.get_level_values(0)
        if "Close" in levels:
            frame = frame["Close"]
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=symbol)
    if symbol in frame.columns:
        series = frame[symbol]
    elif len(frame.columns) == 1:
        series = frame.iloc[:, 0]
    else:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    return series if not series.empty else None


def _download_factor(symbols: tuple[str, ...], start_date) -> pd.DataFrame:
    with YFINANCE_REQUEST_LOCK:
        return yf.download(
            symbols,
            start=start_date,
            end=dt.date.today(),
            progress=False,
            auto_adjust=False,
            timeout=_YF_TIMEOUT_SECONDS,
            threads=False,
        )


@cache_data(ttl=3600)
def get_macro_factor_series(start_date) -> tuple[pd.DataFrame, list[str]]:
    """Daily factor close levels and the ids that could not be fetched.

    A combined Yahoo request is used first; symbols absent from it are retried
    one by one so one bad factor never discards the others.
    """
    start = pd.Timestamp(start_date).date()
    combined = pd.DataFrame()
    try:
        combined = _download_factor(tuple(FACTOR_SYMBOLS.values()), start)
    except Exception:
        logger.warning("Macro factor download failed; trying each factor separately", exc_info=True)

    frame = pd.DataFrame()
    for key, symbol in FACTOR_SYMBOLS.items():
        series = _close_series(combined, symbol)
        if series is None:
            try:
                series = _close_series(_download_factor((symbol,), start), symbol)
            except Exception:
                logger.warning("Macro factor %s unavailable", symbol, exc_info=True)
                series = None
        if series is not None:
            frame[key] = series
    frame = frame.sort_index()
    missing = [key for key in FACTOR_SYMBOLS if key not in frame.columns]
    return frame, missing


def _factor_moves(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily factor moves: returns for prices and percentage points for US 10Y."""
    if levels is None or levels.empty:
        return pd.DataFrame()
    moves = pd.DataFrame(index=levels.index)
    for key in levels.columns:
        if key not in FACTOR_SYMBOLS:
            continue
        series = pd.to_numeric(levels[key], errors="coerce").astype(float)
        moves[key] = series.diff() if key == "us10y" else series.pct_change(fill_method=None)
    return moves.replace([np.inf, -np.inf], np.nan)


def _local_factor_residual(moves: pd.DataFrame) -> pd.DataFrame:
    """Orthogonalize IBOVESPA against the global factors used by the scenario.

    IBOVESPA is a local market factor with no direct shock in the input vector.
    Removing its global component keeps the S&P 500 beta from being split with
    a collinear local factor, so the global shock is not understated.
    """
    if "ibov" not in moves.columns:
        return moves
    others = [key for key in ("usdbrl", "sp500", "us10y", "dxy") if key in moves.columns]
    if not others:
        return moves
    aligned = moves[["ibov", *others]].dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        return moves
    design = np.column_stack([np.ones(len(aligned)), aligned[others].to_numpy(dtype=float)])
    target = aligned["ibov"].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    result = moves.copy()
    result.loc[aligned.index, "ibov"] = target - design @ coefficients
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _resolve_column(ticker: str, returns: pd.DataFrame) -> str | None:
    for candidate in (ticker, f"{ticker}.SA", provider_symbol(ticker)):
        if candidate in returns.columns:
            return candidate
    return None


def _ridge_betas(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float | None, int]:
    """Ridge coefficients in raw factor units with a centered intercept."""
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    scale = centered_x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = centered_x / scale
    gram = standardized.T @ standardized
    penalty = RIDGE_LAMBDA * np.eye(gram.shape[0])
    target = standardized.T @ (y - y_mean)
    try:
        standardized_betas = np.linalg.solve(gram + penalty, target)
    except np.linalg.LinAlgError:
        standardized_betas = np.linalg.lstsq(gram + penalty, target, rcond=None)[0]
    betas = standardized_betas / scale
    intercept = y_mean - float(x_mean @ betas)
    fitted = intercept + x @ betas
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-18 else None
    return betas, intercept, r_squared, int(len(y))


def _treasury_impact(ticker: str, shock: MacroShockInput, as_of: dt.date) -> dict:
    title, maturity = TREASURY_TITLES[ticker]
    years = max((maturity - as_of).days / 365.25, 0.0)
    if title == "Tesouro Selic":
        modified_duration = SELIC_MODIFIED_DURATION
        convexity = 0.0
        delta_bps = shock.selic_bps
        parts = {"selic_bps": 1.0}
    elif title == "Tesouro Prefixado":
        modified_duration = years / (1 + PREFIXED_YIELD)
        convexity = years * (years + 1) / (1 + PREFIXED_YIELD) ** 2
        delta_bps = shock.selic_bps
        parts = {"selic_bps": 1.0}
    else:
        modified_duration = years / (1 + REAL_YIELD)
        convexity = years * (years + 1) / (1 + REAL_YIELD) ** 2
        delta_bps = shock.selic_bps - shock.ipca_bps
        if abs(delta_bps) > 1e-12:
            parts = {
                "selic_bps": shock.selic_bps / delta_bps,
                "ipca_bps": -shock.ipca_bps / delta_bps,
            }
        else:
            parts = {"selic_bps": 0.0, "ipca_bps": 0.0}
    rate_move = delta_bps / 10000.0
    raw_impact = -modified_duration * rate_move + 0.5 * convexity * rate_move ** 2
    impact = _clamp(raw_impact, TD_IMPACT_FLOOR, TD_IMPACT_CAP)
    return {
        "impact": impact,
        "method": "analytical_duration",
        "r2_or_duration": float(modified_duration),
        "parts": {key: impact * share for key, share in parts.items()},
        "detail": (
            f"Duration modificada {modified_duration:.2f} anos; convexidade {convexity:.2f}; "
            f"choque de {delta_bps:+.0f} bps."
        ),
        "betas": {},
        "factors_used": ["selic_bps", "ipca_bps"] if title not in {"Tesouro Selic", "Tesouro Prefixado"} else ["selic_bps"],
        "observations": None,
    }


def _market_impact(ticker: str, returns: pd.DataFrame, shock: MacroShockInput, moves: pd.DataFrame) -> dict:
    is_usd = asset_currency(ticker) == "USD"
    direct_fx = shock.usd_brl_pct / 100.0 if is_usd else 0.0
    column = _resolve_column(ticker, returns)
    regressors = [key for key in moves.columns if not (is_usd and key == "usdbrl")]
    scenario = {
        "usdbrl": shock.usd_brl_pct / 100.0,
        "sp500": shock.sp500_pct / 100.0,
        "us10y": shock.us10y_bps / 100.0,
        "dxy": shock.dxy_pct / 100.0,
        "ibov": 0.0,
    }

    if column is not None and regressors:
        aligned = pd.concat([returns[column].rename("asset"), moves[regressors]], axis=1).dropna()
        if len(aligned) >= MIN_OBSERVATIONS:
            y = aligned["asset"].to_numpy(dtype=float)
            x = aligned[regressors].to_numpy(dtype=float)
            if np.isfinite(y).all() and np.isfinite(x).all():
                coefficients, _, r_squared, observations = _ridge_betas(y, x)
                betas = {key: float(value) for key, value in zip(regressors, coefficients)}
                parts = {
                    key: float(betas.get(key, 0.0) * scenario[key])
                    for key in ("usdbrl", "sp500", "us10y", "dxy")
                }
                parts["usdbrl"] = parts.get("usdbrl", 0.0) + direct_fx
                raw_impact = sum(parts.values())
                impact = _clamp(raw_impact, MARKET_IMPACT_FLOOR, MARKET_IMPACT_CAP)
                if abs(raw_impact) > 1e-12 and impact != raw_impact:
                    parts = {key: value * impact / raw_impact for key, value in parts.items()}
                method = "fx_direct" if is_usd else "factor_regression"
                detail = (
                    f"Ridge (lambda=1,0) em {observations} pregões; R² {r_squared:.2f}. "
                    if r_squared is not None
                    else f"Ridge (lambda=1,0) em {observations} pregões. "
                )
                detail += (
                    f"Tradução cambial direta soma {shock.usd_brl_pct:+.1f}% em USD/BRL."
                    if is_usd
                    else "Betas aplicados aos fatores de mercado do cenário."
                )
                return {
                    "impact": impact,
                    "method": method,
                    "r2_or_duration": r_squared,
                    "parts": parts,
                    "detail": detail,
                    "betas": betas,
                    "factors_used": regressors,
                    "observations": observations,
                }

    if is_usd:
        impact = _clamp(direct_fx, MARKET_IMPACT_FLOOR, MARKET_IMPACT_CAP)
        return {
            "impact": impact,
            "method": "fx_direct",
            "r2_or_duration": None,
            "parts": {"usdbrl": impact},
            "detail": (
                "Sem betas suficientes para os fatores disponíveis; o impacto usa apenas a "
                f"tradução cambial direta de {shock.usd_brl_pct:+.1f}% em USD/BRL."
            ),
            "betas": {},
            "factors_used": ["usdbrl"],
            "observations": 0,
        }

    return {
        "impact": 0.0,
        "method": "insufficient_history",
        "r2_or_duration": None,
        "parts": {},
        "detail": (
            "Menos de 30 pregões em comum com os fatores de mercado disponíveis; impacto zerado."
        ),
        "betas": {},
        "factors_used": [],
        "observations": 0 if column is None else None,
    }


def _impact_chart(rows: list[dict]) -> go.Figure | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: row["impact_brl"])
    values = [row["impact_brl"] for row in ordered]
    colors = [CHART_POSITIVE if value >= 0 else CHART_DANGER for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[row["ticker"] for row in ordered],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+,.0f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Impacto: R$ %{x:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Impacto do choque por ativo (R$)",
        xaxis_title="Impacto (R$)",
        yaxis_title="",
        height=max(260, 44 * len(ordered) + 100),
        margin=dict(l=8, r=24, t=56, b=48),
        showlegend=False,
    )
    apply_plotly_theme(figure)
    figure.update_layout(showlegend=False)
    return figure


def simulate_macro_shock(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    capital: float = 10000.0,
    shock: MacroShockInput | None = None,
    *,
    as_of: dt.date | None = None,
    factor_series: pd.DataFrame | None = None,
) -> dict:
    """Apply one macro shock and return the portfolio-level and per-asset impact.

    ``returns`` is the portfolio's historical daily return frame in BRL (the
    same frame owned by the portfolio snapshot). ``weights`` uses canonical
    tickers; provider symbols are resolved internally. ``factor_series`` is an
    optional close-level frame used by tests and callers that already hold it.
    """
    shock = shock or MacroShockInput()
    as_of = as_of or dt.date.today()
    try:
        capital = float(capital)
    except (TypeError, ValueError) as exc:
        raise ValueError("Capital inválido para simular o choque.") from exc
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Capital inválido para simular o choque.")

    normalized_weights = {
        normalize_ticker(ticker): float(weight)
        for ticker, weight in (weights or {}).items()
        if weight is not None and float(weight) > 0
    }
    total_weight = sum(normalized_weights.values())
    if not normalized_weights or not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("A carteira não tem pesos positivos para simular.")
    normalized_weights = {
        ticker: weight / total_weight for ticker, weight in normalized_weights.items()
    }
    returns = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame()

    if factor_series is None:
        start = returns.index.min().date() if len(returns.index) and isinstance(returns.index, pd.DatetimeIndex) else as_of - dt.timedelta(days=730)
        fetched = get_macro_factor_series(start)
        if isinstance(fetched, tuple):
            factor_series, missing = fetched
        else:
            factor_series, missing = fetched, [key for key in FACTOR_SYMBOLS if key not in fetched.columns]
    else:
        missing = [key for key in FACTOR_SYMBOLS if key not in factor_series.columns]
    missing = [key for key in FACTOR_SYMBOLS if key in (missing or [])]
    moves = _factor_moves(factor_series)
    moves = _local_factor_residual(moves)

    rows: list[dict] = []
    factor_totals = {key: 0.0 for key in SHOCK_FIELDS}
    warnings: list[str] = []
    missing_betas: list[str] = []
    fx_only: list[str] = []
    market_assets = [ticker for ticker in normalized_weights if ticker not in TREASURY_TITLES]

    for ticker, weight in normalized_weights.items():
        if ticker in TREASURY_TITLES:
            calculated = _treasury_impact(ticker, shock, as_of)
        else:
            calculated = _market_impact(ticker, returns, shock, moves)
        impact = float(calculated["impact"])
        if not math.isfinite(impact):
            impact = 0.0
            calculated["parts"] = {}
        impact_brl = capital * weight * impact
        for key, part in calculated["parts"].items():
            shock_key = FACTOR_TO_SHOCK.get(key, key)
            if shock_key in factor_totals:
                factor_totals[shock_key] += capital * weight * float(part)
        if calculated["method"] == "insufficient_history" and weight > 0:
            missing_betas.append(ticker)
        if calculated["method"] == "fx_direct" and calculated["r2_or_duration"] is None and weight > 0:
            fx_only.append(ticker)
        rows.append(
            {
                "ticker": ticker,
                "weight": float(weight),
                "impact_pct": impact,
                "impact_brl": float(impact_brl),
                "method": calculated["method"],
                "method_label": METHOD_LABELS[calculated["method"]],
                "method_detail": calculated["detail"],
                "r2_or_duration": calculated["r2_or_duration"],
                "betas": calculated["betas"],
                "factors_used": calculated["factors_used"],
                "observations": calculated["observations"],
            }
        )

    rows.sort(key=lambda row: (row["impact_brl"], row["ticker"]))
    portfolio_impact_pct = sum(row["weight"] * row["impact_pct"] for row in rows)
    portfolio_impact_brl = capital * portfolio_impact_pct
    if not math.isfinite(portfolio_impact_pct):
        raise ValueError("O choque produziu um impacto não finito.")

    if missing and market_assets:
        warnings.append(
            "Fatores de mercado indisponíveis: "
            + ", ".join(FACTOR_LABELS[key] for key in missing)
            + ". Os ativos de mercado usam somente os fatores disponíveis."
        )
    if missing_betas:
        warnings.append(
            "Histórico insuficiente para estimar betas de "
            + ", ".join(missing_betas)
            + " (menos de 30 pregões em comum); impacto zerado."
        )
    if fx_only:
        warnings.append(
            "Sem betas suficientes para "
            + ", ".join(fx_only)
            + "; o impacto considera apenas a tradução cambial USD/BRL."
        )
    if market_assets and abs(shock.selic_bps) + abs(shock.ipca_bps) > 0:
        warnings.append(
            "Selic e IPCA afetam a duration dos títulos públicos; ativos de mercado respondem "
            "aos fatores S&P 500, USD/BRL, US 10Y e DXY informados no cenário."
        )
    if not moves.shape[1] and market_assets:
        warnings.append(
            "Nenhum fator de mercado disponível; ativos em BRL entram com impacto zero e ativos "
            "em USD usam apenas a tradução cambial."
        )

    factor_contributions = {
        key: {
            "label": SHOCK_LABELS[key],
            "impact_brl": float(factor_totals[key]),
            "impact_pct": float(factor_totals[key] / capital),
        }
        for key in SHOCK_FIELDS
    }
    worst = min(rows, key=lambda row: (row["impact_pct"], row["ticker"])) if rows else None
    best = max(rows, key=lambda row: (row["impact_pct"], row["ticker"])) if rows else None

    return {
        "shock": shock.as_dict(),
        "capital": capital,
        "portfolio_impact_pct": float(portfolio_impact_pct),
        "portfolio_impact_brl": float(portfolio_impact_brl),
        "post_shock_capital_brl": float(capital + portfolio_impact_brl),
        "worst_asset": worst,
        "best_asset": best,
        "factor_contributions": factor_contributions,
        "asset_impacts": rows,
        "presets": presets(),
        "missing_factors": missing,
        "warnings": warnings,
        "methodology": (
            "Títulos do Tesouro Direto: duration modificada e convexidade analíticas. "
            "Ativos de mercado: betas ridge (lambda=1,0) nos fatores históricos disponíveis. "
            "Ativos em USD: tradução cambial direta somada ao efeito dos betas locais."
        ),
        "chart": _impact_chart(rows),
        "as_of": as_of.isoformat(),
    }
