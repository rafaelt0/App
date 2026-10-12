"""Owned, portfolio-bound Monte Carlo analysis and its two CSV exports."""
import datetime as dt
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, csv_response, figure
from backend.state import artifact, begin, publish, require_portfolio, session
from utils.charts import (
    CHART_MUTED, CHART_PRIMARY, CHART_PRIMARY_FILL, CHART_PRIMARY_FILL_STRONG,
    CHART_PRIMARY_LIGHT, CHART_PRIMARY_TRACE, CHART_TEXT,
    apply_plotly_theme,
)
from utils.portfolio_data import daily_returns, get_portfolio_prices
from utils.simulation import annualized_log_return_stats, bootstrap_terminal_values, simulate_portfolio

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


class SimulationRequest(StrictModel):
    revision: int = Field(ge=0, strict=True)
    portfolio_id: str = Field(min_length=1)
    calibration: Literal["portfolio", 2, 3, 5] = "portfolio"
    simulations: int = Field(default=2000, ge=10, le=3000, strict=True)
    initial_capital: float = Field(default=10000, ge=100)
    years: int = Field(default=1, ge=1, le=10, strict=True)


@dataclass
class SimulationArtifact:
    id: str
    revision: int
    portfolio_id: str
    result: dict
    summary: pd.DataFrame
    percentiles: pd.DataFrame
    created: dt.date


def _invalid(message):
    raise HTTPException(status_code=422, detail=message)


def calibration_sample(portfolio, calibration, today):
    """Align strictly positive canonical weights; exact zeros do not shorten history."""
    tickers = list(portfolio.tickers)
    try:
        raw = np.asarray([portfolio.weights[ticker] for ticker in tickers], dtype=float)
    except (KeyError, TypeError, ValueError):
        _invalid("Os pesos da carteira analisada estão indisponíveis.")
    total = raw.sum()
    if not len(raw) or not np.isfinite(raw).all() or (raw < 0).any() or not np.isfinite(total) or total <= 0:
        _invalid("Os pesos analisados devem ser finitos, não negativos e somar um valor positivo.")
    normalized = raw / total
    active = raw > 0
    columns = [f"{ticker}.SA" for ticker, included in zip(tickers, active) if included]
    weights = normalized[active]
    returns = portfolio.returns
    if calibration != "portfolio":
        try:
            prices = get_portfolio_prices(tuple(columns), today - dt.timedelta(days=365 * calibration))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Não foi possível carregar o histórico de calibração: {exc}") from exc
        if prices is None or prices.empty:
            raise HTTPException(status_code=502, detail="A fonte não retornou preços para a janela de calibração escolhida.")
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.copy()
            prices.columns = ["_".join(map(str, column)).strip() for column in prices.columns]
        # Only positive-weight columns shape the sample, so a zero-weight ticker's gaps cannot shorten it.
        returns = daily_returns(prices.loc[:, [column for column in columns if column in prices.columns]])
    missing = [column.removesuffix(".SA") for column in columns if column not in returns.columns]
    if missing:
        _invalid(f"Não há retornos disponíveis para todos os ativos com peso positivo: {', '.join(missing)}.")
    aligned = returns.loc[:, columns].dropna()
    if len(aligned) < 30:
        _invalid(f"Histórico insuficiente para calibrar a simulação: apenas {len(aligned)} retornos completos nos ativos com peso positivo.")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        _invalid("Retornos históricos inválidos: verifique preços ausentes ou retornos de -100% ou menos.")
    return aligned, weights, [{"ticker": ticker, "weight": float(weight)} for ticker, weight in zip(tickers, normalized)]


def _period(frame):
    if isinstance(frame.index, pd.DatetimeIndex):
        return {"start": frame.index.min().date().isoformat(), "end": frame.index.max().date().isoformat()}
    return {"start": None, "end": None}


def _themed(fig, title, x_title="Data", y_title="Valor do Portfólio (R$)"):
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    apply_plotly_theme(fig)
    return fig


def compute_simulation(portfolio, params: SimulationRequest, today=None):
    today = today or dt.date.today()
    aligned, weights, allocation = calibration_sample(portfolio, params.calibration, today)
    logs = np.log1p(aligned)
    mu, covariance = logs.mean().to_numpy(), logs.cov().to_numpy()
    if not np.isfinite(mu).all() or not np.isfinite(covariance).all():
        _invalid("Não foi possível estimar retornos e covariância finitos com este histórico.")
    annual_mean, standard_error = annualized_log_return_stats(aligned.to_numpy(), weights)
    calibration = {"window": params.calibration, "count": len(aligned), **_period(aligned),
                   "annual_log_mean": annual_mean, "iid_standard_error": standard_error, "sensitivity": []}
    if isinstance(aligned.index, pd.DatetimeIndex) and (
        params.calibration == 5 or (aligned.index.max() - aligned.index.min()).days >= 365 * 4
    ):
        endpoint = aligned.index.max()
        for years in (2, 3, 5):
            window = aligned.loc[aligned.index >= endpoint - pd.DateOffset(years=years)]
            if len(window) >= 2:
                mean, error = annualized_log_return_stats(window.to_numpy(), weights)
                calibration["sensitivity"].append({"years": years, "count": len(window), **_period(window),
                                                    "annual_log_mean": mean, "iid_standard_error": error})
    days = params.years * 252
    warnings = []
    if len(aligned) < 252 or days > len(aligned):
        warnings.append("A amostra histórica é curta para este horizonte. Médias e riscos estimados podem mudar muito com outra janela; estas trajetórias não são previsões.")
    if params.simulations < 1000:
        tail_count = max(1, int(np.ceil(params.simulations * 0.05)))
        warnings.append(f"Com {params.simulations} trajetórias, apenas cerca de {tail_count} compõem os 5% inferiores; prefira 1.000 ou mais para comparar a cauda.")
    paths = simulate_portfolio(tuple(mu), tuple(map(tuple, covariance)), tuple(weights), days,
                               params.simulations, float(params.initial_capital), today.isoformat(), chunk_size=64)
    bootstrap = bootstrap_terminal_values(aligned.to_numpy(), weights, days, params.simulations,
                                         float(params.initial_capital), chunk_size=64)
    if not np.isfinite(paths.to_numpy()).all() or not np.isfinite(bootstrap).all():
        _invalid("Os parâmetros e o histórico produziram valores não finitos; não é possível apresentar estes cenários.")
    final = paths.iloc[-1]
    p5, p75, p95 = np.percentile(final, [5, 75, 95])
    capital = params.initial_capital
    stats = {
        "mean_terminal": float(final.mean()), "gain_probability": float((final > capital).mean()),
        "mean_implied_cagr": float((final.mean() / capital) ** (1 / params.years) - 1),
        "p75_implied_cagr": float((p75 / capital) ** (1 / params.years) - 1),
        "p5_terminal": float(p5), "p5_return": float(p5 / capital - 1),
        "lower_tail_mean": float(final[final <= p5].mean()), "minimum": float(final.min()), "maximum": float(final.max()),
    }
    comparison = [
        {"model": "Normal multivariado", "p5": float(p5), "p95": float(p95), "gain_probability": stats["gain_probability"]},
        {"model": "Reamostragem histórica", "p5": float(np.percentile(bootstrap, 5)), "p95": float(np.percentile(bootstrap, 95)),
         "gain_probability": float((bootstrap > capital).mean())},
    ]
    distribution = {"minimum": stats["minimum"], "q1": float(final.quantile(.25)), "median": float(final.quantile(.5)),
                    "q3": float(final.quantile(.75)), "maximum": stats["maximum"], "mean": stats["mean_terminal"],
                    "standard_deviation": float(final.std(ddof=1))}
    fan = paths.quantile(q=np.array([5, 25, 50, 75, 95]) / 100, axis=1).T
    fan.columns = ["P5", "P25", "P50", "P75", "P95"]
    individual = go.Figure()
    for i in range(min(50, params.simulations)):
        individual.add_trace(go.Scatter(x=paths.index, y=paths.iloc[:, i], mode="lines", name=f"Simulação {i + 1}",
                                        line={"width": 1, "color": CHART_PRIMARY_TRACE}))
    _themed(individual, "Exemplos de trajetórias simuladas (modelo normal)")
    individual.update_layout(showlegend=False, margin={"b": 55})
    fan_figure = go.Figure()
    for name, fill, fillcolor, label in (
        ("P95", None, None, None), ("P5", "tonexty", CHART_PRIMARY_FILL, "Faixa 5%–95%"),
        ("P75", None, None, None), ("P25", "tonexty", CHART_PRIMARY_FILL_STRONG, "Faixa 25%–75%"),
        ("P50", None, None, "Mediana"),
    ):
        fan_figure.add_trace(go.Scatter(x=fan.index, y=fan[name], fill=fill, fillcolor=fillcolor,
            name=label or name, showlegend=label is not None,
            line={"color": CHART_PRIMARY_LIGHT if name == "P50" else CHART_PRIMARY_FILL_STRONG, "width": 2 if name == "P50" else .75}))
    _themed(fan_figure, "Faixas de percentis (modelo normal)")
    fan_figure.add_hline(y=capital, line_dash="dash", line_color=CHART_MUTED, line_width=1.5,
                       annotation_text=f"Capital inicial: R$ {capital:,.0f}".replace(",", "."), annotation_position="bottom right")
    histogram = go.Figure(go.Histogram(x=final, nbinsx=30, marker_color=CHART_PRIMARY, opacity=.7))
    histogram.update_layout(bargap=.05)
    for key, text, color, position, shift in (
        ("q1", "Q1 (25%)", CHART_MUTED, "top left", 0),
        ("median", "Mediana (50%)", CHART_TEXT, "top", 18),
        ("q3", "Q3 (75%)", CHART_MUTED, "top right", 36),
    ):
        histogram.add_vline(x=distribution[key], line_width=2.5 if key == "median" else 2,
            line_dash="solid" if key == "median" else "dash", line_color=color,
            annotation_text=text, annotation_position=position, annotation_yshift=shift)
    _themed(histogram, "Distribuição dos valores finais (modelo normal)", "Valor final do portfólio (R$)", "Frequência")
    histogram.update_layout(margin={"t": 120}, title={"y": .97, "yanchor": "top"})
    summary_values = [
        ("Valor Esperado Final", f"R$ {stats['mean_terminal']:,.2f}"),
        ("Probabilidade de Ganho", f"{stats['gain_probability'] * 100:.1f}%"),
        ("CAGR do valor final médio", f"{stats['mean_implied_cagr'] * 100:.1f}% a.a."),
        ("CAGR do valor final P75", f"{stats['p75_implied_cagr'] * 100:.1f}% a.a."),
        ("Valor final P5", f"R$ {p5:,.2f}"), ("Retorno final P5", f"{stats['p5_return'] * 100:.1f}%"),
        ("Média dos 5% menores valores", f"R$ {stats['lower_tail_mean']:,.2f}"),
        ("Menor valor simulado", f"R$ {stats['minimum']:,.2f}"), ("Maior valor simulado", f"R$ {stats['maximum']:,.2f}"),
    ]
    for row in comparison:
        for label, value in (("Valor final P5", f"R$ {row['p5']:,.2f}"), ("Valor final P95", f"R$ {row['p95']:,.2f}"),
                             ("Ganho sobre o capital inicial", f"{row['gain_probability'] * 100:.1f}%")):
            summary_values.append((f"{row['model']} — {label}", value))
    result = {
        "id": uuid4().hex, "portfolio_id": portfolio.id, "revision": params.revision,
        "inputs": params.model_dump(), "mode": portfolio.mode, "allocation": allocation, "calibration": calibration,
        "stats": stats, "comparison": comparison, "distribution": distribution, "warnings": warnings,
        "monte_carlo_error_pp": float(98 / np.sqrt(params.simulations)),
        "spread_fraction": float((p95 - p5) / capital),
        "charts": {"individual": figure(individual), "fan": figure(fan_figure), "histogram": figure(histogram)},
    }
    fan.index = fan.index.strftime("%Y-%m-%d")
    return SimulationArtifact(result["id"], params.revision, portfolio.id, clean_json(result),
                              pd.DataFrame(summary_values, columns=["Métrica", "Valor"]), fan, today)


@router.post("/run")
def run_simulation(params: SimulationRequest, request: Request):
    ctx = session(request)
    with ctx.lock:
        portfolio = require_portfolio(ctx, returns=True)
        if portfolio.id != params.portfolio_id:
            raise HTTPException(status_code=409, detail="A carteira mudou. Carregue a análise atual antes de simular.")
        generation = begin(ctx, "simulation", params.revision)
    calculated = compute_simulation(portfolio, params)
    publish(ctx, "simulation", params.revision, generation, calculated, portfolio_id=portfolio.id)
    return calculated.result


@router.get("/{simulation_id}/summary.csv")
def summary_csv(simulation_id: str, request: Request):
    saved = artifact(session(request), "simulation", simulation_id)
    return csv_response(saved.summary, f"monte_carlo_resumo_{saved.created}.csv", index=False)


@router.get("/{simulation_id}/percentiles.csv")
def percentiles_csv(simulation_id: str, request: Request):
    saved = artifact(session(request), "simulation", simulation_id)
    return csv_response(saved.percentiles, f"monte_carlo_percentis_{saved.created}.csv")
