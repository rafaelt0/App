"""Portfolio orchestration; optimizer and historical-return conventions are unchanged."""
from datetime import date, timedelta
import uuid
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, StrictInt
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
import quantstats as qs

from backend.serialization import StrictModel, clean_json, csv_response, figure
from backend.state import PortfolioSnapshot, artifact, begin, publish, session
from utils import db
from utils.market_data import get_listed_stocks
from utils.portfolio_data import (
    align_benchmark_returns, align_daily_rates, align_weights_to_columns, bound_efficient_return,
    calculate_historical_stress, daily_returns, drawdown_from_returns, estimate_markowitz_inputs,
    evaluate_portfolio_health, find_crisis_history_gaps, finite_or_none, get_benchmark_prices,
    get_cdi_daily_rates, get_portfolio_prices, get_portfolio_trade_prices, get_selic_rate,
    historical_var_cvar,
)
from utils.portfolio_charts import plot_efficient_frontier_and_random_portfolios
from utils.charts import (
    CHART_ACCENT, CHART_DANGER, CHART_DANGER_FILL, CHART_GRID, CHART_MUTED,
    CHART_POSITIVE, CHART_PRIMARY, CHART_SECONDARY, apply_plotly_theme,
)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])
Strategy = Literal["min_volatility", "markowitz", "hrp", "manual"]
MODES = {
    "min_volatility": "Otimização de Mínima Volatilidade",
    "markowitz": "Otimização de Markowitz (Média-Variância)",
    "hrp": "Otimização Hierarchical Risk Parity (HRP)",
    "manual": "Alocação Manual",
}
CRISES = {
    "COVID-19 (2020)": ("2020-01-17", "2020-03-23"),
    "Recessão/Impeachment (2015–16)": ("2014-12-31", "2016-12-31"),
    "Joesley Day (2017)": ("2017-05-17", "2017-06-30"),
    "Crise Global (2008–09)": ("2008-08-01", "2009-03-31"),
    "Crise fiscal brasileira (2024)": ("2024-11-26", "2024-12-30"),
}


class AnalyzeRequest(StrictModel):
    tickers: list[str] = Field(min_length=2, max_length=20)
    revision: int = Field(ge=0)
    strategy: Strategy = "min_volatility"
    capital: float = Field(default=10000, ge=100, le=1000000, allow_inf_nan=False)
    lookback_days: int = Field(default=730, ge=60, le=5000)
    manual_shares: dict[str, StrictInt] = Field(default_factory=dict)
    gamma: float = Field(default=1, ge=0, le=3, allow_inf_nan=False)


class RestoreRequest(StrictModel):
    target: Literal["simulation", "news"]
    revision: int = Field(ge=0)


class StressRequest(StrictModel):
    long_history: bool = False


class RefreshRequest(StrictModel):
    pass


def price_frame(prices, tickers):
    if prices is None:
        return pd.DataFrame()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.copy()
        prices.columns = ["_".join(map(str, column)).strip() for column in prices.columns]
    return prices


def history(tickers, start):
    try:
        prices = price_frame(get_portfolio_prices(tuple(tickers), start), tickers)
    except Exception as exc:
        raise HTTPException(422, f"Erro ao buscar cotações no Yahoo Finance: {exc}. Verifique sua conexão e tente novamente.") from exc
    if prices.empty:
        raise HTTPException(422, "Nenhuma cotação retornada para os ativos selecionados. Verifique os tickers e o período escolhido.")
    missing = [ticker.removesuffix(".SA") for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise HTTPException(422, f"Não foi possível obter cotações para: {', '.join(missing)}. Remova esses ativos ou tente novamente mais tarde.")
    prices = prices.loc[:, tickers]
    returns = daily_returns(prices)
    if len(returns) < 30:
        first = prices.apply(lambda column: column.first_valid_index())
        suspects = [f"{ticker.removesuffix('.SA')} (sem cotações)" for ticker in first[first.isna()].index]
        suspects += [f"{ticker.removesuffix('.SA')} (dados desde {value:%d/%m/%Y})" for ticker, value in first.dropna().sort_values(ascending=False).head(5).items()]
        raise HTTPException(422, f"Histórico de cotações em comum insuficiente: apenas {len(returns)} dia(s) com dados completos. IPO recente, deslistagem ou falha na fonte podem reduzir a amostra. Possíveis responsáveis: {'; '.join(suspects)}. Remova ativos com histórico curto ou ajuste o lookback.")
    return prices, returns


def trade_quotes(tickers):
    try:
        prices = price_frame(get_portfolio_trade_prices(tuple(tickers)), tickers)
    except Exception:
        return {}, True
    quotes = {}
    for ticker in tickers:
        if ticker not in prices:
            continue
        values = prices[ticker].dropna()
        if not values.empty:
            value = finite_or_none(values.iloc[-1])
            if value is not None and value > 0:
                quotes[ticker] = (value, values.index[-1])
    return quotes, False


def purchase_suggestion(weights, capital, quotes, failed=False):
    missing = [ticker.removesuffix(".SA") for ticker, weight in weights.items() if weight > 0 and ticker not in quotes]
    unavailable = failed or bool(missing)
    rows, total = [], 0.0
    for ticker, weight in weights.items():
        suggested = weight * capital
        quote = quotes.get(ticker) if weight > 0 else None
        price, stamp, shares, cost = None, None, None, None
        if weight <= 0:
            shares, cost = 0, 0.0
        elif quote is not None:
            price, stamp = quote
            shares = int(np.floor(suggested / price))
            cost = shares * price
            total += cost
        rows.append({"ticker": ticker.removesuffix(".SA"), "price": price, "quote_date": pd.Timestamp(stamp).date().isoformat() if stamp is not None else None, "shares": shares, "effective_value": cost, "suggested_value": suggested, "suggested_weight": weight, "effective_weight": None})
    for row in rows:
        row["effective_weight"] = None if unavailable and row["suggested_weight"] > 0 else row["effective_value"] / total if total > 0 else 0.0
    totals = None if unavailable else {"allocated": total, "cash": capital - total, "efficiency": total / capital}
    return rows, totals, missing


def optimize(strategy, returns, manual_weights, rf, gamma, warnings):
    if strategy == "manual":
        return manual_weights, "Alocação Manual", None, None
    if strategy == "hrp":
        return dict(HRPOpt(returns).optimize()), MODES[strategy], None, None
    mu, covariance = estimate_markowitz_inputs(returns)
    frontier = EfficientFrontier(mu, covariance)
    label = "Mínima Volatilidade"
    if strategy == "min_volatility":
        frontier.min_volatility()
    else:
        try:
            frontier.max_sharpe(risk_free_rate=rf)
            label = "Max Sharpe (Markowitz)"
            if gamma > 0:
                max_return = frontier.portfolio_performance(risk_free_rate=rf)[0]
                floor = EfficientFrontier(mu, covariance)
                floor.min_volatility()
                min_return = floor.portfolio_performance()[0]
                # Level 0 targets the Max Sharpe return; level 3 targets the Min Volatility return.
                level = min(gamma / 3, 1.0)
                target = bound_efficient_return(min_return + (1 - level) * (max_return - min_return), min_return, max_return)
                frontier = EfficientFrontier(mu, covariance)
                if target is None:
                    frontier.min_volatility()
                    label = "Mínima Volatilidade"
                else:
                    frontier.add_objective(objective_functions.L2_reg, gamma=gamma)
                    frontier.efficient_return(target_return=target)
                    label = "Retorno-alvo com L2"
        except Exception as exc:
            warnings.append(f"Otimização de Max Sharpe falhou (motivo: {exc}). Usando carteira de Mínima Volatilidade.")
            frontier = EfficientFrontier(mu, covariance)
            if gamma > 0:
                frontier.add_objective(objective_functions.L2_reg, gamma=gamma)
            frontier.min_volatility()
            label = "Mínima Volatilidade"
    return dict(frontier.clean_weights()), label, mu, covariance


def sample_info(returns):
    return {"count": len(returns), "start": returns.index.min().date().isoformat(), "end": returns.index.max().date().isoformat()}


def historical_risk_free(index, start, daily_rf, warnings):
    """Daily risk-free rates for historical metrics: CDI by date, else a flagged constant Selic."""
    try:
        return align_daily_rates(get_cdi_daily_rates(start), index), "BCB SGS 12 (CDI diário, alinhado por data)"
    except Exception as exc:
        warnings.append(f"Não foi possível buscar o CDI diário no BCB ({exc}). Sharpe, Sortino e alfa históricos usam a Selic atual constante, que não reflete a taxa de cada dia.")
        return pd.Series(daily_rf, index=index), "Selic atual constante (CDI diário indisponível)"


def line_chart(series, title, name, color=CHART_PRIMARY, percent=False, fill=False, reference=None):
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", name=name,
        line=dict(color=color, width=2), fill="tozeroy" if fill else None,
        fillcolor=CHART_DANGER_FILL if fill else None,
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>" if percent else None))
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title=name)
    if percent:
        fig.update_yaxes(tickformat=".0%")
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dash", line_color=CHART_ACCENT if reference else CHART_GRID)
    apply_plotly_theme(fig)
    return fig


def stress_result(prices, benchmark, weights, start, warnings=None):
    warnings = list(warnings or [])
    gaps = find_crisis_history_gaps(prices, CRISES, start)
    for crisis, missing in gaps.items():
        warnings.append(f"{crisis} — histórico incompleto: " + "; ".join(f"{ticker}: {reason}" for ticker, reason in missing.items()))
    raw = calculate_historical_stress(prices, benchmark, weights, CRISES)
    rows = [{"crisis": row["Crise"], "period": row["Período"], "portfolio": row["Portfólio"], "benchmark": row["IBOV"], "difference_pp": (row["Portfólio"] - row["IBOV"]) * 100 if row["IBOV"] is not None else None} for row in raw]
    chart = None
    if len(rows) >= 2:
        names = [row["crisis"].split(" (")[0] for row in rows]
        values = [row["portfolio"] * 100 for row in rows]
        bench_values = [row["benchmark"] * 100 if row["benchmark"] is not None else None for row in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Portfólio", x=names, y=values, marker_color=[CHART_POSITIVE if value >= 0 else CHART_DANGER for value in values], text=[f"{value:+.1f}%" for value in values], textposition="outside"))
        fig.add_trace(go.Bar(name="IBOVESPA", x=names, y=bench_values, marker_color=CHART_MUTED, text=[f"{value:+.1f}%" if value is not None else "" for value in bench_values], textposition="outside"))
        visible = values + [value for value in bench_values if value is not None]
        padding = max((max(visible) - min(visible)) * .12, 1)
        fig.update_layout(title="Portfólio vs IBOVESPA durante Crises Históricas", barmode="group", yaxis_title="Retorno (%)", height=380)
        fig.update_yaxes(range=[min(visible) - padding, max(visible) + padding])
        apply_plotly_theme(fig)
        chart = figure(fig)
    return {"rows": rows, "warnings": warnings, "gaps": gaps, "chart": chart, "history_start": str(start)}


def health_result(metrics, max_weight):
    score, coverage = evaluate_portfolio_health(metrics["sharpe"], metrics["sortino"], metrics["max_drawdown"], metrics["annual_alpha"], max_weight)
    diagnostics = []
    for key, label in (("sharpe", "Sharpe"), ("sortino", "Sortino")):
        value = metrics[key]
        text, status = (f"{label} indisponível — risco não definido", "warning") if value is None else (f"{label} excelente (>1.0)", "positive") if value > 1 else (f"{label} razoável (0.5–1.0)", "warning") if value > .5 else (f"{label} baixo — revise a alocação", "negative")
        diagnostics.append({"text": text, "status": status})
    dd, alpha = metrics["max_drawdown"], metrics["annual_alpha"]
    diagnostics.append({"text": "Drawdown indisponível" if dd is None else "Drawdown controlado (<10%)" if dd > -10 else "Drawdown moderado (10–20%)" if dd > -20 else "Drawdown severo (>20%) — perda de pico a vale elevada", "status": "warning" if dd is None else "positive" if dd > -10 else "warning" if dd > -20 else "negative"})
    diagnostics.append({"text": "Alfa indisponível — variação insuficiente do IBOVESPA" if alpha is None else f"Alfa anual positivo: {alpha:.1f}%" if alpha > 5 else f"Alfa marginal: {alpha:.1f}%" if alpha > 0 else f"Alfa negativo ({alpha:.1f}%) — abaixo do retorno ajustado ao risco do IBOVESPA", "status": "warning" if alpha is None else "positive" if alpha > 5 else "warning" if alpha > 0 else "negative"})
    diagnostics.append({"text": f"Maior posição: {max_weight:.1f}% (concentração " + ("baixa)" if max_weight <= 30 else "elevada)" if max_weight <= 50 else "muito alta)"), "status": "positive" if max_weight <= 30 else "warning" if max_weight <= 50 else "negative"})
    return {"score": score, "coverage": coverage, "label": "Dados insuficientes" if score is None else "Favorável" if score >= 70 else "Intermediário" if score >= 40 else "Desfavorável", "diagnostics": diagnostics}


def calculate_analysis(payload, prices, returns, weights, quotes, quote_failed, daily_rf, rf_source, warnings, label, mu, covariance):
    start = date.today() - timedelta(days=payload.lookback_days)
    daily = returns.dot(align_weights_to_columns(weights, returns.columns))
    benchmark_prices, benchmark = None, None
    try:
        benchmark_prices = get_benchmark_prices(start)
        daily, benchmark = align_benchmark_returns(daily, benchmark_prices)
    except Exception:
        pass
    available = benchmark is not None and not benchmark.empty
    if not available:
        warnings.append("Dados do IBOVESPA indisponíveis no momento; métricas relativas e análises complementares ficam indisponíveis. O portfólio permanece válido para simulação.")
    rf = (1 + daily_rf) ** 252 - 1
    historical_rf, historical_rf_source = historical_risk_free(daily.index, start, daily_rf, warnings)
    excess = daily - historical_rf
    cumulative = (1 + daily).cumprod()
    value = cumulative * payload.capital
    drawdown = drawdown_from_returns(daily)
    metrics = {"total_return": finite_or_none((value.iloc[-1] / payload.capital - 1) * 100), "volatility": finite_or_none(daily.std() * np.sqrt(252) * 100), "sharpe": finite_or_none(qs.stats.sharpe(excess)), "sortino": finite_or_none(qs.stats.sortino(excess)), "max_drawdown": finite_or_none(drawdown.min() * 100)}
    metric_status = {
        "total_return": "N/D" if metrics["total_return"] is None else "Positivo" if metrics["total_return"] > 0 else "Negativo",
        "volatility": "N/D" if metrics["volatility"] is None else "Baixa" if metrics["volatility"] < 15 else "Alta" if metrics["volatility"] > 25 else "Moderada",
        "max_drawdown": "N/D" if metrics["max_drawdown"] is None else "Controlado" if metrics["max_drawdown"] > -15 else "Severo" if metrics["max_drawdown"] < -30 else "Moderado",
    }
    for key in ("sharpe", "sortino"):
        v = metrics[key]
        metric_status[key] = "N/D" if v is None else "Excelente" if v > 1 else "Bom" if v > .5 else "Baixo"
    monthly = daily.resample("ME").apply(lambda values: (1 + values).prod() - 1)
    years = []
    for year, values in monthly.groupby(monthly.index.year):
        months = {int(stamp.month): float(v) * 100 for stamp, v in values.items()}
        years.append({"year": int(year), **{f"m{i}": months.get(i) for i in range(1, 13)}, "ytd": ((1 + daily[daily.index.year == year]).prod() - 1) * 100})
    value_fig = line_chart(value, "Evolução do Valor do Portfólio" + (" vs Benchmark" if available else ""), "Portfólio")
    value_fig.update_yaxes(title="Valor (R$)")
    if available:
        value_fig.add_trace(go.Scatter(x=benchmark.index, y=(1 + benchmark).cumprod() * payload.capital, mode="lines", name="IBOVESPA", line=dict(color=CHART_SECONDARY, width=1.5, dash="dash")))
    charts = {"value": figure(value_fig)}
    if payload.strategy == "markowitz":
        charts["frontier"] = figure(plot_efficient_frontier_and_random_portfolios(mu, covariance, weights, rf, label))
    purchases, totals, missing = purchase_suggestion(weights, payload.capital, quotes, quote_failed)
    if quote_failed or missing:
        warnings.append("Fechamento bruto indisponível para " + ", ".join(missing or [t.removesuffix('.SA') for t, w in weights.items() if w > 0]) + "; cotas e totais aparecem como N/D.")
    result = {"strategy": payload.strategy, "mode": MODES[payload.strategy], "allocation_label": label, "capital": payload.capital, "lookback_days": payload.lookback_days, "tickers": payload.tickers, "weights": [{"ticker": ticker.removesuffix('.SA'), "weight": weight} for ticker, weight in weights.items()], "sample": {**sample_info(returns), "selected_count": len(payload.tickers), "zero_excluded": payload.strategy == "manual" and len(returns.columns) < len(payload.tickers)}, "comparison_sample": sample_info(daily) if available else None, "source": {"adjusted_prices": "Yahoo Finance Close ajustado", "trade_quotes": "Yahoo Finance Close bruto (últimos 14 dias)", "selic_source": rf_source, "selic_daily": daily_rf, "selic_annual": rf, "historical_rf": historical_rf_source}, "trade_suggestions": purchases, "trade_totals": totals, "metrics": metrics, "metric_status": metric_status, "monthly_returns": years, "benchmark_available": available, "charts": charts, "warnings": warnings, "health": None, "stress": None, "regime": None, "concentration": None, "asset_drawdowns": []}
    if not available:
        return result
    cov = np.cov(daily.squeeze(), benchmark.squeeze())
    variance = float(cov[1, 1])
    relative_valid = np.isfinite(variance) and variance > 1e-12
    beta = finite_or_none(cov[0, 1] / variance) if relative_valid else None
    alpha = finite_or_none(excess.mean() - beta * (benchmark - historical_rf).mean()) if beta is not None else None
    r2, ir = None, None
    if relative_valid:
        try:
            r2 = finite_or_none(qs.stats.r_squared(daily, benchmark))
            # qs.information_ratio is daily mean/std; annualize with the same 252-day basis as the other ratios.
            ir = finite_or_none(qs.stats.information_ratio(daily, benchmark) * np.sqrt(252))
        except Exception:
            pass
    else:
        warnings.append("A variação do IBOVESPA é insuficiente; beta, alfa, R² e information ratio ficaram indisponíveis.")
    elapsed = (value.index[-1] - value.index[0]).days
    historical_var, expected_shortfall = historical_var_cvar(daily)
    metrics.update({"annual_return": finite_or_none(((value.iloc[-1] / payload.capital) ** (365.25 / elapsed) - 1) * 100) if elapsed > 0 else None, "beta": beta, "daily_alpha": alpha, "annual_alpha": finite_or_none(alpha * 252 * 100) if alpha is not None else None, "r_squared": r2, "information_ratio": ir, "daily_var": finite_or_none(qs.stats.var(daily) * 100), "historical_var_95": finite_or_none(historical_var * 100), "cvar_95": finite_or_none(expected_shortfall * 100)})
    max_weight = max(weights.values()) * 100
    result["health"] = health_result(metrics, max_weight)
    active = [ticker for ticker, weight in weights.items() if weight > 0]
    result["stress"] = stress_result(prices.reindex(columns=active), benchmark_prices, weights, start)
    recent = (1 + daily.tail(21)).prod() - 1
    recent_bench = (1 + benchmark.tail(21)).prod() - 1
    regime, advice = ("Alta", "Manter exposição. Ativos com momentum positivo podem ser aumentados.") if recent > .03 else ("Queda", "Revise os pesos. Considere reduzir exposição ou adicionar proteção (ex: BOVA11 put).") if recent < -.03 else ("Lateral", "Consolidação. Bom momento para revisar correlações e rebalancear.")
    result["regime"] = {"label": regime, "portfolio": recent * 100, "benchmark": recent_bench * 100, "difference_pp": (recent - recent_bench) * 100, "volatility": daily.tail(21).std() * np.sqrt(252) * 100, "advice": advice}
    squares = sum(weight ** 2 for weight in weights.values())
    hhi, effective = squares * 10000, round(1 / squares, 1)
    diagnosis = "Alta Concentração: Portfólio fortemente concentrado. Adicione ativos ou redistribua os pesos." if hhi > 5000 else f"Concentração Moderada: Apenas {effective:.1f} ativos efetivos de {len(weights)}." if hhi > 2500 else f"Boa Diversificação: {effective:.1f} ativos efetivos — distribuição equilibrada."
    result["concentration"] = {"hhi": hhi, "effective_assets": effective, "selected_assets": len(weights), "largest_weight": max_weight, "level": "Baixo" if hhi < 2500 else "Moderado" if hhi < 5000 else "Alto", "diagnosis": diagnosis, "diagnostic_status": "negative" if hhi > 5000 else "warning" if hhi > 2500 else "positive", "largest_status": "OK" if max_weight <= 35 else "Concentrado"}
    charts["drawdown"] = figure(line_chart(drawdown, "Evolução do Drawdown do Portfólio", "Drawdown", CHART_DANGER, percent=True, fill=True))
    asset_dd = returns.apply(drawdown_from_returns)
    result["asset_drawdowns"] = sorted([{"ticker": ticker.removesuffix('.SA'), "max_drawdown": asset_dd[ticker].min() * 100, "date": asset_dd[ticker].idxmin().date().isoformat()} for ticker in returns.columns], key=lambda row: row["max_drawdown"])
    rolling_var = benchmark.rolling(60).var()
    rolling_beta = daily.rolling(60).cov(benchmark) / rolling_var.where(rolling_var.abs() > 1e-12)
    rolling_excess = excess.rolling(60)
    rolling_sharpe = (rolling_excess.mean() / rolling_excess.std()) * np.sqrt(252)
    charts["rolling_beta"] = figure(line_chart(rolling_beta, "Beta Móvel (60 dias) vs IBOVESPA", "Beta Móvel", CHART_SECONDARY, reference=1))
    charts["rolling_sharpe"] = figure(line_chart(rolling_sharpe, "Índice de Sharpe Móvel Anualizado (60 dias)", "Sharpe anualizado", reference=0))
    return result


@router.post("/analyze")
def analyze(payload: AnalyzeRequest, request: Request):
    ctx = session(request)
    with ctx.lock:
        if payload.tickers != ctx.selection or payload.revision != ctx.revision:
            raise HTTPException(409, "A seleção mudou. Carregue o portfólio novamente.")
        generation = begin(ctx, "portfolio", payload.revision)
    try:
        universe = set(get_listed_stocks()["Ticker"])
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"Não foi possível carregar a lista de ações da B3: {exc}") from exc
    if len(set(payload.tickers)) != len(payload.tickers) or any(ticker not in universe for ticker in payload.tickers):
        raise HTTPException(422, "Selecione de 2 a 20 ações distintas da lista atual da B3.")
    if any(ticker not in payload.tickers or shares < 0 for ticker, shares in payload.manual_shares.items()):
        raise HTTPException(422, "Cotas devem ser inteiros não negativos dos ativos selecionados.")
    tickers = [f"{ticker}.SA" for ticker in payload.tickers]
    required = [ticker for ticker in tickers if payload.manual_shares.get(ticker.removesuffix('.SA'), 0) > 0] if payload.strategy == "manual" else tickers
    if not required:
        raise HTTPException(422, "A alocação manual precisa ter ao menos uma cota positiva.")
    quotes, failed, manual_weights = {}, False, {}
    if payload.strategy == "manual":
        quotes, failed = trade_quotes(required)
        missing = [ticker.removesuffix('.SA') for ticker in required if ticker not in quotes]
        if failed or missing:
            raise HTTPException(422, f"Não foi possível obter cotações atuais para {', '.join(missing or [t.removesuffix('.SA') for t in required])}; elas são necessárias para calcular os pesos por cotas.")
        values = {ticker: payload.manual_shares[ticker.removesuffix('.SA')] * quotes[ticker][0] for ticker in required}
        total = sum(values.values())
        manual_weights = {ticker: values.get(ticker, 0.0) / total for ticker in tickers}
    prices, returns = history(required, date.today() - timedelta(days=payload.lookback_days))
    warnings = []
    try:
        daily_rf, rf_source = get_selic_rate(), "BCB SGS 432"
    except Exception as exc:
        daily_rf, rf_source = (1 + .1375) ** (1 / 252) - 1, "Referência 13,75% a.a. (BCB indisponível)"
        warnings.append(f"Não foi possível buscar a taxa Selic no BCB ({exc}). Usando valor de referência: 13,75% a.a.")
    rf = (1 + daily_rf) ** 252 - 1
    try:
        weights, label, mu, covariance = optimize(payload.strategy, returns, manual_weights, rf, payload.gamma, warnings)
        if payload.strategy != "manual":
            quotes, failed = trade_quotes([ticker for ticker, weight in weights.items() if weight > 0])
        result = calculate_analysis(payload, prices, returns, weights, quotes, failed, daily_rf, rf_source, warnings, label, mu, covariance)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Não foi possível concluir a análise do portfólio: {exc}") from exc
    identifier = uuid.uuid4().hex
    result.update({"id": identifier, "portfolio_id": identifier, "revision": payload.revision, "saved": False})
    snapshot = PortfolioSnapshot(id=identifier, revision=payload.revision, tickers=list(payload.tickers), weights={ticker.removesuffix('.SA'): float(weight) for ticker, weight in weights.items()}, mode=MODES[payload.strategy], returns=returns, prices=prices, lookback_days=payload.lookback_days, result=clean_json(result))
    with ctx.lock:
        publish(ctx, "portfolio", payload.revision, generation, snapshot)
        try:
            saved = db.portfolio_save(ctx.uid, payload.tickers, weights)
            snapshot.result["saved"] = saved is True
        except Exception:
            snapshot.result["saved"] = False
        if not snapshot.result["saved"]:
            snapshot.result["warnings"].append("A análise foi concluída, mas não foi possível salvar a alocação no banco de dados.")
    return snapshot.result


def restored_weights(tickers, saved, target):
    valid = {}
    for ticker, raw in (saved or {}).items():
        value = finite_or_none(raw)
        if value is not None and value >= 0:
            valid[str(ticker).removesuffix('.SA').strip().upper()] = value
    default = 1 / len(tickers)
    weights = {ticker: valid.get(ticker, default if target == "simulation" else 0.0) for ticker in tickers}
    total = sum(weights.values())
    return {ticker: weight / total for ticker, weight in weights.items()} if total > 0 else {ticker: default for ticker in tickers}


def restore_portfolio(ctx, target, revision):
    if target not in ("news", "simulation"):
        raise HTTPException(422, "Destino de restauração inválido.")
    with ctx.lock:
        if revision != ctx.revision:
            raise HTTPException(409, "A seleção mudou.")
        if ctx.portfolio is not None and (target == "news" or not ctx.portfolio.returns.empty):
            return ctx.portfolio
        if not ctx.restored and not (ctx.portfolio is not None and ctx.portfolio.result.get("restored") and ctx.portfolio.returns.empty):
            raise HTTPException(409, "A seleção foi alterada. Carregue a carteira em Portfolio.")
        selection = list(ctx.selection)
        saved_tickers, saved = db.portfolio_get(ctx.uid)
        tickers = [str(ticker).removesuffix('.SA').strip().upper() for ticker in saved_tickers]
        if tickers != selection or len(tickers) < (2 if target == "simulation" else 1):
            raise HTTPException(422, "Não há carteira salva válida para restaurar.")
        generation = begin(ctx, "portfolio", revision)
    weights = restored_weights(tickers, saved, target)
    prices, returns = pd.DataFrame(), pd.DataFrame()
    if target == "simulation":
        prices, returns = history([f"{ticker}.SA" for ticker in tickers], date.today() - timedelta(days=730))
    # The saved record stores only tickers and weights, so the original strategy and lookback are unknown.
    mode = "Carteira restaurada (janela padrão de 730 dias)" if target == "simulation" else "Carteira restaurada"
    identifier = uuid.uuid4().hex
    result = {"id": identifier, "portfolio_id": identifier, "revision": revision, "restored": True, "target": target, "mode": mode, "weights": [{"ticker": ticker, "weight": weight} for ticker, weight in weights.items()], "sample": sample_info(returns) if not returns.empty else None, "warnings": ["Carteira salva restaurada. Confira os ativos e pesos antes de continuar."]}
    snapshot = PortfolioSnapshot(id=identifier, revision=revision, tickers=tickers, weights=weights, mode=result["mode"], returns=returns, prices=prices, lookback_days=730, result=result)
    publish(ctx, "portfolio", revision, generation, snapshot)
    return snapshot


@router.post("/restore")
def restore(payload: RestoreRequest, request: Request):
    return clean_json(restore_portfolio(session(request), payload.target, payload.revision).result)


@router.post("/refresh")
def refresh(payload: RefreshRequest, request: Request):
    ctx = session(request)
    with ctx.lock:
        begin(ctx, "portfolio", ctx.revision)
        ctx.restored = False
    for provider in (get_portfolio_prices, get_portfolio_trade_prices, get_benchmark_prices, get_selic_rate, get_cdi_daily_rates):
        provider.clear()
    return {"message": "Cotações atualizadas. Clique em Carregar portfólio para recalcular a análise."}


@router.post("/{identifier}/stress")
def stress(identifier: str, payload: StressRequest, request: Request):
    ctx = session(request)
    snapshot = artifact(ctx, "portfolio", identifier)
    if not snapshot.result.get("benchmark_available"):
        raise HTTPException(422, "Stress test indisponível enquanto o IBOVESPA não responder.")
    weights = {f"{ticker}.SA": weight for ticker, weight in snapshot.weights.items()}
    active = [ticker for ticker, weight in weights.items() if weight > 0]
    prices = snapshot.prices.reindex(columns=active)
    start = date.today() - timedelta(days=snapshot.lookback_days)
    benchmark, warnings = None, []
    try:
        benchmark = get_benchmark_prices(start)
    except Exception:
        warnings.append("IBOVESPA indisponível para comparação.")
    if payload.long_history:
        long_start = min(pd.Timestamp(value[0]) for value in CRISES.values()).date()
        try:
            prices = price_frame(get_portfolio_prices(tuple(active), long_start), active).reindex(columns=active)
            start = long_start
        except Exception:
            warnings.append("Não foi possível carregar o histórico longo para o stress test.")
        try:
            benchmark = get_benchmark_prices(long_start)
        except Exception:
            warnings.append("IBOV histórico indisponível; comparação limitada ao lookback selecionado.")
    result = clean_json(stress_result(prices, benchmark, weights, start, warnings))
    with ctx.lock:
        if ctx.portfolio is None or ctx.portfolio.id != identifier or ctx.revision != snapshot.revision:
            raise HTTPException(409, "O portfólio mudou durante o stress test.")
    return result

@router.get("/{identifier}/allocation.csv")
def allocation_csv(identifier: str, request: Request):
    snapshot = artifact(session(request), "portfolio", identifier)
    frame = pd.DataFrame({"Peso (%)": [round(weight * 100, 2) for weight in snapshot.weights.values()]}, index=list(snapshot.weights))
    return csv_response(frame, f"portfolio_alocacao_{date.today()}.csv")
