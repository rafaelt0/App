"""Matplotlib/Plotly chart builders for pages/1_Portfolio.py."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pypfopt.efficient_frontier import EfficientFrontier

from utils.charts import apply_plotly_theme


def apply_matplotlib_theme(fig):
    fig.set_facecolor("#080c14")
    for ax in fig.axes:
        ax.set_facecolor("#0e1524")
        ax.tick_params(colors="#94a3b8", which="both")
        ax.yaxis.label.set_color("#f8fafc")
        ax.xaxis.label.set_color("#f8fafc")
        if ax.title:
            ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#0e1524")
            legend.get_frame().set_edgecolor("#1e293b")
            for text in legend.get_texts():
                text.set_color("#f8fafc")
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_frontier_data(mu_tuple, S_tuple, weights_tuple, rf, num_portfolios=5000):
    """Numeric core of the efficient-frontier chart — cached because it's
    expensive (5,000 simulated portfolios + 25 quadratic programs) and its
    inputs (mu/S/weights/rf) are identical across unrelated widget reruns.
    Takes plain tuples instead of pandas/dict objects so Streamlit hashes
    the cache key cheaply and deterministically.
    """
    mu = np.array(mu_tuple)
    S = np.array(S_tuple)

    # Vectorized Monte Carlo: draw all 5,000 portfolios' weights at once
    # instead of looping in Python, then compute return/vol/Sharpe for the
    # whole batch with matrix ops (np.einsum for the quadratic form w'Sw).
    rng = np.random.default_rng(42)
    weights = rng.random((num_portfolios, len(mu)))
    weights /= weights.sum(axis=1, keepdims=True)
    port_returns = weights @ mu
    port_variances = np.einsum("ij,jk,ik->i", weights, S, weights)
    port_stddevs = np.sqrt(port_variances)
    port_sharpe = np.divide(
        port_returns - rf,
        port_stddevs,
        out=np.zeros_like(port_returns),
        where=port_stddevs > 0,
    )
    results = np.vstack([port_stddevs, port_returns, port_sharpe])

    opt_weights = np.array(weights_tuple)
    opt_return = np.sum(opt_weights * mu)
    opt_stddev = np.sqrt(np.dot(opt_weights.T, np.dot(S, opt_weights)))
    opt_sharpe = (opt_return - rf) / opt_stddev if opt_stddev > 0 else 0

    try:
        ef_min = EfficientFrontier(mu, S)
        min_weights = ef_min.min_volatility()
        min_weights_arr = np.array(list(min_weights.values()))
        min_return = np.sum(min_weights_arr * mu)
        min_stddev = np.sqrt(np.dot(min_weights_arr.T, np.dot(S, min_weights_arr)))
    except Exception:
        min_return = min(mu)
        min_stddev = np.sqrt(np.min(np.diag(S)))

    efficient_vols = []
    target_returns = np.linspace(min_return, max(mu), 25)
    for target in target_returns:
        try:
            ef_target = EfficientFrontier(mu, S)
            ef_target.efficient_return(target)
            w_target = np.array(list(ef_target.clean_weights().values()))
            vol = np.sqrt(np.dot(w_target.T, np.dot(S, w_target)))
            efficient_vols.append(vol)
        except Exception:
            pass

    return (
        results,
        np.array(efficient_vols),
        target_returns,
        opt_return,
        opt_stddev,
        opt_sharpe,
        min_return,
        min_stddev,
    )


def plot_efficient_frontier_and_random_portfolios(mu, S, cleaned_weights, rf):
    (
        results,
        efficient_vols,
        target_returns,
        opt_return,
        opt_stddev,
        opt_sharpe,
        min_return,
        min_stddev,
    ) = _compute_frontier_data(
        tuple(mu.values if hasattr(mu, "values") else mu),
        tuple(map(tuple, S.values if hasattr(S, "values") else S)),
        tuple(cleaned_weights.values()),
        rf,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results[0, :],
            y=results[1, :],
            mode="markers",
            marker=dict(
                size=4,
                color=results[2, :],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Índice Sharpe", font=dict(color="#f8fafc")),
                    tickfont=dict(color="#f8fafc"),
                ),
                opacity=0.6,
            ),
            name="Portfólios Aleatórios",
            text=[
                f"Retorno Anual: {r:.2%}<br>Vol Anual: {v:.2%}<br>Sharpe: {s:.2f}"
                for v, r, s in zip(results[0, :], results[1, :], results[2, :])
            ],
            hoverinfo="text",
        )
    )

    if len(efficient_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=efficient_vols,
                y=target_returns[: len(efficient_vols)],
                mode="lines",
                line=dict(color="#00ff87", width=3),
                name="Fronteira Eficiente",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[opt_stddev],
            y=[opt_return],
            mode="markers",
            marker=dict(
                color="#ff1744",
                size=12,
                symbol="star",
                line=dict(color="#f8fafc", width=2),
            ),
            name="Max Sharpe (Markowitz)",
            text=[
                f"Max Sharpe<br>Retorno: {opt_return:.2%}<br>Vol: {opt_stddev:.2%}<br>Sharpe: {opt_sharpe:.2f}"
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_stddev],
            y=[min_return],
            mode="markers",
            marker=dict(
                color="#ffd600",
                size=10,
                symbol="diamond",
                line=dict(color="#f8fafc", width=1.5),
            ),
            name="Mínima Volatilidade",
            text=[
                f"Mínima Volatilidade<br>Retorno: {min_return:.2%}<br>Vol: {min_stddev:.2%}"
            ],
            hoverinfo="text",
        )
    )

    # LAC — Linha de Alocação de Capital (EAE1242 — Tobin, 1958)
    # Parte do Rf (volatilidade=0) e passa pela carteira tangente (Max Sharpe)
    if opt_stddev > 0:
        lac_slope = (opt_return - rf) / opt_stddev
        lac_x_end = opt_stddev * 1.8
        fig.add_trace(
            go.Scatter(
                x=[0, lac_x_end],
                y=[rf, rf + lac_slope * lac_x_end],
                mode="lines",
                line=dict(color="#ffd600", width=2, dash="dash"),
                name="LAC — Linha de Alocação de Capital",
                hovertemplate="LAC<br>Vol: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[rf],
                mode="markers+text",
                marker=dict(color="#ffd600", size=8, symbol="circle"),
                text=["Rf (Selic)"],
                textposition="top right",
                textfont=dict(color="#ffd600", size=10),
                name="Taxa Livre de Risco (Rf)",
                hovertemplate=f"Rf (Selic) = {rf:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Fronteira Eficiente de Markowitz · LAC · Carteira Tangente",
        xaxis_title="Volatilidade Anualizada (Desvio Padrão)",
        yaxis_title="Retorno Esperado Anualizado",
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=550, margin=dict(b=140))
    return fig
