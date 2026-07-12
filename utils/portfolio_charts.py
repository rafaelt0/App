"""Matplotlib/Plotly chart builders for pages/1_Portfolio.py."""

import numpy as np
import plotly.graph_objects as go
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


def plot_efficient_frontier_and_random_portfolios(mu, S, returns, cleaned_weights, rf):
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mu)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

        results[0, i] = portfolio_stddev
        results[1, i] = portfolio_return
        results[2, i] = (
            (portfolio_return - rf) / portfolio_stddev if portfolio_stddev > 0 else 0
        )

    opt_weights = np.array(list(cleaned_weights.values()))
    opt_return = np.sum(opt_weights * mu)
    opt_stddev = np.sqrt(np.dot(opt_weights.T, np.dot(S, opt_weights)))
    opt_sharpe = (opt_return - rf) / opt_stddev if opt_stddev > 0 else 0

    try:
        ef_min = EfficientFrontier(mu, S)
        min_weights = ef_min.min_volatility()
        min_weights_arr = np.array(list(min_weights.values()))
        min_return = np.sum(min_weights_arr * mu)
        min_stddev = np.sqrt(np.dot(min_weights_arr.T, np.dot(S, min_weights_arr)))
    except:
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
        except:
            pass

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
