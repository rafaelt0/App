from typing import Final

import plotly.graph_objects as go


CHART_PRIMARY: Final = "#76d5c5"
CHART_SECONDARY: Final = "#91b9e7"
CHART_ACCENT: Final = "#e7bd69"
CHART_DANGER: Final = "#ef969d"
CHART_INFO: Final = "#b7a3e2"
CHART_TEXT: Final = "#eef4f2"
CHART_MUTED: Final = "#9cafae"
CHART_GRID: Final = "#63787c"
CHART_BORDER: Final = "#33464a"
CHART_SURFACE: Final = "#142126"
CHART_SECONDARY_FILL: Final = "rgba(145, 185, 231, 0.14)"
CHART_SECONDARY_FILL_STRONG: Final = "rgba(145, 185, 231, 0.24)"
CHART_DANGER_FILL: Final = "rgba(239, 150, 157, 0.16)"
CHART_INFO_FILL: Final = "rgba(183, 163, 226, 0.12)"
CHART_COLORWAY: Final = (
    CHART_PRIMARY,
    CHART_SECONDARY,
    CHART_ACCENT,
    CHART_DANGER,
    CHART_INFO,
)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        colorway=CHART_COLORWAY,
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            color=CHART_TEXT,
        ),
        xaxis=dict(
            gridcolor=CHART_GRID,
            linecolor=CHART_BORDER,
            zerolinecolor=CHART_GRID,
            tickfont=dict(
                family='"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace',
                color=CHART_MUTED,
            ),
        ),
        yaxis=dict(
            gridcolor=CHART_GRID,
            linecolor=CHART_BORDER,
            zerolinecolor=CHART_GRID,
            tickfont=dict(
                family='"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace',
                color=CHART_MUTED,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor=CHART_SURFACE,
            bordercolor=CHART_BORDER,
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor=CHART_SURFACE,
            bordercolor=CHART_BORDER,
            font=dict(color=CHART_TEXT),
        ),
        margin=dict(b=80),
    )
    return fig
