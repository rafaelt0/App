from typing import Final

import plotly.graph_objects as go


CHART_PRIMARY: Final = "#8b93ff"
CHART_SECONDARY: Final = "#4cc9f0"
CHART_ACCENT: Final = "#f5b544"
CHART_DANGER: Final = "#f2757a"
CHART_INFO: Final = "#e879c6"
CHART_POSITIVE: Final = "#3ecf8e"
CHART_TEXT: Final = "#f3f5f9"
CHART_MUTED: Final = "#9aa2b4"
CHART_GRID: Final = "#646e88"
CHART_BORDER: Final = "#2b313d"
CHART_SURFACE: Final = "#161a22"
CHART_SECONDARY_FILL: Final = "rgba(76, 201, 240, 0.14)"
CHART_SECONDARY_FILL_STRONG: Final = "rgba(76, 201, 240, 0.24)"
CHART_DANGER_FILL: Final = "rgba(242, 117, 122, 0.16)"
CHART_INFO_FILL: Final = "rgba(232, 121, 198, 0.12)"
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
            family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
            color=CHART_TEXT,
        ),
        xaxis=dict(
            gridcolor=CHART_GRID,
            linecolor=CHART_BORDER,
            zerolinecolor=CHART_GRID,
            tickfont=dict(
                family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
                color=CHART_MUTED,
            ),
        ),
        yaxis=dict(
            gridcolor=CHART_GRID,
            linecolor=CHART_BORDER,
            zerolinecolor=CHART_GRID,
            tickfont=dict(
                family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
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
