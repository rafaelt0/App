def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            color="#f0f4f8",
        ),
        xaxis=dict(
            gridcolor="#34465b",
            linecolor="#34465b",
            tickfont=dict(
                family='"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace',
                color="#aebaca",
            ),
        ),
        yaxis=dict(
            gridcolor="#34465b",
            linecolor="#34465b",
            tickfont=dict(
                family='"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace',
                color="#aebaca",
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(21, 29, 42, 0.9)",
            bordercolor="#34465b",
            borderwidth=1,
        ),
        margin=dict(b=80),
    )
    return fig
