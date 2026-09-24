def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        colorway=["#61d4c6", "#8cb4f2", "#e7b96b", "#e58a93", "#b7a2e6"],
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            color="#f0f4f8",
        ),
        xaxis=dict(
            gridcolor="#28384b",
            linecolor="#405873",
            zerolinecolor="#34465b",
            tickfont=dict(
                family='"SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace',
                color="#aebaca",
            ),
        ),
        yaxis=dict(
            gridcolor="#28384b",
            linecolor="#405873",
            zerolinecolor="#34465b",
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
            bgcolor="rgba(21, 29, 42, 0.8)",
            bordercolor="rgba(52, 70, 91, 0.65)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#151d2a",
            bordercolor="#34465b",
            font=dict(color="#f0f4f8"),
        ),
        margin=dict(b=80),
    )
    return fig
