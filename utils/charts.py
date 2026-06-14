def apply_plotly_theme(fig):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Grotesk, sans-serif", color="#f8fafc"),
        xaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(14, 21, 36, 0.8)',
            bordercolor='#1e293b',
            borderwidth=1
        ),
        margin=dict(b=80)
    )
    return fig
