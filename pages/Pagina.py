fig_fan = go.Figure()

# Faixa 5-95%
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P95"],
    line=dict(color='rgba(0,100,200,0)'), showlegend=False,
    hoverinfo='skip'
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P5"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
    line=dict(color='rgba(0,100,200,0)'), name='Faixa 5%-95%',
    hoverinfo='skip'
))

# Faixa 25-75%
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P75"],
    line=dict(color='rgba(0,100,200,0)'), showlegend=False,
    hoverinfo='skip'
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P25"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.4)',
    line=dict(color='rgba(0,100,200,0)'), name='Faixa 25%-75%',
    hoverinfo='skip'
))

# Linha Mediana
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P50"],
    line=dict(color='blue', width=3), name='Mediana'
))

# Linha de capital inicial
fig_fan.add_hline(y=valor_inicial, line=dict(color='red', dash='dash'),
                  annotation_text='Capital Inicial', annotation_position='top left')

fig_fan.update_layout(
    title="Simulação Monte Carlo – Fan Chart com Cenários de Portfólio",
    xaxis_title="Dia",
    yaxis_title="Valor do Portfólio (R$)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig_fan, use_container_width=True)

    
    

