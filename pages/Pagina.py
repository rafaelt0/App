import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.hierarchical_portfolio import HRPOpt
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
from scipy.stats import kurtosis, skew
import quantstats as qs
import matplotlib.ticker as mtick
import io
st.header("Opções Simulação 👨‍🔬")
n_simulations = st.slider("Número de Simulações", 10, 500, 200)  # Limite para performance
valor = st.number_input("Capital Inicial (R$)", min_value=100)
years = int(st.number_input("Anos", min_value=1))
st.header("Simulação 🧪")

n_dias = years * 252  # Usar 252 dias úteis para ser mais realista
valor_inicial = valor

# Retornos históricos do portfólio alinhados
aligned = portfolio_returns.dropna()
mu_p = aligned.mean()
sigma_p = aligned.std()

# Simulação vetorizada GBM (Geometric Brownian Motion)
np.random.seed(42)  # para reprodutibilidade
rand = np.random.normal(size=(n_dias, n_simulations))
growth_factors = np.exp((mu_p - 0.5 * sigma_p ** 2) + sigma_p * rand)
simulacoes = np.vstack([np.ones(n_simulations) * valor_inicial, valor_inicial * growth_factors.cumprod(axis=0)])

# DataFrame para facilitar manipulação
sim_df = pd.DataFrame(simulacoes)
sim_df.index.name = "Dia"

# Estatísticas finais da simulação
valor_esperado = sim_df.iloc[-1].mean()
var_5 = np.percentile(sim_df.iloc[-1], 5)
cvar_5 = sim_df.iloc[-1][sim_df.iloc[-1] <= var_5].mean()
pior_cenario = sim_df.iloc[-1].min()
melhor_cenario = sim_df.iloc[-1].max()

sim_stats = pd.DataFrame({
    "Valor Esperado Final (R$)": [valor_esperado],
    "VaR 5% (R$)": [var_5],
    "CVaR 5% (R$)": [cvar_5],
    "Pior Cenário (R$)": [pior_cenario],
    "Melhor Cenário (R$)": [melhor_cenario]
})

st.subheader("📊 Estatísticas da Simulação Monte Carlo")
st.dataframe(sim_stats.style.format("{:,.2f}"))

# Fan chart com faixas de confiança
percentis = [5, 25, 50, 75, 95]
fan_chart = sim_df.quantile(q=np.array(percentis) / 100, axis=1).T
fan_chart.columns = [f"P{p}" for p in percentis]

fig_fan = go.Figure()

# Faixa 5%-95%
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P95"],
    line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P5"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
    line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 5%-95%'
))

# Faixa 25%-75%
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P75"],
    line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P25"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.4)',
    line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 25%-75%'
))

# Linha mediana
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P50"],
    line=dict(color='blue', width=2), name='Mediana'
))

# Layout final
fig_fan.update_layout(
    title="Simulação Monte Carlo - Fan Chart com Faixas de Confiança",
    xaxis_title="Dia",
    yaxis_title="Valor do Portfólio (R$)",
    template="plotly_white"
)

st.plotly_chart(fig_fan, use_container_width=True)

# Histograma do valor final do portfólio
st.subheader("Distribuição do Valor Final do Portfólio")
import seaborn as sns
import matplotlib.pyplot as plt

fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
sns.histplot(sim_df.iloc[-1], bins=50, kde=True, color='skyblue', ax=ax_hist)
ax_hist.set_xlabel("Valor Final do Portfólio (R$)")
ax_hist.set_ylabel("Frequência")
st.pyplot(fig_hist)
