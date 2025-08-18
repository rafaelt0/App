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
import base64








st.header("Simulação Monte Carlo por Ativos (Multivariada) 👨‍🔬")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# converte a imagem para base64
img_base64 = get_base64_of_bin_file("b3explorer.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        st.warning("⚠️ Configure primeiro seu portfólio na aba 1 para liberar a simulação Monte Carlo.")
        st.stop()

# Recupera as variáveis da aba 1
modo = st.session_state["modo"]
returns = st.session_state["returns"]
pesos_manuais = st.session_state["pesos_manuais"]
peso_manual_df = st.session_state["peso_manual_df"]

with st.form("form_simulacao"):
    n_simulations = st.slider("Número de Simulações", 10, 500, 200,
                              help="Quantidade de trajetórias simuladas para o portfólio.")
    valor = st.number_input("Capital Inicial (R$)", min_value=100,
                            help="Valor inicial investido no portfólio.")
    years = int(st.number_input("Anos", min_value=1,
                                help="Horizonte da simulação em anos."))
    
    submitted = st.form_submit_button("Rodar Simulação")

if not submitted:
    st.info("Configure os parâmetros acima e clique em 'Rodar Simulação' para ver os resultados.")
    st.stop()

st.header("Simulação 🧪")

n_dias = years * 252  # 252 dias úteis no ano
valor_inicial = valor

# Garante que temos um dicionário de pesos, independente do modo escolhido
if modo == "Alocação Manual":
    pesos_dict = pesos_manuais
else:
    pesos_dict = dict(zip(peso_manual_df.index + ".SA", peso_manual_df["Peso"].values))

# Remove ativos com peso zero (se houver)
pesos_dict = {k: v for k, v in pesos_dict.items() if v > 1e-6}

aligned_returns = returns.loc[:, pesos_dict.keys()].dropna()

pesos = np.array(list(pesos_dict.values()))

mu = aligned_returns.mean().values  # vetor média de retorno diário
cov = aligned_returns.cov().values  # matriz covariância diária

np.random.seed(42)  # para reprodutibilidade

# Simular retornos multivariados normais correlacionados
retornos_simulados = np.random.multivariate_normal(mu, cov, size=(n_dias, n_simulations))

# Calcular trajetórias para cada ativo em cada simulação
precos_simulados = np.exp(retornos_simulados.cumsum(axis=0))

# Calcular valor do portfólio: soma ponderada dos ativos para cada dia e simulação
valor_portfolio = (precos_simulados * pesos).sum(axis=2) * valor_inicial

# Criar DataFrame para facilitar manipulação e plotagem
datas = pd.date_range(start=datetime.date.today(), periods=n_dias+1, freq='B')
valor_portfolio = np.vstack([np.ones(n_simulations)*valor_inicial, valor_portfolio])
sim_df = pd.DataFrame(valor_portfolio, index=datas)

# Estatísticas finais da simulação
valores_finais = sim_df.iloc[-1]
valor_esperado = valores_finais.mean()
var_5 = np.percentile(valores_finais, 5)
cvar_5 = valores_finais[valores_finais <= var_5].mean()
pior_cenario = valores_finais.min()
melhor_cenario = valores_finais.max()

sim_stats = pd.DataFrame({
    "Valor Esperado Final (R$)": [valor_esperado],
    "VaR 5% (R$)": [var_5],
    "CVaR 5% (R$)": [cvar_5],
    "Pior Cenário (R$)": [pior_cenario],
    "Melhor Cenário (R$)": [melhor_cenario]
})

st.subheader("📊 Estatísticas da Simulação Monte Carlo por Ativos")
st.dataframe(sim_stats.style.format("{:,.2f}"))

st.markdown("""
<small><b>VaR 5%</b>: Valor máximo esperado que você pode perder em 5% dos piores casos.<br>
<b>CVaR 5%</b>: Média das perdas nos piores 5% dos casos, mostrando um risco mais extremo.</small>
""", unsafe_allow_html=True)

# Gráfico com algumas trajetórias individuais para ilustrar a dispersão
st.subheader("Trajetórias Individuais das Simulações (Exemplos)")

fig_individual = go.Figure()
n_plot = min(20, n_simulations)  # limitar para 20 linhas para visualização limpa

for i in range(n_plot):
    fig_individual.add_trace(go.Scatter(
        x=sim_df.index,
        y=sim_df.iloc[:, i],
        mode='lines',
        name=f'Simulação {i+1}',
        line=dict(width=1),
        opacity=0.6
    ))
fig_individual.update_layout(
    title="Exemplos de Trajetórias Simuladas do Valor do Portfólio",
    xaxis_title="Data",
    yaxis_title="Valor do Portfólio (R$)",
    template="plotly_white"
)
st.plotly_chart(fig_individual, use_container_width=True)

# Fan chart com percentis
percentis = [5, 25, 50, 75, 95]
fan_chart = sim_df.quantile(q=np.array(percentis) / 100, axis=1).T
fan_chart.columns = [f"P{p}" for p in percentis]

fig_fan = go.Figure()
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P95"],
    line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P5"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
    line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 5%-95%'
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P75"],
    line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P25"],
    fill='tonexty', fillcolor='rgba(0,100,200,0.4)',
    line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 25%-75%'
))
fig_fan.add_trace(go.Scatter(
    x=fan_chart.index, y=fan_chart["P50"],
    line=dict(color='blue', width=2), name='Mediana'
))
fig_fan.update_layout(
    title="Simulação Monte Carlo por Ativos - Fan Chart com Faixas de Confiança",
    xaxis_title="Data",
    yaxis_title="Valor do Portfólio (R$)",
    template="plotly_white"
)
st.plotly_chart(fig_fan, use_container_width=True)

# Histograma valor final
q1 = valores_finais.quantile(0.25)
q2 = valores_finais.quantile(0.50)
q3 = valores_finais.quantile(0.75)

st.subheader("Distribuição do Valor Final do Portfólio")
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(valores_finais, bins=30, kde=True, color='skyblue', edgecolor='black', ax=ax)
ax.axvline(q1, color='red', linestyle='--', label='Q1 (25%)')
ax.axvline(q2, color='green', linestyle='-', label='Mediana (50%)')
ax.axvline(q3, color='orange', linestyle='--', label='Q3 (75%)')
ax.set_title('Distribuição dos Valores Finais da Simulação Monte Carlo')
ax.set_xlabel('Valor Final do Portfólio (R$)')
ax.set_ylabel('Frequência')
ax.legend()
st.pyplot(fig)

# Estatísticas da distribuição final
estatisticas = {
    "Mínimo": valores_finais.min(),
    "Q1 (25%)": q1,
    "Mediana (50%)": q2,
    "Q3 (75%)": q3,
    "Máximo": valores_finais.max(),
    "Média": valores_finais.mean(),
    "Desvio Padrão": valores_finais.std()
}
df_estatisticas = pd.DataFrame(estatisticas, index=["Valores (R$)"])
st.subheader("Estatísticas da Distribuição Final da Simulação Monte Carlo")
st.dataframe(df_estatisticas.style.format("{:,.2f}"))





