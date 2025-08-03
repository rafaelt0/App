import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import fundamentus
import yfinance as yf

st.sidebar.header("Opções Simulação 👨‍🔬")
n_simulations = st.sidebar.slider("Número de Simulações",10,1000,100)
valor = st.sidebar.number_input("Capital Inicial", min_value=100)
periodos = int(st.sidebar.number_input("Meses", value=12, min_value=1))
years = int(st.sidebar.number_input("Anos", min_value=1))         
data_inicio = st.sidebar.date_input("Data Inicial📅", value=datetime.date(2019,1,1),min_value=datetime.date(2000,1,1))


st.header("Simulação 🧪")

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")


with col3:
    st.write("")

data = pd.read_csv('acoes-listadas-b3.csv')
stocks = data['Ticker'].values
ticker = st.selectbox('Escolha uma ação para simular', stocks)+'.SA'
ticker = yf.Ticker(ticker)
data = ticker.history(start=data_inicio, end=datetime.datetime.now(),interval='1mo')
data = data.Close
returns= data.pct_change()
mean = float(returns.mean())
mu_selected = (1+mean)**12-1

sigma_selected = returns.std()*np.sqrt(12)
    




mu = mu_selected
n = periodos
M = n_simulations
S0 = valor
sigma = sigma_selected
T = years

dt = T/n

St = np.exp(
    (mu - sigma**2/2)*dt
    * sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

St = np.vstack([np.ones(M), St])

St = S0 * St.cumprod(axis=0)

time = np.linspace(0,T,n+1)

tt = np.full(shape=(M,n+1), fill_value=time).T

fig=px.line(St, title="Simulação por Movimento Browniano Geométrico")
fig.update_layout(
                  xaxis = dict(
                    tickmode='array', #change 1
                    tickvals = np.arange(0,n*years,2)))
fig.update_yaxes(title="Portfolio/Ação")
fig.update_xaxes(title="Período")
st.plotly_chart(fig)
mean=St[-1][:].mean()
max=St[-1][:].max()
min=St[-1][:].min()
array = np.array(St[-1][:])
summary=pd.DataFrame([mean,max,min, np.percentile(array,25), np.median(array), np.percentile(array,75), np.std(array)])
summary = summary.rename({0:"Média", 1:"Máximo", 2:"Mínimo", 3:"Primeiro Quartil", 4:"Mediana (Segundo Quartil)", 5:"Terceiro Quartil", 6:"Desvio Padrão"}, axis=0)
summary = summary.rename({0:"Resultados"}, axis=1)
st.subheader("Resultados 🔬")
st.table(summary.T)
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Supondo que sim_df seja seu DataFrame com simulações
# sim_df.index = dias, colunas = simulações

# Calcula percentis para faixas
percentis = [5, 25, 50, 75, 95]
fan_chart = St.quantile(q=np.array(percentis)/100, axis=1).T
fan_chart.columns = [f"P{p}" for p in percentis]

# Cria figura do fan chart
fig_fan = go.Figure()

# Adiciona faixas sombreadas
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






