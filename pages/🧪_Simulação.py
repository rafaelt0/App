import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import fundamentus
import yfinance as yf

st.sidebar.header("Opções Simulação 👨‍🔬")
n_simulations = st.sidebar.slider("Número de Simulações",10,1000,10)
valor = st.sidebar.number_input("Capital Inicial", min_value=10)
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





