import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.sidebar.header("Opções Simulação 👨‍🔬")
mu_selected = st.sidebar.slider("Média",-4.00,4.00,0.01)
sigma_selected = st.sidebar.slider("Volatilidade",-4.00,4.00,0.001)
n_simulations = st.sidebar.slider("Número de Simulações",10,1000,10)
valor = st.sidebar.number_input("Capital Inicial", min_value=10)
periodos = int(st.sidebar.number_input("Passos", value=12, min_value=1))
years = int(st.sidebar.number_input("Anos", min_value=1))         



st.header("Simulação 🧪")

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")


with col3:
    st.write("")

data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = list(st.multiselect('Escolha uma ação para simular', stocks))

try:
    df = fundamentus.get_papel(list(tickers)[0])
    i=1
    for i in range(len(tickers)):
            df = pd.concat([df,fundamentus.get_papel(list(tickers)[i])])
    tickers = [ticker+".SA" for ticker in tickers]
    ticker = yf.Tickers(tickers)
except:
    pass

try:
    data = ticker.history(start=data_inicio, end=datetime.datetime.now(),period=period_dict[period_selected]\
                          ,interval=interval_dict[interval_selected],rounding=True)
    data = data.Close
    returns= data.pct_change()
    returns = returns.dropna()*100
    mu_selected = returns.mean()
    sigma_selected = returns.std()
except:
    pass
    




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





