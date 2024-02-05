import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import sklearn
import pandas as pd
import seaborn as sns
from quantstats.stats import max_drawdown
import warnings
import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
import datetime
from scipy.stats import kurtosis
from scipy.stats import skew
from pypfopt import plotting
import plotly.express as px

st.sidebar.header('Configurações ⚙️')
lista=list(np.arange(2024,2000,-1))
lista.append("None")
period_selected = st.sidebar.selectbox('Período ⏰', ['diário','semanal','trimestral','semestral','mensal','anual'])
period_dict = {'diário':'1d','semanal':'1w','mensal':'1mo','trimestral':'3mo','semestral':'6mo','anual':'1y'}
data_inicio = st.sidebar.date_input("Data Inicial📅", datetime.date(2024,1,1),min_value=datetime.date(2000,1,1))
interval_selected = st.sidebar.selectbox('Intervalo 📊', ['dia','3 meses','mês','semana','hora','minuto'])
interval_dict={'dia':'1d','3 meses':'3mo', 'mês':'1mo','hora':'1h','minuto':'1m','semana':'1wk'}
valor_inicial = st.sidebar.number_input("Valor Investido 💵", min_value=10, max_value=1_000_000)

stocks = fundamentus.list_papel_all()
tickers = list(st.sidebar.multiselect('Monte seu Portfolio (Escolha mais de uma ação)',stocks))
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
    returns_percentage = np.round(returns,2)
    returns_string = returns_percentage.astype(str)+'%'
    dictionary = dict(ticker.tickers)
    lista=list(dictionary.keys())
    factors=[]
    for _ in lista:
        factors.append(yf.Ticker(_).info['longBusinessSummary'])
    df = pd.DataFrame(factors, index=lista)
    df = df.set_axis(["Descrição"], axis=1)
except:
     pass

mu = mean_historical_return(data)
S = CovarianceShrinkage(data).ledoit_wolf()
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
w = ef.max_sharpe()
weights=pd.DataFrame(ef.clean_weights(), index=[0])
weights=weights.rename({0:"Pesos"}, axis=0)
weights=round(weights,4)
weights_graph=np.array(weights).ravel()
weights_string= (weights*100).astype("str")+"%"
st.subheader("Porcentagem ótima no Portfolio")
fig, ax = plt.subplots(figsize=(12,5))
ax.pie(weights_graph,labels=weights.columns.values,autopct='%1.1f%%')
fig = px.pie(weights_graph, values=weights_graph, names=weights.columns.values)
st.plotly_chart(fig)
st.dataframe(weights_string)
weights=(weights*1_000_000).astype("int").T
returns_calc=(returns*1000_000).astype("int")
returns_calc=np.dot(returns_calc,weights)
returns=returns.index
returns_calc=pd.DataFrame(returns_calc/1_000_000_000_000)
returns_calc_non_pct=returns_calc/100
returns_calc.rename({0:"Retornos do Portfolio"},axis=1, inplace=True)
returns_calc.index=returns
returns_calc=round(returns_calc,3)
returns_calc_string = returns_calc.astype("str")+"%"
st.dataframe(returns_calc_string)
cum_return=(1+returns_calc_non_pct).cumprod()-1
cum_returns=round(cum_return*100,3)
cum_returns_string=cum_returns.astype("str")+"%"
cum_returns_df=pd.DataFrame(cum_returns_string)
cum_returns_df.index=returns
cum_returns_df=cum_returns_df.rename({0:"Retornos Acumulados Portfolio"}, axis=1)
st.dataframe(cum_returns_df)
valor=valor_inicial
portfolio_value=(1+cum_return)*valor
portfolio_value_df=pd.DataFrame(portfolio_value)
portfolio_value_df.index=returns
portfolio_value_df.rename({0:'Valor do Portfolio'}, axis=1, inplace=True)
st.write(portfolio_value_df)
st.write(max_drawdown(returns_non_pct))





