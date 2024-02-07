import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import pandas as pd
import seaborn as sns
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


warnings.filterwarnings('ignore')
plt.style.use('ggplot')

st.set_page_config(
    page_title="Stock Explorer",
    page_icon="📈"
    )
st.sidebar.success("Select Page")


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.write("""
# **B3 Explorer 📈**
 """)

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")

with col2:
    st.image('b3.webp', width=(400))

with col3:
    st.write("")

data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
st.subheader("Explore ações da B3 🧭")
tickers = list(st.multiselect('Escolha ações para explorar! (2 ou mais ações)',stocks))
try:
    df = fundamentus.get_papel(list(tickers)[0])
    i=1
    for i in range(len(tickers)):
            df = pd.concat([df,fundamentus.get_papel(list(tickers)[i])])
    df['PL'] = df['PL'].astype('float64')/100
    df_basic = df[['Empresa', 'Setor', 'Subsetor']]    
    st.write(df_basic.drop_duplicates(keep='last'))
    df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 'Valor_de_mercado', 'Data_ult_cot']]
    st.dataframe(df_price.drop_duplicates(keep='last'))
    df_indicadores = df[['Marg_Liquida','ROE', 'ROIC', 'Div_Yield', 'Cres_Rec_5a', 'PL']]
    st.dataframe(df_indicadores.drop_duplicates(keep='last'))
    tickers = [ticker+".SA" for ticker in tickers]
    ticker = yf.Tickers(tickers)
except:
     pass

st.sidebar.header('Configurações ⚙️')
lista=list(np.arange(2024,2000,-1))
lista.append("None")
period_selected = st.sidebar.selectbox('Período ⏰', ['diário','semanal','trimestral','semestral','mensal','anual'])
period_dict = {'diário':'1d','semanal':'1w','mensal':'1mo','trimestral':'3mo','semestral':'6mo','anual':'1y'}
data_inicio = st.sidebar.date_input("Data Inicial📅", datetime.date(2024,1,1),min_value=datetime.date(2000,1,1))
interval_selected = st.sidebar.selectbox('Intervalo 📊', ['dia','3 meses','mês','semana','hora','minuto'])
interval_dict={'dia':'1d','3 meses':'3mo', 'mês':'1mo','hora':'1h','minuto':'1m','semana':'1wk'}


try:
    data = ticker.history(start=data_inicio, end=datetime.datetime.now(),period=period_dict[period_selected]\
                          ,interval=interval_dict[interval_selected],rounding=True)
    data = data.Close
    st.write(data)
    returns= data.pct_change()
    returns = returns.dropna()*100
    returns_percentage = np.round(returns,2)
    returns_string = returns_percentage.astype(str)+'%'
    st.write(returns_string)
    prices = data.plot()
    dictionary = dict(ticker.tickers)
    lista=list(dictionary.keys())
    factors=[]
    for _ in lista:
        factors.append(yf.Ticker(_).info['longBusinessSummary'])
    df = pd.DataFrame(factors, index=lista)
    df = df.set_axis(["Descrição"], axis=1)
    st.subheader("Descrição da Empresa")
    st.table(df)
except:
     pass










