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
    page_title="Análise de Ações B3",
    page_icon="📈"
    )
st.sidebar.success("Select Page")


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.write("""
# **B3 Explorer 📈**
 """)

col1, col2, col3 = st.columns([1,3,1])

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
    st.subheader("Setor")
    st.write(df_basic.drop_duplicates(keep='last'))
    st.subheader("Informações")
    df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 'Valor_de_mercado', 'Data_ult_cot']]
    df_price.columns = ["Cotação", "Máximo (52 semanas)", "Mínimo (52 semanas)", "Volume Médio (2 meses)", "Valor de mercado",
                        "Data última cotação"]
    st.dataframe(df_price.drop_duplicates(keep='last'))
    st.subheader("Indicadores")
    df_indicadores = df[['Marg_Liquida','Marg_EBIT','ROE', 'ROIC', 'Div_Yield', 'Cres_Rec_5a', 'PL', 'EV_EBITDA']]
    df_indicadores.columns = ["Margem Líquida", "EBIT", "ROE", "ROIC", "Dividend Yield", "Crescimento Receita 5 anos", 
                              "P/L","EBITDA"]
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
    data = yd.download(tickers, start=data_inicio, end=datetime.datetime.now())
    data = data['Close']
    data = data.reset_index()
    st.subheader("Cotação")
    st.write(data)
    returns= data.pct_change()
    returns = returns.dropna()*100
    returns_percentage = np.round(returns,2)
    returns_string = returns_percentage.astype(str)+'%'
    st.subheader("Retornos")
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











