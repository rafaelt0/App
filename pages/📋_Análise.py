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

st.sidebar.header('Configurações ⚙️')
lista=list(np.arange(2024,2000,-1))
lista.append("None")
period_selected = st.sidebar.selectbox('Período ⏰', ['diário','semanal','trimestral','semestral','mensal','anual'])
period_dict = {'diário':'1d','semanal':'1w','mensal':'1mo','trimestral':'3mo','semestral':'6mo','anual':'1y'}
data_inicio = st.sidebar.date_input("Data Inicial📅", datetime.date(2024,1,1),min_value=datetime.date(2000,1,1))
interval_selected = st.sidebar.selectbox('Intervalo 📊', ['dia','3 meses','mês','semana','hora','minuto'])
interval_dict={'dia':'1d','3 meses':'3mo', 'mês':'1mo','hora':'1h','minuto':'1m','semana':'1wk'}

st.subheader("Análise")
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")

with col2:
    st.image('OIG2.jpeg', width=(400))

with col3:
    st.write("")

data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = list(st.multiselect('Escolha ações para analisar (2 ou mais).', stocks))
               
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
    prices = data.plot()
    dictionary = dict(ticker.tickers)
    lista=list(dictionary.keys())
    st.subheader("Preços")
    st.pyplot(prices.figure)
    returns_plot = returns.plot()
    st.subheader("Retornos")
    st.pyplot(returns_plot.figure)
    heatmap=sns.heatmap(data.corr(), annot=True)
    st.subheader("Matrix de Correlação")
    st.write(data.corr())
    st.subheader("Matrix de Covariância")
    st.write("OBS:.Variância na diagonal e covariância das demais")
    st.write(data.cov())
    st.subheader("Heatmap")
    st.pyplot(heatmap.figure)
except:
    pass

try:
    fig,ax = plt.subplots()
    histograma=sns.histplot(returns, kde=True, fill=False, element='bars')
    st.subheader("Histograma Retornos")
    st.pyplot(histograma.figure)
    i=0
    curtoses = []
    excess_curtoses = []
    for i in range(len(tickers)):
        curtoses.append(kurtosis(np.array(returns.iloc[:,i])))
        excess_curtoses.append(kurtosis(np.array(returns.iloc[:,i]))-3)
    curtoses = np.array(curtoses).reshape(1,2)
    excess_curtoses = np.array(excess_curtoses).reshape(1,2)
    statistics = returns.mean(), returns.median(), returns.std(), returns.max(), returns.min()
    curtoses = pd.DataFrame(curtoses)
    excess_curtoses = pd.DataFrame(excess_curtoses)
    
    curtoses = curtoses.set_axis(tickers, axis=1)
    excess_curtoses = excess_curtoses.set_axis(tickers, axis=1)
    excess_curtoses = excess_curtoses.set_axis(['Excesso de Curtose'], axis=0)
    curtoses = curtoses.set_axis(['Curtose'], axis=0)
    st.subheader("Curtose")
    st.write(curtoses, excess_curtoses)
    st.write("""
                Definição Curtose/Excesso de Curtose:
                1. **Curtose**: Medida de forma que caracteriza o achatamento da curva da função de distribuição de probabilidade 
                2. **Excesso de Curtose** = Curtose - 3(Curtose de uma Normal)
                ###### Categorias 
                * Excesso de Curtose > 0: **Leptocúrtica**, ie. a distribuição apresenta caudas pesadas
                * Excesso de Curtose = 0: **Mesocúrtica**, ie. distribuição Normal
                * Excesso de Curtose < 0: **Platicúrtica**, ie. a distribuição é mais achatada que a Normal
                 """)
except:
    pass



try:
    skewness = []
    for i in range(len(tickers)):
        skewness.append(skew(np.array(returns.iloc[:,i])))
    skewness=np.array(skewness).reshape(1,2)
    skewness=pd.DataFrame(skewness)
    skewness=skewness.set_axis(tickers, axis=1)
    skewness=skewness.set_axis(['Assimetria da Distribuição'], axis=0)
    st.subheader("Assimetria da Distribuição")
    st.write(skewness)
    st.write(
        """
    Definição Assimetria da Distribuição:
    1. Assimetria é uma medida de falta de \
        simetria de uma determinada distribuição de frequência.\
        Mede a asssimetria das caudas da distribuição.
    ######
    * Se v>0, então a distribuição tem uma cauda direita (valores acima da média) mais pesada
    * Se v<0, então a distribuição tem uma cauda esquerda (valores abaixo da média) mais pesada
    * Se v=0, então a distribuição é aproximadamente simétrica (na terceira potência do desvio em relação à média).
    
"""
    )
except:
    pass
    

try:
    statistics_df = pd.DataFrame(statistics)
    st.subheader("Outras Estatísticas Fundamentais")
    statistics_df=statistics_df.set_axis(['Média','Mediana','Volatilidade','Máximo','Mínimo'], axis=0)
    st.write(statistics_df)
except:
    pass

try:
    fig,ax = plt.subplots()
    cum_returns = (returns + 1).cumprod()
    graph=cum_returns.plot()
    st.subheader("Retornos Acumulados")
    st.pyplot(graph.figure)
except:
     pass
        
