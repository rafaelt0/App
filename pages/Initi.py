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
from scipy.stats import kurtosis, skew
from pypfopt import plotting

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Configurações da página
st.set_page_config(
    page_title="Análise de Ações B3",
    page_icon="📈",
    layout="wide"
)
st.sidebar.success("Selecione uma página")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("**B3 Explorer 📈**")

# Carregando as ações com setor
data = pd.read_csv('acoes-listadas-b3.csv')

# Verifique se seu CSV tem a coluna 'Setor'
if 'Setor' not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

stocks = list(data['Ticker'].values)

# Lista única de setores para filtro
setores = sorted(data['Setor'].dropna().unique())

# Sidebar: filtro por setor
setores_selecionados = st.sidebar.multiselect('Filtrar por Setor', setores, default=setores)

# Filtrar os tickers por setor selecionado
tickers_filtrados = data[data['Setor'].isin(setores_selecionados)]['Ticker'].tolist()

st.subheader("Explore ações da B3 🧭")
tickers = st.multiselect('Escolha ações para explorar! (2 ou mais ações)', tickers_filtrados)

if tickers:
    try:
        # Análise Fundamentalitsta
        df = pd.concat([fundamentus.get_papel(t) for t in tickers])
        df['PL'] = pd.to_numeric(df['PL'], errors='coerce') / 100

        st.subheader("Setor")
        st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

        st.subheader("Informações de Mercado")
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']]
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]
        # Converter as colunas para numéricas para evitar erro
        for col in ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)", 
                    "Volume Médio (2 meses)", "Valor de Mercado"]:
            df_price[col] = pd.to_numeric(df_price[col], errors='coerce').fillna(0)
        
        format_dict = {
            "Cotação": "R$ {:,.2f}",
            "Mínimo (52 semanas)": "R$ {:,.2f}",
            "Máximo (52 semanas)": "R$ {:,.2f}",
            "Volume Médio (2 meses)": "{:,.0f}",
            "Valor de Mercado": "R$ {:,.0f}"
        }

        st.dataframe(df_price.style.format(format_dict), use_container_width=True)

        # Indicadores Fundamentalistas
        st.subheader("Indicadores Financeiros")
        
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA"]

        format_ind = {
            "Margem Líquida": "{:.2f}%",
            "Margem EBIT": "{:.2f}%",
            "ROE": "{:.2f}%",
            "ROIC": "{:.2f}%",
            "Dividend Yield": "{:.2f}%",
            "Crescimento Receita 5 anos": "{:.2f}%",
            "P/L": "{:.2f}",
            "EV/EBITDA": "{:.2f}"
        }
        
        st.dataframe(df_ind.style.format(format_ind), use_container_width=True)

        # Formato do yfinance
        tickers_yf = [t + ".SA" for t in tickers]
        data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1),
                                            max_value=datetime.date.today())
        
        st.sidebar.header('Configurações ⚙️')
        interval_selected = st.sidebar.selectbox('Intervalo 📊', 
                                                 ['1d','1wk','1mo','3mo','6mo','1y'])
        
        # Carregando os dados
        data_prices = yf.download(tickers_yf, start=data_inicio, end=datetime.datetime.now(), 
                                  interval=interval_selected)['Close']
        
        # Corrigir erro de multi index
        if isinstance(data_prices.columns, pd.MultiIndex):
            data_prices = data_prices.droplevel(0, axis=1)

        st.subheader("Cotação Histórica")
        st.line_chart(data_prices)

        # Retornos
        returns = data_prices.pct_change().dropna() * 100
        returns_pct = returns.round(2).astype(str) + '%'
        st.subheader("Retornos (%)")
        st.dataframe(returns_pct)

        # Descrição Empresas
        descriptions = []
        for t in tickers_yf:
            try:
                info = yf.Ticker(t).get_info()
                descriptions.append(info.get('longBusinessSummary', 'Não disponível'))
            except:
                descriptions.append('Não disponível')

        df_desc = pd.DataFrame(descriptions, index=tickers, columns=["Descrição"])
        st.subheader("Descrição da Empresa")
        st.table(df_desc)

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")


