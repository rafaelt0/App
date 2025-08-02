import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import pandas as pd
import seaborn as sns
import warnings
import datetime
from scipy.stats import kurtosis, skew
import plotly.graph_objects as go
import plotly.express as px
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def clean_numeric_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r'[^0-9,.\-]', '', regex=True)
    col = col.str.replace(',', '.')
    return pd.to_numeric(col, errors='coerce')

@st.cache_data(ttl=3600)
def get_fundamentus_data(tickers):
    return pd.concat([fundamentus.get_papel(t) for t in tickers])

@st.cache_data(ttl=3600)
def get_news_google(query, num_news=5):
    url = f"https://www.google.com/search?q={query}+site:news.google.com&tbm=nws&hl=pt-BR"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    news_items = []
    for g in soup.find_all('div', class_='Gx5Zad fP1Qef xpd EtOod pkphOe', limit=num_news):
        title = g.find('div', class_='BNeawe vvjwJb AP7Wnd')
        link = g.find('a', href=True)
        source_time = g.find('div', class_='BNeawe UPmit AP7Wnd')
        snippet = g.find('div', class_='BNeawe s3v9rd AP7Wnd')
        
        if title and link:
            news_items.append({
                "Título": title.get_text(),
                "Fonte/Tempo": source_time.get_text() if source_time else "",
                "Resumo": snippet.get_text() if snippet else "",
                "Link": link['href']
            })
    return pd.DataFrame(news_items)

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Análise de Ações B3", page_icon="📈", layout="wide")
st.sidebar.success("Selecione uma página")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("**B3 Explorer 📈**")

data = pd.read_csv('acoes-listadas-b3.csv')

if 'Setor' not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

stocks = list(data['Ticker'].values)
setores = sorted(data['Setor'].dropna().unique())
setores.insert(0, "Todos")

setores_selecionados = st.sidebar.multiselect(
    'Escolha um ou mais setores (deixe vazio ou "Todos" para todos):', setores, default=["Todos"]
)

if "Todos" in setores_selecionados or not setores_selecionados:
    tickers_filtrados = data['Ticker'].tolist()
else:
    tickers_filtrados = data[data['Setor'].isin(setores_selecionados)]['Ticker'].tolist()

st.subheader("Explore ações da B3 🧭")
tickers = st.multiselect('Escolha ações para explorar! (2 ou mais ações)', tickers_filtrados)

if tickers:
    try:
        df = get_fundamentus_data(tickers)
        df['PL'] = clean_numeric_column(df['PL'])

        st.subheader("Setor")
        st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

        # --- Informações de Mercado ---
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']]
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

        for col in ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)", 
                    "Volume Médio (2 meses)", "Valor de Mercado"]:
            df_price[col] = clean_numeric_column(df_price[col]).fillna(0)

        format_dict = {
            "Cotação": "R$ {:,.2f}",
            "Mínimo (52 semanas)": "R$ {:,.2f}",
            "Máximo (52 semanas)": "R$ {:,.2f}",
            "Volume Médio (2 meses)": "{:,.0f}",
            "Valor de Mercado": "R$ {:,.0f}"
        }

        st.dataframe(df_price.style.format(format_dict), use_container_width=True)

        # --- Indicadores Financeiros ---
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA','Empresa']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA", "Empresa"]

        for col in df_ind.columns.drop('Empresa'):
            df_ind[col] = clean_numeric_column(df_ind[col])

        df_ind = df_ind.fillna(0)
        st.dataframe(df_ind, use_container_width=True)

        # --- Cotação Histórica ---
        tickers_yf = [t + ".SA" for t in tickers]
        data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1),
                                            max_value=datetime.date.today())

        interval_selected = st.sidebar.selectbox('Intervalo 📊', ['1d','1wk','1mo'])
        period_selected = st.sidebar.selectbox('Período', ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])

        data_prices = yf.download(tickers_yf, period=period_selected, interval=interval_selected)['Close']
        if isinstance(data_prices.columns, pd.MultiIndex):
            data_prices = data_prices.droplevel(0, axis=1)

        st.subheader("Cotação Histórica")
        st.line_chart(data_prices)

        returns = data_prices.pct_change().dropna()
        
        # --- Comparação de Ações ---
        if len(returns.columns) > 1:
            st.subheader("📊 Comparação de Ações (Risco x Retorno)")
            mean_returns = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)

            comp_df = pd.DataFrame({
                'Ação': returns.columns,
                'Retorno Anual (%)': mean_returns*100,
                'Volatilidade Anual (%)': volatility*100
            })

            fig_comp = px.scatter(
                comp_df, x='Volatilidade Anual (%)', y='Retorno Anual (%)',
                text='Ação', size='Retorno Anual (%)', color='Ação',
                title="Comparação de Ações: Risco vs Retorno"
            )
            fig_comp.update_traces(textposition='top center')
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(comp_df.round(2))

        # --- Notícias ---
        st.subheader("📰 Últimas Notícias das Ações Selecionadas")
        for ticker in tickers:
            st.markdown(f"### {ticker}")
            news_df = get_news_google(f"{ticker} B3")
            if not news_df.empty:
                st.dataframe(news_df)
            else:
                st.info("Nenhuma notícia encontrada.")

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")

