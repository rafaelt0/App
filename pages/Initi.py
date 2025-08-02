import streamlit as st
import yfinance as yf
import fundamentus
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

st.set_page_config(
    page_title="Análise de Ações B3",
    page_icon="📈",
    layout="wide"
)
st.sidebar.success("Selecione uma página")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("**B3 Explorer 📈**")

# Carregar lista de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = data[['Ticker','Setor']]

# --- Filtro por setor ---
setores_unicos = stocks['Setor'].dropna().unique()
setor_selecionado = st.sidebar.selectbox("Filtrar por Setor", options=["Todos"] + list(setores_unicos))

if setor_selecionado != "Todos":
    stocks_filtradas = stocks[stocks['Setor'] == setor_selecionado]
else:
    stocks_filtradas = stocks

tickers = st.multiselect(
    'Escolha ações para explorar! (2 ou mais ações)',
    options=stocks_filtradas['Ticker'].tolist()
)

if tickers:
    @st.cache_data(ttl=3600)
    def get_fundamentus_data(tickers):
        dfs = []
        for t in tickers:
            try:
                df_temp = fundamentus.get_papel(t)
                dfs.append(df_temp)
            except Exception:
                pass
        return pd.concat(dfs) if dfs else pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def get_yf_data(tickers_yf, start, end, interval):
        data_prices = yf.download(tickers_yf, start=start, end=end, interval=interval)['Close']
        if isinstance(data_prices.columns, pd.MultiIndex):
            data_prices = data_prices.droplevel(0, axis=1)
        return data_prices

    with st.spinner('Carregando dados fundamentalistas...'):
        df = get_fundamentus_data(tickers)
    
    if df.empty:
        st.warning("Não foi possível carregar dados fundamentus para as ações selecionadas.")
    else:
        # Limpar e preparar dados fundamentalistas
        cols_to_numeric = ['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                           'Cres_Rec_5a','PL','EV_EBITDA','Cotacao','Min_52_sem',
                           'Max_52_sem','Vol_med_2m','Valor_de_mercado']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        st.subheader("Setor e Subsetor")
        st.write(df[['Empresa','Setor','Subsetor']].drop_duplicates())

        # Informações de mercado
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']].drop_duplicates()
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

        # Formatação elegante
        format_dict = {
            "Cotação": "R$ {:,.2f}",
            "Mínimo (52 semanas)": "R$ {:,.2f}",
            "Máximo (52 semanas)": "R$ {:,.2f}",
            "Volume Médio (2 meses)": "{:,.0f}",
            "Valor de Mercado": "R$ {:,.0f}"
        }

        # Layout com colunas
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Informações de Mercado")
            st.dataframe(df_price.style.format(format_dict), use_container_width=True)

        with col2:
            st.subheader("Indicadores Financeiros")
            df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                         'Cres_Rec_5a','PL','EV_EBITDA']].drop_duplicates()
            df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                              "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA"]
            st.dataframe(df_ind, use_container_width=True)

        # Configurações yfinance
        tickers_yf = [t + ".SA" for t in tickers]
        data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1))
        interval_selected = st.sidebar.selectbox('Intervalo 📊', 
                                                 ['1d','1wk','1mo','3mo','6mo','1y'])

        with st.spinner('Carregando cotações históricas...'):
            data_prices = get_yf_data(tickers_yf, data_inicio, datetime.datetime.now(), interval_selected)

        if data_prices.empty:
            st.warning("Não foi possível carregar os dados históricos.")
        else:
            st.subheader("Cotação Histórica")
            st.line_chart(data_prices)

            # Retornos
            returns = data_prices.pct_change().dropna() * 100
            returns_styled = returns.style.format("{:.2f}%")
            st.subheader("Retornos (%)")
            st.dataframe(returns_styled, use_container_width=True)

            # Estatísticas dos retornos
            st.subheader("Estatísticas dos Retornos")
            stats_df = pd.DataFrame({
                'Média (%)': returns.mean(),
                'Desvio Padrão (%)': returns.std(),
                'Assimetria (Skew)': returns.apply(skew),
                'Curtose (Kurtosis)': returns.apply(kurtosis)
            })
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

            # Gráfico de correlação
            st.subheader("Mapa de Correlação dos Retornos")
            corr = returns.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        # Checkbox para descrição
        if st.checkbox("Mostrar descrição das empresas"):
            descriptions = []
            with st.spinner("Carregando descrições..."):
                for t in tickers_yf:
                    try:
                        info = yf.Ticker(t).get_info()
                        descriptions.append(info.get('longBusinessSummary', 'Não disponível'))
                    except:
                        descriptions.append('Não disponível')

            df_desc = pd.DataFrame(descriptions, index=tickers, columns=["Descrição"])
            st.subheader("Descrição da Empresa")
            st.table(df_desc)

else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")


