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

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Configuração da página
st.set_page_config(
    page_title="Análise de Ações B3",
    page_icon="📈",
    layout="wide"
)
st.sidebar.success("Selecione uma página")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("**B3 Explorer 📈**")

# =============================
# Carregando as ações
# =============================
data = pd.read_csv('acoes-listadas-b3.csv')  # deve ter colunas 'Ticker' e 'Setor'

# Filtro por Setor
st.sidebar.header("Filtro por Setor")
setores_unicos = sorted(data['Setor'].dropna().unique())
setores_selecionados = st.sidebar.multiselect("Selecione setores:", setores_unicos, default=setores_unicos)

filtro_acoes = data[data['Setor'].isin(setores_selecionados)]
stocks_filtrados = list(filtro_acoes['Ticker'].values)

st.subheader("Explore ações da B3 🧭")
tickers = st.multiselect('Escolha ações para explorar! (2 ou mais ações)', stocks_filtrados)

if tickers:
    try:
        # =============================
        # Análise Fundamentalista
        # =============================
        df = pd.concat([fundamentus.get_papel(t) for t in tickers])
        df['PL'] = pd.to_numeric(df['PL'], errors='coerce')

        st.subheader("Setor e Subsetor")
        st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

        st.subheader("Informações de Mercado")
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']]
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

        # Converter colunas numéricas
        for col in ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)", 
                    "Volume Médio (2 meses)", "Valor de Mercado"]:
            df_price[col] = pd.to_numeric(df_price[col], errors='coerce')
        
        format_dict = {
            "Cotação": "R$ {:,.2f}",
            "Mínimo (52 semanas)": "R$ {:,.2f}",
            "Máximo (52 semanas)": "R$ {:,.2f}",
            "Volume Médio (2 meses)": "{:,.0f}",
            "Valor de Mercado": "R$ {:,.0f}"
        }

        st.dataframe(df_price.style.format(format_dict), use_container_width=True)

        st.subheader("Indicadores Financeiros")
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA"]
        st.dataframe(df_ind, use_container_width=True)

        # =============================
        # Cotação Histórica e Retornos
        # =============================
        st.sidebar.header('Configurações ⚙️')
        data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1))
        interval_selected = st.sidebar.selectbox('Intervalo 📊', 
                                                 ['1d','5d','1wk','1mo','3mo'])

        tickers_yf = [t + ".SA" for t in tickers]
        data_prices = yf.download(tickers_yf, start=data_inicio, end=datetime.datetime.now(), 
                                  interval=interval_selected)['Close']

        if isinstance(data_prices.columns, pd.MultiIndex):
            data_prices = data_prices.droplevel(0, axis=1)

        st.subheader("Cotação Histórica")
        st.line_chart(data_prices)

        # Retornos
        returns = data_prices.pct_change().dropna() * 100
        st.subheader("Retornos (%)")
        st.dataframe((returns.round(2).astype(str) + '%'))

        # Estatísticas de Retornos
        st.subheader("Estatísticas dos Retornos")
        stats_df = pd.DataFrame({
            'Média (%)': returns.mean(),
            'Desvio Padrão (%)': returns.std(),
            'Assimetria': returns.apply(skew),
            'Curtose': returns.apply(kurtosis)
        }).round(4)
        st.dataframe(stats_df)

        # =============================
        # Histogramas de Retornos
        # =============================
        st.subheader("Distribuição dos Retornos Individuais")
        for col in returns.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(returns[col], bins=30, kde=True, ax=ax, color='royalblue')
            ax.set_title(f"Histograma de Retornos - {col}")
            ax.set_xlabel("Retorno (%)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")

else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")



