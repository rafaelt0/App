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

# Extraindo setores únicos para filtro
setores_unicos = sorted(data['Setor'].dropna().unique())

st.sidebar.header("Filtro por Setor")
setores_selecionados = st.sidebar.multiselect("Selecione setores:", setores_unicos, default=setores_unicos)

# Filtra ações pelo setor escolhido
filtro_acoes = data[data['Setor'].isin(setores_selecionados)]
stocks_filtrados = list(filtro_acoes['Ticker'].values)

st.subheader("Explore ações da B3 🧭")
tickers = st.multiselect('Escolha ações para explorar! (2 ou mais ações)', stocks_filtrados)

if tickers:
    if len(tickers) < 2:
        st.warning("Selecione pelo menos 2 ações para análise.")
    else:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # 1. Dados Fundamentus
            progress_text.text("Carregando dados fundamentalistas...")
            df_list = []
            total = len(tickers)
            for i, t in enumerate(tickers):
                df_list.append(fundamentus.get_papel(t))
                progress_bar.progress(int((i + 1) / total * 40))  # até 40%

            df = pd.concat(df_list)
            df['PL'] = pd.to_numeric(df['PL'], errors='coerce') / 100

            # Mostra dados fundamentalistas
            st.subheader("Setor")
            st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

            st.subheader("Informações de Mercado")
            df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m',
                           'Valor_de_mercado', 'Data_ult_cot']]
            df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                                "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

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
            df_ind = df[['Marg_Liquida', 'Marg_EBIT', 'ROE', 'ROIC', 'Div_Yield',
                         'Cres_Rec_5a', 'PL', 'EV_EBITDA']].drop_duplicates(keep='last')
            df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                              "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA"]
            st.dataframe(df_ind, use_container_width=True)

            # 2. Dados yfinance
            progress_text.text("Baixando cotações históricas...")
            tickers_yf = [t + ".SA" for t in tickers]
            data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025, 1, 1),
                                                min_value=datetime.date(2000, 1, 1))
            interval_selected = st.sidebar.selectbox('Intervalo 📊',
                                                     ['1d', '1wk', '1mo', '3mo', '6mo', '1y'])

            data_fim = datetime.datetime.now() + datetime.timedelta(days=1)
            data_prices = yf.download(tickers_yf, start=data_inicio, end=data_fim,
                                      interval=interval_selected)['Close']
            progress_bar.progress(80)

            if isinstance(data_prices.columns, pd.MultiIndex):
                data_prices = data_prices.droplevel(0, axis=1)

            progress_text.text("Finalizando carregamento dos dados...")
            progress_bar.progress(100)

            # Mostrar dados históricos
            st.subheader("Cotação Histórica")
            st.line_chart(data_prices)

            # Retornos
            returns = data_prices.pct_change().dropna() * 100
            returns_pct = returns.round(2).astype(str) + '%'
            st.subheader("Retornos (%)")
            st.dataframe(returns_pct)

            # Estatísticas dos retornos
            st.subheader("Estatísticas dos Retornos")
            stats_df = pd.DataFrame({
                'Média (%)': returns.mean(),
                'Desvio Padrão (%)': returns.std(),
                'Assimetria': returns.apply(skew),
                'Curtose': returns.apply(kurtosis)
            }).round(4)
            st.dataframe(stats_df)

            # Matriz de Correlação
            st.subheader("Matriz de Correlação dos Retornos")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            progress_text.empty()
            progress_bar.empty()

            # Otimização do portfólio
            if st.checkbox("Otimizar Portfólio"):
                mu = mean_historical_return(data_prices)
                S = CovarianceShrinkage(data_prices).ledoit_wolf()
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                st.write("Pesos Otimizados da Carteira:")
                st.write(cleaned_weights)
                fig2, ax2 = plt.subplots()
                plotting.plot_weights(cleaned_weights, ax=ax2)
                st.pyplot(fig2)

        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            st.error(f"Erro ao buscar dados: {e}")
else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")



