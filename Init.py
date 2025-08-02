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
import re
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def clean_numeric_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r'[^0-9,.\-]', '', regex=True)
    col = col.str.replace(',', '.')
    return pd.to_numeric(col, errors='coerce')

st.set_page_config(
    page_title="Análise de Ações B3",
    page_icon="📈",
    layout="wide"
)
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
        df = pd.concat([fundamentus.get_papel(t) for t in tickers])
        df['PL'] = clean_numeric_column(df['PL'])

        st.subheader("Setor")
        st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

        st.subheader("Informações de Mercado")
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

        st.subheader("Indicadores Financeiros")
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA','Empresa']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA", "Empresa"]

        for col in df_ind.columns.drop('Empresa'):
            df_ind[col] = clean_numeric_column(df_ind[col])

        pct_cols = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC", "Dividend Yield", "Crescimento Receita 5 anos"]
        for col in pct_cols:
            df_ind[col] = df_ind[col]

        df_ind = df_ind.fillna(0)

        format_ind = {
            "Margem Líquida": "{:.2f}%",
            "Margem EBIT": "{:.2f}%",
            "ROE": "{:.2f}%",
            "ROIC": "{:.2f}%",
            "Dividend Yield": "{:.2f}%",
            "Crescimento Receita 5 anos": "{:.2f}%",
            "P/L": "{:.2f}",
            "EV/EBITDA": "{:.2f}",
            "Empresa": lambda x: x
        }

        # --- FILTROS PERSONALIZADOS ---

        st.sidebar.subheader("Filtros Personalizados")

        min_ebit = st.sidebar.number_input("Margem EBIT mínima (%)", value=0.0, step=0.1)
        min_roe = st.sidebar.number_input("ROE mínimo (%)", value=0.0, step=0.1)
        min_dividend = st.sidebar.number_input("Dividend Yield mínimo (%)", value=0.0, step=0.1)
        max_pl = st.sidebar.number_input("P/L máximo", value=1000.0, step=0.1)

        def highlight_val(val, min_val=None, max_val=None):
            if pd.isna(val):
                return ''
            if min_val is not None and val < min_val:
                return 'background-color: #fbb4ae; color: red;'  # vermelho claro
            if max_val is not None and val > max_val:
                return 'background-color: #fbb4ae; color: red;'
            return 'background-color: #b6d7a8; color: green;'  # verde claro

        def style_indicators(row):
            styles = [''] * len(row)
            col_idx = {col: i for i, col in enumerate(row.index)}

            styles[col_idx['Margem EBIT']] = highlight_val(row['Margem EBIT'], min_val=min_ebit)
            styles[col_idx['ROE']] = highlight_val(row['ROE'], min_val=min_roe)
            styles[col_idx['Dividend Yield']] = highlight_val(row['Dividend Yield'], min_val=min_dividend)
            styles[col_idx['P/L']] = highlight_val(row['P/L'], max_val=max_pl)

            return styles

        styled_ind = df_ind.style.format(format_ind).apply(style_indicators, axis=1)

        st.dataframe(styled_ind, use_container_width=True)

        # Continuação do código com gráficos e estatísticas descritivas

        tickers_yf = [t + ".SA" for t in tickers]
        data_inicio = st.sidebar.date_input("Data Inicial 📅", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1),
                                            max_value=datetime.date.today())

        st.sidebar.header('Configurações ⚙️')
        interval_selected = st.sidebar.selectbox('Intervalo 📊', 
                                                 ['1d','1wk','1mo','3mo','6mo','1y'])

        data_prices = yf.download(tickers_yf, start=data_inicio, end=datetime.datetime.now(), 
                                  interval=interval_selected)['Close']

        if isinstance(data_prices.columns, pd.MultiIndex):
            data_prices = data_prices.droplevel(0, axis=1)

        st.subheader("Cotação Histórica")
        st.line_chart(data_prices)

        returns = data_prices.pct_change().dropna() * 100

        returns_pct = returns.round(2).astype(str) + '%'
        st.subheader("Retornos (%)")
        st.dataframe(returns_pct)

        st.subheader("Histograma Combinado dos Retornos Diários (%)")
        fig_hist_all = px.histogram(
            returns.melt(var_name='Ação', value_name='Retorno (%)'),
            x='Retorno (%)',
            color='Ação',
            barmode='overlay',
            nbins=100,
            opacity=0.6,
            title='Distribuição dos Retornos Diários (%) - Todas as Ações'
        )
        fig_hist_all.update_layout(height=450)
        st.plotly_chart(fig_hist_all, use_container_width=True)

        st.subheader("Estatísticas Descritivas dos Retornos (%)")
        stats_df = pd.DataFrame(index=returns.columns)
        stats_df['Média (%)'] = returns.mean().round(3)
        stats_df['Mediana (%)'] = returns.median().round(3)
        stats_df['Desvio Padrão (%)'] = returns.std().round(3)
        stats_df['Curtose'] = returns.apply(lambda x: kurtosis(x, fisher=True)).round(3)
        stats_df['Assimetria (Skew)'] = returns.apply(lambda x: skew(x)).round(3)
        stats_df['Mínimo (%)'] = returns.min().round(3)
        stats_df['Máximo (%)'] = returns.max().round(3)

        st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)

        quartis_df = pd.DataFrame(index=returns.columns)
        quartis_df['Q1'] = returns.quantile(0.25).round(4)
        quartis_df['Mediana (Q2)'] = returns.quantile(0.5).round(4)
        quartis_df['Q3'] = returns.quantile(0.75).round(4)
        quartis_df['IQR (Q3 - Q1)'] = (quartis_df['Q3'] - quartis_df['Q1']).round(4)
        quartis_df['Limite Inferior'] = (quartis_df['Q1'] - 1.5 * quartis_df['IQR (Q3 - Q1)']).round(4)
        quartis_df['Limite Superior'] = (quartis_df['Q3'] + 1.5 * quartis_df['IQR (Q3 - Q1)']).round(4)

        st.subheader("Tabela dos Quartis, IQR e Limites dos Retornos Diários (%)")
        st.dataframe(quartis_df, use_container_width=True)

        st.subheader("Boxplot dos Retornos Diários (%) por Ação")
        fig_box = px.box(
            returns.melt(var_name='Ação', value_name='Retorno (%)'),
            x='Ação',
            y='Retorno (%)',
            points="outliers",
            title="Distribuição dos Retornos Diários (%)"
        )
        fig_box.update_layout(height=450)
        st.plotly_chart(fig_box, use_container_width=True)

        if not df_ind.empty and len(df_ind) > 1:
            st.subheader("Comparação Radar dos Indicadores Fundamentalistas")

            indicadores_radar = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC", "Dividend Yield", "Crescimento Receita 5 anos"]

            fig = go.Figure()

            for idx, row in df_ind.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row[indicadores_radar].values,
                    theta=indicadores_radar,
                    fill='toself',
                    name=row['Empresa']
                ))

            max_val = max(df_ind[indicadores_radar].max().max(), 100)
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_val]
                    )),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

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
