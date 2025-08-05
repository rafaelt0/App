import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.hierarchical_portfolio import HRPOpt
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
from scipy.stats import kurtosis, skew
import quantstats as qs
from bcb import sgs
import matplotlib.ticker as mtick
import io
import tempfile

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ===== Sidebar com Branding =====
st.sidebar.image("img/logo.png", use_column_width=True)
st.sidebar.title("B3 Portfolio Explorer")
st.sidebar.header("⚙️ Configurações do Portfólio")

data_inicio = st.sidebar.date_input("📅 Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
valor_inicial = st.sidebar.number_input("💰 Valor Investido (R$)", 100, 1_000_000, 10_000)

# Taxa Selic diária
taxa_selic = sgs.get(432, start=data_inicio)
taxa_selic = taxa_selic.iloc[-1,0]
taxa_selic = (1+taxa_selic)**(1/252)-1

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.sidebar.multiselect("🎯 Selecione as ações do portfólio", stocks)

if len(tickers) < 2:
    st.warning("Selecione pelo menos dois ativos.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]
data_yf = yf.download(tickers_yf, start=data_inicio, progress=False)['Close']
if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

returns = data_yf.pct_change().dropna()

# ===== Escolha de modo =====
modo = st.sidebar.radio("Modo de alocação", ("Otimização HRP", "Alocação Manual"))

if modo == "Alocação Manual":
    st.sidebar.subheader("Defina manualmente os pesos (%)")
    pesos_manuais = {}
    total = 0.0
    for ticker in tickers:
        p = st.sidebar.number_input(f"{ticker}", min_value=0.0, max_value=100.0, value=round(100/len(tickers),2), step=0.01)
        pesos_manuais[ticker + ".SA"] = p / 100
        total += p
    if abs(total - 100) > 0.01:
        st.error(f"A soma dos pesos é {total:.2f}%, deve ser 100%")
        st.stop()
    peso_manual_df = pd.DataFrame.from_dict(pesos_manuais, orient='index', columns=["Peso"])
    pesos_manuais_arr = peso_manual_df["Peso"].values
else:
    hrp = HRPOpt(returns)
    weights_hrp = hrp.optimize()
    peso_manual_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
    pesos_manuais_arr = peso_manual_df["Peso"].values

# Ajustar índices para exibição
peso_manual_df.index = peso_manual_df.index.str.replace(".SA","")

# ===== Layout em Abas =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumo", 
    "⚡ Risco & Retorno", 
    "🔎 Análises Avançadas", 
    "📥 Relatórios"
])

# ===== Aba 1 - Resumo =====
with tab1:
    st.subheader("Resumo do Portfólio")
    st.image("img/divider.png", use_column_width=True)

    st.dataframe((peso_manual_df*100).round(2).T)

    fig_pie = px.pie(peso_manual_df.reset_index(), values="Peso", names="index",
                     title="Composição do Portfólio (%)", hole=0.3,
                     color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig_pie, use_container_width=True)

    alloc_df = peso_manual_df.reset_index()
    alloc_df.columns = ["Ativo", "Peso"]
    fig_treemap = px.treemap(alloc_df, path=['Ativo'], values='Peso', color='Peso',
                             color_continuous_scale='Blues',
                             title="Alocação do Portfólio (Treemap)")
    st.plotly_chart(fig_treemap, use_container_width=True)

# ===== Cálculos para todas as abas =====
portfolio_returns = returns.dot(pesos_manuais_arr)
cum_return = (1 + portfolio_returns).cumprod()
portfolio_value = cum_return * valor_inicial

bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close']
retorno_bench = bench.pct_change().dropna()
portfolio_returns = portfolio_returns.loc[retorno_bench.index]

# ===== Aba 2 - Risco & Retorno =====
with tab2:
    st.subheader("Risco & Retorno do Portfólio")
    st.image("img/divider.png", use_column_width=True)

    # Valor Portfólio vs IBOV
    bench_value = (1+retorno_bench).cumprod() * valor_inicial
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, 
                             mode='lines', name='Portfólio',
                             line=dict(width=3, color='green')))
    fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, 
                             mode='lines', name='IBOVESPA',
                             line=dict(width=2, dash='dot', color='gray')))
    fig.update_layout(template='plotly_white', title='Valor do Portfólio vs IBOVESPA')
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown do Portfólio")
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown_portfolio = (cum_returns - rolling_max) / rolling_max

    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.fill_between(drawdown_portfolio.index, drawdown_portfolio.values, 0, 
                     color='red', alpha=0.3)
    ax1.set_title("Drawdown do Portfólio", fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# ===== Aba 3 - Análises Avançadas =====
with tab3:
    st.subheader("Análises Avançadas")
    st.image("img/divider.png", use_column_width=True)

    # Heatmap de Correlação
    fig_corr, ax_corr = plt.subplots(figsize=(10,8))
    sns.heatmap(data_yf.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                cbar_kws={"shrink": 0.7}, ax=ax_corr)
    ax_corr.set_title("Correlação entre Ativos (%)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig_corr)

    # Contribuição de Risco
    cov_matrix = returns.cov()
    port_vol = np.sqrt(np.dot(pesos_manuais_arr.T, np.dot(cov_matrix, pesos_manuais_arr)))
    marginal_contrib = np.dot(cov_matrix, pesos_manuais_arr) / port_vol
    risk_contribution = pesos_manuais_arr * marginal_contrib
    risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100

    risk_df = pd.DataFrame({
        "Ativo": peso_manual_df.index,
        "Peso (%)": (pesos_manuais_arr*100).round(2),
        "RC (%)": risk_contribution_pct.round(2)
    })
    fig_rc = px.bar(risk_df, x="Ativo", y="RC (%)", color="RC (%)",
                    color_continuous_scale="Reds", title="Contribuição de Risco por Ativo (%)")
    st.plotly_chart(fig_rc, use_container_width=True)

# ===== Aba 4 - Relatórios =====
with tab4:
    st.subheader("Relatórios e Exportações")
    st.image("img/report_preview.png", caption="Exemplo de Relatório QuantStats", use_column_width=True)

    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    portfolio_returns = portfolio_returns.tz_localize(None)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        qs.reports.html(
            portfolio_returns,
            benchmark= retorno_bench,
            output=tmpfile.name,
            title="Relatório Completo do Portfólio",
            download_filename="relatorio_portfolio.html"
        )
        st.download_button(
            label="📥 Baixar Relatório HTML Completo (QuantStats)",
            data=open(tmpfile.name, "rb").read(),
            file_name="relatorio_portfolio.html",
            mime="text/html"
        )

