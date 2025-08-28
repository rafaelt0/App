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
import quantstats as qs
from scipy.stats import kurtosis, skew
from bcb import sgs
import matplotlib.ticker as mtick
import base64
import time

# ------------------------
# Configurações iniciais
# ------------------------
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Análise de Portfólio", layout="wide")

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Título
st.markdown("<h1 style='text-align: center;'>Análise e Otimização de Portfólio 💼</h1>", unsafe_allow_html=True)

# Imagem centralizada
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file("portfolio.png")
st.markdown(f"""
<div style="text-align: center;">
    <img src="data:image/png;base64,{img_base64}" width="250">
</div>""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# ------------------------
# Seleção de parâmetros
# ------------------------
data_inicio = st.date_input("Data Inicial 🗓️", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))

# Selic diária
taxa_selic = sgs.get(432, start=data_inicio).iloc[-1, 0] / 100
taxa_selic = (1 + taxa_selic)**(1/252) - 1

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.multiselect("Selecione as ações do portfólio 📊", stocks)

valor_inicial = st.number_input("Valor Investido (R$) 💵", 100, 1_000_000, 10_000)
modo = st.radio("Modo de alocação ⚙️", 
                ("Otimização Hierarchical Risk Parity (Machine Learning) 🤖", "Alocação Manual 🔧"))

if len(tickers) < 2:
    st.warning("Selecione pelo menos dois ativos.")
    st.stop()

# ------------------------
# Botão de Carregamento
# ------------------------
if st.button("💡 Carregar Portfólio"):

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f"Carregando Portfólio... {i}%")
        time.sleep(0.01)  # simula processamento

    progress_bar.empty()
    status_text.empty()
    st.success("✅ Portfólio carregado com sucesso!")

    # ------------------------
    # Download de dados
    # ------------------------
    tickers_yf = [t + ".SA" for t in tickers]
    data_yf = yf.download(tickers_yf, start=data_inicio, progress=False)['Close']
    if isinstance(data_yf.columns, pd.MultiIndex):
        data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

    returns = data_yf.pct_change().dropna()

    # ------------------------
    # Alocação
    # ------------------------
    if modo == "Alocação Manual 🔧":
        st.subheader("Defina manualmente os pesos (%)")
        pesos_manuais = {}
        total = 0.0
        for ticker in tickers:
            p = st.number_input(f"Peso % de {ticker}", min_value=0.0, max_value=100.0, 
                                value=round(100/len(tickers),2), step=0.01)
            pesos_manuais[ticker + ".SA"] = p / 100
            total += p
        if abs(total - 100) > 0.01:
            st.error(f"A soma dos pesos é {total:.2f}%, deve ser 100%")
            st.stop()
        pesos_arr = np.array(list(pesos_manuais.values()))
        peso_df = pd.DataFrame.from_dict(pesos_manuais, orient='index', columns=["Peso"])
    else:
        st.subheader("Otimização Hierarchical Risk Parity (HRP)")
        hrp = HRPOpt(returns)
        weights_hrp = hrp.optimize()
        peso_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
        pesos_arr = peso_df["Peso"].values

    # Exibe pesos
    peso_df.index = peso_df.index.str.replace(".SA","")
    st.subheader("Pesos do Portfólio (%)")
    st.dataframe((peso_df*100).round(2).T)

    # ------------------------
    # Gráficos de alocação
    # ------------------------
    alloc_df = peso_df.reset_index()
    alloc_df.columns = ["Ativo", "Peso"]

    fig_pie = px.pie(alloc_df, values="Peso", names="Ativo", title="Composição do Portfólio (%)",
                     labels={"Ativo": "Ativo", "Peso": "Percentual"})
    st.plotly_chart(fig_pie)

    fig_treemap = px.treemap(alloc_df, path=['Ativo'], values='Peso', color='Peso',
                             color_continuous_scale='Blues', title="Alocação do Portfólio (Treemap)")
    st.plotly_chart(fig_treemap, use_container_width=True)

    # ------------------------
    # Correlação
    # ------------------------
    st.subheader("Matriz de Correlação")
    st.write(data_yf.corr())

    fig_heat, ax_heat = plt.subplots(figsize=(8,6))
    sns.heatmap(data_yf.corr(), annot=True, cmap="Blues", ax=ax_heat)
    st.pyplot(fig_heat)

    # ------------------------
    # Valor do Portfólio
    # ------------------------
    portfolio_returns = returns.dot(pesos_arr)
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close']
    retorno_bench = bench.pct_change().dropna()
    retorno_bench = retorno_bench.loc[portfolio_returns.index]
    bench_value = (1 + retorno_bench).cumprod() * valor_inicial

    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, name='Portfólio'))
    fig_port.add_trace(go.Scatter(x=bench_value.index, y=bench_value, name='IBOVESPA'))
    fig_port.update_layout(title='Valor do Portfólio', xaxis_title='Data', yaxis_title='Valor (R$)')
    st.plotly_chart(fig_port)

    # ------------------------
    # Estatísticas do portfólio
    # ------------------------
    st.subheader("Estatísticas do Portfólio")
    stats = pd.DataFrame([[ 
        sharpe(portfolio_returns, rf=taxa_selic),
        sortino(portfolio_returns, rf=taxa_selic),
        max_drawdown(portfolio_returns),
        var(portfolio_returns),
        cvar(portfolio_returns),
        tail_ratio(portfolio_returns)
    ]], columns=["Sharpe", "Sortino", "Max Drawdown", "VaR", "CVaR", "Tail Ratio"])
    st.dataframe(stats.round(4))

    # ------------------------
    # Drawdown do Portfólio
    # ------------------------
    st.subheader("Drawdown do Portfólio")
    def calcular_drawdown(series):
        cum_returns = (1 + series).cumprod()
        rolling_max = cum_returns.cummax()
        return (cum_returns - rolling_max) / rolling_max
    drawdown = calcular_drawdown(portfolio_returns)

    fig_dd, ax_dd = plt.subplots(figsize=(10,4))
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4)
    ax_dd.set_title("Drawdown do Portfólio")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Data")
    ax_dd.grid(True)
    ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    st.pyplot(fig_dd)

    # ------------------------
    # Rolling Beta
    # ------------------------
    window = 60
    rolling_beta = portfolio_returns.rolling(window).cov(retorno_bench) / retorno_bench.rolling(window).var()

    st.subheader(f"Beta Móvel ({window} dias) vs IBOVESPA")
    fig_beta, ax_beta = plt.subplots(figsize=(10,4))
    ax_beta.plot(rolling_beta.index, rolling_beta.values, color='blue')
    ax_beta.axhline(1, color='gray', linestyle='--', alpha=0.7)
    ax_beta.set_title(f"Rolling Beta {window} dias")
    ax_beta.set_ylabel("Beta")
    ax_beta.set_xlabel("Data")
    ax_beta.grid(True)
    st.pyplot(fig_beta)

    # ------------------------
    # Contribuição de Risco
    # ------------------------
    cov_matrix = returns.cov()
    port_vol = np.sqrt(np.dot(pesos_arr.T, np.dot(cov_matrix, pesos_arr)))
    marginal_contrib = np.dot(cov_matrix, pesos_arr) / port_vol
    risk_contribution = pesos_arr * marginal_contrib
    risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100

    risk_df = pd.DataFrame({
        "Ativo": peso_df.index,
        "Peso (%)": (pesos_arr*100).round(2),
        "Contribuição de Risco (%)": risk_contribution_pct.round(2)
    })
    st.subheader("Contribuição de Risco por Ativo (%)")
    st.dataframe(risk_df)

    fig_rc = px.bar(risk_df, x="Ativo", y="Contribuição de Risco (%)", color="Contribuição de Risco (%)",
                    color_continuous_scale="Reds", title="Contribuição de Risco por Ativo (%)")
    st.plotly_chart(fig_rc, use_container_width=True)

    # ------------------------
    # Salva variáveis para uso posterior
    # ------------------------
    st.session_state["modo"] = modo
    st.session_state["returns"] = returns
    st.session_state["peso_manual_df"] = peso_df
    st.session_state["portfolio_returns"] = portfolio_returns
    st.session_state["retorno_bench"] = retorno_bench
    st.session_state["pesos_manuais"] = pesos_arr if modo!="Alocação Manual 🔧" else pesos_manuais









       










   
