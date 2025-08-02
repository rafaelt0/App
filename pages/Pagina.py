import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import warnings
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
import quantstats as qs
import tempfile

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Análise de Portfólio - B3 Explorer")

# --- Função para filtrar tickers que têm dados ---
def filtrar_tickers_com_dados(tickers, start_date, end_date):
    tickers_validos = []
    for ticker in tickers:
        data = yf.download(ticker + ".SA", start=start_date, end=end_date, progress=False)
        if not data.empty and 'Close' in data.columns:
            tickers_validos.append(ticker)
    return tickers_validos

# --- Sidebar ---
st.sidebar.header('Configurações ⚙️')

period_selected = st.sidebar.selectbox('Período ⏰', ['diário','semanal','trimestral','semestral','mensal','anual'])
period_dict = {'diário':'1d','semanal':'1wk','mensal':'1mo','trimestral':'3mo','semestral':'6mo','anual':'1y'}

data_inicio = st.sidebar.date_input("Data Inicial📅", datetime.date(2014,1,1), min_value=datetime.date(2000,1,1))
interval_selected = st.sidebar.selectbox('Intervalo 📊', ['dia','mês','3 meses','semana','hora','minuto'])
interval_dict = {'dia':'1d','3 meses':'3mo', 'mês':'1mo','hora':'1h','minuto':'1m','semana':'1wk'}

valor_inicial = st.sidebar.number_input("Valor Investido 💵", min_value=100, max_value=1_000_000)
taxa_selic = st.sidebar.number_input("Taxa Selic 🪙 (%)", value=0.04, max_value=15.0)

# --- Carrega tickers ---
data = pd.read_csv('acoes-listadas-b3.csv')
tickers = list(data['Ticker'].values)

# --- Filtra os tickers válidos (pode demorar) ---
st.info('Filtrando ações com dados disponíveis, aguarde...')
tickers_validos = filtrar_tickers_com_dados(tickers, data_inicio, datetime.date.today())

# --- Seleção dos tickers ---
tickers_selecionaveis = st.multiselect('Monte seu Portfolio (ações com dados)', tickers_validos)

if not tickers_selecionaveis:
    st.warning("Selecione pelo menos uma ação válida para continuar.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers_selecionaveis]

try:
    # Baixa dados
    raw_data = yf.download(
        tickers_yf,
        start=data_inicio,
        end=datetime.datetime.now(),
        interval=interval_dict[interval_selected],
        progress=False
    )

    if raw_data.empty or 'Close' not in raw_data:
        st.error("Nenhum dado encontrado para os tickers selecionados.")
        st.stop()

    data_close = raw_data['Close']

    # Remove multiindex se existir
    if isinstance(data_close.columns, pd.MultiIndex):
        data_close.columns = ['_'.join(col).strip() for col in data_close.columns.values]

    st.subheader("Histórico de Fechamento")
    st.dataframe(data_close.tail())

    returns = data_close.pct_change().dropna()

    # Matriz de correlação
    st.subheader("Matriz de Correlação entre Ativos")
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Otimização Portfólio
    mu = mean_historical_return(data_close)
    S = CovarianceShrinkage(data_close).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    weights = ef.max_sharpe(risk_free_rate=taxa_selic/100)
    weights_df = pd.DataFrame(ef.clean_weights(), index=["Peso"]).T
    weights_df = weights_df.round(4)

    st.subheader("Pesos Ótimos do Portfólio (%)")
    st.dataframe(weights_df*100)

    fig_pie = px.pie(weights_df, values="Peso", names=weights_df.index, title="Composição do Portfólio")
    st.plotly_chart(fig_pie)

    weights_array = weights_df.values.flatten()
    portfolio_returns = returns.dot(weights_array)
    portfolio_returns.name = "Portfolio"
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    # Evolução do Portfolio
    st.subheader("Evolução do Valor do Portfólio")
    fig_val = px.line(portfolio_value, title="Valor do Portfólio")
    st.plotly_chart(fig_val)

    # Informações do portfólio
    portfolio_info = pd.DataFrame({
        "Valor Inicial": [valor_inicial],
        "Valor Máximo": [portfolio_value.max()],
        "Valor Mínimo": [portfolio_value.min()],
        "Valor Final": [portfolio_value.iloc[-1]],
        "Retorno Total (%)": [(portfolio_value.iloc[-1]/valor_inicial - 1)*100],
        "Retorno Médio Diário (%)": [portfolio_returns.mean()*100],
        "Volatilidade Diária (%)": [portfolio_returns.std()*100]
    })
    st.subheader("Informações do Portfólio")
    st.dataframe(portfolio_info.style.format("{:,.2f}"))

    # Distribuição dos Retornos com Estatísticas
    st.subheader("Distribuição dos Retornos Diários (%) e Estatísticas")
    fig_hist, ax_hist = plt.subplots(figsize=(10,5))
    sns.histplot(portfolio_returns*100, bins=50, kde=True, color='skyblue', ax=ax_hist)
    ax_hist.set_xlabel("Retornos Diários (%)")
    ax_hist.set_ylabel("Frequência")

    from scipy.stats import kurtosis, skew
    media = portfolio_returns.mean()*100
    desvio = portfolio_returns.std()*100
    curtose_val = kurtosis(portfolio_returns, fisher=True)
    assimetria_val = skew(portfolio_returns)

    stats_text = (f"Média: {media:.4f}%\n"
                  f"Desvio Padrão: {desvio:.4f}%\n"
                  f"Curtose (Fisher): {curtose_val:.4f}\n"
                  f"Assimetria: {assimetria_val:.4f}")

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    st.pyplot(fig_hist)

    # Estatísticas do Portfólio
    stats = pd.DataFrame([[ 
        sharpe(portfolio_returns, rf=taxa_selic/100)/np.sqrt(2),
        sortino(portfolio_returns, rf=taxa_selic/100),
        max_drawdown(portfolio_returns),
        var(portfolio_returns),
        cvar(portfolio_returns),
        tail_ratio(portfolio_returns)
    ]], columns=["Índice Sharpe", "Índice Sortino", "Max Drawdown", "VaR", "CVaR", "Tail Ratio"])

    st.subheader("Estatísticas do Portfólio")
    st.dataframe(stats)

    # Relatório PDF com QuantStats
    st.subheader("Relatório Completo do Portfolio (PDF)")
    if st.button("Gerar e baixar relatório PDF"):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmpfile:
            qs.reports.snapshot(portfolio_returns, output=tmpfile.name, benchmark=None, title="Relatório de Portfolio")
            tmpfile.seek(0)
            pdf_data = tmpfile.read()
            st.download_button(label="Download Relatório PDF", data=pdf_data, file_name="relatorio_portfolio.pdf", mime="application/pdf")

except Exception as e:
    st.error(f"Erro durante execução: {e}")







