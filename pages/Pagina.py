import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import pandas as pd
import seaborn as sns
import datetime
import warnings
import plotly.express as px
import tempfile
import quantstats as qs

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader("Análise de Portfolio")
col1, col2, col3 = st.columns([1,3,1])

# Sidebar
st.sidebar.header('Configurações ⚙️')
period_selected = st.sidebar.selectbox('Período ⏰', ['diário','semanal','trimestral','semestral','mensal','anual'])
period_dict = {'diário':'1d','semanal':'1wk','mensal':'1mo','trimestral':'3mo','semestral':'6mo','anual':'1y'}

data_inicio = st.sidebar.date_input("Data Inicial📅", datetime.date(2014,1,1),min_value=datetime.date(2000,1,1))
interval_selected = st.sidebar.selectbox('Intervalo 📊', ['dia','mês','3 meses','semana','hora','minuto'])
interval_dict={'dia':'1d','3 meses':'3mo', 'mês':'1mo','hora':'1h','minuto':'1m','semana':'1wk'}

valor_inicial = st.sidebar.number_input("Valor Investido 💵", min_value=100, max_value=1_000_000)
taxa_selic = st.sidebar.number_input("Taxa Selic 🪙 (%)", value=0.04, max_value=15.0)

# Ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = list(st.multiselect('Monte seu Portfolio (Escolha mais de uma ação)', stocks))

if len(tickers) == 0:
    st.warning("Selecione pelo menos uma ação para continuar.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

try:
    raw_data = yf.download(
        tickers_yf,
        start=data_inicio,
        end=datetime.datetime.now(),
        interval=interval_dict[interval_selected]
    )

    if raw_data.empty:
        st.error("Nenhum dado retornado. Mercado fechado ou ticker inválido.")
        st.stop()

    data_close = raw_data["Close"] if "Close" in raw_data else raw_data
    if isinstance(data_close.columns, pd.MultiIndex):
        data_close.columns = ['_'.join(col).strip() for col in data_close.columns.values]

    st.subheader("Histórico de Fechamento")
    st.dataframe(data_close.tail())

    # Retornos
    returns = data_close.pct_change().dropna()

    # Heatmap correlação
    st.subheader("Matriz de Correlação entre Ativos")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Otimização Portfólio
    mu = mean_historical_return(data_close)
    S = CovarianceShrinkage(data_close).ledoit_wolf()
    # === Otimização Portfólio com restrições recomendadas ===
# Pesos entre 2% e 30% para garantir diversificação
    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.3))
    
    # Mantém regularização L2 para suavizar pesos
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    
    # Maximiza índice Sharpe ajustado pelo risco livre
    weights = ef.max_sharpe(risk_free_rate=taxa_selic/100)
    
    # Limpa e organiza pesos em DataFrame
    weights_df = pd.DataFrame(ef.clean_weights(), index=["Peso"]).T
    weights_df = round(weights_df,4)
    
    st.subheader("Pesos Ótimos do Portfólio (%)")
    st.dataframe(weights_df*100)
    
    # Gráfico de pizza dos pesos
    fig_pie = px.pie(weights_df, values="Peso", names=weights_df.index, title="Composição do Portfólio")
    st.plotly_chart(fig_pie)
    
    # Calcula retorno do portfólio
    weights_array = weights_df.values.flatten()
    portfolio_returns = returns.dot(weights_array)
    portfolio_returns.name = "Portfolio"
    
    # Evolução do valor do portfólio
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    # Informações do Portfólio
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

    # === Relatório completo com QuantStats ===
    st.subheader("Baixar Relatório Completo (QuantStats)")

    # Converte para formato aceito pelo QuantStats
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    portfolio_returns = portfolio_returns.tz_localize(None)  # Remove timezone

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        qs.reports.html(
            portfolio_returns,
            output=tmpfile.name,
            title="Relatório Completo do Portfólio",
            download_filename="relatorio_portfolio.html"
        )
        st.download_button(
            label="Baixar Relatório HTML Completo (QuantStats)",
            data=open(tmpfile.name, "rb").read(),
            file_name="relatorio_portfolio.html",
            mime="text/html"
        )

except Exception as e:
    st.error(f"Erro durante execução: {e}")





