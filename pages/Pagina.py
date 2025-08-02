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
from sklearn.ensemble import IsolationForest
from io import BytesIO
from fpdf import FPDF

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio

warnings.filterwarnings('ignore')

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

    if "Close" in raw_data:
        data_close = raw_data["Close"]
    else:
        st.error("Coluna Close não encontrada.")
        st.stop()

    # Flatten colunas se MultiIndex
    if isinstance(data_close.columns, pd.MultiIndex):
        data_close.columns = ['_'.join(col).strip() for col in data_close.columns.values]

    st.subheader("Histórico de Fechamento")
    st.dataframe(data_close.tail())

    returns = data_close.pct_change().dropna()

    # Heatmap correlação
    st.subheader("Matriz de Correlação entre Ativos")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Otimização Portfólio
    mu = mean_historical_return(data_close)
    S = CovarianceShrinkage(data_close).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    weights = ef.max_sharpe(risk_free_rate=taxa_selic/100)
    weights_df = pd.DataFrame(ef.clean_weights(), index=["Peso"]).T
    weights_df = round(weights_df,4)

    st.subheader("Pesos Ótimos do Portfólio (%)")
    st.dataframe(weights_df*100)
    fig_pie = px.pie(weights_df, values="Peso", names=weights_df.index, title="Composição do Portfólio")
    st.plotly_chart(fig_pie)

    weights_array = weights_df.values.flatten()
    portfolio_returns = returns.dot(weights_array)
    portfolio_returns.name = "Portfolio"
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    st.subheader("Evolução do Valor do Portfólio")
    fig_val = px.line(portfolio_value, title="Valor do Portfólio")
    st.plotly_chart(fig_val)

    # Estatísticas
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

    st.subheader("Retornos diários (%)")
    st.line_chart(portfolio_returns*100)

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
    st.dataframe(portfolio_info.style.format("{:.2f}"))

    # === Detecção de Anomalias com IsolationForest ===
    st.subheader("Detecção de Anomalias nos Retornos")

    iso = IsolationForest(contamination=0.01, random_state=42)
    returns_np = portfolio_returns.values.reshape(-1,1)
    iso.fit(returns_np)
    preds = iso.predict(returns_np)  # 1 = normal, -1 = anomalia

    anomalies = portfolio_returns[preds == -1]
    st.write(f"Foram detectados {len(anomalies)} dias com comportamento anômalo nos retornos.")

    if not anomalies.empty:
        st.dataframe(anomalies)

    # === Exportar dados CSV ===
    csv = portfolio_value.to_frame().to_csv().encode('utf-8')
    st.download_button(label="Baixar histórico do valor do portfólio (.csv)", data=csv, file_name='portfolio_value.csv', mime='text/csv')

    # === Gerar relatório PDF simples ===
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Relatório Resumido do Portfólio", 0, 1, 'C')

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Período: {data_inicio} até {datetime.datetime.now().date()}", 0, 1)
        pdf.cell(0, 10, f"Valor Inicial: R$ {valor_inicial:,.2f}", 0, 1)
        pdf.cell(0, 10, f"Valor Final: R$ {portfolio_value.iloc[-1]:,.2f}", 0, 1)
        pdf.cell(0, 10, f"Retorno Total: {portfolio_info['Retorno Total (%)'].values[0]:.2f} %", 0, 1)

        # Estatísticas chave
        pdf.cell(0, 10, f"Índice Sharpe: {stats['Índice Sharpe'].values[0]:.2f}", 0, 1)
        pdf.cell(0, 10, f"Índice Sortino: {stats['Índice Sortino'].values[0]:.2f}", 0, 1)
        pdf.cell(0, 10, f"Max Drawdown: {stats['Max Drawdown'].values[0]:.2f}", 0, 1)

        # Exportar para bytes
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue()

    pdf_bytes = generate_pdf()
    st.download_button(label="Baixar relatório resumido (PDF)", data=pdf_bytes, file_name="relatorio_portfolio.pdf", mime="application/pdf")

except Exception as e:
    st.error(f"Erro durante execução: {e}")


