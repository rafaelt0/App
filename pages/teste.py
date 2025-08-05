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
import matplotlib.ticker as mtick
import tempfile

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Análise e Otimização de Portfólio - B3 Explorer")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Configurações do Portfólio")

data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
valor_inicial = st.sidebar.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)
taxa_selic = st.sidebar.number_input("Taxa Selic Anual (%)", value=5.56, max_value=15.0)
benchmark_opcao = st.sidebar.selectbox("Selecione seu Benchmark", ["SELIC", "CDI", "IBOVESPA"])

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.multiselect("Selecione as ações do portfólio", stocks)

if len(tickers) < 2:
    st.warning("Selecione pelo menos dois ativos para montar o portfólio.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

# ---------------------------
# Função para carregar benchmark
# ---------------------------
def get_benchmark(benchmark_opcao, data_inicio, taxa_selic):
    if benchmark_opcao == "IBOVESPA":
        bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close'].pct_change().dropna()
    else:
        # Calcula taxa diária
        if benchmark_opcao == "SELIC":
            taxa_diaria = (1 + taxa_selic/100)**(1/252) - 1
        else:  # CDI - assumindo igual à Selic para simplificação
            taxa_diaria = (1 + taxa_selic/100)**(1/252) - 1
        datas = pd.date_range(data_inicio, datetime.date.today(), freq='B')
        bench = pd.Series(taxa_diaria, index=datas, name=benchmark_opcao)
    return bench

# ---------------------------
# Baixar dados das ações
# ---------------------------
data_yf = yf.download(tickers_yf, start=data_inicio, progress=False)['Close']
if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

returns = data_yf.pct_change().dropna()

# ---------------------------
# Modo de alocação
# ---------------------------
modo = st.sidebar.radio("Modo de alocação", ("Otimização Hierarchical Risk Parity (HRP)", "Alocação Manual"))

if modo == "Alocação Manual":
    st.subheader("Defina manualmente a porcentagem de cada ativo (soma deve ser 100%)")
    pesos_manuais = {}
    total = 0.0
    for ticker in tickers:
        p = st.number_input(f"Peso % de {ticker}", min_value=0.0, max_value=100.0, value=round(100/len(tickers),2), step=0.01)
        pesos_manuais[ticker + ".SA"] = p / 100
        total += p
    if abs(total - 100) > 0.01:
        st.error(f"A soma dos pesos é {total:.2f}%, deve ser 100%")
        st.stop()
    pesos_array = np.array(list(pesos_manuais.values()))
    peso_df = pd.DataFrame.from_dict(pesos_manuais, orient='index', columns=["Peso"])
else:
    st.subheader("Otimização Hierarchical Risk Parity (HRP)")
    hrp = HRPOpt(returns)
    weights_hrp = hrp.optimize()
    peso_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
    pesos_array = peso_df["Peso"].values

# ---------------------------
# Exibir pesos
# ---------------------------
st.subheader("Pesos do Portfólio (%)")
peso_df.index = peso_df.index.str.replace(".SA","")
st.dataframe((peso_df*100).round(2).T)

# Gráfico Pizza
fig_pie = px.pie(peso_df.reset_index(), values="Peso", names="index", title="Composição do Portfólio (%)")
st.plotly_chart(fig_pie)

# Treemap
alloc_df = peso_df.reset_index()
alloc_df.columns = ["Ativo", "Peso"]
fig_treemap = px.treemap(alloc_df, path=['Ativo'], values='Peso', color='Peso', color_continuous_scale='Blues')
st.plotly_chart(fig_treemap, use_container_width=True)

# Heatmap de Correlação
st.subheader("Matriz de Correlação e Heatmap")
corr = returns.corr()
st.dataframe(corr)
heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(heatmap.figure)

# ---------------------------
# Cálculo do portfólio
# ---------------------------
portfolio_returns = returns.dot(pesos_array)
cum_return = (1 + portfolio_returns).cumprod()
portfolio_value = cum_return * valor_inicial

# ---------------------------
# Benchmark dinâmico
# ---------------------------
bench_returns = get_benchmark(benchmark_opcao, data_inicio, taxa_selic)
portfolio_returns = portfolio_returns.loc[bench_returns.index.intersection(portfolio_returns.index)]
bench_returns = bench_returns.loc[portfolio_returns.index]

# Valores acumulados
ret_cum_port = (1 + portfolio_returns).cumprod()
ret_cum_bench = (1 + bench_returns).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=ret_cum_port.index, y=ret_cum_port*valor_inicial, name='Portfólio'))
fig.add_trace(go.Scatter(x=ret_cum_bench.index, y=ret_cum_bench*valor_inicial, name=benchmark_opcao))
fig.update_layout(title='Valor do Portfólio vs Benchmark', xaxis_title='Data', yaxis_title='Valor (R$)')
st.plotly_chart(fig)

# ---------------------------
# Estatísticas básicas
# ---------------------------
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

# ---------------------------
# Sharpe Móvel Corrigido
# ---------------------------
window = 60
rf_daily = (taxa_selic/100)/252
rolling_sharpe = (portfolio_returns.rolling(window).mean() - rf_daily) / portfolio_returns.rolling(window).std()

st.subheader(f"Índice de Sharpe Móvel ({window} dias)")
fig_sharpe, ax_sharpe = plt.subplots(figsize=(10,4))
ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe.values, color='green')
ax_sharpe.axhline(0, color='gray', linestyle='--', alpha=0.7)
st.pyplot(fig_sharpe)

# ---------------------------
# Relatório QuantStats
# ---------------------------
st.subheader("Baixar Relatório Completo (QuantStats)")
portfolio_returns.index = pd.to_datetime(portfolio_returns.index).tz_localize(None)
bench_returns.index = pd.to_datetime(bench_returns.index).tz_localize(None)

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
    qs.reports.html(
        portfolio_returns,
        benchmark=bench_returns,
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

