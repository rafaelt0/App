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
# Sidebar - configurações
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
def get_benchmark(benchmark_opcao, data_inicio, taxa_selic, last_date):
    if benchmark_opcao == "IBOVESPA":
        bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close'].pct_change().dropna()
    else:
        if benchmark_opcao == "SELIC":
            taxa_diaria = (1 + taxa_selic/100)**(1/252) - 1
        else:  # CDI - simplificado igual SELIC
            taxa_diaria = (1 + taxa_selic/100)**(1/252) - 1
        datas = pd.date_range(data_inicio, last_date, freq='B')
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

# Última data disponível no portfólio
last_date = portfolio_returns.index[-1]

# Obter benchmark com índice alinhado
bench_returns = get_benchmark(benchmark_opcao, data_inicio, taxa_selic, last_date)

# Alinhar índices para evitar erros
common_index = portfolio_returns.index.intersection(bench_returns.index)
portfolio_returns = portfolio_returns.loc[common_index]
bench_returns = bench_returns.loc[common_index]

if portfolio_returns.empty or bench_returns.empty:
    st.error("Dados insuficientes para análise com o benchmark selecionado.")
    st.stop()

# Valores acumulados
ret_cum_port = (1 + portfolio_returns).cumprod()
ret_cum_bench = (1 + bench_returns).cumprod()
portfolio_value = ret_cum_port * valor_inicial
bench_value = ret_cum_bench * valor_inicial

# ---------------------------
# Plot Valor do Portfólio vs Benchmark
# ---------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfólio'))
fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, mode='lines', name=benchmark_opcao))
fig.update_layout(title='Valor do Portfólio vs Benchmark', xaxis_title='Data', yaxis_title='Valor (R$)')
st.plotly_chart(fig)

# ---------------------------
# Estatísticas básicas do portfólio
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
# Distribuição dos retornos
# ---------------------------
st.subheader("Distribuição dos Retornos Diários (%) e Estatísticas")
fig_hist, ax_hist = plt.subplots(figsize=(10,5))
sns.histplot(portfolio_returns*100, bins=50, kde=True, color='skyblue', ax=ax_hist)
ax_hist.set_xlabel("Retornos Diários (%)")
ax_hist.set_ylabel("Frequência")

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

# ---------------------------
# Estatísticas do portfólio (Sharpe, Sortino etc)
# ---------------------------
stats = pd.DataFrame([[
    sharpe(portfolio_returns, rf=taxa_selic/100/252),
    sortino(portfolio_returns, rf=taxa_selic/100/252),
    max_drawdown(portfolio_returns),
    var(portfolio_returns),
    cvar(portfolio_returns),
    tail_ratio(portfolio_returns)
]], columns=["Índice Sharpe", "Índice Sortino", "Max Drawdown", "VaR", "CVaR", "Tail Ratio"])

st.subheader("Estatísticas do Portfólio")
st.dataframe(stats.round(4))

# ---------------------------
# Retorno acumulado vs Benchmark com QuantStats
# ---------------------------
st.subheader("Retorno Acumulado Portfólio vs Benchmark")
fig_qs = qs.plots.returns(portfolio_returns, benchmark=bench_returns, show=False)
st.pyplot(fig_qs)

# ---------------------------
# Métricas Alfa, Beta, R², Information Ratio
# ---------------------------
cov_matrix = np.cov(portfolio_returns, bench_returns)
beta = cov_matrix[0,1] / cov_matrix[1,1]
alfa = portfolio_returns.mean() - beta * bench_returns.mean()
r_quadrado = qs.stats.r_squared(portfolio_returns, bench_returns)
information_ratio = qs.stats.information_ratio(portfolio_returns, bench_returns)

metricas = pd.DataFrame({
    "Alfa Anual (%)": [float(alfa)*252*100],
    "Beta": [beta],
    "R Quadrado (%)": [r_quadrado*100],
    "Information Ratio": [information_ratio]
})

st.subheader("📊 Métricas do Portfólio em relação ao Benchmark")
st.dataframe(metricas.style.format({
    "Alfa Anual (%)": "{:,.2f}%",
    "Beta": "{:,.2f}",
    "R Quadrado (%)": "{:,.2f}%",
    "Information Ratio": "{:,.2f}"
}))

# ---------------------------
# Drawdown do portfólio
# ---------------------------
st.subheader("Drawdown do Portfólio")
cum_returns = (1 + portfolio_returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max

fig_drawdown, ax_drawdown = plt.subplots(figsize=(10,4))
ax_drawdown.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4)
ax_drawdown.set_title("Drawdown do Portfólio")
ax_drawdown.set_ylabel("Drawdown")
ax_drawdown.set_xlabel("Data")
ax_drawdown.grid(True)
ax_drawdown.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

st.pyplot(fig_drawdown)

# ---------------------------
# Rolling Beta (60 dias)
# ---------------------------
window = 60
rolling_cov = portfolio_returns.rolling(window).cov(bench_returns)
rolling_var = bench_returns.rolling(window).var()
rolling_beta = rolling_cov / rolling_var

st.subheader(f"Beta Móvel ({window} dias) vs {benchmark_opcao}")

fig_beta, ax_beta = plt.subplots(figsize=(10,4))
ax_beta.plot(rolling_beta.index, rolling_beta.values, color='blue')
ax_beta.axhline(1, color='gray', linestyle='--', alpha=0.7)
ax_beta.set_title(f"Rolling Beta {window} dias")
ax_beta.set_ylabel("Beta")
ax_beta.set_xlabel("Data")
ax_beta.grid(True)
fig_beta.autofmt_xdate(rotation=15)

st.pyplot(fig_beta)

# ---------------------------
# Sharpe Móvel (corrigido)
# ---------------------------
rf_daily = taxa_selic/100/252
rolling_sharpe = (portfolio_returns.rolling(window).mean() - rf_daily) / portfolio_returns.rolling(window).std()

st.subheader(f"Índice de Sharpe Móvel ({window} dias)")

fig_sharpe, ax_sharpe = plt.subplots(figsize=(10,4))
ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', label='Sharpe Móvel')
ax_sharpe.axhline(0, color='gray', linestyle='--', alpha=0.7, label='Zero')
ax_sharpe.set_title(f"Índice de Sharpe Móvel ({window} dias) do Portfólio")
ax_sharpe.set_ylabel("Sharpe")
ax_sharpe.set_xlabel("Data")
ax_sharpe.grid(True)
ax_sharpe.legend(loc='upper left')
fig_sharpe.autofmt_xdate(rotation=45)
fig_sharpe.tight_layout()

st.pyplot(fig_sharpe)

# ---------------------------
# Relatório QuantStats completo - botão download
# ---------------------------
st.subheader("Baixar Relatório Completo (QuantStats)")

# Remover timezone para evitar erro
portfolio_returns = portfolio_returns.tz_localize(None)
bench_returns = bench_returns.tz_localize(None)

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


