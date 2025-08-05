import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
from bcb import sgs
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.hierarchical_portfolio import HRPOpt
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
from scipy.stats import kurtosis, skew
import quantstats as qs
import matplotlib.ticker as mtick
import io



warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)




st.title("Análise e Otimização de Portfólio - B3 Explorer")
# Sidebar config
st.sidebar.header("Configurações do Portfólio")

data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
valor_inicial = st.sidebar.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)
taxa_selic = st.sidebar.number_input("Taxa Selic (%)", value=0.0556, max_value=15.0)
benchmark_opcao = st.sidebar.selectbox("Selecione seu Benchmark", ["SELIC", "CDI", "IBOVESPA"])

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.multiselect("Selecione as ações do portfólio", stocks)

if benchmark_opcao == "IBOVESPA":
  bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close']
  st.write(bench)
  retorno_bench = bench.pct_change().dropna()
  bench_cum = (1+retorno_bench).cumprod()-1
  bench_value = bench_cum * valor_inicial
elif benchmark_opcao == "SELIC":
  bench = sgs.get({'selic': 432}, start=data_inicio)
  selic_diario = (1+bench)**(1/252)-1
  retorno_bench = selic_diario.pct_change().dropna()
  bench_cum = (1+selic_diario).cumprod()
  bench_value = bench_cum*valor_inicial
else:
  bench = sgs.get({'CDI': 12}, start=data_inicio)
  cdi_diario = bench/100
  retorno_bench = bench.pct_change().dropna()
  bench_cum = (1+cdi_diario).cumprod()
  bench_value = bench_cum*valor_inicial
  
  

if len(tickers) == 0:
    st.warning("Selecione pelo menos uma ação.")
    st.stop()

if len(tickers) == 1:
    st.warning("Selecione pelo menos dois ativos para montar o portfólio.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

# Baixa dados
data_yf = yf.download(tickers_yf, start=data_inicio, progress=False)['Close']
if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

returns = data_yf.pct_change().dropna()

# Escolha modo: manual ou otimizado
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
    pesos_manuais_arr = np.array(list(pesos_manuais.values()))
    peso_manual_df = pd.DataFrame.from_dict(pesos_manuais, orient='index', columns=["Peso"])
else:
    st.subheader("Otimização Hierarchical Risk Parity (HRP)")
    hrp = HRPOpt(returns)
    weights_hrp = hrp.optimize()
    peso_manual_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
    pesos_manuais_arr = peso_manual_df["Peso"].values

    # Mostrar pesos
st.subheader("Pesos do Portfólio (%)")
peso_manual_df.index = peso_manual_df.index.str.replace(".SA","")
st.dataframe((peso_manual_df*100).round(2).T)

# Gráfico pizza das porcentagens
fig_pie = px.pie(peso_manual_df.reset_index(), values="Peso", names="index",
                 title="Composição do Portfólio (%)",
                 labels={"index": "Ativo", "Peso": "Percentual"})
st.plotly_chart(fig_pie)

alloc_df = peso_manual_df.reset_index()
alloc_df.columns = ["Ativo", "Peso"]

fig_treemap = px.treemap(
    alloc_df,
    path=['Ativo'],
    values='Peso',
    color='Peso',
    color_continuous_scale='Blues',
    title="Alocação do Portfólio (Treemap)"
)
st.plotly_chart(fig_treemap, use_container_width=True)


# Heatmap e Matriz de Correlação
heatmap=sns.heatmap(data_yf.corr(), annot=True)
st.subheader("Matrix de Correlação")
st.write(data_yf.corr())
st.subheader("Heatmap")
st.pyplot(heatmap.figure)

# Cálculo do portfólio com os pesos escolhidos
portfolio_returns = returns.dot(pesos_manuais_arr)
cum_return = (1 + portfolio_returns).cumprod()
portfolio_value = cum_return * valor_inicial

# Mostrar gráfico do valor do portfólio x BOVESPA
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, 
                         mode='lines', name='Portfólio'))
fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, 
                         mode='lines', name='IBOVESPA'))
fig.update_layout(title='Valor do Portfólio',
                  xaxis_title='Data', yaxis_title='Valor (R$)')
st.plotly_chart(fig)
# Retornos mensais
st.subheader("Retornos Mensais do Portfólio")

fig = qs.plots.monthly_returns(portfolio_returns, show=False)
st.pyplot(fig)








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

# Distribuição de retornos com estatísticas
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

# Estatísticas do portfólio
stats = pd.DataFrame([[ 
    sharpe(portfolio_returns, rf=taxa_selic/100),
    sortino(portfolio_returns, rf=taxa_selic/100),
    max_drawdown(portfolio_returns),
    var(portfolio_returns),
    cvar(portfolio_returns),
    tail_ratio(portfolio_returns)
]], columns=["Índice Sharpe", "Índice Sortino", "Max Drawdown", "VaR", "CVaR", "Tail Ratio"])

st.subheader("Estatísticas do Portfólio")
st.dataframe(stats.round(4))

# Gráfico Portfolio vs IBOVESPA
st.subheader("Retorno Acumulado Portfólio vs IBOVESPA")
fig = qs.plots.returns(portfolio_returns, benchmark=retorno_bench, show=False)
st.pyplot(fig)

# Métricas vs bench
bench_returns = bench
cov_matrix = np.cov(portfolio_returns.squeeze(), bench_returns.squeeze())  # matriz de covariância 2x2
beta = cov_matrix[0,1] / cov_matrix[1,1]
alfa = portfolio_returns.mean() - beta * bench_returns.mean()
r_quadrado = qs.stats.r_squared(portfolio_returns, bench)
information_ratio = qs.stats.information_ratio(portfolio_returns, bench)


metricas = pd.DataFrame({
    "Alfa Anual (%)": [alfa.values[0]*252*100],
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
# Retornos Anuais

fig = qs.plots.yearly_returns(portfolio_returns, benchmark=retorno_bench, compounded=True, show=False)
ax = plt.gca()

# Alterar legenda
ax.legend(['Portfólio', 'IBOVESPA'])  # renomeia
ax.set_title('Retornos Anuais (Portfólio vs IBOVESPA)')

st.pyplot(fig)
    


st.subheader("Drawdown do Portfólio")
# Drawdown
cum_returns = (1 + portfolio_returns).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max

# Plot Drawdown
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.4)
ax1.set_title("Drawdown do Portfólio")
ax1.set_ylabel("Drawdown")
ax1.set_xlabel("Data")
ax1.grid(True)

ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
st.pyplot(fig1)

# Rolling Beta (60 dias)
window = 60

rolling_cov = portfolio_returns.rolling(window).cov(retorno_bench)
rolling_var = retorno_bench.rolling(window).var()

rolling_beta = rolling_cov / rolling_var

# Gráfico Rolling Beta

st.subheader(f"Beta Móvel ({window} dias) vs IBOVESPA")

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(rolling_beta.index, rolling_beta.values, color='blue')
ax2.axhline(1, color='gray', linestyle='--', alpha=0.7)
ax2.set_title(f"Rolling Beta {window} dias")
ax2.set_ylabel("Beta")
ax2.set_xlabel("Data")
ax2.grid(True)

fig2.autofmt_xdate(rotation=15)

st.pyplot(fig2)

# Gráfico Sharpe Móvel
rolling_sharpe = (
(portfolio_returns.rolling(window).mean() - taxa_selic) /
portfolio_returns.rolling(window).std())

st.subheader(f"Índice de Sharpe Móvel ({window} dias)")

fig_3, ax_3 = plt.subplots(figsize=(10,4))
ax_3.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', label='Sharpe Móvel')
ax_3.axhline(0, color='gray', linestyle='--', alpha=0.7, label='Zero')
ax_3.set_title(f"Índice de Sharpe Móvel ({window} dias) do Portfólio")
ax_3.set_ylabel("Sharpe")
ax_3.set_xlabel("Data")
ax_3.grid(True)
ax_3.legend(loc='upper left')

fig_3.autofmt_xdate(rotation=45)  # datas na diagonal
fig_3.tight_layout()

st.pyplot(fig_3)





# Botão para gerar PDF via quantstats
import tempfile
st.subheader("Baixar Relatório Completo (QuantStats)")

# Converte para formato aceito pelo QuantStats
portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
portfolio_returns = portfolio_returns.tz_localize(None)  # Remove timezone


with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
    qs.reports.html(
        portfolio_returns,
        benchmark= retorno_bench,
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

# Salva variáveis  para uso na aba Simulação
st.session_state["modo"] = modo
st.session_state["returns"] = returns
st.session_state["peso_manual_df"] = peso_manual_df

# Garante que pesos manuais ficam disponíveis como dicionário
if modo == "Alocação Manual":
    st.session_state["pesos_manuais"] = pesos_manuais
else:
    st.session_state["pesos_manuais"] = peso_manual_df["Peso"].to_dict()


# Separação na sidebar
st.sidebar.markdown("---")

selic = sgs.get({'selic': 432}, start='2022-01-01')

