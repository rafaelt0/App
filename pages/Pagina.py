import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import plotly.express as px
from pypfopt.hierarchical_portfolio import HRPOpt
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
from scipy.stats import kurtosis, skew
import quantstats as qs
import io

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Análise e Otimização de Portfólio - B3 Explorer")

# Sidebar config
st.sidebar.header("Configurações do Portfólio")

data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
valor_inicial = st.sidebar.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)
taxa_selic = st.sidebar.number_input("Taxa Selic (%)", value=0.04, max_value=15.0)

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.multiselect("Selecione as ações do portfólio", stocks)

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
st.dataframe((peso_manual_df*100).round(2))

# Gráfico pizza das porcentagens
fig_pie = px.pie(peso_manual_df.reset_index(), values="Peso", names="index",
                 title="Composição do Portfólio (%)",
                 labels={"index": "Ativo", "Peso": "Percentual"})
st.plotly_chart(fig_pie)

# Cálculo do portfólio com os pesos escolhidos
portfolio_returns = returns.dot(pesos_manuais_arr)
cum_return = (1 + portfolio_returns).cumprod()
portfolio_value = cum_return * valor_inicial

# Mostrar gráfico do valor do portfólio
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

# Botão para gerar PDF via quantstats
import tempfile
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
st.sidebar.header("Opções Simulação 👨‍🔬")
n_simulations = st.sidebar.slider("Número de Simulações",10,1000,100)
valor = st.sidebar.number_input("Capital Inicial", min_value=100)
periodos = int(st.sidebar.number_input("Meses", value=12, min_value=1))
years = int(st.sidebar.number_input("Anos", min_value=1))         
data_inicio = st.sidebar.date_input("Data Inicial📅", value=datetime.date(2019,1,1),min_value=datetime.date(2000,1,1))

st.sidebar.markdown("<hr>", unsafe_allow_html=True)


st.header("Simulação 🧪")

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.write("")


with col3:
    st.write("")
     
mean = float(portfolio_returns.mean())
mu_selected = (1+mean)**12-1

sigma_selected = portfolio_returns.std()*np.sqrt(12)
    




mu = mu_selected
n = periodos
M = n_simulations
S0 = valor
sigma = sigma_selected
T = years

dt = T/n

St = np.exp(
    (mu - sigma**2/2)*dt
    * sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

St = np.vstack([np.ones(M), St])

St = S0 * St.cumprod(axis=0)

time = np.linspace(0,T,n+1)

tt = np.full(shape=(M,n+1), fill_value=time).T

fig=px.line(St, title="Simulação por Movimento Browniano Geométrico")
fig.update_layout(
                  xaxis = dict(
                    tickmode='array', #change 1
                    tickvals = np.arange(0,n*years,2)))
fig.update_yaxes(title="Portfolio/Ação")
fig.update_xaxes(title="Período")
st.plotly_chart(fig)
mean=St[-1][:].mean()
max=St[-1][:].max()
min=St[-1][:].min()
array = np.array(St[-1][:])
summary=pd.DataFrame([mean,max,min, np.percentile(array,25), np.median(array), np.percentile(array,75), np.std(array)])
summary = summary.rename({0:"Média", 1:"Máximo", 2:"Mínimo", 3:"Primeiro Quartil", 4:"Mediana (Segundo Quartil)", 5:"Terceiro Quartil", 6:"Desvio Padrão"}, axis=0)
summary = summary.rename({0:"Resultados"}, axis=1)
st.subheader("Resultados 🔬")
st.table(summary.T)




