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
import base64
import time

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Título da página
st.markdown("<h1 style='text-align: center;'>Análise e Otimização de Portfólio 💼</h1>", unsafe_allow_html=True)

# Centralizar a imagem
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# converte a imagem para base64
img_base64 = get_base64_of_bin_file("portfolio.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

# Adiciona espaço vertical
st.markdown("<br><br>", unsafe_allow_html=True)

# Configurações
data_inicio = st.date_input("Data Inicial 🗓️", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
taxa_selic =  sgs.get(432, start=data_inicio)
taxa_selic = taxa_selic.iloc[-1,0]
taxa_selic = (1+taxa_selic)**(1/252)-1

# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
tickers = st.multiselect("Selecione as ações do portfólio 📊", stocks)
# Valor inicial
valor_inicial = st.number_input("Valor Investido (R$) 💵", 100, 1_000_000, 10_000)

# Escolha modo: manual ou otimizado
modo = st.radio("Modo de alocação ⚙️", ("Otimização Hierarchical Risk Parity (Machine Learning) 🤖", "Alocação Manual 🔧"))

if len(tickers) == 0:
    st.warning("Selecione pelo menos uma ação.")
    st.stop()

if len(tickers) == 1:
    st.warning("Selecione pelo menos dois ativos para montar o portfólio.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

# Botão de execução
if st.button("💡 Rodar Análise do Portfólio"):

    # Barra de progresso global
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown("<h2 style='text-align: center;'>Carregando página...</h2>", unsafe_allow_html=True)
    
    step = 0
    total_steps = 14

    # Baixa dados
    data_yf = yf.download(tickers_yf, start=data_inicio, progress=False)['Close']
    if isinstance(data_yf.columns, pd.MultiIndex):
        data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

    returns = data_yf.pct_change().dropna()
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    if modo == "Alocação Manual":
        st.subheader("Defina manualmente a porcentagem de cada ativo (soma deve ser 100%)")
        pesos_manuais = {}
        total = 0.0
        for ticker in tickers:
            p = round(100/len(tickers),2)
            pesos_manuais[ticker + ".SA"] = p / 100
            total += p
        pesos_manuais_arr = np.array(list(pesos_manuais.values()))
        peso_manual_df = pd.DataFrame.from_dict(pesos_manuais, orient='index', columns=["Peso"])
    else:
        st.subheader("Otimização Hierarchical Risk Parity (HRP)")
        hrp = HRPOpt(returns)
        weights_hrp = hrp.optimize()
        peso_manual_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
        pesos_manuais_arr = peso_manual_df["Peso"].values
    step +=1
    progress_bar.progress(int(step/total_steps*100))

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
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    # Heatmap e Matriz de Correlação
    heatmap=sns.heatmap(data_yf.corr(), annot=True)
    st.subheader("Matrix de Correlação")
    st.write(data_yf.corr())
    st.subheader("Heatmap")
    st.pyplot(heatmap.figure)
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    # Cálculo do portfólio com os pesos escolhidos
    portfolio_returns = returns.dot(pesos_manuais_arr)
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    # Obter os dados de benchmark BOVESPA e calcular o retorno acumulado
    bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close']
    retorno_bench = bench.pct_change().dropna()
    portfolio_returns = portfolio_returns.loc[retorno_bench.index]
    retorno_bench = retorno_bench.loc[portfolio_returns.index]
    retorno_cum_bench = (1+retorno_bench).cumprod()
    bench_value = retorno_cum_bench * valor_inicial

    # Mostrar gráfico do valor do portfólio x BOVESPA
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, 
                             mode='lines', name='Portfólio'))
    fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, 
                             mode='lines', name='IBOVESPA'))
    fig.update_layout(title='Valor do Portfólio',
                      xaxis_title='Data', yaxis_title='Valor (R$)')
    st.plotly_chart(fig)
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    # Retornos mensais
    st.subheader("Retornos Mensais do Portfólio")

    fig = qs.plots.monthly_returns(portfolio_returns, show=False)
    st.pyplot(fig)
    step +=1
    progress_bar.progress(int(step/total_steps*100))

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
    step +=1
    progress_bar.progress(int(step/total_steps*100))

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
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    # Estatísticas do portfólio
    stats = pd.DataFrame([[ 
        sharpe(portfolio_returns, rf=taxa_selic),
        sortino(portfolio_returns, rf=taxa_selic),
        max_drawdown(portfolio_returns),
        var(portfolio_returns),
        cvar(portfolio_returns),
        tail_ratio(portfolio_returns)
    ]], columns=["Índice Sharpe", "Índice Sortino", "Max Drawdown", "VaR", "CVaR", "Tail Ratio"])

    st.subheader("Estatísticas do Portfólio")
    st.dataframe(stats.round(4))
    step +=1
    progress_bar.progress(int(step/total_steps*100))

    # Limpa barra e status
    status_text.empty()
    st.success("✅ Página carregada com sucesso!")

    
    
    
    
    
    
    
    
           
    
    
    
    
    
    
    
    
    
    
       
