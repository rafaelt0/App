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
import tempfile
import io

# ---------------- CONFIGURAÇÕES INICIAIS ----------------
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

aba1, aba2 = st.tabs(["📊 Análise do Portfólio", "🧪 Simulação Monte Carlo Portfolio"])

# ---------------- ABA 1: ANÁLISE ----------------
with aba1:
    st.title("Análise e Otimização de Portfólio - B3 Explorer")

    # Sidebar config
    st.sidebar.header("Configurações do Portfólio")
    data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
    valor_inicial = st.sidebar.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)
    taxa_selic = st.sidebar.number_input("Taxa Selic (%)", value=0.0556, max_value=15.0)

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

    # ✅ Calcula retorno do portfólio com os pesos
    portfolio_returns = returns.dot(pesos_manuais_arr)

    # Mostrar pesos
    st.subheader("Pesos do Portfólio (%)")
    peso_manual_df.index = peso_manual_df.index.str.replace(".SA","")
    st.dataframe((peso_manual_df*100).round(2).T)

    # Gráfico pizza das porcentagens
    fig_pie = px.pie(peso_manual_df.reset_index(), values="Peso", names="index",
                     title="Composição do Portfólio (%)",
                     labels={"index": "Ativo", "Peso": "Percentual"})
    st.plotly_chart(fig_pie)

    # Treemap da Alocação
    st.subheader("Treemap da Alocação do Portfólio")
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
    heatmap = sns.heatmap(data_yf.corr(), annot=True)
    st.subheader("Matriz de Correlação")
    st.write(data_yf.corr())
    st.subheader("Heatmap")
    st.pyplot(heatmap.figure)

    # --- Comparação com Benchmark ---
    cum_return = (1 + portfolio_returns).cumprod()
    portfolio_value = cum_return * valor_inicial

    bench = yf.download("^BVSP", start=portfolio_returns.index[0], progress=False)['Close']
    bench = bench.reindex(portfolio_returns.index).fillna(method='ffill')  # alinhar datas
    retorno_bench = bench.pct_change().dropna()
    retorno_cum_bench = (1 + retorno_bench).cumprod()
    bench_value = retorno_cum_bench * valor_inicial

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfólio'))
    fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, mode='lines', name='IBOVESPA'))
    fig.update_layout(title='Comparação: Portfólio x Benchmark', xaxis_title='Data', yaxis_title='Valor (R$)', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

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
    portfolio_returns = returns.dot(pesos_manuais_arr)  # após definir os pesos
    # Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(10,4))
    ax.fill_between(drawdown.index, drawdown.values*100, 0, color='red', alpha=0.4)
    ax.set_title("Drawdown do Portfólio em (%)")
    st.pyplot(fig)

    # Relatório Quantstats
    st.subheader("Baixar Relatório Completo (QuantStats)")
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
            label="Baixar Relatório HTML Completo (QuantStats)",
            data=open(tmpfile.name, "rb").read(),
            file_name="relatorio_portfolio.html",
            mime="text/html"
        )

st.sidebar.markdown("---")

# ---------------- ABA 2: MONTE CARLO ----------------
with aba2:
    if 'portfolio_returns' not in locals():
        st.warning("Monte Carlo só disponível após calcular o portfólio na Aba 1.")
        st.stop()

    st.header("Opções Simulação 👨‍🔬")
    n_simulations = st.slider("Número de Simulações",10,1000,100)
    valor = st.number_input("Capital Inicial", min_value=100)
    years = int(st.number_input("Anos", min_value=1))  

    st.header("Simulação 🧪")

    n_sim = n_simulations
    n_dias = years*365  # dias
    valor_inicial = valor

    mu_p = portfolio_returns.mean()
    sigma_p = portfolio_returns.std()

    # Simulações Monte Carlo
    simulacoes = np.zeros((n_dias, n_sim))
    simulacoes[0] = valor_inicial
    for sim in range(n_sim):
        for t in range(1, n_dias):
            z = np.random.normal()
            simulacoes[t, sim] = simulacoes[t-1, sim] * np.exp((mu_p - 0.5*sigma_p**2) + sigma_p*z)

    sim_df = pd.DataFrame(simulacoes)
    sim_df.index.name = "Dia"

    # Fan chart
    fig = px.line(sim_df, title="Simulações de Monte Carlo para o Portfólio")
    st.plotly_chart(fig)

    valor_esperado = sim_df.iloc[-1].mean()
    var_5 = np.percentile(sim_df.iloc[-1], 5)
    pior_cenario = sim_df.iloc[-1].min()
    melhor_cenario = sim_df.iloc[-1].max()

    sim_stats = pd.DataFrame({
        "Valor Esperado Final (R$)": [valor_esperado],
        "VaR 5% (R$)": [var_5],
        "Pior Cenário (R$)": [pior_cenario],
        "Melhor Cenário (R$)": [melhor_cenario]
    })
    st.subheader("📊 Estatísticas da Simulação Monte Carlo")
    st.dataframe(sim_stats.style.format("{:,.2f}"))

    # Fan chart com percentis
    percentis = [5, 25, 50, 75, 95]
    fan_chart = sim_df.quantile(q=np.array(percentis)/100, axis=1).T
    fan_chart.columns = [f"P{p}" for p in percentis]
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=fan_chart.index, y=fan_chart["P95"], line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=fan_chart.index, y=fan_chart["P5"], fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
                                 line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 5%-95%'))
    fig_fan.add_trace(go.Scatter(x=fan_chart.index, y=fan_chart["P75"], line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=fan_chart.index, y=fan_chart["P25"], fill='tonexty', fillcolor='rgba(0,100,200,0.4)',
                                 line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 25%-75%'))
    fig_fan.add_trace(go.Scatter(x=fan_chart.index, y=fan_chart["P50"], line=dict(color='blue', width=2), name='Mediana'))
    fig_fan.update_layout(title="Simulação Monte Carlo - Fan Chart com Faixas de Confiança",
                          xaxis_title="Dia", yaxis_title="Valor do Portfólio (R$)", template="plotly_white")
    st.plotly_chart(fig_fan, use_container_width=True)
