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
import io

aba1, aba2 = st.tabs(["📊 Análise do Portfólio", "🧪 Simulação Monte Carlo Portfolio"])

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)



with aba1:
    st.title("Análise e Otimização de Portfólio - B3 Explorer")
    # Sidebar config
    st.sidebar.header("Configurações do Portfólio")
    
    data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025, 1, 1), min_value=datetime.date(2000, 1, 1))
    valor_inicial = st.sidebar.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)
    taxa_selic = st.sidebar.number_input("Taxa Selic (%)", value=0.0556, max_value=15.0)
    benchmark_opcao = st.sidebar.multiselect("Selecione seu Benchmark", ["SELIC", "CDI", "IBOVESPA"])
    
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
    bench = yf.download("^BVSP", start=data_inicio)['Close'].pct_change().dropna()
    fig = qs.plots.returns(portfolio_returns, benchmark=bench, show=False)
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

# Separação na sidebar
st.sidebar.markdown("---")

           
with aba2:
    st.header("Simulação Monte Carlo por Ativos (Multivariada) 👨‍🔬")

    with st.form("form_simulacao"):
        n_simulations = st.slider("Número de Simulações", 10, 500, 200,
                                  help="Quantidade de trajetórias simuladas para o portfólio.")
        valor = st.number_input("Capital Inicial (R$)", min_value=100,
                                help="Valor inicial investido no portfólio.")
        years = int(st.number_input("Anos", min_value=1,
                                    help="Horizonte da simulação em anos."))
        
        submitted = st.form_submit_button("Rodar Simulação")
    
    if not submitted:
        st.info("Configure os parâmetros acima e clique em 'Rodar Simulação' para ver os resultados.")
        st.stop()

    st.header("Simulação 🧪")

    n_dias = years * 252  # 252 dias úteis no ano
    valor_inicial = valor
    
    # Garante que temos um dicionário de pesos, independente do modo escolhido
    if modo == "Alocação Manual":
        pesos_dict = pesos_manuais
    else:
        pesos_dict = dict(zip(peso_manual_df.index + ".SA", peso_manual_df["Peso"].values))
    
    # Remove ativos com peso zero (se houver)
    pesos_dict = {k: v for k, v in pesos_dict.items() if v > 1e-6}
    
    aligned_returns = returns.loc[:, pesos_dict.keys()].dropna()

    pesos = np.array(list(pesos_dict.values()))
    
    mu = aligned_returns.mean().values  # vetor média de retorno diário
    cov = aligned_returns.cov().values  # matriz covariância diária
    
    np.random.seed(42)  # para reprodutibilidade
    
    # Simular retornos multivariados normais correlacionados
    retornos_simulados = np.random.multivariate_normal(mu, cov, size=(n_dias, n_simulations))
    
    # Calcular trajetórias para cada ativo em cada simulação
    precos_simulados = np.exp(retornos_simulados.cumsum(axis=0))
    
    # Calcular valor do portfólio: soma ponderada dos ativos para cada dia e simulação
    valor_portfolio = (precos_simulados * pesos).sum(axis=2) * valor_inicial
    
    # Criar DataFrame para facilitar manipulação e plotagem
    datas = pd.date_range(start=datetime.date.today(), periods=n_dias+1, freq='B')
    valor_portfolio = np.vstack([np.ones(n_simulations)*valor_inicial, valor_portfolio])
    sim_df = pd.DataFrame(valor_portfolio, index=datas)
    
    # Estatísticas finais da simulação
    valores_finais = sim_df.iloc[-1]
    valor_esperado = valores_finais.mean()
    var_5 = np.percentile(valores_finais, 5)
    cvar_5 = valores_finais[valores_finais <= var_5].mean()
    pior_cenario = valores_finais.min()
    melhor_cenario = valores_finais.max()
    
    sim_stats = pd.DataFrame({
        "Valor Esperado Final (R$)": [valor_esperado],
        "VaR 5% (R$)": [var_5],
        "CVaR 5% (R$)": [cvar_5],
        "Pior Cenário (R$)": [pior_cenario],
        "Melhor Cenário (R$)": [melhor_cenario]
    })
    
    st.subheader("📊 Estatísticas da Simulação Monte Carlo por Ativos")
    st.dataframe(sim_stats.style.format("{:,.2f}"))

    st.markdown("""
    <small><b>VaR 5%</b>: Valor máximo esperado que você pode perder em 5% dos piores casos.<br>
    <b>CVaR 5%</b>: Média das perdas nos piores 5% dos casos, mostrando um risco mais extremo.</small>
    """, unsafe_allow_html=True)

    # Gráfico com algumas trajetórias individuais para ilustrar a dispersão
    st.subheader("Trajetórias Individuais das Simulações (Exemplos)")

    fig_individual = go.Figure()
    n_plot = min(20, n_simulations)  # limitar para 20 linhas para visualização limpa

    for i in range(n_plot):
        fig_individual.add_trace(go.Scatter(
            x=sim_df.index,
            y=sim_df.iloc[:, i],
            mode='lines',
            name=f'Simulação {i+1}',
            line=dict(width=1),
            opacity=0.6
        ))
    fig_individual.update_layout(
        title="Exemplos de Trajetórias Simuladas do Valor do Portfólio",
        xaxis_title="Data",
        yaxis_title="Valor do Portfólio (R$)",
        template="plotly_white"
    )
    st.plotly_chart(fig_individual, use_container_width=True)

    # Fan chart com percentis
    percentis = [5, 25, 50, 75, 95]
    fan_chart = sim_df.quantile(q=np.array(percentis) / 100, axis=1).T
    fan_chart.columns = [f"P{p}" for p in percentis]

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(
        x=fan_chart.index, y=fan_chart["P95"],
        line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_chart.index, y=fan_chart["P5"],
        fill='tonexty', fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 5%-95%'
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_chart.index, y=fan_chart["P75"],
        line=dict(color='rgba(0,100,200,0.1)'), showlegend=False
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_chart.index, y=fan_chart["P25"],
        fill='tonexty', fillcolor='rgba(0,100,200,0.4)',
        line=dict(color='rgba(0,100,200,0.1)'), name='Faixa 25%-75%'
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_chart.index, y=fan_chart["P50"],
        line=dict(color='blue', width=2), name='Mediana'
    ))
    fig_fan.update_layout(
        title="Simulação Monte Carlo por Ativos - Fan Chart com Faixas de Confiança",
        xaxis_title="Data",
        yaxis_title="Valor do Portfólio (R$)",
        template="plotly_white"
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # Histograma valor final
    q1 = valores_finais.quantile(0.25)
    q2 = valores_finais.quantile(0.50)
    q3 = valores_finais.quantile(0.75)

    st.subheader("Distribuição do Valor Final do Portfólio")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(valores_finais, bins=30, kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.axvline(q1, color='red', linestyle='--', label='Q1 (25%)')
    ax.axvline(q2, color='green', linestyle='-', label='Mediana (50%)')
    ax.axvline(q3, color='orange', linestyle='--', label='Q3 (75%)')
    ax.set_title('Distribuição dos Valores Finais da Simulação Monte Carlo')
    ax.set_xlabel('Valor Final do Portfólio (R$)')
    ax.set_ylabel('Frequência')
    ax.legend()
    st.pyplot(fig)

    # Estatísticas da distribuição final
    estatisticas = {
        "Mínimo": valores_finais.min(),
        "Q1 (25%)": q1,
        "Mediana (50%)": q2,
        "Q3 (75%)": q3,
        "Máximo": valores_finais.max(),
        "Média": valores_finais.mean(),
        "Desvio Padrão": valores_finais.std()
    }
    df_estatisticas = pd.DataFrame(estatisticas, index=["Valores (R$)"])
    st.subheader("Estatísticas da Distribuição Final da Simulação Monte Carlo")
    st.dataframe(df_estatisticas.style.format("{:,.2f}"))





