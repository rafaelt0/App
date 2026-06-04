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
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from quantstats.stats import sharpe, sortino, max_drawdown, var, cvar, tail_ratio
from scipy.stats import kurtosis, skew
import quantstats as qs
from bcb import sgs
import matplotlib.ticker as mtick

# ─── SVG Icon Library ─────────────────────────────────────────────────────────
def _svg(body, size=14):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
            f'{body}</svg>')

ICO_OK    = _svg('<circle cx="12" cy="12" r="9" stroke="#00ff87" stroke-width="1.8"/>'
                 '<path d="M8 12l3 3 5-5" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
ICO_WARN  = _svg('<path d="M12 3L22 21H2L12 3Z" stroke="#ffd600" stroke-width="1.8" stroke-linejoin="round"/>'
                 '<line x1="12" y1="10" x2="12" y2="14" stroke="#ffd600" stroke-width="2" stroke-linecap="round"/>'
                 '<circle cx="12" cy="17.5" r="1" fill="#ffd600"/>')
ICO_CRIT  = _svg('<circle cx="12" cy="12" r="9" stroke="#ff3d5a" stroke-width="1.8"/>'
                 '<line x1="9" y1="9" x2="15" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
                 '<line x1="15" y1="9" x2="9" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>')
ICO_CHART    = _svg('<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
                    '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
                    '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>', 16)
ICO_TARGET   = _svg('<circle cx="12" cy="12" r="9" stroke="#00d2ff" stroke-width="1.8"/>'
                    '<circle cx="12" cy="12" r="5" stroke="#ffd600" stroke-width="1.5"/>'
                    '<circle cx="12" cy="12" r="2" fill="#00ff87"/>', 16)
ICO_SIGNAL   = _svg('<path d="M2 12 Q6 4 12 12 Q18 20 22 12" stroke="#00d2ff" stroke-width="2" '
                    'stroke-linecap="round" fill="none"/>'
                    '<circle cx="12" cy="12" r="2" fill="#ffd600"/>', 16)
ICO_RISK     = _svg('<path d="M12 3l9 18H3L12 3z" stroke="#ff3d5a" stroke-width="1.8" stroke-linejoin="round"/>'
                    '<line x1="12" y1="9" x2="12" y2="14" stroke="#ff3d5a" stroke-width="1.8" stroke-linecap="round"/>'
                    '<circle cx="12" cy="17" r="1" fill="#ff3d5a"/>', 16)
ICO_METRICS  = _svg('<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
                    '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                    '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
                    '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>', 16)
ICO_HEATMAP  = _svg('<rect x="3"  y="3"  width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
                    '<rect x="10" y="3"  width="4" height="4" rx="1" fill="#00d2ff" opacity="0.6"/>'
                    '<rect x="17" y="3"  width="4" height="4" rx="1" fill="#ffd600" opacity="0.4"/>'
                    '<rect x="3"  y="10" width="4" height="4" rx="1" fill="#00d2ff" opacity="0.5"/>'
                    '<rect x="10" y="10" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
                    '<rect x="17" y="10" width="4" height="4" rx="1" fill="#a855f7" opacity="0.5"/>'
                    '<rect x="3"  y="17" width="4" height="4" rx="1" fill="#ffd600" opacity="0.3"/>'
                    '<rect x="17" y="17" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>', 16)
ICO_FRONTIER = _svg('<path d="M3 20 Q8 8 14 10 Q18 12 21 4" stroke="#00ff87" stroke-width="2" stroke-linecap="round" fill="none"/>'
                    '<circle cx="18" cy="6" r="2.5" fill="#ff3d5a"/>'
                    '<circle cx="10" cy="17" r="2" fill="#ffd600"/>', 16)
ICO_LINK     = _svg('<circle cx="7"  cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
                    '<circle cx="17" cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
                    '<line x1="10" y1="12" x2="14" y2="12" stroke="#00d2ff" stroke-width="1.8"/>', 16)
ICO_BOX      = _svg('<rect x="3" y="7" width="18" height="14" rx="2" stroke="#94a3b8" stroke-width="1.8"/>'
                    '<path d="M8 7V5a4 4 0 018 0v2" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>'
                    '<line x1="12" y1="12" x2="12" y2="16" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                    '<line x1="10" y1="14" x2="14" y2="14" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>', 16)
ICO_RULER    = _svg('<rect x="2" y="7" width="20" height="10" rx="2" stroke="#ffd600" stroke-width="1.8"/>'
                    '<line x1="6"  y1="7" x2="6"  y2="12" stroke="#ffd600" stroke-width="1.5"/>'
                    '<line x1="10" y1="7" x2="10" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                    '<line x1="14" y1="7" x2="14" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                    '<line x1="18" y1="7" x2="18" y2="12" stroke="#ffd600" stroke-width="1.5"/>', 16)
ICO_IDEA     = _svg('<circle cx="12" cy="10" r="6" stroke="#ffd600" stroke-width="1.8"/>'
                    '<path d="M9 16.5h6M10 19h4" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
                    '<line x1="12" y1="4" x2="12" y2="2" stroke="#ffd600" stroke-width="1.5" stroke-linecap="round"/>', 16)
ICO_UP       = _svg('<path d="M12 20V4M5 11l7-7 7 7" stroke="#00ff87" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>', 14)
ICO_DOWN     = _svg('<path d="M12 4v16M5 13l7 7 7-7" stroke="#ff3d5a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>', 14)
ICO_FLAT     = _svg('<line x1="4" y1="12" x2="20" y2="12" stroke="#ffd600" stroke-width="2.2" stroke-linecap="round"/>'
                    '<path d="M16 8l4 4-4 4" stroke="#ffd600" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>', 14)

def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f'{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True)

def diag_row(icon_svg, text, color):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
        f'color:{color};font-size:0.88rem">{icon_svg}{text}</div>',
        unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_selic_rate(start_date):
    taxa_selic = sgs.get(432, start=start_date)
    val = (taxa_selic.iloc[-1,0])/100
    daily_val = (1+val)**(1/252)-1
    return daily_val

@st.cache_data(ttl=3600)
def get_portfolio_prices(tickers_yf, start_date):
    today = datetime.date.today()
    return yf.download(tickers_yf, start=start_date, end=today, progress=False)['Close']

@st.cache_data(ttl=86400, show_spinner=False)
def get_sorted_tickers_by_liquidity(tickers_list):
    try:
        import fundamentus.resultado as fzr
        df = fzr.get_resultado_raw()
        df = df.sort_values(by='Liq.2meses', ascending=False)
        sorted_all = df.index.tolist()
        sorted_filtered = [t for t in sorted_all if t in tickers_list]
        remaining = [t for t in tickers_list if t not in sorted_filtered]
        return sorted_filtered + remaining
    except Exception:
        return tickers_list

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# Customização do Matplotlib para o tema Obsidian Neo-Financial
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#080c14'
plt.rcParams['axes.facecolor'] = '#0e1524'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#f8fafc'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['font.family'] = 'sans-serif'

def apply_matplotlib_theme(fig):
    fig.set_facecolor('#080c14')
    for ax in fig.axes:
        ax.set_facecolor('#0e1524')
        ax.tick_params(colors='#94a3b8', which='both')
        ax.yaxis.label.set_color('#f8fafc')
        ax.xaxis.label.set_color('#f8fafc')
        if ax.title:
            ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#0e1524')
            legend.get_frame().set_edgecolor('#1e293b')
            for text in legend.get_texts():
                text.set_color('#f8fafc')
    return fig

# Customização do Plotly para o tema Obsidian Neo-Financial
def apply_plotly_theme(fig):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Grotesk, sans-serif", color="#f8fafc"),
        xaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor='#1e293b',
            linecolor='#1e293b',
            tickfont=dict(family="JetBrains Mono, monospace", color="#94a3b8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(14, 21, 36, 0.8)',
            bordercolor='#1e293b',
            borderwidth=1
        ),
        margin=dict(b=80)
    )
    return fig

def plot_efficient_frontier_and_random_portfolios(mu, S, returns, cleaned_weights, rf):
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(weights * mu)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - rf) / portfolio_stddev if portfolio_stddev > 0 else 0
        
    opt_weights = np.array(list(cleaned_weights.values()))
    opt_return = np.sum(opt_weights * mu)
    opt_stddev = np.sqrt(np.dot(opt_weights.T, np.dot(S, opt_weights)))
    opt_sharpe = (opt_return - rf) / opt_stddev if opt_stddev > 0 else 0
    
    try:
        ef_min = EfficientFrontier(mu, S)
        min_weights = ef_min.min_volatility()
        min_weights_arr = np.array(list(min_weights.values()))
        min_return = np.sum(min_weights_arr * mu)
        min_stddev = np.sqrt(np.dot(min_weights_arr.T, np.dot(S, min_weights_arr)))
    except:
        min_return = min(mu)
        min_stddev = np.sqrt(np.min(np.diag(S)))
    
    efficient_vols = []
    target_returns = np.linspace(min_return, max(mu), 25)
    for target in target_returns:
        try:
            ef_target = EfficientFrontier(mu, S)
            ef_target.efficient_return(target)
            w_target = np.array(list(ef_target.clean_weights().values()))
            vol = np.sqrt(np.dot(w_target.T, np.dot(S, w_target)))
            efficient_vols.append(vol)
        except:
            pass
            
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results[0,:],
        y=results[1,:],
        mode='markers',
        marker=dict(
            size=4,
            color=results[2,:],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Índice Sharpe",
                    font=dict(color="#f8fafc")
                ),
                tickfont=dict(color="#f8fafc")
            ),
            opacity=0.6
        ),
        name="Portfólios Aleatórios",
        text=[f"Retorno Anual: {r:.2%}<br>Vol Anual: {v:.2%}<br>Sharpe: {s:.2f}" for v, r, s in zip(results[0,:], results[1,:], results[2,:])],
        hoverinfo='text'
    ))
    
    if len(efficient_vols) > 0:
        fig.add_trace(go.Scatter(
            x=efficient_vols,
            y=target_returns[:len(efficient_vols)],
            mode='lines',
            line=dict(color='#00ff87', width=3),
            name="Fronteira Eficiente"
        ))
        
    fig.add_trace(go.Scatter(
        x=[opt_stddev],
        y=[opt_return],
        mode='markers',
        marker=dict(color='#ff1744', size=12, symbol='star', line=dict(color='#f8fafc', width=2)),
        name="Max Sharpe (Markowitz)",
        text=[f"Max Sharpe<br>Retorno: {opt_return:.2%}<br>Vol: {opt_stddev:.2%}<br>Sharpe: {opt_sharpe:.2f}"],
        hoverinfo='text'
    ))
    
    fig.add_trace(go.Scatter(
        x=[min_stddev],
        y=[min_return],
        mode='markers',
        marker=dict(color='#ffd600', size=10, symbol='diamond', line=dict(color='#f8fafc', width=1.5)),
        name="Mínima Volatilidade",
        text=[f"Mínima Volatilidade<br>Retorno: {min_return:.2%}<br>Vol: {min_stddev:.2%}"],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Fronteira Eficiente de Markowitz e Portfólios Simulados",
        xaxis_title="Volatilidade Anualizada (Desvio Padrão)",
        yaxis_title="Retorno Esperado Anualizado",
        template='plotly_dark'
    )
    apply_plotly_theme(fig)
    return fig


# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- Donut allocation chart — 4 arcs representing diversified assets -->
          <path d="M30 6 A24 24 0 0 1 54 30" stroke="#00ff87" stroke-width="5.5" stroke-linecap="round"/>
          <path d="M54 30 A24 24 0 0 1 30 54" stroke="#00d2ff" stroke-width="5.5" stroke-linecap="round"/>
          <path d="M30 54 A24 24 0 0 1 6 30" stroke="#ffd600" stroke-width="5.5" stroke-linecap="round"/>
          <path d="M6 30 A24 24 0 0 1 30 6" stroke="#a855f7" stroke-width="5.5" stroke-linecap="round"/>
          <!-- inner ring -->
          <circle cx="30" cy="30" r="11" fill="#080c14" stroke="#1e293b" stroke-width="1"/>
          <!-- center pulse -->
          <circle cx="30" cy="30" r="4" fill="#00ff87" opacity="0.9"/>
          <!-- tick marks at each junction -->
          <circle cx="30" cy="6" r="2" fill="#0e1524" stroke="#00ff87" stroke-width="1.5"/>
          <circle cx="54" cy="30" r="2" fill="#0e1524" stroke="#00d2ff" stroke-width="1.5"/>
          <circle cx="30" cy="54" r="2" fill="#0e1524" stroke="#ffd600" stroke-width="1.5"/>
          <circle cx="6"  cy="30" r="2" fill="#0e1524" stroke="#a855f7" stroke-width="1.5"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">Análise &amp; Otimização de Portfólio</h1>
        <p class="page-hero-subtitle">Construa carteiras de alta performance com Markowitz, HRP e métricas institucionais de risco/retorno.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Configurações
col_config1, col_config2 = st.columns(2)
with col_config1:
    lookback_opcao = st.selectbox(
        "Lookback da Otimização",
        ("2 Anos (Padrão)", "1 Ano", "3 Anos", "5 Anos", "6 Meses", "Personalizado (Dias)")
    )
with col_config2:
    today = datetime.date.today()
    if lookback_opcao == "6 Meses":
        data_inicio = today - datetime.timedelta(days=180)
        st.info(f"Data Inicial calculada: {data_inicio.strftime('%d/%m/%Y')}")
    elif lookback_opcao == "1 Ano":
        data_inicio = today - datetime.timedelta(days=365)
        st.info(f"Data Inicial calculada: {data_inicio.strftime('%d/%m/%Y')}")
    elif lookback_opcao == "2 Anos (Padrão)":
        data_inicio = today - datetime.timedelta(days=365 * 2)
        st.info(f"Data Inicial calculada: {data_inicio.strftime('%d/%m/%Y')}")
    elif lookback_opcao == "3 Anos":
        data_inicio = today - datetime.timedelta(days=365 * 3)
        st.info(f"Data Inicial calculada: {data_inicio.strftime('%d/%m/%Y')}")
    elif lookback_opcao == "5 Anos":
        data_inicio = today - datetime.timedelta(days=365 * 5)
        st.info(f"Data Inicial calculada: {data_inicio.strftime('%d/%m/%Y')}")
    else:
        lookback_dias = st.number_input("Dias de Lookback", min_value=30, max_value=5000, value=500, step=10)
        data_inicio = today - datetime.timedelta(days=lookback_dias)

# Exibir spinner rápido para o SELIC se necessário
with st.spinner("Buscando taxa SELIC acumulada..."):
    taxa_selic = get_selic_rate(data_inicio)


# Seleção de ações
data = pd.read_csv('acoes-listadas-b3.csv')
stocks = list(data['Ticker'].values)
stocks = get_sorted_tickers_by_liquidity(stocks)
tickers = st.multiselect("Selecione as ações do portfólio", stocks)
# Valor inicial
valor_inicial = st.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)

# Escolha modo: manual ou otimizado
modo = st.radio("Modo de alocação", (
    "Otimização de Markowitz (Média-Variância / Sharpe Máximo)",
    "Otimização Hierarchical Risk Parity (Machine Learning)",
    "Alocação Manual"
))

if len(tickers) == 0:
    st.warning("Selecione pelo menos uma ação.")
    st.stop()

if len(tickers) == 1:
    st.warning("Selecione pelo menos dois ativos para montar o portfólio.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

# Baixa dados com tela de carregamento glassmorphic
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Buscando cotações históricas dos ativos na B3...</div>
    </div>
    """, unsafe_allow_html=True)

data_yf = get_portfolio_prices(tickers_yf, data_inicio)
if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ['_'.join(col).strip() for col in data_yf.columns.values]

returns = data_yf.pct_change().dropna()
loading_placeholder.empty()

page_container = st.empty()

if st.button("Carregar Portfolio", type="primary", use_container_width=True):
# Spinner global
    with st.spinner('Carregando dados, aguarde...'):
        
    
       
        if "Manual" in modo:
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
        elif "Hierarchical" in modo:
            st.subheader("Otimização Hierarchical Risk Parity (HRP)")
            hrp = HRPOpt(returns)
            weights_hrp = hrp.optimize()
            peso_manual_df = pd.DataFrame.from_dict(weights_hrp, orient='index', columns=["Peso"])
            pesos_manuais_arr = peso_manual_df["Sample_Vol" if "Sample_Vol" in peso_manual_df.columns else "Peso"].values
        else:
            st.subheader("Otimização de Markowitz (Média-Variância)")
            mu = expected_returns.mean_historical_return(data_yf, frequency=252)
            S = risk_models.sample_cov(data_yf, frequency=252)
            selic_anual = (1 + taxa_selic) ** 252 - 1
            try:
                ef = EfficientFrontier(mu, S)
                raw_weights = ef.max_sharpe(risk_free_rate=selic_anual)
                cleaned_weights = ef.clean_weights()
            except Exception as e:
                st.warning(f"Otimização de Max Sharpe falhou (motivo: {str(e)}). Usando carteira de Mínima Volatilidade.")
                ef = EfficientFrontier(mu, S)
                raw_weights = ef.min_volatility()
                cleaned_weights = ef.clean_weights()
            peso_manual_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=["Peso"])
            pesos_manuais_arr = peso_manual_df["Peso"].values
        
            # Mostrar pesos
        st.subheader("Pesos do Portfólio (%)")
        peso_manual_df.index = peso_manual_df.index.str.replace(".SA","")
        st.dataframe((peso_manual_df*100).round(2).T)
        
        
        
        
        
        alloc_df = peso_manual_df.reset_index()
        alloc_df.columns = ["Ativo", "Peso"]
        
        fig_treemap = px.treemap(
            alloc_df,
            path=['Ativo'],
            values='Peso',
            color='Peso',
            color_continuous_scale='GnBu',
            title="Alocação do Portfólio (Treemap)"
        )
        apply_plotly_theme(fig_treemap)
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        if "Markowitz" in modo:
            section_header(ICO_FRONTIER, "Gráfico da Fronteira Eficiente", "h3")
            with st.spinner("Gerando fronteira eficiente e simulando portfólios..."):
                selic_anual = (1 + taxa_selic) ** 252 - 1
                fig_frontier = plot_efficient_frontier_and_random_portfolios(mu, S, returns, cleaned_weights, selic_anual)
                st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Heatmap de Correlação Interativo (Plotly)
        section_header(ICO_HEATMAP, "Heatmap de Correlação entre Ativos", "h3")
        corr_df = data_yf.corr()
        fig_corr = px.imshow(
            corr_df,
            x=corr_df.index.str.replace(".SA", "", regex=False),
            y=corr_df.columns.str.replace(".SA", "", regex=False),
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Correlação de Retornos Históricos",
            text_auto=".2f"
        )
        apply_plotly_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Cálculo do portfólio com os pesos escolhidos
        portfolio_returns = returns.dot(pesos_manuais_arr)
        
        # Obter os dados de benchmark BOVESPA e calcular o retorno acumulado
        bench = yf.download("^BVSP", start=data_inicio, progress=False)['Close'].squeeze()
        retorno_bench = bench.pct_change().dropna()
        
        # Alinhar datas do portfólio e do benchmark
        comum_idx = portfolio_returns.index.intersection(retorno_bench.index)
        portfolio_returns = portfolio_returns.loc[comum_idx]
        retorno_bench = retorno_bench.loc[comum_idx]
        
        # Calcular os retornos acumulados correspondentes
        cum_return = (1 + portfolio_returns).cumprod()
        portfolio_value = cum_return * valor_inicial
        
        retorno_cum_bench = (1 + retorno_bench).cumprod()
        bench_value = retorno_cum_bench * valor_inicial
        
        
        # Mostrar gráfico do valor do portfólio x BOVESPA
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, 
                                 mode='lines', name='Portfólio', line=dict(color='#00ff87', width=2.5)))
        fig.add_trace(go.Scatter(x=bench_value.index, y=bench_value, 
                                 mode='lines', name='IBOVESPA', line=dict(color='#ffd600', width=1.5, dash='dash')))
        fig.update_layout(title='Evolução do Valor do Portfólio vs Benchmark',
                          xaxis_title='Data', yaxis_title='Valor (R$)')
        apply_plotly_theme(fig)
        st.plotly_chart(fig)
        # Retornos mensais
        st.subheader("Retornos Mensais do Portfólio")
        
        fig = qs.plots.monthly_returns(portfolio_returns, show=False)
        apply_matplotlib_theme(fig)
        st.pyplot(fig)
        
        
        
        
        
        
        
        
        # Cálculos de Métricas
        total_return = (portfolio_value.iloc[-1]/valor_inicial - 1)*100
        vol_anual = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_val = sharpe(portfolio_returns, rf=taxa_selic)
        sortino_val = sortino(portfolio_returns, rf=taxa_selic)
        max_dd = max_drawdown(portfolio_returns) * 100
        
        cov_matrix = np.cov(portfolio_returns.squeeze(), retorno_bench.squeeze())  # matriz de covariância 2x2
        beta = cov_matrix[0,1] / cov_matrix[1,1]
        alfa = portfolio_returns.mean() - beta * retorno_bench.mean()
        alfa_val = alfa.values[0] if hasattr(alfa, "values") and len(alfa.values) > 0 else alfa
        r_quadrado = qs.stats.r_squared(portfolio_returns, retorno_bench)
        information_ratio = qs.stats.information_ratio(portfolio_returns, retorno_bench)
        
        # Desempenho Resumido em Cards (st.metric)
        section_header(ICO_CHART, "Desempenho Resumido da Carteira", "h3")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Retorno Total", f"{total_return:.2f}%")
        col_m2.metric("Volatilidade Anual", f"{vol_anual:.2f}%")
        col_m3.metric("Índice Sharpe", f"{sharpe_val:.2f}")
        col_m4.metric("Índice Sortino", f"{sortino_val:.2f}")
        col_m5.metric("Max Drawdown", f"{max_dd:.2f}%")

        # ── Painel de Decisão do Investidor ───────────────────────────────────
        st.markdown("---")
        section_header(ICO_TARGET, "Painel de Decisão do Investidor", "h3")

        # ── Score de Saúde do Portfólio ──────────────────────────────────────
        score = 0
        health_detalhes = []

        if sharpe_val > 1.0:
            score += 40; health_detalhes.append((ICO_OK,   "Sharpe excelente (>1.0)", "#00ff87"))
        elif sharpe_val > 0.5:
            score += 20; health_detalhes.append((ICO_WARN, "Sharpe razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append((ICO_CRIT, "Sharpe baixo (<0.5) — revise a alocação", "#ff3d5a"))

        if sortino_val > 1.0:
            score += 20; health_detalhes.append((ICO_OK,   "Sortino excelente (>1.0)", "#00ff87"))
        elif sortino_val > 0.5:
            score += 10; health_detalhes.append((ICO_WARN, "Sortino razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append((ICO_CRIT, "Sortino baixo — retornos negativos relevantes", "#ff3d5a"))

        if max_dd > -10:
            score += 20; health_detalhes.append((ICO_OK,   "Drawdown controlado (<10%)", "#00ff87"))
        elif max_dd > -20:
            score += 10; health_detalhes.append((ICO_WARN, "Drawdown moderado (10–20%)", "#ffd600"))
        else:
            health_detalhes.append((ICO_CRIT, "Drawdown severo (>20%) — risco de ruína elevado", "#ff3d5a"))

        alfa_anual = alfa_val * 252 * 100
        if alfa_anual > 5:
            score += 20; health_detalhes.append((ICO_OK,   f"Alfa anual positivo: {alfa_anual:.1f}%", "#00ff87"))
        elif alfa_anual > 0:
            score += 10; health_detalhes.append((ICO_WARN, f"Alfa marginal: {alfa_anual:.1f}%", "#ffd600"))
        else:
            health_detalhes.append((ICO_CRIT, f"Alfa negativo ({alfa_anual:.1f}%) — portfólio perde pro índice", "#ff3d5a"))

        pesos_arr_dec = np.array(pesos_manuais_arr)  # sempre definido independente do modo
        max_peso = pesos_arr_dec.max() * 100
        if max_peso <= 30:
            score += 10; health_detalhes.append((ICO_OK,   f"Concentração saudável (máx: {max_peso:.1f}%)", "#00ff87"))
        elif max_peso <= 50:
            health_detalhes.append((ICO_WARN, f"Concentração elevada (máx: {max_peso:.1f}%)", "#ffd600"))
        else:
            health_detalhes.append((ICO_CRIT, f"Hiper-concentração (máx: {max_peso:.1f}%) — diversifique", "#ff3d5a"))

        score_color = "#00ff87" if score >= 70 else "#ffd600" if score >= 40 else "#ff1744"
        score_label = "SAUDÁVEL" if score >= 70 else "ATENÇÃO" if score >= 40 else "CRÍTICO"

        col_score, col_details = st.columns([1, 2])
        with col_score:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:2px solid {score_color};
                        border-radius:16px;padding:1.5rem;text-align:center;
                        box-shadow:0 0 20px {score_color}33;">
                <div style="font-size:3.5rem;font-weight:900;color:{score_color};
                            font-family:'JetBrains Mono',monospace;
                            text-shadow:0 0 15px {score_color}66;">{score}</div>
                <div style="font-size:0.65rem;color:#94a3b8;letter-spacing:0.12em;margin-top:0.2rem;">DE 110 PONTOS</div>
                <div style="font-size:0.9rem;font-weight:700;color:{score_color};margin-top:0.5rem;
                            letter-spacing:0.08em;">{score_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_details:
            st.markdown("**Diagnóstico por indicador:**")
            for ico, msg, color in health_detalhes:
                diag_row(ico, msg, color)

        # ── Regime de Mercado (últimos 21 dias) ──────────────────────────────
        section_header(ICO_SIGNAL, "Regime de Mercado (Últimos 21 Dias)", "h4")
        retorno_recente = (1 + portfolio_returns.tail(21)).prod() - 1
        retorno_recente_bench = (1 + retorno_bench.tail(21)).prod() - 1
        outperformance_recente = (retorno_recente - retorno_recente_bench) * 100

        if retorno_recente > 0.03:
            regime_ico, regime_txt, regime_acao = ICO_UP,   "Alta",   "Manter exposição. Ativos com momentum positivo podem ser aumentados."
        elif retorno_recente < -0.03:
            regime_ico, regime_txt, regime_acao = ICO_DOWN, "Queda",  "Revise os pesos. Considere reduzir exposição ou adicionar proteção (ex: BOVA11 put)."
        else:
            regime_ico, regime_txt, regime_acao = ICO_FLAT, "Lateral","Consolidação. Bom momento para revisar correlações e rebalancear."

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Regime", regime_txt, delta=f"{retorno_recente*100:.2f}% (21d)")
        col_r2.metric("vs IBOVESPA (21d)", f"{outperformance_recente:+.2f}%",
                      delta="Outperform" if outperformance_recente > 0 else "Underperform")
        col_r3.metric("Volatilidade (21d)", f"{portfolio_returns.tail(21).std()*np.sqrt(252)*100:.1f}% a.a.")
        diag_row(ICO_IDEA, f"<b>Sugestão:</b> {regime_acao}", "#ffd600")

        # ── Índice HHI de Concentração ────────────────────────────────────────
        hhi = (pesos_arr_dec ** 2).sum() * 10000
        hhi_equiv = int(1 / (pesos_arr_dec ** 2).sum())
        n_ativos = len(pesos_arr_dec)

        col_hhi1, col_hhi2, col_hhi3 = st.columns(3)
        col_hhi1.metric("HHI Concentração", f"{hhi:.0f}",
                        delta="Baixo" if hhi < 2500 else "Moderado" if hhi < 5000 else "Alto",
                        delta_color="normal" if hhi < 2500 else "inverse")
        col_hhi2.metric("Ativos Efetivos", f"{hhi_equiv}/{n_ativos}")
        col_hhi3.metric("Maior Peso", f"{max_peso:.1f}%",
                        delta="OK" if max_peso <= 35 else "Concentrado",
                        delta_color="normal" if max_peso <= 35 else "inverse")

        if hhi > 5000:
            diag_row(ICO_CRIT, f"<b>Alta Concentração (HHI={hhi:.0f}):</b> Portfólio fortemente concentrado. Adicione ativos ou redistribua os pesos.", "#ff3d5a")
        elif hhi > 2500:
            diag_row(ICO_WARN, f"<b>Concentração Moderada (HHI={hhi:.0f}):</b> Apenas {hhi_equiv} ativos efetivos de {n_ativos}.", "#ffd600")
        else:
            diag_row(ICO_OK, f"<b>Boa Diversificação (HHI={hhi:.0f}):</b> {hhi_equiv} ativos efetivos — distribuição equilibrada.", "#00ff87")

        st.markdown("---")

        # Distribuição de retornos com estatísticas

        st.subheader("Distribuição dos Retornos Diários (%) e Estatísticas")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(portfolio_returns * 100, bins=50, kde=True, color='#00ff87', edgecolor='#080c14', alpha=0.65, ax=ax_hist)
        ax_hist.set_xlabel("Retornos Diários (%)")
        ax_hist.set_ylabel("Frequência")
        
        media = portfolio_returns.mean() * 100
        desvio = portfolio_returns.std() * 100
        curtose_val = kurtosis(portfolio_returns, fisher=True)
        assimetria_val = skew(portfolio_returns)
        
        stats_text = (f"Média: {media:.4f}%\n"
                      f"Desvio Padrão: {desvio:.4f}%\n"
                      f"Curtose (Fisher): {curtose_val:.4f}\n"
                      f"Assimetria: {assimetria_val:.4f}")
        
        props = dict(boxstyle='round', facecolor='#0e1524', edgecolor='#1e293b', alpha=0.95)
        ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes,
                     fontsize=10, color='#f8fafc', verticalalignment='top', horizontalalignment='right', bbox=props)
        ax_hist.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        apply_matplotlib_theme(fig_hist)
        st.pyplot(fig_hist)
        
        # Métricas Consolidadas
        detailed_stats = pd.DataFrame({
            "Métrica": [
                "Valor Inicial", "Valor Máximo Alcançado", "Valor Mínimo Alcançado", "Valor Final",
                "Retorno Médio Diário", "Alfa Anualizado vs IBOV", "Beta vs IBOVESPA",
                "R² vs IBOVESPA (%)", "Information Ratio", "VaR Diário (95%)", "CVaR Diário (95%)", "Tail Ratio"
            ],
            "Valor": [
                f"R$ {valor_inicial:,.2f}",
                f"R$ {portfolio_value.max():,.2f}",
                f"R$ {portfolio_value.min():,.2f}",
                f"R$ {portfolio_value.iloc[-1]:,.2f}",
                f"{portfolio_returns.mean()*100:.4f}%",
                f"{alfa_val*252*100:.2f}%",
                f"{beta:.4f}",
                f"{r_quadrado*100:.2f}%",
                f"{information_ratio:.4f}",
                f"{var(portfolio_returns)*100:.2f}%",
                f"{cvar(portfolio_returns)*100:.2f}%",
                f"{tail_ratio(portfolio_returns):.4f}"
            ]
        })
        section_header(ICO_METRICS, "Métricas Detalhadas do Portfólio", "h3")
        st.dataframe(detailed_stats, use_container_width=True, hide_index=True)
        # Retornos Anuais
        
        fig = qs.plots.yearly_returns(portfolio_returns, benchmark=retorno_bench, compounded=True, show=False)
        ax = fig.gca() if hasattr(fig, 'gca') else fig.axes[0]
        
        # Alterar legenda
        ax.legend(['Portfólio', 'IBOVESPA'])  # renomeia
        ax.set_title('Retornos Anuais (Portfólio vs IBOVESPA)')
        
        apply_matplotlib_theme(fig)
        st.pyplot(fig)
            
        
        
        st.subheader("Drawdown do Portfólio")
        # Drawdown
        st.subheader("Análise de Drawdown por Ativo")
        
        # Função para calcular drawdown
        def calcular_drawdown(series):
            cum_returns = (1 + series).cumprod()
            rolling_max = cum_returns.cummax()
            drawdown = (cum_returns - rolling_max) / rolling_max
            return drawdown
        
        # Calcula drawdown para cada ativo
        drawdowns_ativos = returns.apply(calcular_drawdown)
        
        # Calcula máximo drawdown e a data que ocorreu para cada ativo
        max_drawdowns = drawdowns_ativos.min()
        data_max_drawdowns = drawdowns_ativos.idxmin()
        
        df_drawdowns = pd.DataFrame({
            'Máximo Drawdown (%)': max_drawdowns * 100,
            'Data do Máximo Drawdown': data_max_drawdowns
        }).sort_values(by='Máximo Drawdown (%)')
        
        # Ajusta índice para mostrar ticker sem ".SA"
        df_drawdowns.index = df_drawdowns.index.str.replace(".SA", "", regex=False)
        
        st.dataframe(df_drawdowns.style.format({
            'Máximo Drawdown (%)': '{:.2f}%',
            'Data do Máximo Drawdown': lambda x: x.strftime('%Y-%m-%d')
        }))
        
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
            
        # Plot Drawdown
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.fill_between(drawdown.index, drawdown.values, 0, color='#ff1744', alpha=0.35)
        ax1.plot(drawdown.index, drawdown.values, color='#ff1744', linewidth=1.5)
        ax1.set_title("Drawdown do Portfólio", fontsize=12, fontweight='bold', pad=10)
        ax1.set_ylabel("Drawdown")
        ax1.set_xlabel("Data")
        ax1.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        apply_matplotlib_theme(fig1)
        st.pyplot(fig1)
            
        # Rolling Beta (60 dias)
        window = 60
        rolling_cov = portfolio_returns.rolling(window).cov(retorno_bench)
        rolling_var = retorno_bench.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        
        # Gráfico Rolling Beta
        st.subheader(f"Beta Móvel ({window} dias) vs IBOVESPA")
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        ax2.plot(rolling_beta.index, rolling_beta.values, color='#00d2ff', linewidth=2)
        ax2.axhline(1, color='#ffd600', linestyle='--', alpha=0.7, linewidth=1.5, label='Beta = 1')
        ax2.set_title(f"Rolling Beta {window} dias vs IBOVESPA", fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylabel("Beta")
        ax2.set_xlabel("Data")
        ax2.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        ax2.legend()
        fig2.autofmt_xdate(rotation=15)
        apply_matplotlib_theme(fig2)
        st.pyplot(fig2)
        
        # Gráfico Sharpe Móvel
        rolling_sharpe = (
            (portfolio_returns.rolling(window).mean() - taxa_selic) /
            portfolio_returns.rolling(window).std()
        )
        
        st.subheader(f"Índice de Sharpe Móvel ({window} dias)")
        fig_3, ax_3 = plt.subplots(figsize=(10, 4.5))
        ax_3.plot(rolling_sharpe.index, rolling_sharpe.values, color='#00ff87', linewidth=2, label='Sharpe Móvel')
        ax_3.axhline(0, color='#94a3b8', linestyle='--', alpha=0.7)
        ax_3.set_title(f"Índice de Sharpe Móvel ({window} dias) do Portfólio", fontsize=12, fontweight='bold', pad=10)
        ax_3.set_ylabel("Sharpe")
        ax_3.set_xlabel("Data")
        ax_3.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        ax_3.legend()
        fig_3.autofmt_xdate(rotation=45)
        fig_3.tight_layout()
        apply_matplotlib_theme(fig_3)
        st.pyplot(fig_3)
        
        section_header(ICO_RISK, "Análise de Contribuição de Risco", "h3")
        
        cov_matrix_rc = returns.cov()
        port_vol = np.sqrt(np.dot(pesos_manuais_arr.T, np.dot(cov_matrix_rc, pesos_manuais_arr)))
        marginal_contrib = np.dot(cov_matrix_rc, pesos_manuais_arr) / port_vol
        risk_contribution = pesos_manuais_arr * marginal_contrib  # risco absoluto de cada ativo
        risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100
        
        risk_df = pd.DataFrame({
            "Ativo": peso_manual_df.index,
            "Peso (%)": (pesos_manuais_arr*100).round(2),
            "RC (%)": risk_contribution_pct.round(2)
        })
        
        fig_rc = px.bar(
            risk_df,
            x="Ativo",
            y="RC (%)",
            color="RC (%)",
            color_continuous_scale="Viridis",
            title="Contribuição de Risco por Ativo (%)"
        )
        apply_plotly_theme(fig_rc)
        st.plotly_chart(fig_rc, use_container_width=True)
        
        # Salva variáveis para uso na aba Simulação e Relatório
        if "Manual" in modo:
            clean_modo = "Alocação Manual"
        elif "Hierarchical" in modo:
            clean_modo = "Otimização Hierarchical Risk Parity (HRP)"
        else:
            clean_modo = "Otimização de Markowitz (Média-Variância)"
            
        st.session_state["modo"] = clean_modo
        st.session_state["returns"] = returns
        st.session_state["peso_manual_df"] = peso_manual_df
        st.session_state["portfolio_returns"] = portfolio_returns
        st.session_state["retorno_bench"] = retorno_bench
        st.session_state["lookback"] = lookback_opcao
        
        # Garante que pesos manuais ficam disponíveis como dicionário
        if "Manual" in modo:
            st.session_state["pesos_manuais"] = pesos_manuais
        else:
            st.session_state["pesos_manuais"] = peso_manual_df["Peso"].to_dict()
                    
                    
                    
                    
                    
                    
                    
                    
                           
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                       
