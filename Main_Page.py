import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import pandas as pd
import seaborn as sns
import warnings
import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
import datetime
from scipy.stats import kurtosis, skew
from pypfopt import plotting
import re
import traceback
import time
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Configurar temas de plotagem escuros
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#080c14'
plt.rcParams['axes.facecolor'] = '#0e1524'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['font.family'] = 'sans-serif'

# Customização do Plotly para o tema Obsidian Neo-Financial
def apply_plotly_theme(fig):
    fig.update_layout(
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
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(14, 21, 36, 0.8)',
            bordercolor='#1e293b',
            borderwidth=1
        ),
        margin=dict(b=80)
    )
    return fig

def clean_numeric_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r'[^0-9,.\-]', '', regex=True)
    col = col.str.replace(',', '.')
    return pd.to_numeric(col, errors='coerce')

@st.cache_data(ttl=3600)
def get_fundamentus_data(tickers):
    """Busca dados fundamentalistas com retry automático."""
    for attempt in range(3):
        try:
            return pd.concat([fundamentus.get_papel(t) for t in tickers])
        except OSError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

@st.cache_data(ttl=3600)
def get_yfinance_data(tickers_yf, start, interval):
    """Busca cotações do Yahoo Finance com retry automático."""
    today = datetime.date.today()
    for attempt in range(3):
        try:
            return yf.download(tickers_yf, start=start, end=today, interval=interval)['Close']
        except OSError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

@st.cache_data(ttl=86400)
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

st.set_page_config(
    page_title="B3 Explorer — Análise Quantitativa de Ações",
    page_icon="b3.webp",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SVG Icon Library ─────────────────────────────────────────────────────────
def _svg(body, size=14):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
            f'{body}</svg>')

ICO_COMPASS = _svg('<circle cx="12" cy="12" r="10" stroke="#00ff87" stroke-width="1.8"/>'
                   '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="#00ff87"/>', 16)
ICO_SECTOR  = _svg('<rect x="3" y="3" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="14" y="3" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="3" y="16" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="14" y="12" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>', 16)
ICO_MARKET  = _svg('<path d="M3 3v18h18" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
                   '<path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>', 16)
ICO_METRICS = _svg('<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
                   '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
                   '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>', 16)
ICO_CHART   = _svg('<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
                   '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
                   '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>', 16)
ICO_STATS   = _svg('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" stroke="#a855f7" stroke-width="1.8"/>'
                   '<line x1="4" y1="22" x2="4" y2="15" stroke="#a855f7" stroke-width="1.8"/>', 16)
ICO_RULER   = _svg('<rect x="2" y="7" width="20" height="10" rx="2" stroke="#ffd600" stroke-width="1.8"/>'
                   '<line x1="6"  y1="7" x2="6"  y2="12" stroke="#ffd600" stroke-width="1.5"/>'
                   '<line x1="10" y1="7" x2="10" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                   '<line x1="14" y1="7" x2="14" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                   '<line x1="18" y1="7" x2="18" y2="12" stroke="#ffd600" stroke-width="1.5"/>', 16)
ICO_BOX     = _svg('<rect x="3" y="7" width="18" height="14" rx="2" stroke="#94a3b8" stroke-width="1.8"/>'
                   '<path d="M8 7V5a4 4 0 018 0v2" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="12" y1="12" x2="12" y2="16" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="10" y1="14" x2="14" y2="14" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>', 16)
ICO_RADAR   = _svg('<polygon points="12 2 22 8.5 22 19.5 12 22 2 19.5 2 8.5" stroke="#a855f7" stroke-width="1.8" fill="none"/>'
                   '<polygon points="12 6 18 10 18 17 12 19 6 17 6 10" stroke="#a855f7" stroke-width="1.2" fill="none" opacity="0.6"/>'
                   '<line x1="12" y1="2" x2="12" y2="22" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
                   '<line x1="2" y1="8.5" x2="22" y2="19.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
                   '<line x1="2" y1="19.5" x2="22" y2="8.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>', 16)
ICO_INFO    = _svg('<circle cx="12" cy="12" r="10" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<line x1="12" y1="16" x2="12" y2="12" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>'
                   '<line x1="12" y1="8" x2="12" y2="8.01" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>', 16)

def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f'{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True)

# Sidebar Principal
st.sidebar.markdown("""
<div style="padding:0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem">
  <span style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase">
    Navegação
  </span>
</div>
""", unsafe_allow_html=True)

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero main-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- X-axis baseline -->
          <line x1="4" y1="54" x2="56" y2="54" stroke="#1e293b" stroke-width="1.5"/>
          <!-- Candle 1 — bullish green -->
          <line x1="14" y1="10" x2="14" y2="50" stroke="#334155" stroke-width="1.5"/>
          <rect x="10" y="22" width="8" height="18" rx="2" fill="#00ff87"/>
          <!-- Candle 2 — bearish red -->
          <line x1="30" y1="8" x2="30" y2="46" stroke="#334155" stroke-width="1.5"/>
          <rect x="26" y="16" width="8" height="20" rx="2" fill="#ff3d5a"/>
          <!-- Candle 3 — bullish green, stronger -->
          <line x1="46" y1="6" x2="46" y2="44" stroke="#334155" stroke-width="1.5"/>
          <rect x="42" y="12" width="8" height="22" rx="2" fill="#00ff87"/>
          <!-- Trend line (cyan) -->
          <path d="M6 50 L22 36 L38 42 L54 18"
                stroke="#00d2ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          <!-- Arrow head -->
          <path d="M48 14 L54 18 L50 24"
                stroke="#00d2ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">B3 Explorer</h1>
        <p class="page-hero-subtitle">Plataforma quantitativa de análise de ações brasileiras — Fundamentalismo, Otimização de Portfólio e Simulação Monte Carlo.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Carrega lista de ações da B3 com setores para filtragem inicial

data = pd.read_csv('acoes-listadas-b3.csv')

if 'Setor' not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

# Cria listas de tickers e setores para seleção
stocks = list(data['Ticker'].values)
setores = sorted(data['Setor'].dropna().unique())
setores.insert(0, "Todos")

st.sidebar.subheader("Escolha o setor.")

# Permite filtro por setor na barra lateral
setores_selecionados = st.sidebar.multiselect(
    'Escolha um ou mais setores:', setores, default=[]
)


#selecionar Todos ou nada, mostra todos os tickers
if "Todos" in setores_selecionados or not setores_selecionados:
    tickers_filtrados = data['Ticker'].tolist()
else:
    tickers_filtrados = data[data['Setor'].isin(setores_selecionados)]['Ticker'].tolist()

# Ordenar por liquidez para colocar maiores empresas no topo
tickers_filtrados = get_sorted_tickers_by_liquidity(tickers_filtrados)

section_header(ICO_COMPASS, "Escolha ações para explorar", "h3")
tickers = st.multiselect('Escolha sua ação. Selecione a página desejada e as configurações na barra lateral.', tickers_filtrados)


# Só executa análise se houver pelo menos uma ação selecionada
if tickers:
    try:
        # 1. Configurações da barra lateral (movidas para o início para permitir o download conjunto)
        data_inicio = st.sidebar.date_input("Data Inicial", datetime.date(2025,1,1),
                                            min_value=datetime.date(2000,1,1),
                                            max_value=datetime.date.today())
        st.sidebar.header('Configurações')
        interval_selected = st.sidebar.selectbox('Intervalo', 
                                                 ['1d','1wk','1mo','3mo','6mo','1y'])

        # 2. Exibir tela de carregamento glassmorphic
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Buscando cotações e indicadores fundamentalistas na B3...</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Buscar dados usando funções cacheadas
        df = get_fundamentus_data(tickers)

        tickers_yf = [t + ".SA" for t in tickers]
        data_prices = get_yfinance_data(tickers_yf, data_inicio, interval_selected)

        # Se for Series (ticker único), converte para DataFrame e renomeia a coluna para o ticker
        if isinstance(data_prices, pd.Series):
            data_prices = data_prices.to_frame(name=tickers[0])
            data_prices.index = pd.to_datetime(data_prices.index)

        # 4. Limpar tela de carregamento
        loading_placeholder.empty()

        section_header(ICO_SECTOR, "Setor", "h3")
        st.write(df[['Empresa', 'Setor', 'Subsetor']].drop_duplicates(keep='last'))

        # Dataframe estatísticas básicas
        section_header(ICO_MARKET, "Informações de Mercado", "h3")
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']]
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

        # Limpa colunas numéricas para evitar erros de formatação
        for col in ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)", 
                    "Volume Médio (2 meses)", "Valor de Mercado"]:
            df_price[col] = clean_numeric_column(df_price[col]).fillna(0)

        format_dict = {
            "Cotação": "R$ {:,.2f}",
            "Mínimo (52 semanas)": "R$ {:,.2f}",
            "Máximo (52 semanas)": "R$ {:,.2f}",
            "Volume Médio (2 meses)": "{:,.0f}",
            "Valor de Mercado": "R$ {:,.0f}"
        }

        st.dataframe(df_price.style.format(format_dict), use_container_width=True)

        # Indicadores Fundamentalistas
        section_header(ICO_METRICS, "Indicadores Financeiros", "h3")
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA','PVP','Empresa']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA", "P/VP", "Empresa"]

        # Transforma tudo em numérico para poder filtrar e aplicar estilos
        for col in df_ind.columns.drop('Empresa'):
            df_ind[col] = clean_numeric_column(df_ind[col])

        # Corrige o bug de parsing da biblioteca fundamentus (removeu o decimal)
        for col in ["P/L", "EV/EBITDA", "P/VP"]:
            if col in df_ind.columns:
                df_ind[col] = df_ind[col] / 100.0

        # Colunas percentuais
        pct_cols = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC", "Dividend Yield", "Crescimento Receita 5 anos"]
        for col in pct_cols:
            df_ind[col] = df_ind[col]

        df_ind = df_ind.fillna(0)

        format_ind = {
            "Margem Líquida": "{:.2f}%",
            "Margem EBIT": "{:.2f}%",
            "ROE": "{:.2f}%",
            "ROIC": "{:.2f}%",
            "Dividend Yield": "{:.2f}%",
            "Crescimento Receita 5 anos": "{:.2f}%",
            "P/L": "{:.2f}",
            "EV/EBITDA": "{:.2f}",
            "P/VP": "{:.2f}",
            "Empresa": lambda x: x
        }
        
        # Filtro de indicadores
        st.markdown("""
<h4 style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
  <span>Filtros</span>
</h4>
""", unsafe_allow_html=True)

        # Organização das colunas
        col1, col2 = st.columns(2)

        with col1:
            min_ebit = st.number_input("Margem EBIT mínima (%)", value=0.0, step=0.1)
            min_roe = st.number_input("ROE mínimo (%)", value=0.0, step=0.1)
            min_margem_liq = st.number_input("Margem Líquida Mínima (%)", value=0.0, step=0.1)
            min_cresc_5a = st.number_input("Crescimento Receita 5 Anos Mínima (%)", value=0.0, step=0.1)
            
        with col2:
            min_dividend = st.number_input("Dividend Yield mínimo (%)", value=0.0, step=0.1)
            max_pl = st.number_input("P/L máximo", value=1000.0, step=0.1)
            min_roic = st.number_input("ROIC mínimo (%)", value=0.0, step=0.1)
            max_ev_ebitda = st.number_input("EV/EBITDA Máximo", value=1000.0, step=0.1)
            max_pvp = st.number_input("P/VP máximo", value=1000.0, step=0.1)
            

        # Formatação Condicional
        def highlight_val(val, min_val=None, max_val=None):
            if pd.isna(val):
                return ''
            if min_val is not None and val < min_val:
                return 'background-color: rgba(255, 23, 68, 0.15); color: #ff1744;'  # vermelho neon translúcido
            if max_val is not None and val > max_val:
                return 'background-color: rgba(255, 23, 68, 0.15); color: #ff1744;'
            return 'background-color: rgba(0, 230, 118, 0.15); color: #00ff87;'  # verde neon translúcido

        # Define as cores
        def style_indicators(row):
            styles = [''] * len(row)
            col_idx = {col: i for i, col in enumerate(row.index)}

            styles[col_idx['Margem EBIT']] = highlight_val(row['Margem EBIT'], min_val=min_ebit)
            styles[col_idx['ROE']] = highlight_val(row['ROE'], min_val=min_roe)
            styles[col_idx['Margem Líquida']] = highlight_val(row['Margem Líquida'], min_val=min_margem_liq)
            styles[col_idx['Crescimento Receita 5 anos']] = highlight_val(row['Crescimento Receita 5 anos'], min_val=min_cresc_5a)
            styles[col_idx['Dividend Yield']] = highlight_val(row['Dividend Yield'], min_val=min_dividend)
            styles[col_idx['EV/EBITDA']] = highlight_val(row['EV/EBITDA'], max_val=max_ev_ebitda)
            styles[col_idx['P/L']] = highlight_val(row['P/L'], max_val=max_pl)
            styles[col_idx['ROIC']] = highlight_val(row['ROIC'], min_val=min_roic)
            styles[col_idx['P/VP']] = highlight_val(row['P/VP'], max_val=max_pvp)
            

            return styles

        styled_ind = df_ind.style.format(format_ind).apply(style_indicators, axis=1)
        st.dataframe(styled_ind, use_container_width=True)

        # ── Comparação Visual de Múltiplos ───────────────────────────────────
        if len(tickers) > 1:
            st.markdown("""
<h4 style="display:flex;align-items:center;gap:6px;margin-top:1.5rem;margin-bottom:.4rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"></line>
    <line x1="12" y1="20" x2="12" y2="4"></line>
    <line x1="6" y1="20" x2="6" y2="14"></line>
  </svg>
  <span>Comparação Gráfica de Múltiplos</span>
</h4>
""", unsafe_allow_html=True)
            
            comparable_indicators = [
                "P/L", "P/VP", "EV/EBITDA", "ROE", "ROIC", "Dividend Yield", 
                "Margem Líquida", "Margem EBIT", "Crescimento Receita 5 anos"
            ]
            
            col_chart_sel, _ = st.columns([1, 1])
            with col_chart_sel:
                selected_comp_mult = st.selectbox(
                    "Selecione o indicador para o gráfico comparativo",
                    comparable_indicators,
                    key="comp_mult_select_key"
                )
            
            df_chart = df_ind[[selected_comp_mult]].copy()
            
            is_pct = selected_comp_mult in ["ROE", "ROIC", "Dividend Yield", "Margem Líquida", "Margem EBIT", "Crescimento Receita 5 anos"]
            text_labels = []
            for v in df_chart[selected_comp_mult].values:
                if pd.isna(v):
                    text_labels.append("N/D")
                elif is_pct:
                    text_labels.append(f"{v:.2f}%")
                else:
                    text_labels.append(f"{v:.2f}")
            
            fig_comp = go.Figure(go.Bar(
                x=df_chart.index.tolist(),
                y=df_chart[selected_comp_mult].values,
                marker=dict(
                    color=df_chart[selected_comp_mult].values,
                    colorscale="Viridis",
                    showscale=False
                ),
                text=text_labels,
                textposition="outside",
                textfont=dict(size=10, color="#f8fafc")
            ))
            
            fig_comp.update_layout(
                title=dict(
                    text=f"Comparativo de {selected_comp_mult} — Ações Selecionadas",
                    font=dict(size=14, color="#f8fafc")
                ),
                xaxis_title="Ação",
                yaxis_title=f"{selected_comp_mult} (%)" if is_pct else selected_comp_mult,
                height=350,
                margin=dict(t=50, b=40, l=40, r=40)
            )
            apply_plotly_theme(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Comparação de Múltiplos do Setor ──────────────────────────────────
        st.markdown("---")
        st.markdown("""
<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:.25rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
       style="vertical-align:-3px">
    <circle cx="7"  cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>
    <circle cx="17" cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>
    <line x1="10" y1="12" x2="14" y2="12" stroke="#00d2ff" stroke-width="1.8"/>
    <rect x="4" y="3" width="6" height="4" rx="1" fill="#00ff87" opacity="0.7"/>
    <rect x="14" y="3" width="6" height="4" rx="1" fill="#ffd600" opacity="0.7"/>
    <rect x="4" y="17" width="6" height="4" rx="1" fill="#a855f7" opacity="0.7"/>
    <rect x="14" y="17" width="6" height="4" rx="1" fill="#ff3d5a" opacity="0.7"/>
  </svg>
  Comparação de Múltiplos do Setor
</h3>
<p style="color:#94a3b8;font-size:0.85rem;margin-top:0;margin-bottom:1rem">
  Posicionamento da(s) ação(ões) selecionada(s) em relação a todos os pares do setor na B3.
</p>
""", unsafe_allow_html=True)

        # Múltiplos a comparar: (coluna fundamentus, nome display, menor=melhor?)
        MULTIPLES_CFG = [
            ("PL",         "P/L",          True,  "Valuation"),
            ("PVP",        "P/VP",         True,  "Valuation"),
            ("EV_EBITDA",  "EV/EBITDA",    True,  "Valuation"),
            ("EV_EBIT",    "EV/EBIT",      True,  "Valuation"),
            ("PSR",        "PSR",          True,  "Valuation"),
            ("ROE",        "ROE (%)",      False, "Rentabilidade"),
            ("ROIC",       "ROIC (%)",     False, "Rentabilidade"),
            ("Marg_EBIT",  "Margem EBIT (%)", False, "Rentabilidade"),
            ("Marg_Liquida","Marg. Líq. (%)", False, "Rentabilidade"),
            ("Div_Yield",  "Div. Yield (%)", False, "Yield"),
        ]
        COLS_NEEDED = [c[0] for c in MULTIPLES_CFG]

        # Detecta setor(es) das ações selecionadas
        setores_ativas = df['Setor'].dropna().unique().tolist()

        # Mapa ticker -> setor individual
        ticker_setor_map = {}
        for t in tickers:
            sr = data[data['Ticker'] == t]
            if not sr.empty:
                ticker_setor_map[t] = sr['Setor'].values[0]

        if not setores_ativas:
            st.info("Setor não identificado para comparação.")
        else:
            # Mostra setor de cada ativo individualmente
            setor_info = " | ".join([f"**{t}**: {ticker_setor_map.get(t, '?')}" for t in tickers])
            st.caption(f"Setores detectados: {setor_info}")

            # Busca todos os tickers do setor via fundamentus
            @st.cache_data(ttl=3600, show_spinner=False)
            def get_sector_peers(setores):
                import fundamentus.resultado as fzr
                try:
                    raw = fzr.get_resultado_raw()
                    # Mapeia as colunas do raw para os identificadores de MULTIPLES_CFG
                    rename_map = {
                        'P/L': 'PL',
                        'P/VP': 'PVP',
                        'EV/EBITDA': 'EV_EBITDA',
                        'EV/EBIT': 'EV_EBIT',
                        'PSR': 'PSR',
                        'ROE': 'ROE',
                        'ROIC': 'ROIC',
                        'Mrg Ebit': 'Marg_EBIT',
                        'Mrg. Líq.': 'Marg_Liquida',
                        'Div.Yield': 'Div_Yield'
                    }
                    df2 = pd.DataFrame(index=raw.index)
                    for src, dest in rename_map.items():
                        if src in raw.columns:
                            df2[dest] = raw[src]
                    df2 = df2.drop_duplicates(keep='first')
                    return df2
                except Exception:
                    return pd.DataFrame()

            with st.spinner("Buscando peers do setor..."):
                peers_raw = get_sector_peers(tuple(setores_ativas))

            if peers_raw.empty:
                st.warning("Não foi possível buscar os pares do setor.")
            else:
                # fundamentus.get_resultado() retorna DataFrame com tickers como índice
                # Filtra somente os múltiplos que existem
                cols_available = [c for c in COLS_NEEDED if c in peers_raw.columns]
                peers_df = peers_raw[cols_available].copy()

                # Converte tudo para numérico
                for col in cols_available:
                    peers_df[col] = pd.to_numeric(
                        peers_df[col].astype(str).str.replace(',', '.').str.replace(r'[^\d.\-]', '', regex=True),
                        errors='coerce'
                    )

                # Limpa outliers e valores não aplicáveis (por célula, sem remover a linha inteira)
                # Fundamentus usa 0 para indicar "não aplicável" em muitos múltiplos
                for mult, name, lower_better, _ in MULTIPLES_CFG:
                    if mult not in peers_df.columns:
                        continue
                    # Converte zeros exatos em NaN (0 = não aplicável no fundamentus)
                    peers_df.loc[peers_df[mult] == 0, mult] = pd.NA
                    if lower_better:
                        # Remove apenas valores negativos ou absurdamente altos (> 500)
                        peers_df.loc[
                            (peers_df[mult].notna()) &
                            ((peers_df[mult] < 0) | (peers_df[mult] >= 500)),
                            mult
                        ] = pd.NA

                peers_df = peers_df.dropna(how='all')

                # Tickers selecionados no setor
                selected_in_peers = [t for t in tickers if t in peers_df.index]
                if not selected_in_peers:
                    # tenta com .SA removido
                    selected_in_peers = tickers

                # ── Tabela de percentis ─────────────────────────────────────
                st.markdown("#### Posicionamento por Percentil")
                st.caption("Verde = favorável | Vermelho = desfavorável | Cinza = neutro. "
                           "O percentil indica onde a ação está no ranking do setor (100 = melhor).")

                rows = []
                for mult, name, lower_better, categoria in MULTIPLES_CFG:
                    if mult not in peers_df.columns:
                        continue
                    col_data = peers_df[mult].dropna()
                    if col_data.empty:
                        continue

                    for t in selected_in_peers:
                        if t not in peers_df.index or pd.isna(peers_df.loc[t, mult]):
                            continue
                        val = peers_df.loc[t, mult]

                        # Obtém o setor da empresa individual
                        setor_row = data[data['Ticker'] == t]
                        if not setor_row.empty:
                            setor_t = setor_row['Setor'].values[0]
                            tickers_do_setor_t = data[data['Setor'] == setor_t]['Ticker'].tolist()
                            if t not in tickers_do_setor_t:
                                tickers_do_setor_t.append(t)
                            col_data_sector = col_data[col_data.index.isin(tickers_do_setor_t)]
                        else:
                            col_data_sector = col_data

                        if col_data_sector.empty:
                            col_data_sector = col_data

                        setor_med = col_data_sector.median()
                        setor_mean = col_data_sector.mean()
                        n_peers = len(col_data_sector)

                        # Percentil: % de peers piores que esta ação neste múltiplo
                        if lower_better:
                            pct = (col_data_sector > val).sum() / n_peers * 100
                        else:
                            pct = (col_data_sector < val).sum() / n_peers * 100

                        # Veredicto
                        if pct >= 70:
                            veredicto = "Favorável"
                            cor = "#00ff87"
                        elif pct >= 40:
                            veredicto = "Neutro"
                            cor = "#ffd600"
                        else:
                            veredicto = "Desfavorável"
                            cor = "#ff3d5a"

                        rows.append({
                            "Ação": t,
                            "Setor": ticker_setor_map.get(t, "N/D"),
                            "Categoria": categoria,
                            "Múltiplo": name,
                            "Valor": round(val, 2),
                            "Mediana Setor": round(setor_med, 2),
                            "Média Setor": round(setor_mean, 2),
                            "Peers (n)": n_peers,
                            "Percentil": round(pct, 1),
                            "Veredicto": veredicto,
                            "_cor": cor,
                        })

                if rows:
                    rank_df = pd.DataFrame(rows)

                    def color_veredicto(val):
                        m = {"Favorável": "#00ff8722", "Neutro": "#ffd60022", "Desfavorável": "#ff3d5a22"}
                        c = {"Favorável": "#00ff87",   "Neutro": "#ffd600",   "Desfavorável": "#ff3d5a"}
                        return f'background-color:{m.get(val,"")};color:{c.get(val,"")};font-weight:600'

                    def color_pct(val):
                        if val >= 70: return 'color:#00ff87;font-weight:700'
                        if val >= 40: return 'color:#ffd600;font-weight:700'
                        return 'color:#ff3d5a;font-weight:700'

                    display_df = rank_df.drop(columns=["_cor"])
                    styled_rank = (
                        display_df.style
                        .applymap(color_veredicto, subset=["Veredicto"])
                        .applymap(color_pct,       subset=["Percentil"])
                        .format({"Valor": "{:.2f}", "Mediana Setor": "{:.2f}",
                                 "Média Setor": "{:.2f}", "Percentil": "{:.1f}%"})
                        .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "0.82rem"})
                    )
                    st.dataframe(styled_rank, use_container_width=True, hide_index=True)

                    # ── Gráfico Comparativo de Percentis no Setor ───────────
                    st.markdown("#### Performance Relativa no Setor")
                    st.caption("Percentil de posicionamento no setor para cada indicador (100% representa o melhor posicionamento no setor).")
                    
                    tab_labels_perf = [f"{t} ({ticker_setor_map.get(t, '?')})" for t in selected_in_peers]
                    tabs_perf = st.tabs(tab_labels_perf)
                    for i, ticker_s in enumerate(selected_in_peers):
                        with tabs_perf[i]:
                            setor_name = ticker_setor_map.get(ticker_s, 'N/D')
                            sub_rank = rank_df[rank_df["Ação"] == ticker_s]
                            n_peers_display = sub_rank["Peers (n)"].max() if not sub_rank.empty else 0
                            st.caption(f"Comparado contra **{int(n_peers_display)} peers** do setor **{setor_name}**")
                            fig_pct = px.bar(
                                sub_rank,
                                x="Múltiplo",
                                y="Percentil",
                                color="Ação",
                                color_discrete_sequence=['#00ff87'],
                                title=None,
                                labels={"Percentil": "Percentil no Setor (%)", "Múltiplo": "Múltiplo / Indicador"}
                            )
                            fig_pct.update_layout(
                                yaxis=dict(range=[0, 105], ticksuffix="%"),
                                height=380,
                                margin=dict(t=30, b=40, l=40, r=40),
                                showlegend=False
                            )
                            apply_plotly_theme(fig_pct)
                            st.plotly_chart(fig_pct, use_container_width=True)

                    # ── Gráfico de barras por múltiplo ───────────────────────
                    st.markdown("#### Distribuição do Setor por Múltiplo")
                    mult_opcoes = [m[1] for m in MULTIPLES_CFG if m[0] in peers_df.columns]
                    mult_sel = st.selectbox("Selecione o múltiplo para visualizar", mult_opcoes, key="mult_sel")

                    mult_key = next((m[0] for m in MULTIPLES_CFG if m[1] == mult_sel), None)
                    if mult_key and mult_key in peers_df.columns:
                        tab_labels_dist = [f"{t} ({ticker_setor_map.get(t, '?')})" for t in selected_in_peers]
                        tabs_dist = st.tabs(tab_labels_dist)
                        for i, ticker_s in enumerate(selected_in_peers):
                            with tabs_dist[i]:
                                setor_row = data[data['Ticker'] == ticker_s]
                                if not setor_row.empty:
                                    setor_t = setor_row['Setor'].values[0]
                                    tickers_do_setor_t = data[data['Setor'] == setor_t]['Ticker'].tolist()
                                    if ticker_s not in tickers_do_setor_t:
                                        tickers_do_setor_t.append(ticker_s)
                                    
                                    col_plot = peers_df[peers_df.index.isin(tickers_do_setor_t)][mult_key].dropna().sort_values()
                                    
                                    if col_plot.empty:
                                        st.warning(f"Dados indisponíveis para o setor de {ticker_s}.")
                                        continue
                                        
                                    bar_colors = ['#1e293b'] * len(col_plot)
                                    if ticker_s in col_plot.index:
                                        bar_colors[list(col_plot.index).index(ticker_s)] = '#00ff87'
                                        
                                    fig_mult = go.Figure(go.Bar(
                                        x=col_plot.index.tolist(),
                                        y=col_plot.values,
                                        marker_color=bar_colors,
                                        marker_line_width=0,
                                        text=[f"{v:.1f}" for v in col_plot.values],
                                        textposition="outside",
                                        textfont=dict(size=8, color="#94a3b8"),
                                    ))

                                    # Mediana
                                    med_val = col_plot.median()
                                    fig_mult.add_hline(
                                        y=med_val,
                                        line_dash="dash", line_color="#ffd600", line_width=1.5,
                                        annotation_text=f"Mediana: {med_val:.1f}",
                                        annotation_position="top left",
                                        annotation_font=dict(color="#ffd600", size=11)
                                    )
                                    fig_mult.update_layout(
                                        title=f"{mult_sel} — Peers de {ticker_s} no Setor: {setor_t} ({len(col_plot)} empresas)",
                                        xaxis_title="Ticker",
                                        yaxis_title=mult_sel,
                                        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                                        height=420,
                                        showlegend=False,
                                    )
                                    apply_plotly_theme(fig_mult)
                                    st.plotly_chart(fig_mult, use_container_width=True)

                    # ── Scorecard de valuation ────────────────────────────────
                    st.markdown("#### Scorecard de Valuation do Setor")
                    score_cols = st.columns(len(selected_in_peers)) if len(selected_in_peers) > 1 else [st]

                    for i, ticker_s in enumerate(selected_in_peers):
                        sub = rank_df[rank_df["Ação"] == ticker_s]
                        if sub.empty:
                            continue
                        total_score = sub["Percentil"].mean()
                        fav  = (sub["Veredicto"] == "Favorável").sum()
                        neut = (sub["Veredicto"] == "Neutro").sum()
                        desf = (sub["Veredicto"] == "Desfavorável").sum()

                        sc = "#00ff87" if total_score >= 60 else "#ffd600" if total_score >= 40 else "#ff3d5a"
                        label = "ATRATIVO" if total_score >= 60 else "NEUTRO" if total_score >= 40 else "CARO/FRACO"

                        with score_cols[i] if len(selected_in_peers) > 1 else score_cols[0]:
                            st.markdown(f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:2px solid {sc};
            border-radius:14px;padding:1.2rem;text-align:center;
            box-shadow:0 0 18px {sc}33;margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;letter-spacing:.06em">{ticker_s}</div>
  <div style="font-size:2.8rem;font-weight:900;color:{sc};font-family:'JetBrains Mono',monospace;
              text-shadow:0 0 12px {sc}66;line-height:1.1">{total_score:.0f}</div>
  <div style="font-size:0.6rem;color:#64748b;letter-spacing:.12em">PERCENTIL MÉDIO</div>
  <div style="font-size:0.8rem;font-weight:700;color:{sc};margin-top:.4rem;letter-spacing:.06em">{label}</div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:.6rem;display:flex;justify-content:center;gap:.8rem">
    <span style="color:#00ff87">✓ {fav} fav.</span>
    <span style="color:#ffd600">~ {neut} neut.</span>
    <span style="color:#ff3d5a">✗ {desf} desf.</span>
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    st.info("Nenhum ticker selecionado encontrado nos dados de peers do setor.")

        st.markdown("---")

        # Os dados de cotação (data_prices) já foram baixados e processados no início do bloco

        
        section_header(ICO_CHART, "Cotação Histórica", "h3")
        st.write(data_prices)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Custom palette with vibrant tones
        palette = sns.color_palette("muted", len(data_prices.columns)) if len(data_prices.columns) > 1 else ['#00d2ff']
        sns.lineplot(data=data_prices, ax=ax, palette=palette, linewidth=2)
        ax.set_xlabel("Período")
        ax.set_ylabel("Cotação (R$)")
        ax.set_title("Cotação do(s) Ativo(s)", fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
        st.pyplot(fig)

        returns = data_prices.pct_change()

        # Descrições yfinance
        descriptions = []
        for t in tickers_yf:
            try:
                info = yf.Ticker(t).get_info()
                descriptions.append(info.get('longBusinessSummary', 'Não disponível'))
            except:
                descriptions.append('Não disponível')

        df_desc = pd.DataFrame(descriptions, index=tickers, columns=["Descrição"])
        section_header(ICO_INFO, "Descrição da Empresa", "h3")
        st.table(df_desc)

    except OSError as e:
        st.cache_data.clear()
        st.error(f"Erro de I/O ao buscar dados. O cache foi limpo automaticamente.")
        st.caption(f"Detalhe técnico: {e}")
        if st.button("🔄 Tentar novamente", key="retry_os_error"):
            st.rerun()
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        st.caption(f"```\n{traceback.format_exc()}\n```")
else:
    st.info("Selecione pelo menos uma ação para iniciar a análise.")






