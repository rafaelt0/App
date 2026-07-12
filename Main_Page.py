import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import datetime
import traceback

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.ui import load_css, loading_overlay, render_flow_sidebar, section_header
from utils.market_data import (
    clean_numeric_column,
    get_full_market_data,
    get_sorted_tickers_by_liquidity,
)
from utils.icons import (
    ICO_ALERT,
    ICO_BOLT,
    ICO_BULB,
    ICO_CHART,
    ICO_CHECK_SM,
    ICO_COMPASS,
    ICO_FILTER,
    ICO_INFO,
    ICO_MARKET,
    ICO_METRICS,
    ICO_NEWS,
    ICO_RADAR,
    ICO_SECTOR,
    ICO_SHIELD,
    ICO_STAR,
    ICO_STATS,
    ICO_X_SM,
)
from utils.formatting import (
    extract_debt_metric,
    format_large_br_currency,
    format_large_number,
    get_ev_ebitda_context,
)
from utils.home_data import (
    COLS_NEEDED,
    MULTIPLES_CFG,
    get_fundamentus_data,
    get_sector_peers,
    get_yfinance_data,
)
from utils.home_render import (
    color_pct,
    color_veredicto,
    get_ticker_setor,
    render_debt_panel,
    render_hist_section,
    render_price_cards,
    render_sector_cards,
    render_star_button,
    render_ticker_cards,
)

import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# Configurar temas de plotagem escuros
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#080c14"
plt.rcParams["axes.facecolor"] = "#0e1524"
plt.rcParams["text.color"] = "#f8fafc"
plt.rcParams["axes.labelcolor"] = "#94a3b8"
plt.rcParams["xtick.color"] = "#94a3b8"
plt.rcParams["ytick.color"] = "#94a3b8"
plt.rcParams["grid.color"] = "#1e293b"
plt.rcParams["font.family"] = "sans-serif"


st.set_page_config(
    page_title="B3Lab — Análise Quantitativa de Ações",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.logo("logo.svg", icon_image="favicon.svg")

render_flow_sidebar(active_step=1, pending_opacities=[0.45, 0.35, 0.25])

# CSS customizado
load_css()


# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# Carrega lista de ações da B3 com setores para filtragem inicial

data = pd.read_csv("acoes-listadas-b3.csv")

if "Setor" not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

# Cria listas de tickers e setores para seleção
stocks = list(data["Ticker"].values)
setores = sorted(data["Setor"].dropna().unique())
setores.insert(0, "Todos")
_ticker_setor = dict(zip(data["Ticker"], data["Setor"]))

# ── Watchlist (sidebar) ───────────────────────────────────────────────────────
_watchlist = _db.wl_get()
if _watchlist:
    st.sidebar.markdown(
        f'<div class="sidebar-section-label" style="color:#ffd600">{ICO_STAR} Favoritos</div>',
        unsafe_allow_html=True,
    )
    for _wt in _watchlist:
        _c1, _c2 = st.sidebar.columns([5, 1])
        with _c1:
            if st.button(
                _wt,
                key=f"wl_load_{_wt}",
                use_container_width=True,
                help=_ticker_setor.get(_wt, ""),
            ):
                _cur = st.session_state.get("selected_tickers", [])
                if _wt not in _cur:
                    st.session_state["selected_tickers"] = _cur + [_wt]
                st.rerun()
        with _c2:
            if st.button("✕", key=f"wl_rm_{_wt}", help="Remover dos favoritos"):
                _db.wl_remove(_wt)
                st.rerun()
    st.sidebar.markdown(
        "<div style='margin-bottom:0.75rem'></div>", unsafe_allow_html=True
    )

st.sidebar.markdown(
    f'<div class="sidebar-section-label">{ICO_FILTER} Escolha o setor</div>',
    unsafe_allow_html=True,
)

# Aplica qualquer setor pré-selecionado pelos atalhos "Explorar por setor"
# (que rodam depois deste widget ser instanciado, por isso usam staging)
if "_pending_sectors" in st.session_state:
    st.session_state["setores_selecionados"] = st.session_state.pop("_pending_sectors")

# Permite filtro por setor na barra lateral
setores_selecionados = st.sidebar.multiselect(
    "Escolha um ou mais setores:", setores, default=[], key="setores_selecionados"
)

if st.session_state.get("selected_tickers"):
    if st.sidebar.button("Limpar seleção de ações", use_container_width=True):
        st.session_state["selected_tickers"] = []
        st.rerun()

# selecionar Todos ou nada, mostra todos os tickers
if "Todos" in setores_selecionados or not setores_selecionados:
    tickers_filtrados = data["Ticker"].tolist()
else:
    tickers_filtrados = data[data["Setor"].isin(setores_selecionados)][
        "Ticker"
    ].tolist()

# Ordenar por liquidez para colocar maiores empresas no topo
tickers_filtrados = get_sorted_tickers_by_liquidity(tickers_filtrados)

section_header(ICO_COMPASS, "Escolha ações para explorar", "h3")
n_disponíveis = len(tickers_filtrados)
setor_label = (
    "todos os setores"
    if (not setores_selecionados or "Todos" in setores_selecionados)
    else ", ".join(setores_selecionados[:2])
    + ("..." if len(setores_selecionados) > 2 else "")
)
st.caption(
    f"{n_disponíveis} ações disponíveis em {setor_label}, ordenadas por liquidez."
)

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []

# Apply any ticker selection staged by widgets below (which run after this
# key's widget is already instantiated, so they can't write to it directly)
if "_pending_tickers" in st.session_state:
    st.session_state["selected_tickers"] = st.session_state.pop("_pending_tickers")

# Remove any stale tickers that are no longer in the filtered list
st.session_state["selected_tickers"] = [
    t for t in st.session_state["selected_tickers"] if t in tickers_filtrados
]

tickers = st.multiselect(
    "Escolha sua ação. Selecione a página desejada e as configurações na barra lateral.",
    options=tickers_filtrados,
    format_func=lambda t: f"{t}  ·  {_ticker_setor.get(t, '')}",
    key="selected_tickers",
)

if not tickers:
    # ── Onboarding ─────────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="margin:1.5rem 0 0.5rem 0;padding:1.5rem;background:linear-gradient(135deg,rgba(0,255,135,0.04) 0%,rgba(0,210,255,0.04) 100%);border:1px solid rgba(0,255,135,0.12);border-radius:14px;">
  <h3 style="margin:0 0 0.3rem 0;font-size:1.1rem;background:linear-gradient(135deg,#f8fafc 40%,#00ff87 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Bem-vindo ao B3 Explorer
  </h3>
  <p style="color:#94a3b8;font-size:0.85rem;margin:0 0 1.2rem 0;line-height:1.6">
    Plataforma de análise quantitativa de ações da B3. Escolha um ticker no campo acima
    ou clique em uma ação abaixo para começar.
  </p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00ff87;text-transform:uppercase;margin-bottom:0.5rem">① Escolha</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Filtre por setor na barra lateral ou digite o código da ação no campo de busca acima.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00d2ff;text-transform:uppercase;margin-bottom:0.5rem">② Analise</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Veja P/L, EV/EBITDA, ROE, ROIC, endividamento e posição no setor vs peers automaticamente.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a855f7;text-transform:uppercase;margin-bottom:0.5rem">③ Valoração</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Use a página <b style="color:#a855f7">Valuation</b> para o DCF completo McKinsey/Koller em 8 etapas.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Quick-start grid
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#64748b;'
        'text-transform:uppercase;margin:1.2rem 0 0.6rem 0">Início rápido — ações populares</div>',
        unsafe_allow_html=True,
    )

    _QUICK = [
        ("WEGE3", "Máquinas", "#00ff87"),
        ("PETR4", "Petróleo", "#ffd600"),
        ("VALE3", "Mineração", "#f87171"),
        ("ITUB4", "Banco", "#00d2ff"),
        ("RENT3", "Locação", "#a855f7"),
        ("ABEV3", "Bebidas", "#fb923c"),
        ("EGIE3", "Energia", "#34d399"),
        ("RADL3", "Farmácia", "#60a5fa"),
    ]
    _cols = st.columns(4)
    for i, (tkr, setor, cor) in enumerate(_QUICK):
        with _cols[i % 4]:
            st.markdown(
                f'<div style="text-align:center;padding:0.2rem 0 0.1rem 0;'
                f"font-size:0.62rem;color:{cor};font-weight:700;text-transform:uppercase;"
                f'letter-spacing:0.06em">{setor}</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                tkr,
                key=f"qs_{tkr}",
                use_container_width=True,
                help=f"Selecionar {tkr} — {setor}",
            ):
                st.session_state["_pending_tickers"] = [tkr]
                st.rerun()

    # Sector shortcuts
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#64748b;'
        'text-transform:uppercase;margin:1.4rem 0 0.6rem 0">Explorar por setor</div>',
        unsafe_allow_html=True,
    )
    # Rótulos amigáveis mapeados para os valores reais da coluna "Setor" do CSV
    _SETORES_DEST = [
        ("Bancos", "Intermediários Financeiros"),
        ("Petróleo e Gás", "Petróleo, Gás e Biocombustíveis"),
        ("Mineração", "Mineração"),
        ("Energia Elétrica", "Energia Elétrica"),
        ("Tecnologia", "Programas e Serviços"),
        ("Bebidas", "Bebidas"),
        ("Saúde", "Serv.Méd.Hospit. Análises e Diagnósticos"),
        ("Varejo", "Comércio"),
    ]
    _sc = st.columns(4)
    for i, (_label, _real_setor) in enumerate(_SETORES_DEST):
        with _sc[i % 4]:
            if st.button(_label, key=f"setor_qs_{_label}", use_container_width=True):
                if _real_setor in setores:
                    st.session_state["_pending_sectors"] = [_real_setor]
                st.rerun()

    st.markdown(
        '<div style="margin-top:1.2rem;padding:0.75rem 1rem;background:rgba(0,0,0,0.2);'
        'border-radius:8px;border-left:3px solid #334155">'
        f'<span style="font-size:0.78rem;color:#64748b">{ICO_BULB} <b style="color:#94a3b8">Dica:</b> '
        "Use o campo de busca acima para digitar qualquer ticker da B3. "
        "O filtro de setor na barra lateral reduz a lista para facilitar a escolha.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

if tickers:
    st.markdown(
        f"""
    <div style="background:rgba(0,255,135,0.05);border:1px solid rgba(0,255,135,0.2);border-radius:10px;padding:0.6rem 1rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
        <span style="font-size:0.85rem;color:#94a3b8;">{len(tickers)} ação(ões) selecionada(s) — analise o portfólio completo na página <strong style="color:#00ff87">Portfolio</strong></span>
        <span style="font-size:0.75rem;color:#00ff87;font-family:'JetBrains Mono',monospace;font-weight:700;">← barra lateral</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Analisar",
        type="primary",
        use_container_width=True,
        key="btn_analisar",
    ):
        st.session_state["analyzed_tickers"] = list(tickers)

# Só executa análise depois que o usuário clica em "Analisar" para a seleção atual
analyzed_tickers = st.session_state.get("analyzed_tickers", [])
ready_to_analyze = bool(tickers) and analyzed_tickers == tickers

if tickers and not ready_to_analyze:
    st.info("Clique em **Analisar** para buscar os indicadores das ações selecionadas.")

if ready_to_analyze:
    try:
        # 1. Buscar dados usando funções cacheadas, com animação de carregamento
        with loading_overlay(
            "Buscando indicadores fundamentalistas na B3...", tickers=tickers
        ):
            df = get_fundamentus_data(tickers)

        tickers_yf = [t + ".SA" for t in tickers]

        # Export button for fundamental data
        try:
            export_cols = [
                "Empresa",
                "Setor",
                "Subsetor",
                "Cotacao",
                "Min_52_sem",
                "Max_52_sem",
                "Marg_Liquida",
                "Marg_EBIT",
                "ROE",
                "ROIC",
                "Div_Yield",
                "Cres_Rec_5a",
                "PL",
                "EV_EBITDA",
                "PVP",
            ]
            export_cols_exist = [c for c in export_cols if c in df.columns]
            df_export = df[export_cols_exist].copy()
            csv_data = df_export.to_csv().encode("utf-8")
            st.download_button(
                label="Exportar dados fundamentalistas (CSV)",
                data=csv_data,
                file_name=f"fundamentus_{'+'.join(tickers)}_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        except Exception:
            pass

        section_header(ICO_SECTOR, "Setor", "h3")
        df_sector = df[["Empresa", "Setor", "Subsetor"]]

        if len(tickers) > 1:
            tabs_sector = st.tabs(tickers)
            for tab, ticker in zip(tabs_sector, tickers):
                with tab:
                    if ticker in df_sector.index:
                        row_s = df_sector.loc[ticker]
                        if isinstance(row_s, pd.DataFrame):
                            row_s = row_s.iloc[-1]
                        render_sector_cards(ticker, row_s)
                    else:
                        st.warning(f"Sem dados de setor para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df_sector.index:
                row_s = df_sector.loc[ticker]
                if isinstance(row_s, pd.DataFrame):
                    row_s = row_s.iloc[-1]
                render_sector_cards(ticker, row_s)
            else:
                st.warning(f"Sem dados de setor para {ticker}")

        # Informações de mercado em caixas estilizadas
        section_header(ICO_MARKET, "Informações de Mercado", "h3")
        df_price = df[
            [
                "Cotacao",
                "Min_52_sem",
                "Max_52_sem",
                "Vol_med_2m",
                "Valor_de_mercado",
                "Data_ult_cot",
            ]
        ]
        df_price.columns = [
            "Cotação",
            "Mínimo (52 semanas)",
            "Máximo (52 semanas)",
            "Volume Médio (2 meses)",
            "Valor de Mercado",
            "Data Última Cotação",
        ]

        # Limpa colunas numéricas para evitar erros de formatação
        for col in [
            "Cotação",
            "Mínimo (52 semanas)",
            "Máximo (52 semanas)",
            "Volume Médio (2 meses)",
            "Valor de Mercado",
        ]:
            df_price[col] = clean_numeric_column(df_price[col]).fillna(0)

        if len(tickers) > 1:
            tabs_price = st.tabs(tickers)
            for tab, ticker in zip(tabs_price, tickers):
                with tab:
                    if ticker in df_price.index:
                        row_p = df_price.loc[ticker]
                        if isinstance(row_p, pd.DataFrame):
                            row_p = row_p.iloc[-1]
                        render_price_cards(ticker, row_p)
                    else:
                        st.warning(f"Sem dados de mercado para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df_price.index:
                row_p = df_price.loc[ticker]
                if isinstance(row_p, pd.DataFrame):
                    row_p = row_p.iloc[-1]
                render_price_cards(ticker, row_p)
            else:
                st.warning(f"Sem dados de mercado para {ticker}")

        # Indicadores Fundamentalistas
        section_header(ICO_METRICS, "Indicadores Financeiros", "h3")
        df_ind = df[
            [
                "Marg_Liquida",
                "Marg_EBIT",
                "ROE",
                "ROIC",
                "Div_Yield",
                "Cres_Rec_5a",
                "PL",
                "EV_EBITDA",
                "PVP",
                "Empresa",
            ]
        ].drop_duplicates(keep="last")
        df_ind.columns = [
            "Margem Líquida",
            "Margem EBIT",
            "ROE",
            "ROIC",
            "Dividend Yield",
            "Crescimento Receita 5 anos",
            "P/L",
            "EV/EBITDA",
            "P/VP",
            "Empresa",
        ]

        # Transforma tudo em numérico para poder filtrar e aplicar estilos
        for col in df_ind.columns.drop("Empresa"):
            df_ind[col] = clean_numeric_column(df_ind[col])

        # Corrige o bug de parsing da biblioteca fundamentus (removeu o decimal)
        for col in ["P/L", "EV/EBITDA", "P/VP"]:
            if col in df_ind.columns:
                df_ind[col] = df_ind[col] / 100.0

        # Colunas percentuais
        pct_cols = [
            "Margem Líquida",
            "Margem EBIT",
            "ROE",
            "ROIC",
            "Dividend Yield",
            "Crescimento Receita 5 anos",
        ]
        for col in pct_cols:
            df_ind[col] = df_ind[col]

        df_ind = df_ind.fillna(0)

        # Remove duplicate indices if any
        df_ind = df_ind[~df_ind.index.duplicated(keep="last")]

        # Exibição dos cards
        if len(tickers) > 1:
            tabs_tickers = st.tabs(tickers)
            for idx, ticker in enumerate(tickers):
                with tabs_tickers[idx]:
                    render_star_button(ticker)
                    if ticker in df_ind.index:
                        render_ticker_cards(
                            df_ind.loc[ticker], setor=get_ticker_setor(df, ticker)
                        )
        else:
            ticker = tickers[0]
            render_star_button(ticker)
            if ticker in df_ind.index:
                render_ticker_cards(df_ind.loc[ticker], setor=get_ticker_setor(df, ticker))

        # ── Histórico Fundamentalista ─────────────────────────────────────────
        st.markdown("---")
        section_header(ICO_CHART, "Histórico Fundamentalista", "h3")
        st.caption(
            "Evolução anual de receita, lucro, margens e ROE — últimos 4 anos (fonte: yfinance / relatórios anuais)."
        )

        if len(tickers) > 1:
            tabs_hist = st.tabs(tickers)
            for _h_idx, _h_tkr in enumerate(tickers):
                with tabs_hist[_h_idx]:
                    render_hist_section(_h_tkr)
        else:
            render_hist_section(tickers[0])

        # ── Saúde Financeira ─────────────────────────────────────────────────
        st.markdown("---")
        section_header(ICO_SHIELD, "Saúde Financeira", "h3")
        st.caption(
            "Endividamento e liquidez da empresa. "
            "Dívida/PL acima de 3x e Liquidez abaixo de 1x são sinais de alerta."
        )

        if len(tickers) > 1:
            tabs_debt = st.tabs(tickers)
            for tab, ticker in zip(tabs_debt, tickers):
                with tab:
                    if ticker in df.index:
                        row_debt = df.loc[ticker]
                        if isinstance(row_debt, pd.DataFrame):
                            row_debt = row_debt.iloc[-1]
                        render_debt_panel(ticker, row_debt)
                    else:
                        st.warning(f"Sem dados para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df.index:
                row_debt = df.loc[ticker]
                if isinstance(row_debt, pd.DataFrame):
                    row_debt = row_debt.iloc[-1]
                render_debt_panel(ticker, row_debt)

        # ── Link para página de Valuation ────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.25);'
            'border-radius:10px;padding:1rem 1.25rem;margin:0.5rem 0">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" '
            'stroke="#a855f7" stroke-width="2" stroke-linecap="round"><line x1="12" y1="1" x2="12" y2="23"/>'
            '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" fill="none"/></svg>'
            '<span style="color:#a855f7;font-weight:700;font-size:0.95rem">Valuation McKinsey/Koller</span>'
            "</div>"
            '<p style="color:#94a3b8;font-size:0.82rem;margin:0">'
            "O módulo de valuation completo (Enterprise DCF, NOPLAT, ROIC, Continuing Value, WACC, "
            'Sensitivity, Múltiplos) foi movido para a página <b style="color:#f8fafc">Valuation</b> '
            "no menu lateral — implementado com a metodologia McKinsey/Koller "
            "(<i>Measuring and Managing the Value of Companies</i>)."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Comparação Visual de Múltiplos ───────────────────────────────────
        if len(tickers) > 1:
            st.markdown(
                """
<h4 style="display:flex;align-items:center;gap:6px;margin-top:1.5rem;margin-bottom:.4rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"></line>
    <line x1="12" y1="20" x2="12" y2="4"></line>
    <line x1="6" y1="20" x2="6" y2="14"></line>
  </svg>
  <span>Comparação Gráfica de Múltiplos</span>
</h4>
""",
                unsafe_allow_html=True,
            )

            comparable_indicators = [
                "P/L",
                "P/VP",
                "EV/EBITDA",
                "ROE",
                "ROIC",
                "Dividend Yield",
                "Margem Líquida",
                "Margem EBIT",
                "Crescimento Receita 5 anos",
            ]

            col_chart_sel, _ = st.columns([1, 1])
            with col_chart_sel:
                selected_comp_mult = st.selectbox(
                    "Selecione o indicador para o gráfico comparativo",
                    comparable_indicators,
                    key="comp_mult_select_key",
                )

            df_chart = df_ind[[selected_comp_mult]].copy()

            is_pct = selected_comp_mult in [
                "ROE",
                "ROIC",
                "Dividend Yield",
                "Margem Líquida",
                "Margem EBIT",
                "Crescimento Receita 5 anos",
            ]
            text_labels = []
            for v in df_chart[selected_comp_mult].values:
                if pd.isna(v):
                    text_labels.append("N/D")
                elif selected_comp_mult == "EV/EBITDA" and abs(v) < 0.01:
                    text_labels.append("N/A")
                elif is_pct:
                    text_labels.append(f"{v:.2f}%")
                else:
                    text_labels.append(f"{v:.2f}")

            fig_comp = go.Figure(
                go.Bar(
                    x=df_chart.index.tolist(),
                    y=df_chart[selected_comp_mult].values,
                    marker=dict(
                        color=df_chart[selected_comp_mult].values,
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    text=text_labels,
                    textposition="outside",
                    textfont=dict(size=10, color="#f8fafc"),
                )
            )

            fig_comp.update_layout(
                title=dict(
                    text=f"Comparativo de {selected_comp_mult} — Ações Selecionadas",
                    font=dict(size=14, color="#f8fafc"),
                ),
                xaxis_title="Ação",
                yaxis_title=f"{selected_comp_mult} (%)"
                if is_pct
                else selected_comp_mult,
                height=350,
                margin=dict(t=50, b=40, l=40, r=40),
            )
            apply_plotly_theme(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Comparação de Múltiplos do Setor ──────────────────────────────────
        st.markdown("---")
        st.markdown(
            """
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
""",
            unsafe_allow_html=True,
        )

        rank_df = pd.DataFrame()  # Inicializa; será populado se houver dados de peers

        # Detecta setor(es) das ações selecionadas
        setores_ativas = df["Setor"].dropna().unique().tolist()

        # Mapa ticker -> setor individual
        ticker_setor_map = {}
        for t in tickers:
            sr = data[data["Ticker"] == t]
            if not sr.empty:
                ticker_setor_map[t] = sr["Setor"].values[0]

        if not setores_ativas:
            st.info("Setor não identificado para comparação.")
        else:
            # Mostra setor de cada ativo individualmente
            setor_info = " | ".join(
                [f"**{t}**: {ticker_setor_map.get(t, '?')}" for t in tickers]
            )
            st.caption(f"Setores detectados: {setor_info}")

            # Busca todos os tickers do setor via fundamentus (função cacheada no nível do módulo)
            with loading_overlay("Buscando peers do setor..."):
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
                    peers_df[col] = clean_numeric_column(peers_df[col])

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
                            (peers_df[mult].notna())
                            & ((peers_df[mult] < 0) | (peers_df[mult] >= 500)),
                            mult,
                        ] = pd.NA

                peers_df = peers_df.dropna(how="all")

                # Tickers selecionados no setor
                selected_in_peers = [t for t in tickers if t in peers_df.index]
                if not selected_in_peers:
                    # tenta com .SA removido
                    selected_in_peers = tickers

                # ── Tabela de percentis ─────────────────────────────────────
                st.markdown("#### Posicionamento por Percentil")
                st.caption(
                    "Verde = favorável | Vermelho = desfavorável | Cinza = neutro. "
                    "O percentil indica onde a ação está no ranking do setor (100 = melhor)."
                )

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
                        setor_row = data[data["Ticker"] == t]
                        if not setor_row.empty:
                            setor_t = setor_row["Setor"].values[0]
                            tickers_do_setor_t = data[data["Setor"] == setor_t][
                                "Ticker"
                            ].tolist()
                            if t not in tickers_do_setor_t:
                                tickers_do_setor_t.append(t)
                            col_data_sector = col_data[
                                col_data.index.isin(tickers_do_setor_t)
                            ]
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

                        rows.append(
                            {
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
                            }
                        )

                if rows:
                    rank_df = pd.DataFrame(rows)

                    display_df = rank_df.drop(columns=["_cor"])
                    styled_rank = (
                        display_df.style.map(color_veredicto, subset=["Veredicto"])
                        .map(color_pct, subset=["Percentil"])
                        .format(
                            {
                                "Valor": "{:.2f}",
                                "Mediana Setor": "{:.2f}",
                                "Média Setor": "{:.2f}",
                                "Percentil": "{:.1f}%",
                            }
                        )
                        .set_properties(
                            **{
                                "font-family": "JetBrains Mono, monospace",
                                "font-size": "0.82rem",
                            }
                        )
                    )
                    st.dataframe(styled_rank, use_container_width=True, hide_index=True)

                    # ── Gráfico Comparativo de Percentis no Setor ───────────
                    st.markdown("#### Performance Relativa no Setor")
                    st.caption(
                        "Percentil de posicionamento no setor para cada indicador (100% representa o melhor posicionamento no setor)."
                    )

                    tab_labels_perf = [
                        f"{t} ({ticker_setor_map.get(t, '?')})"
                        for t in selected_in_peers
                    ]
                    tabs_perf = st.tabs(tab_labels_perf)
                    for i, ticker_s in enumerate(selected_in_peers):
                        with tabs_perf[i]:
                            setor_name = ticker_setor_map.get(ticker_s, "N/D")
                            sub_rank = rank_df[rank_df["Ação"] == ticker_s]
                            n_peers_display = (
                                sub_rank["Peers (n)"].max() if not sub_rank.empty else 0
                            )
                            st.caption(
                                f"Comparado contra **{int(n_peers_display)} peers** do setor **{setor_name}**"
                            )
                            fig_pct = px.bar(
                                sub_rank,
                                x="Múltiplo",
                                y="Percentil",
                                color="Ação",
                                color_discrete_sequence=["#00ff87"],
                                title=None,
                                labels={
                                    "Percentil": "Percentil no Setor (%)",
                                    "Múltiplo": "Múltiplo / Indicador",
                                },
                            )
                            fig_pct.update_layout(
                                yaxis=dict(range=[0, 105], ticksuffix="%"),
                                height=380,
                                margin=dict(t=30, b=40, l=40, r=40),
                                showlegend=False,
                            )
                            apply_plotly_theme(fig_pct)
                            st.plotly_chart(fig_pct, use_container_width=True)

                    # ── Gráfico de barras por múltiplo ───────────────────────
                    st.markdown("#### Distribuição do Setor por Múltiplo")
                    mult_opcoes = [
                        m[1] for m in MULTIPLES_CFG if m[0] in peers_df.columns
                    ]
                    mult_sel = st.selectbox(
                        "Selecione o múltiplo para visualizar",
                        mult_opcoes,
                        key="mult_sel",
                    )

                    mult_key = next(
                        (m[0] for m in MULTIPLES_CFG if m[1] == mult_sel), None
                    )
                    if mult_key and mult_key in peers_df.columns:
                        tab_labels_dist = [
                            f"{t} ({ticker_setor_map.get(t, '?')})"
                            for t in selected_in_peers
                        ]
                        tabs_dist = st.tabs(tab_labels_dist)
                        for i, ticker_s in enumerate(selected_in_peers):
                            with tabs_dist[i]:
                                setor_row = data[data["Ticker"] == ticker_s]
                                if not setor_row.empty:
                                    setor_t = setor_row["Setor"].values[0]
                                    tickers_do_setor_t = data[data["Setor"] == setor_t][
                                        "Ticker"
                                    ].tolist()
                                    if ticker_s not in tickers_do_setor_t:
                                        tickers_do_setor_t.append(ticker_s)

                                    col_plot = (
                                        peers_df[
                                            peers_df.index.isin(tickers_do_setor_t)
                                        ][mult_key]
                                        .dropna()
                                        .sort_values()
                                    )

                                    if col_plot.empty:
                                        st.warning(
                                            f"Dados indisponíveis para o setor de {ticker_s}."
                                        )
                                        continue

                                    bar_colors = ["#1e293b"] * len(col_plot)
                                    if ticker_s in col_plot.index:
                                        bar_colors[
                                            list(col_plot.index).index(ticker_s)
                                        ] = "#00ff87"

                                    fig_mult = go.Figure(
                                        go.Bar(
                                            x=col_plot.index.tolist(),
                                            y=col_plot.values,
                                            marker_color=bar_colors,
                                            marker_line_width=0,
                                            text=[f"{v:.1f}" for v in col_plot.values],
                                            textposition="outside",
                                            textfont=dict(size=8, color="#94a3b8"),
                                        )
                                    )

                                    # Mediana
                                    med_val = col_plot.median()
                                    fig_mult.add_hline(
                                        y=med_val,
                                        line_dash="dash",
                                        line_color="#ffd600",
                                        line_width=1.5,
                                        annotation_text=f"Mediana: {med_val:.1f}",
                                        annotation_position="top left",
                                        annotation_font=dict(color="#ffd600", size=11),
                                    )
                                    fig_mult.update_layout(
                                        title=f"{mult_sel} — Peers de {ticker_s} no Setor: {setor_t} ({len(col_plot)} empresas)",
                                        xaxis_title="Ticker",
                                        yaxis_title=mult_sel,
                                        xaxis=dict(
                                            tickangle=-45, tickfont=dict(size=9)
                                        ),
                                        height=420,
                                        showlegend=False,
                                    )
                                    apply_plotly_theme(fig_mult)
                                    st.plotly_chart(fig_mult, use_container_width=True)

                    # ── Scorecard de valuation ────────────────────────────────
                    st.markdown("#### Scorecard de Valuation do Setor")
                    score_cols = st.columns(len(selected_in_peers))

                    for i, ticker_s in enumerate(selected_in_peers):
                        sub = rank_df[rank_df["Ação"] == ticker_s]
                        if sub.empty:
                            continue
                        total_score = sub["Percentil"].mean()
                        fav = (sub["Veredicto"] == "Favorável").sum()
                        neut = (sub["Veredicto"] == "Neutro").sum()
                        desf = (sub["Veredicto"] == "Desfavorável").sum()

                        sc = (
                            "#00ff87"
                            if total_score >= 60
                            else "#ffd600"
                            if total_score >= 40
                            else "#ff3d5a"
                        )
                        label = (
                            "ATRATIVO"
                            if total_score >= 60
                            else "NEUTRO"
                            if total_score >= 40
                            else "CARO/FRACO"
                        )

                        with score_cols[i]:
                            st.markdown(
                                f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1.5px solid {sc};
            border-radius:10px;padding:0.65rem 0.85rem;margin-bottom:.5rem;
            box-shadow:0 0 10px {sc}26">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem">
    <div>
      <div style="font-size:0.85rem;font-weight:700;color:#94a3b8;letter-spacing:.04em">{ticker_s}</div>
      <div style="font-size:0.6rem;font-weight:700;color:{sc};letter-spacing:.08em;margin-top:.15rem">{label}</div>
    </div>
    <div title="Percentil médio" style="font-size:1.7rem;font-weight:900;color:{sc};
                font-family:'JetBrains Mono',monospace;text-shadow:0 0 8px {sc}55;line-height:1">{total_score:.0f}</div>
  </div>
  <div style="font-size:0.68rem;color:#94a3b8;margin-top:.45rem;display:flex;gap:.7rem;
              border-top:1px solid rgba(255,255,255,0.06);padding-top:.4rem">
    <span title="{fav} indicador(es) favorável(eis)" style="display:flex;align-items:center;gap:3px;color:#00ff87">{ICO_CHECK_SM} {fav}</span>
    <span title="{neut} indicador(es) neutro(s)" style="color:#ffd600">~ {neut}</span>
    <span title="{desf} indicador(es) desfavorável(eis)" style="display:flex;align-items:center;gap:3px;color:#ff3d5a">{ICO_X_SM} {desf}</span>
  </div>
</div>
""",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info(
                        "Nenhum ticker selecionado encontrado nos dados de peers do setor."
                    )

        # ── Síntese do Analista ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            """
<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:.5rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="#a855f7" stroke-width="1.8"/>
    <path d="M12 8v4l3 3" stroke="#a855f7" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span style="background:linear-gradient(135deg,#f8fafc,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Síntese do Analista</span>
</h3>
""",
            unsafe_allow_html=True,
        )

        usar_percentis = not rank_df.empty

        sintese_items = []
        for t in tickers:
            if t not in df_ind.index:
                continue
            r = df_ind.loc[t]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[-1]
            nome = df.loc[t, "Empresa"] if t in df.index else t
            if isinstance(nome, pd.Series):
                nome = nome.iloc[0]

            pontos_pos = []
            pontos_neg = []
            alertas = []
            fonte_label = ""

            if usar_percentis:
                # Usa percentis do setor já calculados
                ticker_rank = rank_df[rank_df["Ação"] == t]
                fonte_label = "vs. peers"
                for _, rr in ticker_rank.iterrows():
                    mult = rr["Múltiplo"]
                    pct = rr["Percentil"]
                    val = rr["Valor"]
                    med = rr["Mediana Setor"]
                    tip = f'title="mediana do setor: {med:.2f}"'
                    if pct >= 70:
                        pontos_pos.append((f"{mult} {val:.2f} · p{pct:.0f}", tip))
                    elif pct < 30:
                        pontos_neg.append((f"{mult} {val:.2f} · p{pct:.0f}", tip))
                # Alertas extras de P/L absoluto
                pl = float(r.get("P/L", 0) or 0)
                if pl < 0:
                    alertas.append((f"P/L {pl:.1f}x — prejuízo", ""))
            else:
                # Fallback: thresholds absolutos com contexto
                fonte_label = "thresholds"
                roe = float(r.get("ROE", 0) or 0)
                roic = float(r.get("ROIC", 0) or 0)
                pl = float(r.get("P/L", 0) or 0)
                dy = float(r.get("Dividend Yield", 0) or 0)
                ml = float(r.get("Margem Líquida", 0) or 0)
                cr = float(r.get("Crescimento Receita 5 anos", 0) or 0)

                if roe > 15:
                    pontos_pos.append((f"ROE {roe:.1f}%", 'title="acima dos 15% de referência"'))
                elif roe < 5:
                    pontos_neg.append((f"ROE {roe:.1f}%", 'title="abaixo dos 5% mínimos"'))

                if roic > 12:
                    pontos_pos.append((f"ROIC {roic:.1f}%", 'title="acima dos 12% de referência"'))
                elif roic < 5:
                    pontos_neg.append((f"ROIC {roic:.1f}%", ""))

                if 0 < pl < 15:
                    pontos_pos.append((f"P/L {pl:.1f}x", 'title="abaixo de 15x"'))
                elif pl > 30:
                    pontos_neg.append((f"P/L {pl:.1f}x", 'title="acima de 30x"'))
                elif pl < 0:
                    alertas.append((f"P/L {pl:.1f}x — prejuízo", ""))

                if dy > 5:
                    pontos_pos.append((f"DY {dy:.1f}%", ""))

                if cr > 10:
                    pontos_pos.append((f"Cresc.Rec {cr:.1f}%", ""))
                elif cr < 0:
                    pontos_neg.append((f"Cresc.Rec {cr:.1f}%", ""))

                if ml > 15:
                    pontos_pos.append((f"Mrg.Líq {ml:.1f}%", ""))
                elif 0 <= ml < 5:
                    alertas.append((f"Mrg.Líq {ml:.1f}%", 'title="margem comprimida"'))
                elif ml < 0:
                    pontos_neg.append((f"Mrg.Líq {ml:.1f}%", ""))

            n_pos = len(pontos_pos)
            n_neg = len(pontos_neg)
            if n_pos >= 3 or (n_pos > n_neg and n_pos >= 2):
                veredicto = ("ATRATIVO", "#00ff87")
            elif n_neg >= 3 or (n_neg > n_pos and n_neg >= 2):
                veredicto = ("FRACO", "#ff3d5a")
            else:
                veredicto = ("NEUTRO", "#ffd600")

            def _chip(item, color):
                text, tip = item
                return (
                    f'<span {tip} style="display:inline-block;background:{color}14;'
                    f'border:1px solid {color}40;color:{color};border-radius:999px;'
                    f'padding:1px 8px;font-size:0.7rem;font-weight:600;margin:0 4px 4px 0;'
                    f'white-space:nowrap;">{text}</span>'
                )

            chips_html = "".join(_chip(p, "#00ff87") for p in pontos_pos)
            chips_html += "".join(_chip(p, "#ff3d5a") for p in pontos_neg)
            chips_html += "".join(_chip(p, "#ffd600") for p in alertas)
            if not chips_html:
                chips_html = '<span style="color:#64748b;font-size:0.72rem;">Dados de peers insuficientes.</span>'

            sintese_items.append(f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:10px;padding:0.55rem 0.85rem;margin-bottom:0.4rem;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.3rem;">
    <div>
      <span style="font-family:'JetBrains Mono',monospace;font-weight:800;color:#00d2ff;font-size:0.88rem;">{t}</span>
      <span style="font-size:0.7rem;color:#64748b;margin-left:0.4rem;">{nome}</span>
      <span style="font-size:0.62rem;color:#475569;font-style:italic;margin-left:0.4rem;">{fonte_label}</span>
    </div>
    <span style="background:rgba(0,0,0,0.3);border:1px solid {veredicto[1]}40;border-radius:6px;padding:0.1rem 0.6rem;font-size:0.66rem;font-weight:800;color:{veredicto[1]};letter-spacing:0.06em;">{veredicto[0]}</span>
  </div>
  <div>{chips_html}</div>
</div>
""")

        if sintese_items:
            st.markdown("".join(sintese_items), unsafe_allow_html=True)

        # ── Próximo Passo ────────────────────────────────────────────────────
        tickers_str = ", ".join(tickers[:3]) + ("..." if len(tickers) > 3 else "")
        st.markdown(
            f"""
<div style="background:linear-gradient(135deg,rgba(0,255,135,0.06),rgba(0,210,255,0.03));border:1px solid rgba(0,255,135,0.25);border-radius:14px;padding:1.2rem 1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:0.5rem;">
  <div>
    <div style="font-size:0.72rem;color:#64748b;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">Próximo Passo</div>
    <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;">Abra <span style="color:#00ff87">Portfolio</span> na barra lateral</div>
    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">Monte e otimize a carteira com {tickers_str}</div>
  </div>
  <div style="font-size:1.8rem;opacity:0.6;">→</div>
</div>
""",
            unsafe_allow_html=True,
        )

    except OSError as e:
        st.cache_data.clear()
        st.error("Erro de I/O ao buscar dados. O cache foi limpo automaticamente.")
        st.caption(f"Detalhe técnico: {e}")
        if st.button("Tentar novamente", key="retry_os_error"):
            st.rerun()
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        st.caption(f"```\n{traceback.format_exc()}\n```")
