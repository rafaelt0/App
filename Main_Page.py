import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import datetime
import traceback

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.ui import load_css, loading_overlay, render_flow_sidebar, section_header
from utils.market_data import clean_numeric_column, get_sorted_tickers_by_liquidity
from utils.icons import (
    ICO_BULB,
    ICO_COMPASS,
    ICO_FILTER,
    ICO_MARKET,
    ICO_METRICS,
    ICO_SECTOR,
    ICO_SHIELD,
    ICO_STAR,
)
from utils.home_data import get_fundamentus_data
from utils.home_render import (
    get_ticker_setor,
    render_debt_panel,
    render_price_cards,
    render_sector_cards,
    render_star_button,
    render_ticker_cards,
)

import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

        rank_df = pd.DataFrame()  # peers do setor agora ficam em Valuation; mantém fallback por thresholds

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
