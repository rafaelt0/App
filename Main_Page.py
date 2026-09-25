import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import datetime
import logging
import math
from html import escape

from urllib.parse import quote

logger = logging.getLogger(__name__)

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.identity import get_browser_uid
from utils.ui import (
    load_css,
    loading_overlay,
    next_step_card,
    render_page_header,
    section_header,
)
from utils.market_data import (
    clean_numeric_column,
    compute_target_upside,
    get_listed_stocks,
    get_market_target_data,
    get_sorted_tickers_by_liquidity,
)
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
from utils.home_data import (
    clear_fundamentus_cache,
    get_fundamentus_data,
    get_sector_peers,
)
from utils.home_render import (
    get_ticker_setor,
    render_analyst_synthesis,
    render_debt_panel,
    render_price_cards,
    render_sector_cards,
    render_star_button,
    render_ticker_cards,
)

import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _money(value, symbol):
    return f"{symbol} {value:,.2f}" if value is not None else "—"


def _recommendation_label(key, score):
    labels = {
        "strongbuy": "Compra forte",
        "buy": "Compra",
        "hold": "Manter",
        "underperform": "Abaixo da média",
        "sell": "Venda",
        "strongsell": "Venda forte",
    }
    normalized = str(key or "").lower().replace("_", "").replace("-", "")
    if normalized in labels:
        return labels[normalized]
    if normalized and normalized != "none":
        return str(key).replace("_", " ").title()

    score = _positive_number(score)
    if score is None or score > 5:
        return "—"
    if score <= 1.5:
        return "Compra forte"
    if score <= 2.5:
        return "Compra"
    if score <= 3.5:
        return "Manter"
    if score <= 4.5:
        return "Abaixo da média"
    return "Venda"


def _render_market_target_panel(tickers, uid):
    handoff_ticker = str(
        st.session_state.pop("_market_target_handoff_ticker", "")
    ).strip().upper()
    query_ticker = str(st.query_params.get("market_target_ticker", "")).strip().upper()
    if handoff_ticker:
        st.session_state["market_target_ticker"] = handoff_ticker
    elif "market_target_ticker" not in st.session_state:
        st.session_state["market_target_ticker"] = query_ticker or (
            tickers[0] if tickers else ""
        )

    section_header(ICO_MARKET, "Preço-alvo e consenso", "h2")
    input_col, refresh_col = st.columns([5, 1], vertical_alignment="bottom")
    with input_col:
        ticker = st.text_input(
            "Ticker para preço-alvo",
            placeholder="Digite um ticker, ex.: PETR4",
            help="Digite o código de uma ação listada na B3.",
            key="market_target_ticker",
        ).strip().upper().removesuffix(".SA")

    if ticker:
        st.query_params["market_target_ticker"] = ticker
    elif "market_target_ticker" in st.query_params:
        del st.query_params["market_target_ticker"]

    if not ticker:
        return
    if not ticker.isalnum() or not 4 <= len(ticker) <= 8:
        st.warning("Digite um ticker B3 válido (4 a 8 letras ou números).")
        return

    with refresh_col:
        st.button(
            "Atualizar dados",
            key=f"market_target_refresh_{ticker}",
            use_container_width=True,
            on_click=lambda value=ticker: get_market_target_data.clear(value),
            help="Busca novamente os dados de mercado no Yahoo Finance.",
        )

    with loading_overlay(f"Buscando preços-alvo de {ticker}…", tickers=[ticker]):
        market = get_market_target_data(ticker)
    if not market or "_error" in market:
        st.error(f"Não foi possível carregar os dados de mercado de {ticker}.")
        return

    price = _positive_number(market.get("price")) or _positive_number(
        market.get("regular_price")
    )
    target_low = _positive_number(market.get("target_low"))
    target_mean = _positive_number(market.get("target_mean"))
    target_median = _positive_number(market.get("target_median"))
    target_high = _positive_number(market.get("target_high"))
    upside = compute_target_upside(price, target_mean)
    currency = str(market.get("currency") or "BRL").upper()
    currency_symbol = {"BRL": "R$", "USD": "US$", "EUR": "€"}.get(currency, currency)
    company_name = market.get("company_name") or ticker
    st.markdown(f"**{company_name} · {ticker}**")

    metrics = st.columns(3)
    with metrics[0]:
        st.metric("Cotação atual", _money(price, currency_symbol))
    with metrics[1]:
        st.metric("Preço-alvo médio", _money(target_mean, currency_symbol))
    with metrics[2]:
        st.metric(
            "Potencial até o alvo médio",
            f"{upside:+.1f}%" if upside is not None else "—",
            help="(Preço-alvo médio ÷ cotação atual − 1) × 100.",
        )

    if target_mean is None:
        st.info("O Yahoo Finance não informa um preço-alvo médio para este ticker.")

    st.markdown("#### Faixa de preços-alvo dos analistas")
    for column, label, value in zip(
        st.columns(3),
        ("Alvo mínimo", "Alvo mediano", "Alvo máximo"),
        (target_low, target_median, target_high),
    ):
        with column:
            st.metric(label, _money(value, currency_symbol))

    if target_low is not None and target_high is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sorted((target_low, target_high)),
                y=[0, 0],
                mode="lines",
                line={"color": "#475569", "width": 12},
                hovertemplate="Faixa estimada: %{x:,.2f}<extra></extra>",
                showlegend=False,
            )
        )
        for value, label, color in (
            (price, "Cotação atual", "#94a3b8"),
            (target_mean, "Alvo médio", "#a855f7"),
            (target_median, "Alvo mediano", "#00d2ff"),
        ):
            if value is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[value],
                        y=[0],
                        mode="markers",
                        name=label,
                        marker={"color": color, "size": 12},
                        hovertemplate=f"{label}: %{{x:,.2f}}<extra></extra>",
                    )
                )
        apply_plotly_theme(fig)
        fig.update_layout(
            height=230,
            margin={"t": 40, "b": 60, "l": 15, "r": 15},
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.08,
                "yanchor": "bottom",
            },
            xaxis={"title": f"Preço por ação ({currency_symbol})"},
            yaxis={"visible": False, "fixedrange": True},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("A faixa visual exige que a fonte informe os alvos mínimo e máximo.")

    analyst_count = _positive_number(market.get("analyst_count"))
    recommendation = _recommendation_label(
        market.get("recommendation"), market.get("recommendation_mean")
    )
    context_cols = st.columns(2)
    with context_cols[0]:
        st.metric(
            "Opiniões de analistas (Yahoo)",
            str(int(analyst_count)) if analyst_count else "—",
            help=(
                "Contagem agregada informada pelo Yahoo Finance; não é uma contagem verificada "
                "de colaboradores da faixa de preços-alvo."
            ),
        )
    with context_cols[1]:
        st.metric("Recomendação agregada", recommendation)
    st.caption(
        "Fonte: Yahoo Finance. O preço-alvo reflete estimativas de analistas disponíveis na fonte; "
        "é uma referência de mercado, não uma previsão garantida nem recomendação de investimento."
    )

    is_starred = _db.wl_has(uid, ticker)
    if st.button(
        "★ Remover dos Favoritos" if is_starred else "☆ Salvar nos Favoritos",
        key="market_target_watchlist_btn",
        help="Ticker salvo na watchlist da página principal.",
    ):
        if is_starred:
            _db.wl_remove(uid, ticker)
        else:
            _db.wl_add(uid, ticker)
        st.rerun()


# Configurar temas de plotagem escuros
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#0b111a"
plt.rcParams["axes.facecolor"] = "#151d2a"
plt.rcParams["text.color"] = "#f0f4f8"
plt.rcParams["axes.labelcolor"] = "#aebaca"
plt.rcParams["xtick.color"] = "#aebaca"
plt.rcParams["ytick.color"] = "#aebaca"
plt.rcParams["grid.color"] = "#34465b"
plt.rcParams["font.family"] = "sans-serif"


st.set_page_config(
    page_title="B3Lab — Análise Quantitativa de Ações",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo("logo.svg", icon_image="favicon.svg")


# CSS customizado
load_css()


# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "B3 Explorer",
    "Compare fundamentos, preço e risco das empresas listadas na B3.",
    "home",
)


# Carrega lista de ações da B3 com setores para filtragem inicial
data = pd.DataFrame()
try:
    data = get_listed_stocks()
except (OSError, ValueError) as exc:
    st.error(f"Não foi possível carregar a lista de ações da B3: {exc}")
    st.stop()

if "Setor" not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

# Cria listas de tickers e setores para seleção
stocks = list(data["Ticker"].values)
setores = sorted(data["Setor"].dropna().unique())
setores.insert(0, "Todos")
SECTOR_AUTOPICK_LIMIT = 8
MAX_ANALYSIS_TICKERS = 20
_ticker_setor = dict(zip(data["Ticker"], data["Setor"]))

_uid = get_browser_uid()

if "selected_tickers" not in st.session_state:
    _saved_tickers, _ = _db.portfolio_get(_uid)
    _saved_tickers = [
        str(ticker).replace(".SA", "")
        for ticker in _saved_tickers
        if str(ticker).replace(".SA", "") in stocks
    ]
    if len(_saved_tickers) > MAX_ANALYSIS_TICKERS:
        st.session_state["_analysis_limit_notice"] = len(_saved_tickers)
    st.session_state["selected_tickers"] = _saved_tickers
    if _saved_tickers:
        st.session_state["_saved_selection_notice"] = True

# ── Watchlist (sidebar) ───────────────────────────────────────────────────────
_watchlist = _db.wl_get(_uid)
if _watchlist:
    st.sidebar.markdown(
        f'<div class="sidebar-section-label" style="color:#ffd600">{ICO_STAR} Favoritos</div>',
        unsafe_allow_html=True,
    )
    for _wt in _watchlist:
        _c1, _c2 = st.sidebar.columns([5, 1])
        with _c1:
            if _wt in stocks:
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
            elif st.button(
                f"{_wt} · Preço-alvo",
                key=f"wl_val_{_wt}",
                use_container_width=True,
                help="Ticker fora da lista local; carregar o preço-alvo nesta página.",
            ):
                st.session_state["_market_target_handoff_ticker"] = _wt
                st.session_state["market_target_ticker"] = _wt
                st.query_params["uid"] = _uid
                st.query_params["market_target_ticker"] = _wt
                st.rerun()
        with _c2:
            if st.button("✕", key=f"wl_rm_{_wt}", help="Remover dos favoritos"):
                _db.wl_remove(_uid, _wt)
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
    "Escolha um ou mais setores:",
    setores,
    default=[],
    key="setores_selecionados",
    placeholder="Selecione os setores…",
)

if st.sidebar.button(
    "Atualizar dados Fundamentus",
    help="Limpa o cache dos indicadores e busca dados atualizados na próxima análise.",
):
    removed = clear_fundamentus_cache()
    get_sorted_tickers_by_liquidity.clear()
    st.session_state["fund_refresh_removed"] = removed
    st.rerun()

_refresh_removed = st.session_state.pop("fund_refresh_removed", None)
if _refresh_removed is not None:
    st.sidebar.success(f"Cache Fundamentus atualizado ({_refresh_removed} entradas removidas).")

# Detecta se o usuário mudou a seleção de setores nesta interação. Quando muda,
# os tickers do(s) setor(es) são autoselecionados abaixo (após a filtragem).
_prev_setores = st.session_state.get("_prev_setores_selecionados")
_sector_changed = _prev_setores is not None and _prev_setores != setores_selecionados
st.session_state["_prev_setores_selecionados"] = list(setores_selecionados)

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


# Ao escolher um setor, autoselecionar apenas os ativos mais líquidos. O widget
# de tickers ainda não foi instanciado, então escrevemos direto no session_state.
# O usuário continua podendo adicionar outros ativos manualmente.
if _sector_changed and setores_selecionados and "Todos" not in setores_selecionados:
    _auto_selected = tickers_filtrados[:SECTOR_AUTOPICK_LIMIT]
    st.session_state["selected_tickers"] = list(_auto_selected)
    if len(tickers_filtrados) > len(_auto_selected):
        st.session_state["_sector_autopick_notice"] = (
            len(_auto_selected),
            len(tickers_filtrados),
        )

section_header(ICO_COMPASS, "Selecione ativos para analisar", "h2")
n_disponíveis = len(tickers_filtrados)
setor_label = escape(
    "todos os setores"
    if (not setores_selecionados or "Todos" in setores_selecionados)
    else ", ".join(setores_selecionados[:2])
    + ("…" if len(setores_selecionados) > 2 else "")
)
st.markdown(
    f"""
<div class="selection-summary" role="status">
  <div class="selection-summary-main">
    <span class="selection-summary-count">{n_disponíveis} ações</span>
    <span class="selection-summary-context">disponíveis em {setor_label}</span>
  </div>
  <span class="selection-summary-source">Fundamentus <i>·</i> ordenação por liquidez</span>
</div>
""",
    unsafe_allow_html=True,
)

_sector_autopick_notice = st.session_state.pop("_sector_autopick_notice", None)
if _sector_autopick_notice:
    _selected_count, _available_count = _sector_autopick_notice
    st.info(
        f"Selecionamos os {_selected_count} ativos mais líquidos de "
        f"{_available_count} disponíveis. Você pode adicionar outros pelo campo abaixo."
    )

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []

def _clear_main_selection():
    """Clear the visible home-page selection before the next widget rerun."""
    st.session_state["selected_tickers"] = []
    st.session_state["analyzed_tickers"] = []



# Apply any ticker selection staged by widgets below (which run after this
# key's widget is already instantiated, so they can't write to it directly)
if "_pending_tickers" in st.session_state:
    st.session_state["selected_tickers"] = st.session_state.pop("_pending_tickers")

# Remove any stale tickers that are no longer in the filtered list
st.session_state["selected_tickers"] = [
    t for t in st.session_state["selected_tickers"] if t in tickers_filtrados
][:MAX_ANALYSIS_TICKERS]

tickers = st.multiselect(
    "Escolha ações para analisar",
    options=tickers_filtrados,
    format_func=lambda t: (
        f"{t}  ·  {_ticker_setor[t]}" if _ticker_setor.get(t) else t
    ),
    placeholder="Digite o ticker ou selecione na lista…",
    max_selections=MAX_ANALYSIS_TICKERS,
    help=(
        f"Selecione até {MAX_ANALYSIS_TICKERS} ações. "
        "Use o filtro de setor para reduzir a lista."
    ),
    key="selected_tickers",
)

if st.session_state.pop("_saved_selection_notice", False):
    st.info(
        "Seleção da carteira salva carregada. "
        "Revise os ativos e clique em **Analisar** para atualizar os fundamentos."
    )

_saved_count = st.session_state.pop("_analysis_limit_notice", None)
if _saved_count:
    st.info(
        f"A seleção salva tinha {_saved_count} ações. "
        f"Carregamos as primeiras {MAX_ANALYSIS_TICKERS} para manter a análise estável."
    )

_render_market_target_panel(tickers, _uid)

# A new selection must always require an explicit analysis click. This avoids
# reusing results from an earlier selection when the user returns to it later.
if st.session_state.get("analyzed_tickers", []) != list(tickers):
    st.session_state["analyzed_tickers"] = []

if not tickers:
    # ── Onboarding ─────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="onboarding-card">
  <div class="onboarding-card-header">
    <div>
      <div class="onboarding-eyebrow">Comece por aqui</div>
      <div class="onboarding-title">Escolha um ativo para começar</div>
      <p class="onboarding-description">
        Pesquise pelo ticker, filtre por setor ou use um dos ativos líquidos abaixo.
      </p>
    </div>
    <div class="onboarding-signal" aria-hidden="true">
      <span></span><span></span><span></span>
    </div>
  </div>
  <div class="onboarding-tip-grid">
    <div class="onboarding-tip onboarding-tip-green">
      <div class="onboarding-tip-label"><span class="onboarding-tip-badge">1</span>Escolha</div>
      <div class="onboarding-tip-copy">Filtre por setor ou pesquise diretamente pelo ticker.</div>
    </div>
    <div class="onboarding-tip onboarding-tip-blue">
      <div class="onboarding-tip-label"><span class="onboarding-tip-badge">2</span>Compare</div>
      <div class="onboarding-tip-copy">Compare fundamentos, rentabilidade, crescimento e endividamento lado a lado.</div>
    </div>
    <div class="onboarding-tip onboarding-tip-purple">
      <div class="onboarding-tip-label"><span class="onboarding-tip-badge">3</span>Aprofunde</div>
      <div class="onboarding-tip-copy">Use o Portfolio e o Screener; consulte preços-alvo e consenso nesta página.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Quick-start grid
    st.markdown(
        '<div class="onboarding-subheading">Seleção rápida <span>· ações líquidas</span></div>',
        unsafe_allow_html=True,
    )

    _QUICK = [
        ("WEGE3", "Máquinas", "#61d4c6"),
        ("PETR4", "Petróleo", "#e7b96b"),
        ("VALE3", "Mineração", "#e58a93"),
        ("ITUB4", "Banco", "#8cb4f2"),
        ("RENT3", "Locação", "#b7a2e6"),
        ("ABEV3", "Bebidas", "#d79b6f"),
        ("EGIE3", "Energia", "#7fcea3"),
        ("RADL3", "Farmácia", "#84b8e8"),
    ]
    _cols = st.columns(4)
    for i, (tkr, setor, cor) in enumerate(_QUICK):
        with _cols[i % 4]:
            st.markdown(
                f'<div class="quick-start-sector" style="--quick-color:{cor}">{setor}</div>',
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
        '<div class="onboarding-subheading onboarding-subheading-spaced">Explorar <span>· setores</span></div>',
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
        '<div class="onboarding-hint">'
        f"{ICO_BULB} <b>Atalho</b><span>Digite qualquer ticker da B3 no campo de busca ou filtre por setor na barra lateral.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

if tickers:
    st.markdown(
        f"""
    <div class="selection-active-note">
      <span>{len(tickers)} ação(ões) selecionada(s) — use <strong>Portfolio</strong> para analisar a carteira completa.</span>
      <span class="selection-active-note-hint">seleção atual</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_analyze, col_clear = st.columns([4, 1])
    with col_analyze:
        if st.button(
            "Analisar",
            type="primary",
            use_container_width=True,
            key="btn_analisar",
            help="Busca os indicadores fundamentalistas dos ativos selecionados.",
        ):
            st.session_state["analyzed_tickers"] = list(tickers)
    with col_clear:
        st.button(
            "Limpar",
            use_container_width=True,
            key="btn_limpar_selecao",
            on_click=_clear_main_selection,
            help="Remove os ativos selecionados e volta ao início.",
        )
# Só executa análise depois que o usuário clica em "Analisar" para a seleção atual
analyzed_tickers = st.session_state.get("analyzed_tickers", [])
ready_to_analyze = bool(tickers) and analyzed_tickers == tickers

if tickers and not ready_to_analyze:
    st.info("Clique em **Analisar** para buscar os indicadores das ações selecionadas.")

if ready_to_analyze:
    try:
        sector_values = tuple(
            sorted(
                {
                    str(sector).strip()
                    for sector in data.loc[
                        data["Ticker"].isin(tickers), "Setor"
                    ].dropna()
                    if str(sector).strip()
                }
            )
        )
        # 1. Buscar dados usando funções cacheadas, com animação de carregamento
        with loading_overlay(
            "Buscando indicadores fundamentalistas na B3…", tickers=tickers
        ):
            df = get_fundamentus_data(tickers)
            peers_raw = get_sector_peers(sector_values)
        _fundamentus_index = {
            str(index).replace(".SA", "").strip().upper() for index in df.index
        }
        _missing_fundamentus = [
            ticker for ticker in tickers if ticker.upper() not in _fundamentus_index
        ]
        if _missing_fundamentus:
            logger.warning(
                "fundamentus returned partial data for %s; clearing Streamlit cache",
                _missing_fundamentus,
            )
            clear_fundamentus_cache()

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
            csv_data = df_export.to_csv().encode("utf-8-sig")
            st.download_button(
                label="Exportar dados fundamentalistas (CSV)",
                data=csv_data,
                file_name=f"fundamentus_{'+'.join(tickers)}_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        except Exception:
            logger.debug("fundamentus CSV export failed", exc_info=True)

        section_header(ICO_SECTOR, "Setor", "h2")
        df_sector = df[["Empresa", "Setor", "Subsetor"]]

        if len(tickers) > 1:
            df_sector_rows = df_sector[
                ~df_sector.index.duplicated(keep="last")
            ].reindex(tickers)
            st.dataframe(
                df_sector_rows.rename_axis("Ticker").reset_index(),
                hide_index=True,
                use_container_width=True,
            )
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
        section_header(ICO_MARKET, "Informações de Mercado", "h2")
        df_price = df[
            [
                "Cotacao",
                "Min_52_sem",
                "Max_52_sem",
                "Vol_med_2m",
                "Valor_de_mercado",
                "Data_ult_cot",
            ]
        ].copy()
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
            df_price_rows = df_price[
                ~df_price.index.duplicated(keep="last")
            ].reindex(tickers)
            st.dataframe(
                df_price_rows.rename_axis("Ticker").reset_index(),
                hide_index=True,
                use_container_width=True,
            )
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
        section_header(ICO_METRICS, "Indicadores Financeiros", "h2")
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

        # Remove duplicate indices if any
        df_ind = df_ind[~df_ind.index.duplicated(keep="last")]

        if len(tickers) > 1:
            comparison = (
                df_ind.reindex(tickers)
                .drop(columns="Empresa")
                .rename_axis("Ticker")
                .reset_index()
                .rename(
                    columns={
                        "Margem Líquida": "Margem Líquida (%)",
                        "Margem EBIT": "Margem EBIT (%)",
                        "ROE": "ROE (%)",
                        "ROIC": "ROIC (%)",
                        "Dividend Yield": "Dividend Yield (%)",
                        "Crescimento Receita 5 anos": "Receita 5a (%)",
                    }
                )
            )
            percent_columns = [
                "Margem Líquida (%)",
                "Margem EBIT (%)",
                "ROE (%)",
                "ROIC (%)",
                "Dividend Yield (%)",
                "Receita 5a (%)",
            ]
            st.caption("Percentuais em % · múltiplos em × · sem dados ficam em branco.")
            st.dataframe(
                comparison,
                hide_index=True,
                use_container_width=True,
                height=min(35 * (len(tickers) + 1) + 8, 760),
                column_config={
                    "Ticker": st.column_config.TextColumn(width="small"),
                    **{
                        name: st.column_config.NumberColumn(
                            format="%.1f%%", width="small"
                        )
                        for name in percent_columns
                    },
                    **{
                        name: st.column_config.NumberColumn(
                            format="%.2f×", width="small"
                        )
                        for name in ("P/L", "EV/EBITDA", "P/VP")
                    },
                },
            )
            st.markdown("**Favoritos rápidos**")
            for start in range(0, len(tickers), 4):
                favorite_columns = st.columns(min(4, len(tickers) - start))
                for column, ticker in zip(
                    favorite_columns, tickers[start : start + 4]
                ):
                    with column:
                        render_star_button(ticker, _uid)
        else:
            ticker = tickers[0]
            render_star_button(ticker, _uid)
            if ticker in df_ind.index:
                render_ticker_cards(
                    df_ind.loc[ticker].fillna(0), setor=get_ticker_setor(df, ticker)
                )

        # ── Saúde Financeira ─────────────────────────────────────────────────
        st.markdown("---")
        section_header(ICO_SHIELD, "Saúde Financeira", "h2")
        st.caption(
            "Endividamento e liquidez da empresa. "
            "Dívida/PL acima de 3x e Liquidez abaixo de 1x são sinais de alerta."
        )

        for ticker in tickers:
            if ticker in df.index:
                row_debt = df.loc[ticker]
                if isinstance(row_debt, pd.DataFrame):
                    row_debt = row_debt.iloc[-1]
                if len(tickers) > 1:
                    st.markdown(f"**{ticker}**")
                render_debt_panel(ticker, row_debt)
            else:
                st.warning(f"Sem dados para {ticker}")

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

        # ── Síntese do Analista ──────────────────────────────────────────────
        st.markdown("---")
        render_analyst_synthesis(df_ind, df, tickers, peers_raw, data)

        # ── Próximo Passo ────────────────────────────────────────────────────
        tickers_str = ", ".join(tickers[:3]) + ("…" if len(tickers) > 3 else "")
        next_step_card(
            message=f"Monte e otimize a carteira com {tickers_str}.",
            accent="var(--brand-primary)",
            cta_label="Abrir Portfolio",
            cta_page="pages/1_Portfolio.py",
            cta_url=(
                "Portfolio?"
                f"uid={quote(_uid, safe='')}"
                f"&portfolio_tickers={quote(','.join(tickers), safe='')}"
            ),
        )

    except OSError as e:
        st.cache_data.clear()
        st.error("Erro de I/O ao buscar dados. O cache foi limpo automaticamente.")
        st.caption(f"Detalhe técnico: {e}")
        if st.button("Tentar novamente", key="retry_os_error"):
            st.rerun()
    except Exception:
        logger.exception("failed to fetch/render main page data")
        st.error("Não foi possível concluir a análise agora.")
        st.caption(
            "A fonte de dados pode estar temporariamente indisponível. "
            "Verifique a seleção e tente novamente."
        )
        if st.button("Tentar novamente", key="retry_main_analysis"):
            st.rerun()
