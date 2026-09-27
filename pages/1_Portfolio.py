import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
import logging
from html import escape

logger = logging.getLogger(__name__)
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from quantstats.stats import sharpe, sortino, max_drawdown, var
import quantstats as qs
from utils import db as _db
from utils.charts import (
    CHART_ACCENT,
    CHART_DANGER,
    CHART_DANGER_FILL,
    CHART_GRID,
    CHART_MUTED,
    CHART_PRIMARY,
    CHART_SECONDARY,
    CHART_TEXT,
    apply_plotly_theme,
)
from utils.identity import get_browser_uid
from utils.ui import (
    diag_row,
    empty_state_card,
    load_css,
    loading_overlay,
    next_step_card,
    render_cards_grid,
    render_page_header,
    section_header,
)
from utils.market_data import get_listed_stocks
from utils.icons import (
    ICO_BOX,
    ICO_CHART,
    ICO_CRIT,
    ICO_DOWN,
    ICO_FLAT,
    ICO_FRONTIER,
    ICO_HEATMAP,
    ICO_IDEA,
    ICO_LINK,
    ICO_METRICS,
    ICO_OK,
    ICO_RISK,
    ICO_RULER,
    ICO_SIGNAL,
    ICO_STRESS,
    ICO_TARGET,
    ICO_UP,
    ICO_WARN,
)
from utils.portfolio_data import (
    align_benchmark_returns,
    align_weights_to_columns,
    calculate_historical_stress,
    finite_or_none,
    evaluate_portfolio_health,
    find_crisis_history_gaps,
    bound_efficient_return,
    get_benchmark_prices,
    get_portfolio_prices,
    get_portfolio_trade_prices,
    get_selic_rate,
    estimate_markowitz_inputs,
)
from utils.portfolio_charts import (
    apply_matplotlib_theme,
    plot_efficient_frontier_and_random_portfolios,
)


# CSS customizado
load_css()
st.markdown(
    """
    <style>
    body:has(.page-hero[data-page="portfolio"]) .main {
      color-scheme: dark;
      background: var(--bg-color) !important;
      color: var(--text-main) !important;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
    }
    body:has(.page-hero[data-page="portfolio"]) [data-testid="stHeader"] {
      background: var(--bg-color) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .block-container {
      max-width: 1460px;
      padding-bottom: 4rem;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
      margin: 0.7rem 0 1rem;
      padding: 0.2rem 1.1rem 0.9rem;
      border: 1px solid var(--panel-border) !important;
      border-radius: 8px !important;
      background: var(--panel-bg) !important;
      box-shadow: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .ui-section-heading {
      margin-top: 1.5rem !important;
      padding-bottom: 0.65rem !important;
      border-bottom-color: var(--panel-border) !important;
      color: var(--text-main) !important;
      font-size: clamp(1.05rem, 1.5vw, 1.25rem) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .ui-section-heading svg {
      color: var(--brand-secondary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .ui-section-heading svg [stroke] {
      stroke: var(--brand-secondary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] .ui-section-heading {
      margin-top: 0.55rem !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetric"] {
      min-height: 92px;
      padding: 0.75rem 0.8rem !important;
      border: 0 !important;
      border-top: 2px solid var(--panel-border) !important;
      border-radius: 0 !important;
      background: transparent !important;
      box-shadow: none !important;
      transform: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetricValue"],
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetricValue"] > div {
      color: var(--brand-primary) !important;
      font-family: inherit !important;
      font-size: clamp(1.2rem, 2vw, 1.65rem) !important;
      font-variant-numeric: tabular-nums;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .mcard-value {
      font-family: inherit !important;
      font-variant-numeric: tabular-nums;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetricLabel"],
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetricLabel"] > div {
      color: var(--text-muted) !important;
      font-size: 0.76rem !important;
      font-weight: 600 !important;
      letter-spacing: 0 !important;
      text-transform: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stPlotlyChart"] {
      margin: 0.35rem 0 1rem;
      overflow: hidden;
      border: 1px solid var(--panel-border);
      border-radius: 7px;
      background: var(--panel-bg);
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stDataFrame"] {
      border: 1px solid var(--panel-border);
      border-radius: 7px;
      background: var(--panel-bg);
      box-shadow: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main table {
      color: var(--text-main) !important;
      font-family: inherit !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main th,
    body:has(.page-hero[data-page="portfolio"]) .main td {
      border-color: var(--panel-border) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stSelectbox label,
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect label,
    body:has(.page-hero[data-page="portfolio"]) .main .stNumberInput label,
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stRadio"] > label {
      color: var(--text-muted) !important;
      font-size: 0.82rem !important;
      font-weight: 650 !important;
      letter-spacing: 0 !important;
      text-transform: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stTextInput input,
    body:has(.page-hero[data-page="portfolio"]) .main .stNumberInput input,
    body:has(.page-hero[data-page="portfolio"]) .main .stSelectbox [data-baseweb="select"] > div,
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect [data-baseweb="select"] > div {
      border-color: var(--border-strong) !important;
      border-radius: 6px !important;
      background: var(--panel-bg) !important;
      color: var(--text-main) !important;
      font-family: inherit !important;
      font-variant-numeric: tabular-nums;
      box-shadow: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect [data-baseweb="tag"] {
      border: 1px solid var(--panel-border);
      background: var(--panel-raised);
      color: var(--brand-primary);
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect input {
      color: var(--text-main) !important;
      caret-color: var(--brand-primary) !important;
      font-family: inherit !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect input::placeholder {
      color: var(--text-faint) !important;
      opacity: 1 !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main .stMultiSelect [data-baseweb="select"] input + div {
      color: var(--text-faint) !important;
      opacity: 1 !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stRadio"] label[data-baseweb="radio"] p,
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stRadio"] label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] {
      color: var(--text-main) !important;
      opacity: 1 !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {
      background: var(--brand-primary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) [data-baseweb="popover"] [data-baseweb="menu"] {
      border-color: var(--panel-border) !important;
      background: var(--panel-bg) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) [data-baseweb="popover"] [role="option"] {
      color: var(--text-main) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) [data-baseweb="popover"] [role="option"]:hover,
    body:has(.page-hero[data-page="portfolio"]) [data-baseweb="popover"] [role="option"][aria-selected="true"] {
      background: var(--panel-raised) !important;
      color: var(--brand-primary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid^="stNumberInput-Step"] {
      background: var(--panel-raised) !important;
      color: var(--brand-primary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main button[data-testid^="baseButton-"] {
      min-height: 2.6rem;
      border: 1px solid var(--border-strong) !important;
      border-radius: 6px !important;
      background: var(--panel-raised) !important;
      color: var(--brand-primary) !important;
      box-shadow: none !important;
      transform: none !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main button[data-testid^="baseButton-"]:hover {
      border-color: var(--brand-primary) !important;
      background: var(--surface-hover) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stPageLink"] a {
      border-color: var(--border-strong) !important;
      background: var(--panel-bg) !important;
      color: var(--brand-secondary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="baseButton-primary"] {
      border-color: var(--brand-primary) !important;
      background: var(--brand-primary) !important;
      color: var(--panel-bg) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [data-testid="baseButton-primary"]:hover {
      background: var(--brand-primary) !important;
    }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#00ff87"],
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#4ade80"] { color: var(--brand-primary) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#ffd600"] { color: var(--brand-accent) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#ff3d5a"],
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#ff1744"],
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#f87171"] { color: var(--brand-danger) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#00d2ff"],
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#38bdf8"] { color: var(--brand-secondary) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#f8fafc"] { color: var(--text-main) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#94a3b8"],
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#64748b"] { color: var(--text-faint) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#475569"] { color: var(--text-muted) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main [style*="#1e293b"] { border-color: var(--panel-border) !important; }
    body:has(.page-hero[data-page="portfolio"]) .main :focus-visible {
      outline: none !important;
      box-shadow: var(--focus-ring) !important;
    }
    @media (max-width: 640px) {
      body:has(.page-hero[data-page="portfolio"]) .main .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
      }
      body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.1rem 0.7rem 0.65rem;
      }
      body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stMetric"] {
        min-height: 78px;
        padding: 0.6rem 0.35rem !important;
      }
      body:has(.page-hero[data-page="portfolio"]) .main [data-testid="stPlotlyChart"] {
        margin-left: -0.2rem;
        margin-right: -0.2rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _get_trade_quotes(tickers):
    try:
        prices = get_portfolio_trade_prices(tuple(tickers))
    except Exception:
        logger.warning("unadjusted trade quotes unavailable", exc_info=True)
        return {}, True
    if prices is None:
        return {}, False
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [
            "_".join(map(str, column)).strip() for column in prices.columns
        ]
    quotes = {}
    for ticker in tickers:
        if ticker in prices.columns:
            available = prices[ticker].dropna()
            if not available.empty:
                latest_price = finite_or_none(available.iloc[-1])
                if latest_price is not None and latest_price > 0:
                    quotes[ticker] = (latest_price, available.index[-1])
    return quotes, False


# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "Otimização de portfólio",
    "Monte uma carteira, compare risco e retorno e revise os pesos antes de investir.",
    "portfolio",
)
section_header(ICO_RULER, "Montagem da carteira", "h2")
st.caption(
    "Escolha o histórico e os papéis; depois defina o capital e a forma de alocação."
)

def _refresh_portfolio_data() -> None:
    get_portfolio_prices.clear()
    get_portfolio_trade_prices.clear()
    get_benchmark_prices.clear()
    get_selic_rate.clear()
    st.session_state["portfolio_loaded"] = False
    st.session_state["portfolio_loaded_tickers"] = []
    st.session_state["portfolio_analysis_tickers"] = []
    st.session_state["_portfolio_refresh_notice"] = True


with st.container(border=True):
    period_title, refresh_col = st.columns([2, 1])
    with period_title:
        section_header(ICO_RULER, "Janela de análise", "h3")
    with refresh_col:
        st.button(
            "Atualizar cotações",
            key="portfolio_refresh",
            use_container_width=True,
            on_click=_refresh_portfolio_data,
            help=(
                "Limpa o cache de cotações, benchmark e Selic. "
                "A análise precisa ser carregada novamente."
            ),
        )
    if st.session_state.pop("_portfolio_refresh_notice", False):
        st.info("Cotações atualizadas. Clique em **Carregar portfólio** para recalcular a análise.")

    col_config1, col_config2 = st.columns([1, 2], gap="large")
    with col_config1:
        lookback_opcao = st.selectbox(
            "Período de análise",
            (
                "2 Anos (Padrão)",
                "1 Ano",
                "3 Anos",
                "5 Anos",
                "6 Meses",
                "Personalizado (Dias)",
            ),
        )
    with col_config2:
        today = datetime.date.today()
        if lookback_opcao == "6 Meses":
            data_inicio = today - datetime.timedelta(days=180)
            st.info(f"Dados a partir de {data_inicio.strftime('%d/%m/%Y')}")
        elif lookback_opcao == "1 Ano":
            data_inicio = today - datetime.timedelta(days=365)
            st.info(f"Dados a partir de {data_inicio.strftime('%d/%m/%Y')}")
        elif lookback_opcao == "2 Anos (Padrão)":
            data_inicio = today - datetime.timedelta(days=365 * 2)
            st.info(f"Dados a partir de {data_inicio.strftime('%d/%m/%Y')}")
        elif lookback_opcao == "3 Anos":
            data_inicio = today - datetime.timedelta(days=365 * 3)
            st.info(f"Dados a partir de {data_inicio.strftime('%d/%m/%Y')}")
        elif lookback_opcao == "5 Anos":
            data_inicio = today - datetime.timedelta(days=365 * 5)
            st.info(f"Dados a partir de {data_inicio.strftime('%d/%m/%Y')}")
        else:
            lookback_dias = st.number_input(
                "Dias de lookback",
                min_value=60,
                max_value=5000,
                value=500,
                step=10,
                help="Mínimo de 60 dias para garantir ao menos 30 retornos úteis.",
            )
            data_inicio = today - datetime.timedelta(days=lookback_dias)

try:
    taxa_selic = get_selic_rate()
except Exception as _selic_err:
    logger.warning("get_selic_rate failed; using reference rate: %s", _selic_err)
    logger.debug("get_selic_rate failure details", exc_info=True)
    st.warning(
        f"Não foi possível buscar a taxa Selic no BCB ({_selic_err}). Usando valor de referência: 13,75% a.a."
    )
    taxa_selic = (1 + 0.1375) ** (1 / 252) - 1
taxa_selic_anual = (1 + taxa_selic) ** 252 - 1


# Seleção de ações
MAX_TICKERS = 20

data = pd.DataFrame()
try:
    data = get_listed_stocks()
except (OSError, ValueError) as exc:
    st.error(f"Não foi possível carregar a lista de ações da B3: {exc}")
    st.stop()

stocks = list(data["Ticker"].values)
_ticker_empresa = dict(zip(data["Ticker"], data["Empresa"])) if "Empresa" in data else {}

_uid = get_browser_uid()
_saved_tickers, _ = _db.portfolio_get(_uid)

_query_handoff = str(st.query_params.get("portfolio_tickers", "")).strip()
if "portfolio_tickers" in st.query_params:
    del st.query_params["portfolio_tickers"]
_handoff_tickers = [
    ticker
    for ticker in (
        item.strip().upper().replace(".SA", "")
        for item in _query_handoff.split(",")
    )
    if ticker in stocks
][:MAX_TICKERS]
_handoff_value = ",".join(_handoff_tickers)
if _handoff_value and st.session_state.get("_portfolio_handoff_value") != _handoff_value:
    st.session_state["_portfolio_handoff_value"] = _handoff_value
    st.session_state["selected_tickers"] = _handoff_tickers
    st.session_state["portfolio_loaded"] = False
    st.session_state["portfolio_loaded_tickers"] = []
    st.session_state["portfolio_analysis_tickers"] = []
    st.session_state["_portfolio_handoff_notice"] = _handoff_tickers

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = [
        t for t in _saved_tickers if t in stocks
    ]
    if st.session_state["selected_tickers"]:
        st.session_state["_portfolio_restore_notice"] = True

# Remove any stale tickers not in the current list, and trim to the cap in
# case it was lowered or the session predates the limit
st.session_state["selected_tickers"] = [
    t for t in st.session_state["selected_tickers"] if t in stocks
][:MAX_TICKERS]

def _clear_saved_portfolio():
    # Runs as a button callback, i.e. BEFORE the widgets are instantiated on the
    # next run. Clearing "selected_tickers" here is allowed; doing it inline after
    # the multiselect below already exists raises a StreamlitAPIException.
    _db.portfolio_clear(_uid)
    st.session_state["selected_tickers"] = []
    st.session_state["portfolio_loaded"] = False
    st.session_state["portfolio_loaded_tickers"] = []
    st.session_state["portfolio_analysis_tickers"] = []


with st.container(border=True):
    section_header(ICO_BOX, "Ativos e distribuição", "h3")
    col_tickers, col_clear = st.columns([2, 1])
    with col_tickers:
        tickers = st.multiselect(
            "Ações da carteira",
            options=stocks,
            format_func=lambda t: f"{t}  ·  {_ticker_empresa[t]}" if _ticker_empresa.get(t) else t,
            placeholder="Busque pelo ticker ou nome da empresa…",
            key="selected_tickers",
            max_selections=MAX_TICKERS,
            help=(
                f"Digite ticker ou empresa. Limite de {MAX_TICKERS} ativos "
                "para manter os cálculos estáveis."
            ),
        )

    if st.session_state.pop("_portfolio_handoff_notice", None):
        st.info(
            "Seleção carregada da análise de ativos. "
            "Confira os papéis e clique em **Carregar portfólio**."
        )

    if st.session_state.pop("_portfolio_restore_notice", False):
        st.info(
            "Carteira salva restaurada. Clique em **Carregar portfólio** para atualizar a análise."
        )
    with col_clear:
        st.write("")
        st.write("")
        st.button(
            "Limpar carteira",
            use_container_width=True,
            help="Remove os ativos salvos desta carteira e limpa a seleção atual.",
            on_click=_clear_saved_portfolio,
        )

    col_capital, col_strategy = st.columns([1.4, 2], gap="large")
    with col_capital:
        valor_inicial = st.number_input(
            "Capital disponível (R$)", 100, 1_000_000, 10_000
        )
    with col_strategy:
        modo = st.radio(
            "Estratégia de alocação",
            (
                "Otimização de Mínima Volatilidade",
                "Otimização de Markowitz (Média-Variância)",
                "Otimização Hierarchical Risk Parity (Machine Learning)",
                "Alocação Manual",
            ),
        )

if len(tickers) == 0:
    empty_state_card(
        ICO_BOX,
        "Monte seu portfólio",
        "Selecione pelo menos duas ações para comparar risco, retorno e diversificação.",
        "Voltar para análise de ativos",
        "Main_Page.py",
    )
    st.stop()

if len(tickers) == 1:
    empty_state_card(
        ICO_BOX,
        "Adicione mais um ativo",
        "A otimização precisa de pelo menos dois ativos para calcular uma carteira.",
        "Voltar para análise de ativos",
        "Main_Page.py",
    )
    st.stop()


# Any portfolio rerun can change lookback, mode, or manual shares. Invalidate
# cross-page results until this run completes successfully.
if st.session_state.get("portfolio_loaded"):
    st.session_state["portfolio_analysis_tickers"] = []
tickers_yf = [t + ".SA" for t in tickers]

# Inputs de cotas manuais devem aparecer ANTES do botão
cotas_manuais_inputs = {}
if "Manual" in modo:
    with st.container(border=True):
        section_header(ICO_BOX, "Cotas por ativo", "h3")
        st.caption(
            "Os pesos serão calculados pelo valor de mercado das cotas informadas; "
            "o capital disponível continua sendo a base da simulação."
        )
        manual_columns = st.columns(min(len(tickers), 4))
        for index, ticker in enumerate(tickers):
            with manual_columns[index % len(manual_columns)]:
                _shares_key = f"cotas_manual_{ticker}"
                st.session_state.setdefault(_shares_key, 0)
                shares = st.number_input(
                    f"Cotas de {ticker}",
                    min_value=0,
                    step=1,
                    key=_shares_key,
                )
            cotas_manuais_inputs[ticker + ".SA"] = int(shares)

# Controle de diversificação para Markowitz. A otimização média-variância é um
# problema de canto: ela concentra o capital em poucos ativos e zera o resto.
# A regularização L2 (add_objective(L2_reg, gamma)) penaliza a concentração,
# reduzindo os pesos zerados de forma suave — gamma maior = mais distribuído.
gamma_l2 = 0.0
if "Markowitz" in modo:
    with st.container(border=True):
        section_header(ICO_RULER, "Diversificação", "h3")
        st.session_state.setdefault("portfolio_gamma_l2_input", 1.0)

        def _set_gamma_preset(value: float) -> None:
            st.session_state["portfolio_gamma_l2_input"] = value

        st.caption("Controle quanto a otimização evita concentrar o capital:")
        _gamma_presets = (
            ("Concentrada", 0.0, "Markowitz puro; pode concentrar em poucos ativos."),
            ("Equilibrada", 1.0, "Ponto de partida balanceado entre retorno e diversificação."),
            ("Diversificada", 2.0, "Penaliza mais a concentração e distribui melhor os pesos."),
        )
        _gamma_cols = st.columns(len(_gamma_presets))
        for _index, (_label, _value, _help) in enumerate(_gamma_presets):
            with _gamma_cols[_index]:
                st.button(
                    _label,
                    key=f"gamma_preset_{_index}",
                    use_container_width=True,
                    help=_help,
                    on_click=_set_gamma_preset,
                    args=(_value,),
                )

        gamma_l2 = st.number_input(
            "Diversificação (regularização L2)",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="portfolio_gamma_l2_input",
            help=(
                "Penaliza a concentração para evitar pesos zerados. "
                "0 = Markowitz puro; valores maiores distribuem o capital "
                "entre mais ativos, aproximando-se de pesos iguais."
            ),
        )

if "Manual" in modo:
    _required_tickers_yf = [
        ticker for ticker in tickers_yf if cotas_manuais_inputs.get(ticker, 0) > 0
    ]
    if not _required_tickers_yf:
        st.error("A alocação manual precisa ter ao menos uma cota positiva.")
        st.stop()
else:
    _required_tickers_yf = tickers_yf

# Keep the analysis visible across widget reruns, but fetch only after loading.
if st.button("Carregar portfólio", type="primary", use_container_width=True):
    st.session_state["portfolio_loaded"] = True
    st.session_state["portfolio_loaded_tickers"] = list(tickers)

_loaded_tickers = st.session_state.get("portfolio_loaded_tickers", [])
if st.session_state.get("portfolio_loaded") and _loaded_tickers != list(tickers):
    st.info("A seleção mudou. Clique em **Carregar portfólio** para atualizar a análise.")
if not st.session_state.get("portfolio_loaded") or _loaded_tickers != list(tickers):
    st.stop()

trade_quotes = {}
quote_fetch_error = False
pesos_manuais = {}
if "Manual" in modo:
    trade_tickers = tuple(_required_tickers_yf)
    trade_quotes, quote_fetch_error = _get_trade_quotes(trade_tickers)
    missing_manual_quotes = [
        ticker for ticker in trade_tickers if ticker not in trade_quotes
    ]
    if quote_fetch_error or missing_manual_quotes:
        missing_labels = ", ".join(
            ticker.removesuffix(".SA") for ticker in missing_manual_quotes
        ) or ", ".join(ticker.removesuffix(".SA") for ticker in trade_tickers)
        st.error(
            f"Não foi possível obter cotações atuais para {missing_labels}; "
            "elas são necessárias para calcular os pesos por cotas."
        )
        st.stop()
    market_values = {
        ticker: cotas_manuais_inputs[ticker] * trade_quotes[ticker][0]
        for ticker in trade_tickers
    }
    total_market_value = sum(market_values.values())
    pesos_manuais = {
        ticker: market_values.get(ticker, 0.0) / total_market_value
        for ticker in tickers_yf
    }

_price_status = st.empty()
_price_status.markdown(
    '<div class="discreet-status">Baixando cotações históricas…</div>',
    unsafe_allow_html=True,
)
try:
    data_yf = get_portfolio_prices(_required_tickers_yf, data_inicio)
except Exception as _price_err:
    logger.warning("get_portfolio_prices failed: %s", _price_err)
    logger.debug("get_portfolio_prices failure details", exc_info=True)
    _price_status.empty()
    st.error(
        f"Erro ao buscar cotações no Yahoo Finance: {_price_err}. Verifique sua conexão e tente novamente."
    )
    st.stop()
_price_status.empty()

if data_yf.empty:
    st.error(
        "Nenhuma cotação retornada para os ativos selecionados. Verifique os tickers e o período escolhido."
    )
    st.stop()

if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ["_".join(col).strip() for col in data_yf.columns.values]
missing_tickers = sorted(
    set(map(str, _required_tickers_yf)).difference(map(str, data_yf.columns))
)
if missing_tickers:
    missing_labels = ", ".join(ticker.replace(".SA", "") for ticker in missing_tickers)
    logger.warning("portfolio price history missing for %s", missing_tickers)
    st.error(
        f"Não foi possível obter cotações para: {missing_labels}. "
        "Remova esses ativos ou tente novamente mais tarde."
    )
    st.stop()

analysis_prices = data_yf.loc[:, _required_tickers_yf]
returns = analysis_prices.pct_change(fill_method=None).dropna()

MIN_RETURN_ROWS = 30
if len(returns) < MIN_RETURN_ROWS:
    # In complete-case returns, one short-history asset can collapse the sample.
    first_valid = analysis_prices.apply(lambda col: col.first_valid_index())
    short_history = first_valid.dropna().sort_values(ascending=False).head(5)
    culprits = ", ".join(
        [f"{col.replace('.SA', '')} (sem cotações)" for col in first_valid[first_valid.isna()].index]
        + [
            f"{col.replace('.SA', '')} (dados desde {date.strftime('%d/%m/%Y')})"
            for col, date in short_history.items()
        ]
    )
    st.error(
        f"Histórico de cotações em comum insuficiente entre os ativos usados "
        f"(apenas {len(returns)} dia(s) com dados completos). "
        "Isso costuma acontecer quando um ou mais ativos têm histórico bem mais curto "
        "que os demais (IPO recente, deslistagem, falha na fonte de dados)."
        + (f" Possíveis responsáveis: {culprits}." if culprits else "")
        + " Remova ativos com histórico curto ou ajuste o período de lookback."
    )
    st.stop()

page_container = st.empty()

if (
    st.session_state.get("portfolio_loaded")
    and st.session_state.get("portfolio_loaded_tickers", []) == list(tickers)
):
    _sample_period = (
        f"{returns.index.min():%d/%m/%Y} a {returns.index.max():%d/%m/%Y}"
        if isinstance(returns.index, pd.DatetimeIndex)
        else "datas indisponíveis"
    )
    _zero_weight_note = (
        " Ativos com peso zero foram excluídos da amostra."
        if "Manual" in modo and len(_required_tickers_yf) < len(tickers)
        else ""
    )
    st.caption(
        f"{len(tickers)} ativos selecionados · {len(returns)} retornos diários completos "
        f"({_sample_period}).{_zero_weight_note}"
    )
    # Overlay de carregamento global
    with loading_overlay("Carregando dados, aguarde…", tickers=tickers):
        if "Manual" in modo:
            pesos_manuais_arr = np.array(list(pesos_manuais.values()))
            peso_manual_df = pd.DataFrame.from_dict(
                pesos_manuais, orient="index", columns=["Peso"]
            )
        elif "Hierarchical" in modo:
            st.subheader("Otimização Hierarchical Risk Parity (HRP)")
            hrp = HRPOpt(returns)
            weights_hrp = hrp.optimize()
            peso_manual_df = pd.DataFrame.from_dict(
                weights_hrp, orient="index", columns=["Peso"]
            )
            pesos_manuais_arr = peso_manual_df[
                "Sample_Vol" if "Sample_Vol" in peso_manual_df.columns else "Peso"
            ].values
        elif "Mínima Volatilidade" in modo:
            st.subheader("Otimização de Mínima Volatilidade")
            mu, S = estimate_markowitz_inputs(returns)
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            peso_manual_df = pd.DataFrame.from_dict(
                cleaned_weights, orient="index", columns=["Peso"]
            )
            pesos_manuais_arr = peso_manual_df["Peso"].values
        else:
            st.subheader("Otimização de Markowitz (Média-Variância)")
            mu, S = estimate_markowitz_inputs(returns)
            selic_anual = (1 + taxa_selic) ** 252 - 1
            try:
                ef = EfficientFrontier(mu, S)
                raw_weights = ef.max_sharpe(risk_free_rate=selic_anual)
                allocation_label = "Max Sharpe (Markowitz)"
                if gamma_l2 > 0:
                    # L2_reg does not compose with max_sharpe (it internally
                    # transforms the problem, so the penalty is ignored). To
                    # regularize while preserving the Max-Sharpe return, re-solve
                    # for that same target return with L2 active — this spreads
                    # the weight across more assets and avoids zeros.
                    tangency_return = ef.portfolio_performance(
                        risk_free_rate=selic_anual
                    )[0]
                    target_return = bound_efficient_return(
                        tangency_return, float(mu.min()), float(mu.max())
                    )
                    ef = EfficientFrontier(mu, S)
                    if target_return is None:
                        # With effectively identical expected returns there is
                        # no meaningful target-return frontier to solve.
                        raw_weights = ef.min_volatility()
                        allocation_label = "Mínima Volatilidade"
                    else:
                        ef.add_objective(objective_functions.L2_reg, gamma=gamma_l2)
                        raw_weights = ef.efficient_return(target_return=target_return)
                        allocation_label = "Retorno-alvo com L2"
                cleaned_weights = ef.clean_weights()
            except Exception as e:
                logger.exception("max_sharpe optimization failed")
                st.warning(
                    f"Otimização de Max Sharpe falhou (motivo: {str(e)}). Usando carteira de Mínima Volatilidade."
                )
                ef = EfficientFrontier(mu, S)
                if gamma_l2 > 0:
                    ef.add_objective(objective_functions.L2_reg, gamma=gamma_l2)
                raw_weights = ef.min_volatility()
                allocation_label = "Mínima Volatilidade"
                cleaned_weights = ef.clean_weights()
            peso_manual_df = pd.DataFrame.from_dict(
                cleaned_weights, orient="index", columns=["Peso"]
            )
            pesos_manuais_arr = peso_manual_df["Peso"].values

        _persisted_weights = {
            str(index): float(weight)
            for index, weight in peso_manual_df["Peso"].items()
            if pd.notna(weight)
        }
        _db.portfolio_save(_uid, tickers, _persisted_weights)
        # Keep market-data ticker labels for calculations; the display frame below
        # strips .SA because the simulation page expects bare ticker names.
        pesos_por_ticker = peso_manual_df["Peso"].to_dict()

        # Mostrar pesos
        st.subheader("Pesos do Portfólio (%)")
        peso_manual_df.index = peso_manual_df.index.str.replace(".SA", "", regex=False)
        pesos_dict = {
            ticker: f"{(row['Peso'] * 100):.2f}%"
            for ticker, row in peso_manual_df.iterrows()
        }
        render_cards_grid(pesos_dict)

        # Download portfolio allocation
        try:
            df_pesos_export = peso_manual_df.copy()
            df_pesos_export.index = df_pesos_export.index.str.replace(
                ".SA", "", regex=False
            )
            df_pesos_export["Peso (%)"] = (df_pesos_export["Peso"] * 100).round(2)
            df_pesos_export = df_pesos_export.drop(columns=["Peso"])
            pesos_csv = df_pesos_export.to_csv().encode("utf-8-sig")
            st.download_button(
                label="⬇ Exportar alocação (CSV)",
                data=pesos_csv,
                file_name=f"portfolio_alocacao_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        except Exception:
            logger.debug("portfolio allocation CSV export failed", exc_info=True)

        # ── Sugestão de Compra de Cotas (Alocação Discreta) ───────────────────
        st.subheader("Sugestão de Compra de Cotas")
        st.caption(
            "Estimativas usam o último fechamento bruto disponível por ativo (pode haver atraso); "
            "confirme o preço de execução. Retornos seguem usando preços ajustados."
        )
        st.markdown(
            f"Estimativa de cotas a comprar considerando o valor total de **R$ {valor_inicial:,.2f}**."
        )

        trade_tickers = tuple(
            ticker for ticker, weight in pesos_por_ticker.items() if weight > 0
        )
        if "Manual" not in modo:
            trade_quotes, quote_fetch_error = _get_trade_quotes(trade_tickers)

        missing_quotes = [
            ticker.removesuffix(".SA")
            for ticker in trade_tickers
            if ticker not in trade_quotes
        ]
        cotas_list = []
        total_efetivo = 0.0
        for ticker, row in peso_manual_df.iterrows():
            weight = float(row["Peso"])
            valor_teorico = weight * valor_inicial
            quote = trade_quotes.get(f"{ticker}.SA") if weight > 0 else None
            if weight <= 0:
                latest_price, quote_date, cotas, valor_efetivo = None, None, 0, 0.0
            elif quote is None:
                latest_price, quote_date, cotas, valor_efetivo = None, None, None, None
            else:
                latest_price, quote_date = quote
                cotas = int(np.floor(valor_teorico / latest_price))
                valor_efetivo = cotas * latest_price
                total_efetivo += valor_efetivo

            cotas_list.append(
                {
                    "Ativo": ticker,
                    "Preço Unitário": (
                        f"R$ {latest_price:,.2f}" if latest_price is not None
                        else "—" if weight <= 0 else "N/D"
                    ),
                    "Data da Cotação": (
                        pd.Timestamp(quote_date).strftime("%d/%m/%Y")
                        if quote_date is not None else "—" if weight <= 0 else "N/D"
                    ),
                    "Cotas a Comprar": f"{cotas:,}" if cotas is not None else "N/D",
                    "Valor Efetivo": (
                        f"R$ {valor_efetivo:,.2f}" if valor_efetivo is not None else "N/D"
                    ),
                    "Valor Sugerido": f"R$ {valor_teorico:,.2f}",
                    "Peso Sugerido (%)": f"{weight * 100:.2f}%",
                    "_Valor Efetivo": valor_efetivo,
                    "_Peso": weight,
                }
            )

        quote_unavailable = quote_fetch_error or bool(missing_quotes)
        if quote_unavailable:
            missing_labels = ", ".join(
                missing_quotes or [ticker.removesuffix(".SA") for ticker in trade_tickers]
            )
            st.warning(
                f"Fechamento bruto indisponível para {missing_labels}; "
                "cotas e totais aparecem como N/D."
            )
        for quote_row in cotas_list:
            effective_value = quote_row.pop("_Valor Efetivo")
            weight = quote_row.pop("_Peso")
            if quote_unavailable and weight > 0:
                quote_row["Peso Efetivo (%)"] = "N/D"
            elif total_efetivo > 0:
                quote_row["Peso Efetivo (%)"] = (
                    f"{effective_value / total_efetivo * 100:.2f}%"
                )
            else:
                quote_row["Peso Efetivo (%)"] = "0.00%"
        st.dataframe(pd.DataFrame(cotas_list), use_container_width=True, hide_index=True)

        if not quote_unavailable:
            sobra_caixa = valor_inicial - total_efetivo
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Total Alocado Efetivo", f"R$ {total_efetivo:,.2f}")
            col_c2.metric("Saldo Restante (Caixa)", f"R$ {sobra_caixa:,.2f}")
            col_c3.metric(
                "Eficiência da Alocação", f"{(total_efetivo / valor_inicial) * 100:.2f}%"
            )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        if "Markowitz" in modo:
            section_header(ICO_FRONTIER, "Gráfico da Fronteira Eficiente", "h2")
            with loading_overlay("Gerando fronteira eficiente e simulando portfólios…"):
                selic_anual = (1 + taxa_selic) ** 252 - 1
                fig_frontier = plot_efficient_frontier_and_random_portfolios(
                    mu, S, cleaned_weights, selic_anual, allocation_label
                )
                st.plotly_chart(fig_frontier, use_container_width=True)

        # Cálculo do portfólio com os pesos escolhidos
        pesos_alinhados = align_weights_to_columns(
            pesos_por_ticker, returns.columns
        )
        portfolio_returns = returns.dot(pesos_alinhados)

        # O benchmark é complementar: se o IBOVESPA falhar, preserve a análise
        # do portfólio e sinalize que o gráfico está sem comparação.
        retorno_bench = None
        bench = None
        try:
            bench = get_benchmark_prices(data_inicio)
            portfolio_returns, retorno_bench = align_benchmark_returns(
                portfolio_returns, bench
            )
        except Exception as _bench_err:
            logger.warning("get_benchmark_prices failed: %s", _bench_err)
            logger.debug("get_benchmark_prices failure details", exc_info=True)

        benchmark_available = retorno_bench is not None and not retorno_bench.empty
        if benchmark_available:
            _benchmark_period = (
                f"{portfolio_returns.index.min():%d/%m/%Y} a "
                f"{portfolio_returns.index.max():%d/%m/%Y}"
                if isinstance(portfolio_returns.index, pd.DatetimeIndex)
                else "datas indisponíveis"
            )
            st.caption(
                f"Comparações com o IBOVESPA usam {len(portfolio_returns)} pregões comuns "
                f"({_benchmark_period})."
            )
        if not benchmark_available:
            st.warning(
                "Dados do IBOVESPA indisponíveis no momento; "
                "o gráfico exibirá apenas o valor do portfólio."
            )

        # Calcular o retorno acumulado do portfólio e, quando disponível, do benchmark.
        cum_return = (1 + portfolio_returns).cumprod()
        portfolio_value = cum_return * valor_inicial
        bench_value = None
        if benchmark_available:
            retorno_cum_bench = (1 + retorno_bench).cumprod()
            bench_value = retorno_cum_bench * valor_inicial

        # Mostrar gráfico do valor do portfólio x BOVESPA
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value,
                mode="lines",
                name="Portfólio",
                line=dict(color=CHART_PRIMARY, width=2.5),
            )
        )
        if benchmark_available:
            fig.add_trace(
                go.Scatter(
                    x=bench_value.index,
                    y=bench_value,
                    mode="lines",
                    name="IBOVESPA",
                    line=dict(color=CHART_SECONDARY, width=1.5, dash="dash"),
                )
            )
        fig.update_layout(
            title=(
                "Evolução do Valor do Portfólio vs Benchmark"
                if benchmark_available
                else "Evolução do Valor do Portfólio"
            ),
            xaxis_title="Data",
            yaxis_title="Valor (R$)",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Retornos históricos hipotéticos, pesos fixos rebalanceados diariamente; taxas e impostos excluídos. "
            + (
                "Markowitz/HRP são avaliados na mesma amostra usada para otimizar os pesos (in-sample), "
                "não é validação fora da amostra."
                if "Manual" not in modo
                else "A alocação manual descreve a amostra histórica, não uma previsão."
            )
        )
        # Retornos mensais
        section_header(ICO_HEATMAP, "Tabela de Retornos Mensais do Portfólio", "h2")
        try:
            monthly_ret = portfolio_returns.resample("ME").apply(
                lambda x: (1 + x).prod() - 1
            )
            monthly_ret_df = (
                monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month])
                .first()
                .unstack(level=1)
                * 100
            )
            monthly_ret_df = monthly_ret_df.reindex(columns=range(1, 13))
            month_cols = {
                1: "Jan",
                2: "Fev",
                3: "Mar",
                4: "Abr",
                5: "Mai",
                6: "Jun",
                7: "Jul",
                8: "Ago",
                9: "Set",
                10: "Out",
                11: "Nov",
                12: "Dez",
            }
            monthly_ret_df = monthly_ret_df.rename(columns=month_cols)

            # YTD
            ytd_ret = (
                portfolio_returns.groupby(portfolio_returns.index.year).apply(
                    lambda x: (1 + x).prod() - 1
                )
                * 100
            )
            monthly_ret_df["YTD"] = ytd_ret

            html_rows = []
            for year, row in monthly_ret_df.iterrows():
                row_html = '<tr style="border-bottom: 1px solid var(--border-default); font-size: 0.88rem;">'
                row_html += f'<td style="padding:var(--space-2);font-weight:700;text-align:left;color:var(--text-primary);font-family:var(--font-mono)">{year}</td>'
                for m in [
                    "Jan",
                    "Fev",
                    "Mar",
                    "Abr",
                    "Mai",
                    "Jun",
                    "Jul",
                    "Ago",
                    "Set",
                    "Out",
                    "Nov",
                    "Dez",
                ]:
                    val = row[m]
                    if pd.isna(val):
                        row_html += '<td style="padding:var(--space-2);color:var(--text-muted);font-family:var(--font-mono)">-</td>'
                    else:
                        color = (
                            CHART_PRIMARY
                            if val > 0
                            else CHART_DANGER
                            if val < 0
                            else CHART_MUTED
                        )
                        sign = "+" if val > 0 else ""
                        row_html += f'<td style="padding:var(--space-2);color:{color};font-family:var(--font-mono);font-weight:700">{sign}{val:.2f}%</td>'

                ytd_val = row["YTD"]
                if pd.isna(ytd_val):
                    row_html += '<td style="padding:var(--space-2);border-left:1px solid var(--border-default);color:var(--text-muted);font-family:var(--font-mono)">-</td>'
                else:
                    ytd_color = (
                        CHART_PRIMARY
                        if ytd_val > 0
                        else CHART_DANGER
                        if ytd_val < 0
                        else CHART_MUTED
                    )
                    ytd_sign = "+" if ytd_val > 0 else ""
                    row_html += f'<td style="padding:var(--space-2);border-left:1px solid var(--border-default);color:{ytd_color};font-family:var(--font-mono);font-weight:700">{ytd_sign}{ytd_val:.2f}%</td>'
                row_html += "</tr>"
                html_rows.append(row_html)

            table_html = f"""
            <div style="background:var(--panel-bg);
                        border: 1px solid var(--panel-border);
                        border-radius: var(--radius-md);
                        padding: var(--space-4);
                        overflow-x: auto; 
                        margin-bottom: var(--space-6);
                        box-shadow: none;">
                <table style="width: 100%; border-collapse: collapse; text-align: center; color: var(--text-primary); font-family: var(--font-display);">
                    <thead>
                        <tr style="border-bottom: 1px solid var(--border-default); color: var(--text-muted); font-size: 0.875rem;">
                            <th style="padding: 0.8rem 0.5rem; text-align: left;">Ano</th>
                            <th style="padding: 0.8rem 0.5rem;">Jan</th>
                            <th style="padding: 0.8rem 0.5rem;">Fev</th>
                            <th style="padding: 0.8rem 0.5rem;">Mar</th>
                            <th style="padding: 0.8rem 0.5rem;">Abr</th>
                            <th style="padding: 0.8rem 0.5rem;">Mai</th>
                            <th style="padding: 0.8rem 0.5rem;">Jun</th>
                            <th style="padding: 0.8rem 0.5rem;">Jul</th>
                            <th style="padding: 0.8rem 0.5rem;">Ago</th>
                            <th style="padding: 0.8rem 0.5rem;">Set</th>
                            <th style="padding: 0.8rem 0.5rem;">Out</th>
                            <th style="padding: 0.8rem 0.5rem;">Nov</th>
                            <th style="padding: 0.8rem 0.5rem;">Dez</th>
                            <th style="padding: var(--space-2); border-left: 1px solid var(--border-default); font-weight: 700; color: var(--brand-secondary);">YTD</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(html_rows)}
                    </tbody>
                </table>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)
        except Exception as e:
            logger.exception("monthly returns table generation failed")
            st.warning(f"Não foi possível gerar a tabela de retornos mensais: {str(e)}")

        if not benchmark_available:
            st.info(
                "Métricas relativas ao IBOVESPA ficam "
                "indisponíveis enquanto a série do benchmark não responder."
            )
            total_return = finite_or_none((portfolio_value.iloc[-1] / valor_inicial - 1) * 100)
            vol_anual = finite_or_none(portfolio_returns.std() * np.sqrt(252) * 100)
            sharpe_val = finite_or_none(sharpe(portfolio_returns, rf=taxa_selic_anual))
            sortino_val = finite_or_none(sortino(portfolio_returns, rf=taxa_selic_anual))
            max_dd = finite_or_none(max_drawdown(portfolio_returns) * 100)

            section_header(ICO_CHART, "Desempenho da Carteira", "h2")
            _fallback_metrics = st.columns(5)
            _fallback_metrics[0].metric(
                "Retorno Total",
                f"{total_return:.2f}%" if total_return is not None else "N/D",
                delta=("Positivo" if total_return > 0 else "Negativo")
                if total_return is not None else None,
                delta_color=("normal" if total_return > 0 else "inverse")
                if total_return is not None else "off",
            )
            _fallback_metrics[1].metric(
                "Volatilidade Anual",
                f"{vol_anual:.2f}%" if vol_anual is not None else "N/D",
                delta=("Baixa" if vol_anual < 15 else "Alta" if vol_anual > 25 else "Moderada")
                if vol_anual is not None else None,
                delta_color=("normal" if vol_anual < 15 else "inverse" if vol_anual > 25 else "off")
                if vol_anual is not None else "off",
            )
            _fallback_metrics[2].metric(
                "Índice Sharpe", f"{sharpe_val:.2f}" if sharpe_val is not None else "N/D"
            )
            _fallback_metrics[3].metric(
                "Índice Sortino", f"{sortino_val:.2f}" if sortino_val is not None else "N/D"
            )
            _fallback_metrics[4].metric(
                "Max Drawdown", f"{max_dd:.2f}%" if max_dd is not None else "N/D"
            )
            st.session_state.update(
                {
                    "modo": modo,
                    "returns": returns,
                    "portfolio_analysis_tickers": list(tickers),
                    "peso_manual_df": peso_manual_df,
                    "pesos_manuais": pesos_por_ticker,
                    "portfolio_returns": portfolio_returns,
                    "retorno_bench": None,
                    "lookback": lookback_opcao,
                    "total_return": total_return,
                    "vol_anual": vol_anual,
                    "sharpe_val": sharpe_val,
                    "sortino_val": sortino_val,
                    "max_dd": max_dd,
                }
            )
            for key in ("beta", "alfa_val", "r_quadrado", "information_ratio"):
                st.session_state.pop(key, None)
            st.caption(
                "Atualize as cotações quando o IBOVESPA estiver disponível para "
                "reativar beta, alfa e as comparações relativas."
            )
            next_step_card(
                message="Portfólio configurado — projete trajetórias com Monte Carlo.",
                accent="var(--brand-secondary)",
                cta_label="Abrir Simulação",
                cta_page="pages/2_Simulação.py",
            )
            st.stop()


        # Cálculos de Métricas
        total_return = finite_or_none((portfolio_value.iloc[-1] / valor_inicial - 1) * 100)
        vol_anual = finite_or_none(portfolio_returns.std() * np.sqrt(252) * 100)
        sharpe_val = finite_or_none(sharpe(portfolio_returns, rf=taxa_selic_anual))
        sortino_val = finite_or_none(sortino(portfolio_returns, rf=taxa_selic_anual))
        max_dd = finite_or_none(max_drawdown(portfolio_returns) * 100)

        cov_matrix = np.cov(
            portfolio_returns.squeeze(), retorno_bench.squeeze()
        )  # matriz de covariância 2x2
        benchmark_variance = float(cov_matrix[1, 1])
        benchmark_metrics_valid = (
            np.isfinite(benchmark_variance) and benchmark_variance > 1e-12
        )
        if not benchmark_metrics_valid:
            st.warning(
                "A variação do IBOVESPA é insuficiente para calcular métricas "
                "relativas; beta, alfa, R² e information ratio ficaram indisponíveis."
            )
            beta = None
        else:
            beta = finite_or_none(cov_matrix[0, 1] / benchmark_variance)
        # Jensen alpha is defined only when the benchmark has usable variance.
        alfa_val = None
        if benchmark_metrics_valid:
            alfa = (portfolio_returns.mean() - taxa_selic) - beta * (
                retorno_bench.mean() - taxa_selic
            )
            alfa_val = finite_or_none(
                alfa.values[0]
                if hasattr(alfa, "values") and len(alfa.values) > 0
                else alfa
            )
        if benchmark_metrics_valid:
            try:
                r_quadrado = finite_or_none(
                    qs.stats.r_squared(portfolio_returns, retorno_bench)
                )
                information_ratio = finite_or_none(
                    qs.stats.information_ratio(portfolio_returns, retorno_bench)
                )
            except Exception:
                logger.warning("relative benchmark metrics failed", exc_info=True)
                r_quadrado = None
                information_ratio = None
        else:
            r_quadrado = None
            information_ratio = None

        _elapsed_days = (
            portfolio_value.index[-1] - portfolio_value.index[0]
        ).days
        ret_anual = (
            finite_or_none(
                ((portfolio_value.iloc[-1] / valor_inicial) ** (365.25 / _elapsed_days) - 1)
                * 100
            )
            if _elapsed_days > 0
            else None
        )
        var_val = finite_or_none(var(portfolio_returns) * 100)
        alfa_anual = (
            finite_or_none(alfa_val * 252 * 100) if alfa_val is not None else None
        )

        # One compact summary replaces the repeated consolidated metric grid below.
        section_header(ICO_METRICS, "Resumo de Desempenho", "h2")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric(
            "Retorno Total",
            f"{total_return:.2f}%" if total_return is not None else "N/D",
            delta=("Positivo" if total_return > 0 else "Negativo")
            if total_return is not None else None,
            delta_color=("normal" if total_return > 0 else "inverse")
            if total_return is not None else "off",
            help="Retorno histórico hipotético, não retorno futuro nem das cotas compradas.",
        )
        col_m2.metric(
            "Volatilidade Anual",
            f"{vol_anual:.2f}%" if vol_anual is not None else "N/D",
            delta=("Baixa" if vol_anual < 15 else "Alta" if vol_anual > 25 else "Moderada")
            if vol_anual is not None else None,
            delta_color=("normal" if vol_anual < 15 else "inverse" if vol_anual > 25 else "off")
            if vol_anual is not None else "off",
            help="Desvio padrão anualizado dos retornos. Mede o risco total.",
        )
        col_m3.metric(
            "Índice Sharpe",
            f"{sharpe_val:.2f}" if sharpe_val is not None else "N/D",
            delta=(
                "Excelente" if sharpe_val > 1
                else "Bom" if sharpe_val > 0.5
                else "Baixo"
            ) if sharpe_val is not None else None,
            delta_color=("normal" if sharpe_val > 0.5 else "inverse")
            if sharpe_val is not None else "off",
            help="Retorno por unidade de risco. Acima de 1.0 é excelente.",
        )
        col_m4.metric(
            "Índice Sortino",
            f"{sortino_val:.2f}" if sortino_val is not None else "N/D",
            delta=(
                "Excelente" if sortino_val > 1
                else "Bom" if sortino_val > 0.5
                else "Baixo"
            ) if sortino_val is not None else None,
            delta_color=("normal" if sortino_val > 0.5 else "inverse")
            if sortino_val is not None else "off",
            help="Igual ao Sharpe mas penaliza apenas volatilidade negativa.",
        )
        col_m5.metric(
            "Max Drawdown",
            f"{max_dd:.2f}%" if max_dd is not None else "N/D",
            delta=("Controlado" if max_dd > -15 else "Severo" if max_dd < -30 else "Moderado")
            if max_dd is not None else None,
            delta_color=("normal" if max_dd > -15 else "inverse" if max_dd < -30 else "off")
            if max_dd is not None else "off",
            help="Maior perda de pico a vale. Quanto mais próximo de 0, melhor.",
        )
        _extra_metrics = st.columns(5)
        _extra_metrics[0].metric(
            "Retorno Anualizado", f"{ret_anual:.2f}%" if ret_anual is not None else "N/D"
        )
        _extra_metrics[1].metric(
            "Beta vs IBOV", f"{beta:.3f}" if beta is not None else "N/D"
        )
        _extra_metrics[2].metric(
            "Alpha Anual", f"{alfa_anual:.2f}%" if alfa_anual is not None else "N/D"
        )
        _extra_metrics[3].metric(
            "VaR Diário (95%)", f"{var_val:.2f}%" if var_val is not None else "N/D"
        )
        _extra_metrics[4].metric(
            "Information Ratio",
            f"{information_ratio:.2f}" if information_ratio is not None else "N/D",
        )

        # ── Painel de Decisão do Investidor ───────────────────────────────────
        section_header(ICO_TARGET, "Painel de Decisão do Investidor", "h2")

        # ── Score de Saúde do Portfólio ──────────────────────────────────────
        health_detalhes = []

        if sharpe_val is None:
            health_detalhes.append(
                (ICO_WARN, "Sharpe indisponível — risco não definido", "#ffd600")
            )
        elif sharpe_val > 1.0:
            health_detalhes.append((ICO_OK, "Sharpe excelente (>1.0)", "#00ff87"))
        elif sharpe_val > 0.5:
            health_detalhes.append((ICO_WARN, "Sharpe razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append(
                (ICO_CRIT, "Sharpe baixo (<0.5) — revise a alocação", "#ff3d5a")
            )

        if sortino_val is None:
            health_detalhes.append(
                (ICO_WARN, "Sortino indisponível — risco de queda não definido", "#ffd600")
            )
        elif sortino_val > 1.0:
            health_detalhes.append((ICO_OK, "Sortino excelente (>1.0)", "#00ff87"))
        elif sortino_val > 0.5:
            health_detalhes.append((ICO_WARN, "Sortino razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append(
                (ICO_CRIT, "Sortino baixo — retornos negativos relevantes", "#ff3d5a")
            )

        if max_dd is None:
            health_detalhes.append((ICO_WARN, "Drawdown indisponível", "#ffd600"))
        elif max_dd > -10:
            health_detalhes.append((ICO_OK, "Drawdown controlado (<10%)", "#00ff87"))
        elif max_dd > -20:
            health_detalhes.append((ICO_WARN, "Drawdown moderado (10–20%)", "#ffd600"))
        else:
            health_detalhes.append(
                (
                    ICO_CRIT,
                    "Drawdown severo (>20%) — perda de pico a vale elevada",
                    "#ff3d5a",
                )
            )

        if alfa_val is None:
            alfa_anual = None
            health_detalhes.append(
                (
                    ICO_WARN,
                    "Alfa indisponível — variação insuficiente do IBOVESPA",
                    "#ffd600",
                )
            )
        else:
            alfa_anual = finite_or_none(alfa_val * 252 * 100)
            if alfa_anual is None:
                health_detalhes.append((ICO_WARN, "Alfa indisponível", "#ffd600"))
            elif alfa_anual > 5:
                health_detalhes.append(
                    (ICO_OK, f"Alfa anual positivo: {alfa_anual:.1f}%", "#00ff87")
                )
            elif alfa_anual > 0:
                health_detalhes.append(
                    (ICO_WARN, f"Alfa marginal: {alfa_anual:.1f}%", "#ffd600")
                )
            else:
                health_detalhes.append(
                    (
                        ICO_CRIT,
                        f"Alfa negativo ({alfa_anual:.1f}%) — "
                        "abaixo do retorno ajustado ao risco do IBOVESPA",
                        "#ff3d5a",
                    )
                )

        pesos_arr_dec = np.array(
            pesos_manuais_arr
        )  # sempre definido independente do modo
        max_peso = pesos_arr_dec.max() * 100
        score, score_coverage = evaluate_portfolio_health(
            sharpe_val, sortino_val, max_dd, alfa_anual, max_peso
        )
        if max_peso <= 30:
            health_detalhes.append(
                (
                    ICO_OK,
                    f"Maior posição: {max_peso:.1f}% (concentração baixa)",
                    "#00ff87",
                )
            )
        elif max_peso <= 50:
            health_detalhes.append(
                (
                    ICO_WARN,
                    f"Maior posição: {max_peso:.1f}% (concentração elevada)",
                    "#ffd600",
                )
            )
        else:
            health_detalhes.append(
                (
                    ICO_CRIT,
                    f"Maior posição: {max_peso:.1f}% (concentração muito alta)",
                    "#ff3d5a",
                )
            )

        score_label = (
            "Dados insuficientes"
            if score is None
            else "Favorável" if score >= 70
            else "Intermediário" if score >= 40
            else "Desfavorável"
        )

        col_score, col_details = st.columns([1, 2], gap="large")
        with col_score:
            st.metric(
                "Indicador heurístico",
                f"{score}/100" if score is not None else "N/D",
            )
            st.caption(f"{score_label} · cobertura de {score_coverage}%")
        with col_details:
            st.markdown("**Indicadores considerados**")
            for ico, msg, color in health_detalhes:
                diag_row(ico, msg, color)


        # ── Stress Test — Crises Históricas ──────────────────────────────────
        st.markdown("---")
        section_header(ICO_STRESS, "Stress Test — Crises Históricas", "h2")
        st.caption(
            "Desempenho hipotético com pesos atuais rebalanceados diariamente, sem taxas ou impostos. "
            "Por padrão usa o lookback selecionado; histórico longo é opcional."
        )

        CRISES_HISTORICAS = {
            "COVID-19 (2020)": ("2020-01-17", "2020-03-23"),
            "Recessão/Impeachment (2015–16)": ("2014-12-31", "2016-12-31"),
            "Joesley Day (2017)": ("2017-05-17", "2017-06-30"),
            "Crise Global (2008–09)": ("2008-08-01", "2009-03-31"),
            "Crise fiscal brasileira (2024)": ("2024-11-26", "2024-12-30"),
        }
        stress_start = min(
            pd.Timestamp(start) for start, _ in CRISES_HISTORICAS.values()
        ).date()

        _stress_tickers_yf = tuple(
            ticker for ticker, weight in pesos_por_ticker.items() if weight > 0
        )
        stress_prices = data_yf.reindex(columns=_stress_tickers_yf)
        stress_benchmark = bench
        stress_history_start = data_inicio
        if st.checkbox(
            f"Carregar histórico longo para crises (desde {stress_start.year})", value=False
        ):
            try:
                stress_prices = get_portfolio_prices(_stress_tickers_yf, stress_start)
                if isinstance(stress_prices.columns, pd.MultiIndex):
                    stress_prices.columns = [
                        "_".join(map(str, col)).strip() for col in stress_prices.columns
                    ]
                stress_prices = stress_prices.reindex(columns=_stress_tickers_yf)
                stress_history_start = stress_start
            except Exception as _stress_err:
                logger.warning("Historical stress prices unavailable: %s", _stress_err)
                st.warning(
                    "Não foi possível carregar o histórico longo para o stress test."
                )
                stress_prices = data_yf
            try:
                stress_benchmark = get_benchmark_prices(stress_start)
            except Exception as _stress_bench_err:
                logger.warning(
                    "Historical stress benchmark unavailable: %s", _stress_bench_err
                )
                st.warning(
                    "IBOV histórico indisponível; comparação limitada ao lookback selecionado."
                )

        for crisis, missing_prices in find_crisis_history_gaps(
            stress_prices, CRISES_HISTORICAS, stress_history_start
        ).items():
            missing = "; ".join(
                f"{ticker}: {reason}" for ticker, reason in missing_prices.items()
            )
            st.warning(f"{crisis} — histórico incompleto: {missing}")

        stress_results = calculate_historical_stress(
            stress_prices, stress_benchmark, pesos_por_ticker, CRISES_HISTORICAS
        )

        if not stress_results:
            st.info(
                "Não há pelo menos 5 pregões completos nas crises para todos os ativos "
                "selecionados. Amplie o histórico acima para incluir crises mais antigas."
            )
        else:
            for r in stress_results:
                port_pct = r["Portfólio"] * 100
                ibov_pct = r["IBOV"] * 100 if r["IBOV"] is not None else None
                diff = (
                    (r["Portfólio"] - r["IBOV"]) * 100
                    if r["IBOV"] is not None
                    else None
                )
                port_color = "#ff3d5a" if port_pct < 0 else "#00ff87"
                ibov_str = f"{ibov_pct:+.1f}%" if ibov_pct is not None else "N/D"
                diff_str = f"{diff:+.1f}pp" if diff is not None else "N/D"
                diff_color = (
                    "#00ff87" if (diff is not None and diff >= 0) else "#ff3d5a"
                )
                cards_html = (
                    f'<div class="mcard"><div class="mcard-label">Portfólio</div>'
                    f'<div class="mcard-value" style="color:{port_color}">{port_pct:+.1f}%</div></div>'
                    f'<div class="mcard"><div class="mcard-label">IBOVESPA</div>'
                    f'<div class="mcard-value" style="color:{"#ff3d5a" if (ibov_pct is not None and ibov_pct < 0) else "#00d2ff"}">{ibov_str}</div></div>'
                    f'<div class="mcard"><div class="mcard-label">vs IBOV</div>'
                    f'<div class="mcard-value" style="color:{diff_color}">{diff_str}</div></div>'
                )
                st.markdown(
                    f'<div style="margin-bottom:0.3rem;font-size:0.78rem;font-weight:700;'
                    f'color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em">'
                    f'{escape(str(r["Crise"]))} <span style="font-weight:400;color:#475569">({escape(str(r["Período"]))})</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mcard-grid">{cards_html}</div>',
                    unsafe_allow_html=True,
                )

            if len(stress_results) >= 2:
                crises_names = [r["Crise"].split(" (")[0] for r in stress_results]
                port_vals = [r["Portfólio"] * 100 for r in stress_results]
                ibov_vals = [
                    r["IBOV"] * 100 if r["IBOV"] is not None else None
                    for r in stress_results
                ]
                stress_values = port_vals + [v for v in ibov_vals if v is not None]
                stress_padding = max((max(stress_values) - min(stress_values)) * 0.12, 1)
                fig_stress = go.Figure()
                fig_stress.add_trace(
                    go.Bar(
                        name="Portfólio",
                        x=crises_names,
                        y=port_vals,
                        marker_color=[
                            "#61d4c6" if v >= 0 else "#e58a93" for v in port_vals
                        ],
                        text=[f"{v:+.1f}%" for v in port_vals],
                        textposition="outside",
                    )
                )
                fig_stress.add_trace(
                    go.Bar(
                        name="IBOVESPA",
                        x=crises_names,
                        y=ibov_vals,
                        marker_color=[
                            "rgba(138,177,188,0.8)"
                            if v is not None and v >= 0
                            else "rgba(169,116,67,0.72)"
                            for v in ibov_vals
                        ],
                        text=[f"{v:+.1f}%" if v is not None else "" for v in ibov_vals],
                        textposition="outside",
                    )
                )
                fig_stress.update_layout(
                    barmode="group",
                    title="Portfólio vs IBOVESPA durante Crises Históricas",
                    yaxis_title="Retorno (%)",
                    height=380,
                    margin=dict(t=50, b=40, l=40, r=20),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                fig_stress.update_yaxes(
                    range=[
                        min(stress_values) - stress_padding,
                        max(stress_values) + stress_padding,
                    ]
                )
                apply_plotly_theme(fig_stress)
                st.plotly_chart(fig_stress, use_container_width=True)

        # ── Regime de Mercado (últimos 21 dias) ──────────────────────────────
        section_header(ICO_SIGNAL, "Regime de Mercado (Últimos 21 Dias)", "h3")
        retorno_recente = (1 + portfolio_returns.tail(21)).prod() - 1
        retorno_recente_bench = (1 + retorno_bench.tail(21)).prod() - 1
        outperformance_recente = (retorno_recente - retorno_recente_bench) * 100

        if retorno_recente > 0.03:
            regime_ico, regime_txt, regime_acao = (
                ICO_UP,
                "Alta",
                "Manter exposição. Ativos com momentum positivo podem ser aumentados.",
            )
        elif retorno_recente < -0.03:
            regime_ico, regime_txt, regime_acao = (
                ICO_DOWN,
                "Queda",
                "Revise os pesos. Considere reduzir exposição ou adicionar proteção (ex: BOVA11 put).",
            )
        else:
            regime_ico, regime_txt, regime_acao = (
                ICO_FLAT,
                "Lateral",
                "Consolidação. Bom momento para revisar correlações e rebalancear.",
            )

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Regime", regime_txt, delta=f"{retorno_recente * 100:.2f}% (21d)")
        col_r2.metric(
            "vs IBOVESPA (21d)",
            f"{outperformance_recente:+.2f}%",
            delta="Outperform" if outperformance_recente > 0 else "Underperform",
        )
        col_r3.metric(
            "Volatilidade (21d)",
            f"{portfolio_returns.tail(21).std() * np.sqrt(252) * 100:.1f}% a.a.",
        )
        diag_row(ICO_IDEA, f"<b>Sugestão:</b> {regime_acao}", "#ffd600")

        # ── Índice HHI de Concentração ────────────────────────────────────────
        hhi = (pesos_arr_dec**2).sum() * 10000
        hhi_equiv = int(1 / (pesos_arr_dec**2).sum())
        n_ativos = len(pesos_arr_dec)

        col_hhi1, col_hhi2, col_hhi3 = st.columns(3)
        col_hhi1.metric(
            "HHI Concentração",
            f"{hhi:.0f}",
            delta="Baixo" if hhi < 2500 else "Moderado" if hhi < 5000 else "Alto",
            delta_color="normal" if hhi < 2500 else "inverse",
        )
        col_hhi2.metric("Ativos Efetivos", f"{hhi_equiv}/{n_ativos}")
        col_hhi3.metric(
            "Maior Peso",
            f"{max_peso:.1f}%",
            delta="OK" if max_peso <= 35 else "Concentrado",
            delta_color="normal" if max_peso <= 35 else "inverse",
        )

        if hhi > 5000:
            diag_row(
                ICO_CRIT,
                f"<b>Alta Concentração (HHI={hhi:.0f}):</b> Portfólio fortemente concentrado. Adicione ativos ou redistribua os pesos.",
                "#ff3d5a",
            )
        elif hhi > 2500:
            diag_row(
                ICO_WARN,
                f"<b>Concentração Moderada (HHI={hhi:.0f}):</b> Apenas {hhi_equiv} ativos efetivos de {n_ativos}.",
                "#ffd600",
            )
        else:
            diag_row(
                ICO_OK,
                f"<b>Boa Diversificação (HHI={hhi:.0f}):</b> {hhi_equiv} ativos efetivos — distribuição equilibrada.",
                "#00ff87",
            )

        st.markdown(
            """
<div class="portfolio-section-divider"></div>
""",
            unsafe_allow_html=True,
        )
        section_header(ICO_RISK, "Análise de Drawdown", "h2")

        # 1. Gráfico de Drawdown do Portfólio
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                fillcolor=CHART_DANGER_FILL,
                line=dict(color=CHART_DANGER, width=1.5),
                name="Drawdown",
                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>",
            )
        )
        fig1.update_layout(
            title="Evolução do Drawdown do Portfólio",
            xaxis_title="Data",
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
        )
        apply_plotly_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Tabela de Drawdown por Ativo
        st.subheader("Máximo Drawdown por Ativo Individual")

        def calcular_drawdown(series):
            cum_returns_act = (1 + series).cumprod()
            rolling_max_act = cum_returns_act.cummax()
            drawdown_act = (cum_returns_act - rolling_max_act) / rolling_max_act
            return drawdown_act

        drawdowns_ativos = returns.apply(calcular_drawdown)
        max_drawdowns = drawdowns_ativos.min()
        data_max_drawdowns = drawdowns_ativos.idxmin()

        df_drawdowns = pd.DataFrame(
            {
                "Máximo Drawdown (%)": max_drawdowns * 100,
                "Data do Máximo Drawdown": data_max_drawdowns,
            }
        ).sort_values(by="Máximo Drawdown (%)")

        df_drawdowns.index = df_drawdowns.index.str.replace(".SA", "", regex=False)

        items_dd = list(df_drawdowns.iterrows())
        num_cols = 4
        for i in range(0, len(items_dd), num_cols):
            chunk = items_dd[i : i + num_cols]
            cols = st.columns(len(chunk))
            for col, (ticker, row_d) in zip(cols, chunk):
                m_dd = row_d["Máximo Drawdown (%)"]
                dt_dd = row_d["Data do Máximo Drawdown"].strftime("%Y-%m-%d")
                with col:
                    st.markdown(
                        f"""
                    <div style="background:var(--panel-bg);
                                border: 1px solid var(--panel-border);
                                border-radius: var(--radius-md);
                                padding: var(--space-3);
                                text-align: center; 
                                box-shadow: none;
                                margin-bottom: var(--space-2);
                                min-height: 110px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;">
                        <div style="font-size: 1.1rem; color: var(--brand-secondary); font-weight: 700; font-family: var(--font-mono); margin-bottom: var(--space-1);">{ticker}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; margin-bottom: var(--space-1);">Máx. Drawdown</div>
                        <div style="font-size: 1.2rem; color: var(--brand-danger); font-weight: 700; font-family: var(--font-mono); margin-bottom: var(--space-1);">{m_dd:.2f}%</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">{dt_dd}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Rolling Beta (60 dias)
        window = 60
        rolling_cov = portfolio_returns.rolling(window).cov(retorno_bench)
        rolling_var = retorno_bench.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var.where(rolling_var.abs() > 1e-12)

        # Gráfico Rolling Beta
        st.subheader(f"Beta Móvel ({window} dias) vs IBOVESPA")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=rolling_beta.index,
                y=rolling_beta.values,
                line=dict(color=CHART_SECONDARY, width=2),
                name="Beta Móvel",
                hovertemplate="%{x|%d/%m/%Y}<br>β=%{y:.3f}<extra></extra>",
            )
        )
        fig2.add_hline(
            y=1,
            line_dash="dash",
            line_color=CHART_ACCENT,
            line_width=1.5,
            annotation_text="β = 1",
            annotation_font=dict(color=CHART_ACCENT, size=10),
        )
        fig2.update_layout(
            title=f"Beta Móvel ({window} dias) vs IBOVESPA",
            xaxis_title="Data",
            yaxis_title="Beta",
        )
        apply_plotly_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # Gráfico Sharpe Móvel
        rolling_sharpe = (
            (
                portfolio_returns.rolling(window).mean() - taxa_selic
            )
            / portfolio_returns.rolling(window).std()
        ) * np.sqrt(252)

        st.subheader(f"Índice de Sharpe Móvel Anualizado ({window} dias)")
        fig_3 = go.Figure()
        fig_3.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                line=dict(color=CHART_PRIMARY, width=2),
                name="Sharpe Móvel",
                hovertemplate="%{x|%d/%m/%Y}<br>Sharpe=%{y:.2f}<extra></extra>",
            )
        )
        fig_3.add_hline(y=0, line_dash="dash", line_color=CHART_GRID, line_width=1)
        fig_3.update_layout(
            title=f"Índice de Sharpe Móvel Anualizado ({window} dias)",
            xaxis_title="Data",
            yaxis_title="Sharpe anualizado",
        )
        apply_plotly_theme(fig_3)
        st.plotly_chart(fig_3, use_container_width=True)

        # Salva variáveis para uso na aba Simulação e Relatório
        if "Manual" in modo:
            clean_modo = "Alocação Manual"
        elif "Hierarchical" in modo:
            clean_modo = "Otimização Hierarchical Risk Parity (HRP)"
        elif "Mínima Volatilidade" in modo:
            clean_modo = "Otimização de Mínima Volatilidade"
        else:
            clean_modo = "Otimização de Markowitz (Média-Variância)"
        st.session_state["modo"] = clean_modo
        st.session_state["returns"] = returns
        st.session_state["portfolio_analysis_tickers"] = list(tickers)
        st.session_state["peso_manual_df"] = peso_manual_df
        st.session_state["portfolio_returns"] = portfolio_returns
        st.session_state["retorno_bench"] = retorno_bench
        st.session_state["lookback"] = lookback_opcao
        st.session_state["total_return"] = total_return
        st.session_state["vol_anual"] = vol_anual
        st.session_state["sharpe_val"] = sharpe_val
        st.session_state["sortino_val"] = sortino_val
        st.session_state["max_dd"] = max_dd
        st.session_state["beta"] = beta
        st.session_state["alfa_val"] = alfa_val
        st.session_state["r_quadrado"] = r_quadrado
        st.session_state["information_ratio"] = information_ratio

        # Garante que pesos manuais ficam disponíveis como dicionário
        if "Manual" in modo:
            st.session_state["pesos_manuais"] = pesos_manuais
        else:
            st.session_state["pesos_manuais"] = pesos_por_ticker

        # ── Próximo Passo ────────────────────────────────────────────────────
        st.markdown("---")
        sharpe_txt = (
            f"Sharpe de {sharpe_val:.2f}" if sharpe_val > 0 else "portfólio configurado"
        )
        next_step_card(
            message=f"Com {sharpe_txt} — projete trajetórias com Monte Carlo.",
            accent="var(--brand-secondary)",
            cta_label="Abrir Simulação",
            cta_page="pages/2_Simulação.py",
        )
