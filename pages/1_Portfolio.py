import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from quantstats.stats import sharpe, sortino, max_drawdown, var
import quantstats as qs
from bcb import sgs
from utils.charts import apply_plotly_theme
from utils.ui import load_css, loading_overlay, render_flow_sidebar, svg_icon
from utils.market_data import get_sorted_tickers_by_liquidity


# ─── SVG Icon Library ─────────────────────────────────────────────────────────
_svg = svg_icon

ICO_OK = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#00ff87" stroke-width="1.8"/>'
    '<path d="M8 12l3 3 5-5" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
)
ICO_WARN = _svg(
    '<path d="M12 3L22 21H2L12 3Z" stroke="#ffd600" stroke-width="1.8" stroke-linejoin="round"/>'
    '<line x1="12" y1="10" x2="12" y2="14" stroke="#ffd600" stroke-width="2" stroke-linecap="round"/>'
    '<circle cx="12" cy="17.5" r="1" fill="#ffd600"/>'
)
ICO_CRIT = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#ff3d5a" stroke-width="1.8"/>'
    '<line x1="9" y1="9" x2="15" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="15" y1="9" x2="9" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
)
ICO_CHART = _svg(
    '<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
    '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
    '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>',
    16,
)
ICO_TARGET = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#00d2ff" stroke-width="1.8"/>'
    '<circle cx="12" cy="12" r="5" stroke="#ffd600" stroke-width="1.5"/>'
    '<circle cx="12" cy="12" r="2" fill="#00ff87"/>',
    16,
)
ICO_SIGNAL = _svg(
    '<path d="M2 12 Q6 4 12 12 Q18 20 22 12" stroke="#00d2ff" stroke-width="2" '
    'stroke-linecap="round" fill="none"/>'
    '<circle cx="12" cy="12" r="2" fill="#ffd600"/>',
    16,
)
ICO_RISK = _svg(
    '<path d="M12 3l9 18H3L12 3z" stroke="#ff3d5a" stroke-width="1.8" stroke-linejoin="round"/>'
    '<line x1="12" y1="9" x2="12" y2="14" stroke="#ff3d5a" stroke-width="1.8" stroke-linecap="round"/>'
    '<circle cx="12" cy="17" r="1" fill="#ff3d5a"/>',
    16,
)
ICO_METRICS = _svg(
    '<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
    '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>',
    16,
)
ICO_HEATMAP = _svg(
    '<rect x="3"  y="3"  width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
    '<rect x="10" y="3"  width="4" height="4" rx="1" fill="#00d2ff" opacity="0.6"/>'
    '<rect x="17" y="3"  width="4" height="4" rx="1" fill="#ffd600" opacity="0.4"/>'
    '<rect x="3"  y="10" width="4" height="4" rx="1" fill="#00d2ff" opacity="0.5"/>'
    '<rect x="10" y="10" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
    '<rect x="17" y="10" width="4" height="4" rx="1" fill="#a855f7" opacity="0.5"/>'
    '<rect x="3"  y="17" width="4" height="4" rx="1" fill="#ffd600" opacity="0.3"/>'
    '<rect x="17" y="17" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>',
    16,
)
ICO_FRONTIER = _svg(
    '<path d="M3 20 Q8 8 14 10 Q18 12 21 4" stroke="#00ff87" stroke-width="2" stroke-linecap="round" fill="none"/>'
    '<circle cx="18" cy="6" r="2.5" fill="#ff3d5a"/>'
    '<circle cx="10" cy="17" r="2" fill="#ffd600"/>',
    16,
)
ICO_LINK = _svg(
    '<circle cx="7"  cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
    '<circle cx="17" cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
    '<line x1="10" y1="12" x2="14" y2="12" stroke="#00d2ff" stroke-width="1.8"/>',
    16,
)
ICO_BOX = _svg(
    '<rect x="3" y="7" width="18" height="14" rx="2" stroke="#94a3b8" stroke-width="1.8"/>'
    '<path d="M8 7V5a4 4 0 018 0v2" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="12" y1="12" x2="12" y2="16" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="10" y1="14" x2="14" y2="14" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>',
    16,
)
ICO_RULER = _svg(
    '<rect x="2" y="7" width="20" height="10" rx="2" stroke="#ffd600" stroke-width="1.8"/>'
    '<line x1="6"  y1="7" x2="6"  y2="12" stroke="#ffd600" stroke-width="1.5"/>'
    '<line x1="10" y1="7" x2="10" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
    '<line x1="14" y1="7" x2="14" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
    '<line x1="18" y1="7" x2="18" y2="12" stroke="#ffd600" stroke-width="1.5"/>',
    16,
)
ICO_IDEA = _svg(
    '<circle cx="12" cy="10" r="6" stroke="#ffd600" stroke-width="1.8"/>'
    '<path d="M9 16.5h6M10 19h4" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="12" y1="4" x2="12" y2="2" stroke="#ffd600" stroke-width="1.5" stroke-linecap="round"/>',
    16,
)
ICO_UP = _svg(
    '<path d="M12 20V4M5 11l7-7 7 7" stroke="#00ff87" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)
ICO_DOWN = _svg(
    '<path d="M12 4v16M5 13l7 7 7-7" stroke="#ff3d5a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)
ICO_FLAT = _svg(
    '<line x1="4" y1="12" x2="20" y2="12" stroke="#ffd600" stroke-width="2.2" stroke-linecap="round"/>'
    '<path d="M16 8l4 4-4 4" stroke="#ffd600" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)


def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f"{icon_svg}<span>{text}</span></{tag}>",
        unsafe_allow_html=True,
    )


def diag_row(icon_svg, text, color):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
        f'color:{color};font-size:0.88rem">{icon_svg}{text}</div>',
        unsafe_allow_html=True,
    )


def render_cards_grid(data_dict, colors_sequence=None):
    if not colors_sequence:
        colors_sequence = [
            "#38bdf8",
            "#4ade80",
            "#fbbf24",
            "#fb7185",
            "#c084fc",
            "#f472b6",
            "#34d399",
            "#60a5fa",
        ]
    items = list(data_dict.items())
    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{lbl}</div>'
        f'<div class="mcard-value" style="color:{colors_sequence[i % len(colors_sequence)]}">{val}</div></div>'
        for i, (lbl, val) in enumerate(items)
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_selic_rate(start_date):
    taxa_selic = sgs.get(432, start=start_date)
    val = (taxa_selic.iloc[-1, 0]) / 100
    daily_val = (1 + val) ** (1 / 252) - 1
    return daily_val


@st.cache_data(ttl=3600)
def get_portfolio_prices(tickers_yf, start_date):
    today = datetime.date.today()
    return yf.download(tickers_yf, start=start_date, end=today, progress=False)["Close"]


@st.cache_data(ttl=3600)
def get_benchmark_prices(start_date):
    return yf.download("^BVSP", start=start_date, progress=False)["Close"].squeeze()


# CSS customizado
load_css()

render_flow_sidebar(active_step=2, pending_opacities=[0.4, 0.25])

warnings.filterwarnings("ignore")

# Customização do Matplotlib para o tema Obsidian Neo-Financial
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#080c14"
plt.rcParams["axes.facecolor"] = "#0e1524"
plt.rcParams["text.color"] = "#f8fafc"
plt.rcParams["axes.labelcolor"] = "#f8fafc"
plt.rcParams["xtick.color"] = "#94a3b8"
plt.rcParams["ytick.color"] = "#94a3b8"
plt.rcParams["grid.color"] = "#1e293b"
plt.rcParams["font.family"] = "sans-serif"


def apply_matplotlib_theme(fig):
    fig.set_facecolor("#080c14")
    for ax in fig.axes:
        ax.set_facecolor("#0e1524")
        ax.tick_params(colors="#94a3b8", which="both")
        ax.yaxis.label.set_color("#f8fafc")
        ax.xaxis.label.set_color("#f8fafc")
        if ax.title:
            ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#0e1524")
            legend.get_frame().set_edgecolor("#1e293b")
            for text in legend.get_texts():
                text.set_color("#f8fafc")
    return fig


# Customização do Plotly para o tema Obsidian Neo-Financial
def plot_efficient_frontier_and_random_portfolios(mu, S, returns, cleaned_weights, rf):
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mu)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

        results[0, i] = portfolio_stddev
        results[1, i] = portfolio_return
        results[2, i] = (
            (portfolio_return - rf) / portfolio_stddev if portfolio_stddev > 0 else 0
        )

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

    fig.add_trace(
        go.Scatter(
            x=results[0, :],
            y=results[1, :],
            mode="markers",
            marker=dict(
                size=4,
                color=results[2, :],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Índice Sharpe", font=dict(color="#f8fafc")),
                    tickfont=dict(color="#f8fafc"),
                ),
                opacity=0.6,
            ),
            name="Portfólios Aleatórios",
            text=[
                f"Retorno Anual: {r:.2%}<br>Vol Anual: {v:.2%}<br>Sharpe: {s:.2f}"
                for v, r, s in zip(results[0, :], results[1, :], results[2, :])
            ],
            hoverinfo="text",
        )
    )

    if len(efficient_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=efficient_vols,
                y=target_returns[: len(efficient_vols)],
                mode="lines",
                line=dict(color="#00ff87", width=3),
                name="Fronteira Eficiente",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[opt_stddev],
            y=[opt_return],
            mode="markers",
            marker=dict(
                color="#ff1744",
                size=12,
                symbol="star",
                line=dict(color="#f8fafc", width=2),
            ),
            name="Max Sharpe (Markowitz)",
            text=[
                f"Max Sharpe<br>Retorno: {opt_return:.2%}<br>Vol: {opt_stddev:.2%}<br>Sharpe: {opt_sharpe:.2f}"
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_stddev],
            y=[min_return],
            mode="markers",
            marker=dict(
                color="#ffd600",
                size=10,
                symbol="diamond",
                line=dict(color="#f8fafc", width=1.5),
            ),
            name="Mínima Volatilidade",
            text=[
                f"Mínima Volatilidade<br>Retorno: {min_return:.2%}<br>Vol: {min_stddev:.2%}"
            ],
            hoverinfo="text",
        )
    )

    # LAC — Linha de Alocação de Capital (EAE1242 — Tobin, 1958)
    # Parte do Rf (volatilidade=0) e passa pela carteira tangente (Max Sharpe)
    if opt_stddev > 0:
        lac_slope = (opt_return - rf) / opt_stddev
        lac_x_end = opt_stddev * 1.8
        fig.add_trace(
            go.Scatter(
                x=[0, lac_x_end],
                y=[rf, rf + lac_slope * lac_x_end],
                mode="lines",
                line=dict(color="#ffd600", width=2, dash="dash"),
                name="LAC — Linha de Alocação de Capital",
                hovertemplate="LAC<br>Vol: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[rf],
                mode="markers+text",
                marker=dict(color="#ffd600", size=8, symbol="circle"),
                text=["Rf (Selic)"],
                textposition="top right",
                textfont=dict(color="#ffd600", size=10),
                name="Taxa Livre de Risco (Rf)",
                hovertemplate=f"Rf (Selic) = {rf:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Fronteira Eficiente de Markowitz · LAC · Carteira Tangente",
        xaxis_title="Volatilidade Anualizada (Desvio Padrão)",
        yaxis_title="Retorno Esperado Anualizado",
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=550, margin=dict(b=140))
    return fig


# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Configurações
col_config1, col_config2 = st.columns(2)
with col_config1:
    lookback_opcao = st.selectbox(
        "Lookback da Otimização",
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
        lookback_dias = st.number_input(
            "Dias de Lookback", min_value=30, max_value=5000, value=500, step=10
        )
        data_inicio = today - datetime.timedelta(days=lookback_dias)

try:
    taxa_selic = get_selic_rate(data_inicio)
except Exception as _selic_err:
    st.warning(
        f"Não foi possível buscar a taxa Selic no BCB ({_selic_err}). Usando valor de referência: 13,75% a.a."
    )
    taxa_selic = (1 + 0.1375) ** (1 / 252) - 1


# Seleção de ações
MAX_TICKERS = 20

data = pd.read_csv("acoes-listadas-b3.csv")
stocks = list(data["Ticker"].values)
stocks = get_sorted_tickers_by_liquidity(stocks)

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []

# Remove any stale tickers not in the current list, and trim to the cap in
# case it was lowered or the session predates the limit
st.session_state["selected_tickers"] = [
    t for t in st.session_state["selected_tickers"] if t in stocks
][:MAX_TICKERS]

tickers = st.multiselect(
    "Selecione as ações do portfólio",
    options=stocks,
    key="selected_tickers",
    max_selections=MAX_TICKERS,
    help=f"Limite de {MAX_TICKERS} ativos para manter o download de cotações e a otimização estáveis.",
)
# Valor inicial
valor_inicial = st.number_input("Valor Investido (R$)", 100, 1_000_000, 10_000)

# Escolha modo: manual ou otimizado
modo = st.radio(
    "Modo de alocação",
    (
        "Otimização de Markowitz (Média-Variância / Sharpe Máximo)",
        "Otimização Hierarchical Risk Parity (Machine Learning)",
        "Alocação Manual",
    ),
)

if len(tickers) == 0:
    st.warning("Selecione pelo menos uma ação.")
    st.stop()

if len(tickers) == 1:
    st.warning("Selecione pelo menos dois ativos para montar o portfólio.")
    st.stop()

tickers_yf = [t + ".SA" for t in tickers]

# Inputs de peso manual devem aparecer ANTES do botão
pesos_manuais_inputs = {}
if "Manual" in modo:
    st.markdown("---")
    section_header(ICO_BOX, "Alocação Manual dos Pesos", "h4")
    total_pesos = 0.0
    for ticker in tickers:
        p = st.number_input(
            f"Peso % de {ticker}",
            min_value=0.0,
            max_value=100.0,
            value=round(100 / len(tickers), 2),
            step=0.01,
            key=f"peso_manual_{ticker}",
        )
        pesos_manuais_inputs[ticker + ".SA"] = p / 100
        total_pesos += p

    if abs(total_pesos - 100) > 0.01:
        st.error(
            f"Soma dos pesos: **{total_pesos:.2f}%** — ajuste para exatamente 100%."
        )
    else:
        st.success(f"Soma dos pesos: {total_pesos:.2f}% ✓")
    st.markdown("---")

# Pré-carrega cotações (resultado cacheado após primeira execução)
with loading_overlay("Baixando cotações históricas...", tickers=tickers):
    try:
        data_yf = get_portfolio_prices(tickers_yf, data_inicio)
    except Exception as _price_err:
        st.error(
            f"Erro ao buscar cotações no Yahoo Finance: {_price_err}. Verifique sua conexão e tente novamente."
        )
        st.stop()

if data_yf.empty:
    st.error(
        "Nenhuma cotação retornada para os ativos selecionados. Verifique os tickers e o período escolhido."
    )
    st.stop()

if isinstance(data_yf.columns, pd.MultiIndex):
    data_yf.columns = ["_".join(col).strip() for col in data_yf.columns.values]

returns = data_yf.pct_change().dropna()

MIN_RETURN_ROWS = 30
if len(returns) < MIN_RETURN_ROWS:
    # Rows with a gap in ANY ticker's price history get dropped by dropna()
    # above, so one short-history ticker (recent IPO, delisting, missing
    # data) can collapse `returns` to near-empty even when most tickers
    # have plenty of data. Surface which ones before optimizers choke on it.
    first_valid = data_yf.apply(lambda col: col.first_valid_index())
    short_history = first_valid.sort_values(ascending=False).head(5)
    culprits = ", ".join(
        f"{col.replace('.SA', '')} (dados desde {date.strftime('%d/%m/%Y')})"
        for col, date in short_history.items()
        if pd.notna(date)
    )
    st.error(
        f"Histórico de cotações em comum insuficiente entre os ativos selecionados "
        f"(apenas {len(returns)} dia(s) com dados completos para todos os ativos). "
        "Isso costuma acontecer quando um ou mais ativos têm histórico bem mais curto "
        "que os demais (IPO recente, deslistagem, falha na fonte de dados)."
        + (f" Possíveis responsáveis: {culprits}." if culprits else "")
        + " Remova esses ativos ou reduza o período de lookback."
    )
    st.stop()

page_container = st.empty()

if st.button("Carregar Portfolio", type="primary", use_container_width=True):
    # Overlay de carregamento global
    with loading_overlay("Carregando dados, aguarde...", tickers=tickers):
        if "Manual" in modo:
            pesos_manuais = {}
            total = 0.0
            for ticker in tickers:
                p = st.session_state.get(
                    f"peso_manual_{ticker}", round(100 / len(tickers), 2)
                )
                pesos_manuais[ticker + ".SA"] = p / 100
                total += p
            if abs(total - 100) > 0.01:
                st.error(f"A soma dos pesos é {total:.2f}%, deve ser 100%")
                st.stop()
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
                st.warning(
                    f"Otimização de Max Sharpe falhou (motivo: {str(e)}). Usando carteira de Mínima Volatilidade."
                )
                ef = EfficientFrontier(mu, S)
                raw_weights = ef.min_volatility()
                cleaned_weights = ef.clean_weights()
            peso_manual_df = pd.DataFrame.from_dict(
                cleaned_weights, orient="index", columns=["Peso"]
            )
            pesos_manuais_arr = peso_manual_df["Peso"].values

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
            pesos_csv = df_pesos_export.to_csv().encode("utf-8")
            st.download_button(
                label="⬇ Exportar alocação (CSV)",
                data=pesos_csv,
                file_name=f"portfolio_alocacao_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        except Exception:
            pass

        # ── Sugestão de Compra de Cotas (Alocação Discreta) ───────────────────
        st.subheader("Sugestão de Compra de Cotas")
        st.markdown(
            f"Estimativa de cotas a comprar considerando o valor total de **R$ {valor_inicial:,.2f}**."
        )

        cotas_list = []
        total_efetivo = 0.0

        for ticker, row in peso_manual_df.iterrows():
            col_name = ticker + ".SA"
            latest_price = 0.0

            # Busca o preço mais recente válido
            if col_name in data_yf.columns:
                valid_prices = data_yf[col_name].dropna()
                latest_price = valid_prices.iloc[-1] if not valid_prices.empty else 0.0
            elif ticker in data_yf.columns:
                valid_prices = data_yf[ticker].dropna()
                latest_price = valid_prices.iloc[-1] if not valid_prices.empty else 0.0
            else:
                for col in data_yf.columns:
                    if ticker in col:
                        valid_prices = data_yf[col].dropna()
                        latest_price = (
                            valid_prices.iloc[-1] if not valid_prices.empty else 0.0
                        )
                        break

            weight = row["Peso"]
            valor_teorico = weight * valor_inicial

            if latest_price > 0:
                cotas = int(np.floor(valor_teorico / latest_price))
                valor_efetivo = cotas * latest_price
            else:
                cotas = 0
                valor_efetivo = 0.0

            total_efetivo += valor_efetivo

            cotas_list.append(
                {
                    "Ativo": ticker,
                    "Preço Unitário": latest_price,
                    "Peso Sugerido (%)": weight * 100,
                    "Valor Sugerido": valor_teorico,
                    "Cotas a Comprar": cotas,
                    "Valor Efetivo": valor_efetivo,
                }
            )

        df_cotas = pd.DataFrame(cotas_list)
        if total_efetivo > 0:
            df_cotas["Peso Efetivo (%)"] = (
                df_cotas["Valor Efetivo"] / total_efetivo
            ) * 100
        else:
            df_cotas["Peso Efetivo (%)"] = 0.0

        for idx, row_c in df_cotas.iterrows():
            ticker_name = row_c["Ativo"]
            pu = row_c["Preço Unitário"]
            p_sug = row_c["Peso Sugerido (%)"]
            val_sug = row_c["Valor Sugerido"]
            cotas = row_c["Cotas a Comprar"]
            val_ef = row_c["Valor Efetivo"]
            p_ef = row_c["Peso Efetivo (%)"]

            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, #0e1726, #070c14); 
                        border: 1px solid #1e293b; 
                        border-radius: 12px; 
                        padding: 1rem; 
                        margin-bottom: 1rem; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 0.5rem; margin-bottom: 0.8rem;">
                    <span style="font-size: 1.2rem; color: #38bdf8; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{ticker_name}</span>
                    <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Preço Unitário: <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">R$ {pu:,.2f}</strong></span>
                </div>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 1rem; margin-bottom: 0.8rem;">
                    <div style="flex: 1; min-width: 120px; background: rgba(30, 41, 59, 0.5); padding: 0.6rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem;">Cotas a Comprar</div>
                        <div style="font-size: 1.2rem; color: #4ade80; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{int(cotas):,}</div>
                    </div>
                    <div style="flex: 1; min-width: 120px; background: rgba(30, 41, 59, 0.5); padding: 0.6rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem;">Valor Efetivo</div>
                        <div style="font-size: 1.1rem; color: #fbbf24; font-weight: 800; font-family: 'JetBrains Mono', monospace;">R$ {val_ef:,.2f}</div>
                    </div>
                    <div style="flex: 1; min-width: 120px; background: rgba(30, 41, 59, 0.5); padding: 0.6rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem;">Valor Sugerido</div>
                        <div style="font-size: 1.1rem; color: #a78bfa; font-weight: 800; font-family: 'JetBrains Mono', monospace;">R$ {val_sug:,.2f}</div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; padding-top: 0.2rem;">
                    <span>Peso Sugerido: <strong style="color: #e2e8f0; font-family: 'JetBrains Mono', monospace;">{p_sug:.2f}%</strong></span>
                    <span>Peso Efetivo: <strong style="color: #e2e8f0; font-family: 'JetBrains Mono', monospace;">{p_ef:.2f}%</strong></span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Resumo financeiro do rebalanceamento/compra
        sobra_caixa = valor_inicial - total_efetivo

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Total Alocado Efetivo", f"R$ {total_efetivo:,.2f}")
        col_c2.metric("Saldo Restante (Caixa)", f"R$ {sobra_caixa:,.2f}")
        col_c3.metric(
            "Eficiência da Alocação", f"{(total_efetivo / valor_inicial) * 100:.2f}%"
        )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        alloc_df = peso_manual_df.reset_index()
        alloc_df.columns = ["Ativo", "Peso"]

        fig_donut = px.pie(
            alloc_df,
            names="Ativo",
            values="Peso",
            hole=0.4,
            title="Distribuição de Alocação da Carteira",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        apply_plotly_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

        if "Markowitz" in modo:
            section_header(ICO_FRONTIER, "Gráfico da Fronteira Eficiente", "h3")
            with loading_overlay("Gerando fronteira eficiente e simulando portfólios..."):
                selic_anual = (1 + taxa_selic) ** 252 - 1
                fig_frontier = plot_efficient_frontier_and_random_portfolios(
                    mu, S, returns, cleaned_weights, selic_anual
                )
                st.plotly_chart(fig_frontier, use_container_width=True)

            # ── Teorema da Separação de Tobin (1958) ─────────────────────────
            st.markdown(
                """
<div style="background:linear-gradient(135deg,rgba(245,158,11,0.07),rgba(168,85,247,0.04));
border:1px solid rgba(245,158,11,0.25);border-radius:14px;padding:1rem 1.2rem;margin:0.5rem 0 1rem 0">
<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#f59e0b;text-transform:uppercase;margin-bottom:0.5rem">
📐 Teorema da Separação de Tobin (1958)
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
  <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:0.7rem">
    <div style="font-size:0.68rem;font-weight:700;color:#00ff87;margin-bottom:0.3rem">① Decisão Objetiva — igual para todos</div>
    <div style="font-size:0.72rem;color:#94a3b8;line-height:1.5">
    Qual é a carteira ótima de ativos de risco?<br>
    Resposta: sempre a <b style="color:#ff1744">Carteira Tangente</b> (Max Sharpe).<br>
    Independente da sua aversão ao risco.
    </div>
  </div>
  <div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:0.7rem">
    <div style="font-size:0.68rem;font-weight:700;color:#a855f7;margin-bottom:0.3rem">② Decisão Subjetiva — depende do perfil</div>
    <div style="font-size:0.72rem;color:#94a3b8;line-height:1.5">
    Quanto alocar entre Rf e a Carteira Tangente?<br>
    Conservador: mais Rf. Agressivo: mais tangente.<br>
    Todos os pontos ótimos estão na <b style="color:#ffd600">LAC</b>.
    </div>
  </div>
</div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Calcula parâmetros da carteira tangente
            _opt_w = np.array(list(cleaned_weights.values()))
            _et = float(np.sum(_opt_w * mu))  # E[R] tangente
            _st = float(np.sqrt(np.dot(_opt_w.T, np.dot(S, _opt_w))))  # σ tangente
            _sharpe_t = (_et - selic_anual) / _st if _st > 0 else 0

            st.markdown("**Simulador da LAC — quanto você aloca em ativos de risco?**")
            st.caption(
                "Mova o slider para ver como o retorno esperado e o risco do seu portfólio mudam "
                "ao longo da Linha de Alocação de Capital. "
                "O Sharpe permanece constante — essa é a essência do Teorema da Separação."
            )

            _w = (
                st.slider(
                    "% em ativos de risco (Carteira Tangente)",
                    min_value=0,
                    max_value=150,
                    value=100,
                    step=5,
                    format="%d%%",
                    help="0% = 100% na Selic (sem risco). 100% = Carteira Tangente pura. >100% = alavancagem.",
                )
                / 100.0
            )

            _ep = selic_anual + _w * (_et - selic_anual)  # E[Rp] na LAC
            _sp = abs(_w) * _st  # σp na LAC (Rf tem σ=0)
            _shp = (_ep - selic_anual) / _sp if _sp > 0 else 0

            _perfil = (
                "🏦 Conservador — grande parte em Rf (Selic)"
                if _w < 0.4
                else "⚖️ Moderado — equilíbrio entre Rf e Carteira Tangente"
                if _w < 0.8
                else "🚀 Arrojado — próximo ou na Carteira Tangente"
                if _w <= 1.0
                else "⚡ Alavancado — tomou emprestado ao Rf para investir mais"
            )

            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("% em Rf (Selic)", f"{(1 - _w) * 100:.0f}%")
            tc2.metric(
                "E[Retorno] a.a.",
                f"{_ep * 100:.2f}%",
                delta=f"{(_ep - selic_anual) * 100:+.2f}% acima do Rf",
            )
            tc3.metric("Volatilidade a.a.", f"{_sp * 100:.2f}%")
            tc4.metric(
                "Sharpe do Portfólio",
                f"{_shp:.3f}",
                delta=f"= Sharpe tangente ({_sharpe_t:.3f})",
                delta_color="off",
                help="O Sharpe é CONSTANTE em toda a LAC — essa é a prova do Teorema da Separação.",
            )

            st.markdown(
                f"""
<div style="background:rgba(245,158,11,0.06);border-left:3px solid #f59e0b;
border-radius:0 8px 8px 0;padding:0.6rem 1rem;font-size:0.78rem;color:#cbd5e1;margin-top:0.3rem">
<b style="color:#f59e0b">Perfil:</b> {_perfil}<br>
<span style="color:#64748b;font-size:0.68rem">
Sharpe constante = {_sharpe_t:.3f} ao longo de toda a LAC.
Rf = {selic_anual * 100:.2f}% · E[R tangente] = {_et * 100:.2f}% · σ tangente = {_st * 100:.2f}%
</span>
</div>
""",
                unsafe_allow_html=True,
            )

        # Heatmap de Correlação Interativo (Plotly)
        section_header(ICO_HEATMAP, "Heatmap de Correlação entre Ativos", "h3")
        corr_df = returns.corr()
        fig_corr = px.imshow(
            corr_df,
            x=corr_df.index.str.replace(".SA", "", regex=False),
            y=corr_df.columns.str.replace(".SA", "", regex=False),
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlação de Retornos Históricos",
            text_auto=".2f",
        )
        apply_plotly_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Cálculo do portfólio com os pesos escolhidos
        portfolio_returns = returns.dot(pesos_manuais_arr)

        # Obter os dados de benchmark BOVESPA e calcular o retorno acumulado
        bench = get_benchmark_prices(data_inicio)
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
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value,
                mode="lines",
                name="Portfólio",
                line=dict(color="#00ff87", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bench_value.index,
                y=bench_value,
                mode="lines",
                name="IBOVESPA",
                line=dict(color="#ffd600", width=1.5, dash="dash"),
            )
        )
        fig.update_layout(
            title="Evolução do Valor do Portfólio vs Benchmark",
            xaxis_title="Data",
            yaxis_title="Valor (R$)",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        # Retornos mensais
        section_header(ICO_HEATMAP, "Tabela de Retornos Mensais do Portfólio", "h3")
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
                row_html = '<tr style="border-bottom: 1px solid #1e293b; font-size: 0.88rem;">'
                row_html += f"<td style=\"padding: 0.7rem 0.5rem; font-weight: 700; text-align: left; color: #f8fafc; font-family: 'JetBrains Mono', monospace;\">{year}</td>"
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
                        row_html += "<td style=\"padding: 0.7rem 0.5rem; color: #475569; font-family: 'JetBrains Mono', monospace;\">-</td>"
                    else:
                        color = (
                            "#4ade80"
                            if val > 0
                            else "#f87171"
                            if val < 0
                            else "#94a3b8"
                        )
                        sign = "+" if val > 0 else ""
                        row_html += f"<td style=\"padding: 0.7rem 0.5rem; color: {color}; font-family: 'JetBrains Mono', monospace; font-weight: 700;\">{sign}{val:.2f}%</td>"

                ytd_val = row["YTD"]
                if pd.isna(ytd_val):
                    row_html += "<td style=\"padding: 0.7rem 0.5rem; border-left: 1px solid #1e293b; color: #475569; font-family: 'JetBrains Mono', monospace;\">-</td>"
                else:
                    ytd_color = (
                        "#4ade80"
                        if ytd_val > 0
                        else "#f87171"
                        if ytd_val < 0
                        else "#94a3b8"
                    )
                    ytd_sign = "+" if ytd_val > 0 else ""
                    row_html += f"<td style=\"padding: 0.7rem 0.5rem; border-left: 1px solid #1e293b; color: {ytd_color}; font-family: 'JetBrains Mono', monospace; font-weight: 800;\">{ytd_sign}{ytd_val:.2f}%</td>"
                row_html += "</tr>"
                html_rows.append(row_html)

            table_html = f"""
            <div style="background: linear-gradient(135deg, #0e1726, #070c14); 
                        border: 1px solid #1e293b; 
                        border-radius: 12px; 
                        padding: 1.2rem; 
                        overflow-x: auto; 
                        margin-bottom: 1.5rem;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <table style="width: 100%; border-collapse: collapse; text-align: center; color: #f8fafc; font-family: 'Space Grotesk', sans-serif;">
                    <thead>
                        <tr style="border-bottom: 2px solid #1e293b; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">
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
                            <th style="padding: 0.8rem 0.5rem; border-left: 1px solid #1e293b; font-weight: 700; color: #38bdf8;">YTD</th>
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
            st.warning(f"Não foi possível gerar a tabela de retornos mensais: {str(e)}")

        # Cálculos de Métricas
        total_return = (portfolio_value.iloc[-1] / valor_inicial - 1) * 100
        vol_anual = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_val = sharpe(portfolio_returns, rf=taxa_selic)
        sortino_val = sortino(portfolio_returns, rf=taxa_selic)
        max_dd = max_drawdown(portfolio_returns) * 100

        cov_matrix = np.cov(
            portfolio_returns.squeeze(), retorno_bench.squeeze()
        )  # matriz de covariância 2x2
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        alfa = portfolio_returns.mean() - beta * retorno_bench.mean()
        alfa_val = (
            alfa.values[0] if hasattr(alfa, "values") and len(alfa.values) > 0 else alfa
        )
        r_quadrado = qs.stats.r_squared(portfolio_returns, retorno_bench)
        information_ratio = qs.stats.information_ratio(portfolio_returns, retorno_bench)

        # Desempenho Resumido em Cards (st.metric)
        section_header(ICO_CHART, "Desempenho Resumido da Carteira", "h3")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric(
            "Retorno Total",
            f"{total_return:.2f}%",
            delta="Positivo" if total_return > 0 else "Negativo",
            delta_color="normal" if total_return > 0 else "inverse",
            help="Retorno total do portfólio no período selecionado.",
        )
        col_m2.metric(
            "Volatilidade Anual",
            f"{vol_anual:.2f}%",
            delta="Baixa"
            if vol_anual < 15
            else ("Alta" if vol_anual > 25 else "Moderada"),
            delta_color="normal"
            if vol_anual < 15
            else ("inverse" if vol_anual > 25 else "off"),
            help="Desvio padrão anualizado dos retornos. Mede o risco total.",
        )
        col_m3.metric(
            "Índice Sharpe",
            f"{sharpe_val:.2f}",
            delta="Excelente"
            if sharpe_val > 1
            else ("Bom" if sharpe_val > 0.5 else "Baixo"),
            delta_color="normal" if sharpe_val > 0.5 else "inverse",
            help="Retorno por unidade de risco. Acima de 1.0 é excelente.",
        )
        col_m4.metric(
            "Índice Sortino",
            f"{sortino_val:.2f}",
            delta="Excelente"
            if sortino_val > 1
            else ("Bom" if sortino_val > 0.5 else "Baixo"),
            delta_color="normal" if sortino_val > 0.5 else "inverse",
            help="Igual ao Sharpe mas penaliza apenas volatilidade negativa.",
        )
        col_m5.metric(
            "Max Drawdown",
            f"{max_dd:.2f}%",
            delta="Controlado"
            if max_dd > -15
            else ("Severo" if max_dd < -30 else "Moderado"),
            delta_color="normal"
            if max_dd > -15
            else ("inverse" if max_dd < -30 else "off"),
            help="Maior perda de pico a vale. Quanto mais próximo de 0, melhor.",
        )

        # ── Painel de Decisão do Investidor ───────────────────────────────────
        st.markdown("---")
        section_header(ICO_TARGET, "Painel de Decisão do Investidor", "h3")

        # ── Score de Saúde do Portfólio ──────────────────────────────────────
        score = 0
        health_detalhes = []

        if sharpe_val > 1.0:
            score += 35
            health_detalhes.append((ICO_OK, "Sharpe excelente (>1.0)", "#00ff87"))
        elif sharpe_val > 0.5:
            score += 17
            health_detalhes.append((ICO_WARN, "Sharpe razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append(
                (ICO_CRIT, "Sharpe baixo (<0.5) — revise a alocação", "#ff3d5a")
            )

        if sortino_val > 1.0:
            score += 20
            health_detalhes.append((ICO_OK, "Sortino excelente (>1.0)", "#00ff87"))
        elif sortino_val > 0.5:
            score += 10
            health_detalhes.append((ICO_WARN, "Sortino razoável (0.5–1.0)", "#ffd600"))
        else:
            health_detalhes.append(
                (ICO_CRIT, "Sortino baixo — retornos negativos relevantes", "#ff3d5a")
            )

        if max_dd > -10:
            score += 20
            health_detalhes.append((ICO_OK, "Drawdown controlado (<10%)", "#00ff87"))
        elif max_dd > -20:
            score += 10
            health_detalhes.append((ICO_WARN, "Drawdown moderado (10–20%)", "#ffd600"))
        else:
            health_detalhes.append(
                (ICO_CRIT, "Drawdown severo (>20%) — risco de ruína elevado", "#ff3d5a")
            )

        alfa_anual = alfa_val * 252 * 100
        if alfa_anual > 5:
            score += 15
            health_detalhes.append(
                (ICO_OK, f"Alfa anual positivo: {alfa_anual:.1f}%", "#00ff87")
            )
        elif alfa_anual > 0:
            score += 7
            health_detalhes.append(
                (ICO_WARN, f"Alfa marginal: {alfa_anual:.1f}%", "#ffd600")
            )
        else:
            health_detalhes.append(
                (
                    ICO_CRIT,
                    f"Alfa negativo ({alfa_anual:.1f}%) — portfólio perde pro índice",
                    "#ff3d5a",
                )
            )

        pesos_arr_dec = np.array(
            pesos_manuais_arr
        )  # sempre definido independente do modo
        max_peso = pesos_arr_dec.max() * 100
        if max_peso <= 30:
            score += 10
            health_detalhes.append(
                (ICO_OK, f"Concentração saudável (máx: {max_peso:.1f}%)", "#00ff87")
            )
        elif max_peso <= 50:
            health_detalhes.append(
                (ICO_WARN, f"Concentração elevada (máx: {max_peso:.1f}%)", "#ffd600")
            )
        else:
            health_detalhes.append(
                (
                    ICO_CRIT,
                    f"Hiper-concentração (máx: {max_peso:.1f}%) — diversifique",
                    "#ff3d5a",
                )
            )

        score_color = (
            "#00ff87" if score >= 70 else "#ffd600" if score >= 40 else "#ff1744"
        )
        score_label = (
            "SAUDÁVEL" if score >= 70 else "ATENÇÃO" if score >= 40 else "CRÍTICO"
        )

        col_score, col_details = st.columns([1, 2])
        with col_score:
            st.markdown(
                f"""
            <div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:2px solid {score_color};
                        border-radius:16px;padding:1.5rem;text-align:center;
                        box-shadow:0 0 20px {score_color}33;">
                <div style="font-size:3.5rem;font-weight:900;color:{score_color};
                            font-family:'JetBrains Mono',monospace;
                            text-shadow:0 0 15px {score_color}66;">{score}</div>
                <div style="font-size:0.65rem;color:#94a3b8;letter-spacing:0.12em;margin-top:0.2rem;">DE 100 PONTOS</div>
                <div style="font-size:0.9rem;font-weight:700;color:{score_color};margin-top:0.5rem;
                            letter-spacing:0.08em;">{score_label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col_details:
            st.markdown("**Diagnóstico por indicador:**")
            for ico, msg, color in health_detalhes:
                diag_row(ico, msg, color)

        # ── Análise CAPM por Ativo — EAE1242 (Sharpe, 1964) ─────────────────
        st.markdown("---")
        ICO_CAPM = _svg(
            '<line x1="3" y1="21" x2="21" y2="3" stroke="#a855f7" stroke-width="1.8" stroke-linecap="round"/>'
            '<circle cx="8" cy="16" r="2.5" fill="#00ff87"/>'
            '<circle cx="14" cy="10" r="2.5" fill="#ff3d5a"/>'
            '<circle cx="18" cy="6" r="2.5" fill="#ffd600"/>'
            '<line x1="3" y1="21" x2="21" y2="3" stroke="#a855f7" stroke-width="1.8" stroke-linecap="round" stroke-dasharray="3 2"/>',
            16,
        )
        section_header(ICO_CAPM, "Análise CAPM por Ativo", "h3")
        st.caption(
            "**CAPM (Sharpe, 1964):** E[Rᵢ] = Rƒ + βᵢ × (E[Rₘ] − Rƒ)  ·  "
            "β > 1 = mais volátil que o mercado  ·  α > 0 = retorno acima do esperado pelo risco"
        )

        selic_anual_capm = (1 + taxa_selic) ** 252 - 1
        bench_sq = retorno_bench.squeeze()
        rm_anual = bench_sq.mean() * 252

        def _clean_col(c):
            return (
                c.replace("Close_", "").replace("close_", "").replace(".SA", "").strip()
            )

        capm_rows = []
        for col in returns.columns:
            ri = returns[col].dropna()
            rm = bench_sq.reindex(ri.index).dropna()
            ri = ri.reindex(rm.index)
            if len(ri) < 20:
                continue
            cov_mat = np.cov(ri.values, rm.values)
            var_m = cov_mat[1, 1]
            if var_m <= 0:
                continue
            beta_i = cov_mat[0, 1] / var_m
            alpha_i = (ri.mean() - beta_i * rm.mean()) * 252  # Jensen's alpha a.a.
            ri_var = np.var(ri.values)
            r2 = (cov_mat[0, 1] ** 2) / (ri_var * var_m) if ri_var * var_m > 0 else 0
            ri_anual = ri.mean() * 252
            er_capm = selic_anual_capm + beta_i * (rm_anual - selic_anual_capm)
            treynor = (
                (ri_anual - selic_anual_capm) / beta_i if beta_i != 0 else float("nan")
            )
            capm_rows.append(
                {
                    "Ativo": _clean_col(col),
                    "Beta (β)": round(beta_i, 3),
                    "Alpha Jensen (a.a.)": round(alpha_i * 100, 2),
                    "R² (Risco Sist.)": round(r2 * 100, 1),
                    "E[Ri] CAPM": round(er_capm * 100, 2),
                    "Retorno Real": round(ri_anual * 100, 2),
                    "Treynor": round(treynor, 3),
                    "_alpha_sign": alpha_i,
                }
            )

        if capm_rows:
            df_capm = pd.DataFrame(capm_rows)

            # Tabela CAPM
            df_display = df_capm.drop(columns=["_alpha_sign"]).copy()
            df_display["Alpha Jensen (a.a.)"] = df_display["Alpha Jensen (a.a.)"].apply(
                lambda x: f"{x:+.2f}%"
            )
            df_display["R² (Risco Sist.)"] = df_display["R² (Risco Sist.)"].apply(
                lambda x: f"{x:.1f}%"
            )
            df_display["E[Ri] CAPM"] = df_display["E[Ri] CAPM"].apply(
                lambda x: f"{x:.2f}%"
            )
            df_display["Retorno Real"] = df_display["Retorno Real"].apply(
                lambda x: f"{x:.2f}%"
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.caption(
                "**Beta (β):** sensibilidade ao mercado.  "
                "**Alpha (α):** retorno acima do previsto pelo CAPM — Jensen's Alpha.  "
                "**R²:** % do risco total explicado pelo mercado (risco sistemático).  "
                "**Treynor:** retorno excedente por unidade de risco sistemático."
            )

            # SML — Security Market Line
            st.markdown("##### Security Market Line (SML)")
            betas = df_capm["Beta (β)"].values
            b_min = min(-0.2, betas.min() - 0.2)
            b_max = max(1.6, betas.max() + 0.3)
            b_line = np.linspace(b_min, b_max, 120)
            sml_y = (selic_anual_capm + b_line * (rm_anual - selic_anual_capm)) * 100

            fig_sml = go.Figure()
            fig_sml.add_trace(
                go.Scatter(
                    x=b_line,
                    y=sml_y,
                    mode="lines",
                    line=dict(color="#a855f7", width=2),
                    name="SML — Retorno Esperado CAPM",
                    hovertemplate="β=%{x:.2f}<br>E[R]=%{y:.2f}%<extra></extra>",
                )
            )
            # Reference lines
            fig_sml.add_vline(
                x=1.0, line_dash="dot", line_color="#334155", line_width=1
            )
            fig_sml.add_hline(
                y=selic_anual_capm * 100,
                line_dash="dot",
                line_color="#334155",
                line_width=1,
            )
            # Rf point
            fig_sml.add_trace(
                go.Scatter(
                    x=[0],
                    y=[selic_anual_capm * 100],
                    mode="markers+text",
                    marker=dict(color="#ffd600", size=9, symbol="diamond"),
                    text=["Rf"],
                    textposition="top right",
                    textfont=dict(color="#ffd600", size=10),
                    name="Rf (Selic)",
                    hovertemplate=f"Rf = {selic_anual_capm:.2%}<extra></extra>",
                )
            )
            # IBOVESPA (β=1)
            fig_sml.add_trace(
                go.Scatter(
                    x=[1.0],
                    y=[rm_anual * 100],
                    mode="markers+text",
                    marker=dict(color="#00d2ff", size=10, symbol="star"),
                    text=["IBOV"],
                    textposition="top right",
                    textfont=dict(color="#00d2ff", size=10),
                    name="IBOVESPA (β=1)",
                    hovertemplate=f"IBOV<br>β=1.00<br>Retorno={rm_anual:.2%}<extra></extra>",
                )
            )
            # Each asset
            for _, row in df_capm.iterrows():
                cor = "#00ff87" if row["_alpha_sign"] > 0 else "#ff3d5a"
                fig_sml.add_trace(
                    go.Scatter(
                        x=[row["Beta (β)"]],
                        y=[row["Retorno Real"]],
                        mode="markers+text",
                        marker=dict(
                            size=11, color=cor, line=dict(color="#f8fafc", width=1)
                        ),
                        text=[row["Ativo"]],
                        textposition="top center",
                        textfont=dict(size=9, color=cor),
                        name=row["Ativo"],
                        hovertemplate=(
                            f"<b>{row['Ativo']}</b><br>"
                            f"β = {row['Beta (β)']:.3f}<br>"
                            f"Retorno Real: {row['Retorno Real']:.2f}%<br>"
                            f"E[R] CAPM: {row['E[Ri] CAPM']:.2f}%<br>"
                            f"Alpha: {row['Alpha Jensen (a.a.)']:+.2f}%<extra></extra>"
                        ),
                    )
                )
            fig_sml.update_layout(
                xaxis_title="Beta (β) — Risco Sistemático vs IBOVESPA",
                yaxis_title="Retorno Anualizado (%)",
                showlegend=False,
                annotations=[
                    dict(
                        x=b_max - 0.1,
                        y=sml_y[-1] + 4,
                        text="Acima da SML: α > 0",
                        showarrow=False,
                        font=dict(color="#00ff87", size=10),
                        xanchor="right",
                    ),
                    dict(
                        x=b_max - 0.1,
                        y=sml_y[-1] - 4,
                        text="Abaixo da SML: α < 0",
                        showarrow=False,
                        font=dict(color="#ff3d5a", size=10),
                        xanchor="right",
                    ),
                ],
            )
            apply_plotly_theme(fig_sml)
            st.plotly_chart(fig_sml, use_container_width=True)
            st.caption(
                "🟢 Acima da SML = Alpha positivo (gerou valor além do risco assumido)  ·  🔴 Abaixo = Alpha negativo"
            )

            # Decomposição de risco
            st.markdown("##### Decomposição de Risco: Sistemático vs Não-Sistemático")
            st.caption(
                "R² = risco sistemático (mercado) / risco total.  1 − R² = risco idiossincrático (diversificável)."
            )
            risco_cols = st.columns(len(capm_rows))
            for i, row in enumerate(capm_rows):
                with risco_cols[i]:
                    r2_val = row["R² (Risco Sist.)"]
                    st.markdown(
                        f"""
<div style="text-align:center;padding:0.6rem;border:1px solid #1e293b;border-radius:10px">
  <div style="font-size:0.75rem;font-weight:700;color:#e2e8f0;margin-bottom:0.4rem">{row["Ativo"]}</div>
  <div style="font-size:1.1rem;font-weight:700;color:#a855f7">{r2_val:.0f}%</div>
  <div style="font-size:0.6rem;color:#64748b">sistemático</div>
  <div style="font-size:0.9rem;color:#334155">{100 - r2_val:.0f}%</div>
  <div style="font-size:0.6rem;color:#64748b">idiossincrático</div>
</div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Dados insuficientes para calcular CAPM individual por ativo.")

        # ── Stress Test — Crises Históricas ──────────────────────────────────
        st.markdown("---")
        ICO_STRESS = _svg(
            '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" '
            'stroke="#ff3d5a" stroke-width="1.8" fill="none"/>'
            '<line x1="12" y1="9" x2="12" y2="13" stroke="#ff3d5a" stroke-width="1.8" stroke-linecap="round"/>'
            '<line x1="12" y1="17" x2="12.01" y2="17" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>',
            16,
        )
        section_header(ICO_STRESS, "Stress Test — Crises Históricas", "h3")
        st.caption(
            "Desempenho estimado do portfólio durante períodos de turbulência histórica. "
            "Exibe apenas crises dentro do período de dados selecionado."
        )

        CRISES_HISTORICAS = {
            "COVID-19 (2020)": ("2020-01-17", "2020-03-23"),
            "Recessão/Impeachment (2015–16)": ("2014-12-31", "2016-12-31"),
            "Joesley Day (2017)": ("2017-05-17", "2017-06-30"),
            "Crise Global (2008–09)": ("2008-08-01", "2009-03-31"),
        }

        data_start = data_yf.index.min()
        data_end = data_yf.index.max()
        stress_results = []

        for crise_nome, (s_str, e_str) in CRISES_HISTORICAS.items():
            s_ts = pd.Timestamp(s_str)
            e_ts = pd.Timestamp(e_str)
            if s_ts > data_end or e_ts < data_start:
                continue
            s_clip = max(s_ts, data_start)
            e_clip = min(e_ts, data_end)
            mask_p = (data_yf.index >= s_clip) & (data_yf.index <= e_clip)
            period_prices = data_yf[mask_p]
            if len(period_prices) < 5:
                continue
            period_ret = period_prices.pct_change().dropna()
            pesos_dict = dict(zip(data_yf.columns, pesos_manuais_arr))
            pesos_period = np.array([pesos_dict.get(c, 0) for c in period_ret.columns])
            if pesos_period.sum() > 0:
                pesos_period = pesos_period / pesos_period.sum()
            port_ret_period = period_ret.dot(pesos_period)
            cum_port = (1 + port_ret_period).prod() - 1
            mask_b = (retorno_bench.index >= s_clip) & (retorno_bench.index <= e_clip)
            ibov_period = retorno_bench[mask_b]
            cum_ibov = (1 + ibov_period).prod() - 1 if len(ibov_period) >= 5 else None
            stress_results.append(
                {
                    "Crise": crise_nome,
                    "Período": f"{s_clip.strftime('%b/%Y')} → {e_clip.strftime('%b/%Y')}",
                    "Portfólio": cum_port,
                    "IBOV": cum_ibov,
                }
            )

        if not stress_results:
            st.info(
                "Nenhuma crise histórica está no intervalo de dados selecionado. "
                "Selecione um período mais longo (3+ anos) para ativar o stress test."
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
                    f'{r["Crise"]} <span style="font-weight:400;color:#475569">({r["Período"]})</span></div>',
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
                    (r["IBOV"] * 100 if r["IBOV"] is not None else 0)
                    for r in stress_results
                ]
                fig_stress = go.Figure()
                fig_stress.add_trace(
                    go.Bar(
                        name="Portfólio",
                        x=crises_names,
                        y=port_vals,
                        marker_color=[
                            "#00ff87" if v >= 0 else "#ff3d5a" for v in port_vals
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
                            "rgba(0,210,255,0.6)" if v >= 0 else "rgba(255,150,0,0.6)"
                            for v in ibov_vals
                        ],
                        text=[f"{v:+.1f}%" for v in ibov_vals],
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
                apply_plotly_theme(fig_stress)
                st.plotly_chart(fig_stress, use_container_width=True)

        # ── Regime de Mercado (últimos 21 dias) ──────────────────────────────
        section_header(ICO_SIGNAL, "Regime de Mercado (Últimos 21 Dias)", "h4")
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

        st.markdown("---")

        # Métricas Consolidadas Essenciais
        dias_totais = (portfolio_value.index[-1] - portfolio_value.index[0]).days
        ret_total = (portfolio_value.iloc[-1] / valor_inicial - 1) * 100
        ret_anual = (
            ((portfolio_value.iloc[-1] / valor_inicial) ** (365.25 / dias_totais) - 1)
            * 100
            if dias_totais > 0
            else 0.0
        )
        vol_anual = portfolio_returns.std() * np.sqrt(252) * 100

        excesso_retorno_diario = portfolio_returns - taxa_selic
        std_dev = portfolio_returns.std()
        sharpe_anual = (
            (excesso_retorno_diario.mean() / std_dev) * np.sqrt(252)
            if std_dev > 0
            else 0.0
        )

        max_dd_val = max_drawdown(portfolio_returns) * 100
        var_val = var(portfolio_returns) * 100
        alfa_anual = alfa_val * 252 * 100

        detailed_stats = pd.DataFrame(
            {
                "Métrica": [
                    "Retorno Total",
                    "Retorno Anualizado",
                    "Volatilidade Anualizada",
                    "Índice de Sharpe",
                    "Beta vs IBOVESPA",
                    "Alfa Anualizado",
                    "Máximo Drawdown",
                    "VaR Diário (95%)",
                    "Information Ratio",
                ],
                "Valor": [
                    f"{ret_total:.2f}%",
                    f"{ret_anual:.2f}%",
                    f"{vol_anual:.2f}%",
                    f"{sharpe_anual:.2f}",
                    f"{beta:.4f}",
                    f"{alfa_anual:.2f}%",
                    f"{max_dd_val:.2f}%",
                    f"{var_val:.2f}%",
                    f"{information_ratio:.2f}",
                ],
            }
        )
        section_header(ICO_METRICS, "Métricas Consolidadas do Portfólio", "h3")
        stats_dict = dict(zip(detailed_stats["Métrica"], detailed_stats["Valor"]))
        render_cards_grid(stats_dict)

        st.markdown(
            """
<div style="background:linear-gradient(90deg,rgba(0,210,255,0.12) 0%,transparent 100%);height:2px;border-radius:2px;margin:2rem 0 1.5rem 0;"></div>
""",
            unsafe_allow_html=True,
        )
        section_header(ICO_RISK, "Análise de Drawdown", "h3")

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
                fillcolor="rgba(255,23,68,0.25)",
                line=dict(color="#ff1744", width=1.5),
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
                    <div style="background: linear-gradient(135deg, #0e1726, #070c14); 
                                border: 1px solid #1e293b; 
                                border-radius: 10px; 
                                padding: 0.8rem; 
                                text-align: center; 
                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                margin-bottom: 0.5rem;
                                min-height: 110px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;">
                        <div style="font-size: 1.1rem; color: #38bdf8; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.2rem;">{ticker}</div>
                        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem;">Máx. Drawdown</div>
                        <div style="font-size: 1.2rem; color: #ff1744; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.3rem;">{m_dd:.2f}%</div>
                        <div style="font-size: 0.65rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">{dt_dd}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Rolling Beta (60 dias)
        window = 60
        rolling_cov = portfolio_returns.rolling(window).cov(retorno_bench)
        rolling_var = retorno_bench.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var

        # Gráfico Rolling Beta
        st.subheader(f"Beta Móvel ({window} dias) vs IBOVESPA")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=rolling_beta.index,
                y=rolling_beta.values,
                line=dict(color="#00d2ff", width=2),
                name="Beta Móvel",
                hovertemplate="%{x|%d/%m/%Y}<br>β=%{y:.3f}<extra></extra>",
            )
        )
        fig2.add_hline(
            y=1,
            line_dash="dash",
            line_color="#ffd600",
            line_width=1.5,
            annotation_text="β = 1",
            annotation_font=dict(color="#ffd600", size=10),
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
            portfolio_returns.rolling(window).mean() - taxa_selic
        ) / portfolio_returns.rolling(window).std()

        st.subheader(f"Índice de Sharpe Móvel ({window} dias)")
        fig_3 = go.Figure()
        fig_3.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                line=dict(color="#00ff87", width=2),
                name="Sharpe Móvel",
                hovertemplate="%{x|%d/%m/%Y}<br>Sharpe=%{y:.2f}<extra></extra>",
            )
        )
        fig_3.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
        fig_3.update_layout(
            title=f"Índice de Sharpe Móvel ({window} dias)",
            xaxis_title="Data",
            yaxis_title="Sharpe",
        )
        apply_plotly_theme(fig_3)
        st.plotly_chart(fig_3, use_container_width=True)

        st.markdown(
            """
<div style="background:linear-gradient(90deg,rgba(0,210,255,0.12) 0%,transparent 100%);height:2px;border-radius:2px;margin:2rem 0 1.5rem 0;"></div>
""",
            unsafe_allow_html=True,
        )
        section_header(ICO_RISK, "Análise de Contribuição de Risco", "h3")

        cov_matrix_rc = returns.cov()
        port_vol = np.sqrt(
            np.dot(pesos_manuais_arr.T, np.dot(cov_matrix_rc, pesos_manuais_arr))
        )
        marginal_contrib = np.dot(cov_matrix_rc, pesos_manuais_arr) / port_vol
        risk_contribution = (
            pesos_manuais_arr * marginal_contrib
        )  # risco absoluto de cada ativo
        risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100

        risk_df = pd.DataFrame(
            {
                "Ativo": peso_manual_df.index,
                "Peso (%)": (pesos_manuais_arr * 100).round(2),
                "RC (%)": risk_contribution_pct.round(2),
            }
        )

        fig_rc = px.bar(
            risk_df,
            x="Ativo",
            y="RC (%)",
            color="RC (%)",
            color_continuous_scale="Viridis",
            title="Contribuição de Risco por Ativo (%)",
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
            st.session_state["pesos_manuais"] = peso_manual_df["Peso"].to_dict()

        # ── Próximo Passo ────────────────────────────────────────────────────
        st.markdown("---")
        sharpe_txt = (
            f"Sharpe de {sharpe_val:.2f}" if sharpe_val > 0 else "portfólio configurado"
        )
        st.markdown(
            f"""
<div style="background:linear-gradient(135deg,rgba(0,210,255,0.06),rgba(0,255,135,0.03));border:1px solid rgba(0,210,255,0.25);border-radius:14px;padding:1.2rem 1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:0.5rem;">
  <div>
    <div style="font-size:0.72rem;color:#64748b;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">Próximo Passo</div>
    <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;">Abra <span style="color:#00d2ff">Simulação</span> na barra lateral</div>
    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">Com {sharpe_txt} — projete trajetórias com Monte Carlo</div>
  </div>
  <div style="font-size:1.8rem;opacity:0.6;">→</div>
</div>
""",
            unsafe_allow_html=True,
        )
