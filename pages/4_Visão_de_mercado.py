import logging
import math

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.identity import get_browser_uid
from utils.market_data import compute_target_upside, get_listed_stocks
from utils.ui import load_css, loading_overlay, render_page_header

logger = logging.getLogger(__name__)
load_css()


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_target_data(ticker_b3):
    try:
        info = yf.Ticker(f"{ticker_b3}.SA").info or {}
        return {
            "company_name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency") or "BRL",
            "price": info.get("currentPrice"),
            "regular_price": info.get("regularMarketPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "target_high": info.get("targetHighPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
        }
    except Exception as exc:
        logger.warning("Market target fetch failed for %s: %s", ticker_b3, exc)
        logger.debug("Market target fetch details", exc_info=True)
        return {"_error": str(exc)}


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


render_page_header(
    "Visão de mercado",
    "Compare a cotação atual com a faixa e o consenso de preços-alvo dos analistas.",
    "valuation",
)

try:
    b3_stocks = sorted(get_listed_stocks()["Ticker"].tolist())
except (OSError, ValueError) as exc:
    st.error(f"Não foi possível carregar a lista de ações da B3: {exc}")
    st.stop()

col_t, col_hint = st.columns([5, 1], vertical_alignment="bottom")
with col_t:
    query_ticker = str(st.query_params.get("valuation_ticker", "")).strip().upper()
    handoff_ticker = str(
        st.session_state.pop("_valuation_handoff_ticker", "")
    ).strip().upper()
    requested_ticker = handoff_ticker or query_ticker
    if requested_ticker:
        b3_stocks = sorted(set(b3_stocks) | {requested_ticker})

    defaults = st.session_state.get("selected_tickers", [])
    default_ticker = defaults[0] if defaults else None
    if default_ticker not in b3_stocks:
        saved_tickers, _ = _db.portfolio_get(get_browser_uid())
        default_ticker = next(
            (
                str(saved).replace(".SA", "")
                for saved in saved_tickers
                if str(saved).replace(".SA", "") in b3_stocks
            ),
            None,
        )
    if handoff_ticker:
        default_ticker = handoff_ticker
        st.session_state["valuation_ticker"] = handoff_ticker

    if "valuation_ticker" not in st.session_state:
        initial_ticker = requested_ticker or default_ticker
        st.session_state["valuation_ticker"] = (
            initial_ticker if initial_ticker in b3_stocks else ""
        )

    ticker = st.text_input(
        "Ticker B3",
        placeholder="Digite um ticker, ex.: PETR4",
        help="Digite o código de uma ação listada na B3.",
        key="valuation_ticker",
    ).strip().upper().removesuffix(".SA")
    if ticker and ticker not in b3_stocks:
        st.warning("Ticker não encontrado na lista de ações da B3.")
        st.stop()

    if ticker:
        st.query_params["valuation_ticker"] = ticker
    elif "valuation_ticker" in st.query_params:
        del st.query_params["valuation_ticker"]

if ticker:
    with col_hint:
        st.button(
            "Atualizar dados",
            key=f"valuation_refresh_{ticker}",
            use_container_width=True,
            on_click=lambda value=ticker: get_market_target_data.clear(value),
            help="Busca novamente os dados de mercado no Yahoo Finance.",
        )

if not ticker:
    st.info("Selecione um ticker B3 para ver o preço-alvo de mercado.")
    quick_tickers = [
        item for item in ("PETR4", "WEGE3", "VALE3", "RENT3") if item in b3_stocks
    ]
    quick_cols = st.columns(4)
    for index, quick_ticker in enumerate(quick_tickers):
        with quick_cols[index]:
            st.button(
                quick_ticker,
                key=f"valuation_quick_{quick_ticker}",
                use_container_width=True,
                on_click=lambda value=quick_ticker: st.session_state.update(
                    {"valuation_ticker": value}
                ),
            )
    st.stop()

with loading_overlay(f"Buscando preços-alvo de {ticker}…", tickers=[ticker]):
    market = get_market_target_data(ticker)

if not market or "_error" in market:
    st.error(f"Não foi possível carregar os dados de mercado de {ticker}. Tente atualizar os dados.")
    st.stop()

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
st.subheader(f"{company_name} · {ticker}")

main_cols = st.columns(3)
with main_cols[0]:
    st.metric("Cotação atual", _money(price, currency_symbol))
with main_cols[1]:
    st.metric("Preço-alvo médio", _money(target_mean, currency_symbol))
with main_cols[2]:
    st.metric(
        "Potencial até o alvo médio",
        f"{upside:+.1f}%" if upside is not None else "—",
        help="(Preço-alvo médio ÷ cotação atual − 1) × 100.",
    )

if target_mean is None:
    st.info("O Yahoo Finance não informa um preço-alvo médio para este ticker.")

st.markdown("#### Faixa de preços-alvo dos analistas")
range_cols = st.columns(3)
for column, label, value in zip(
    range_cols,
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

uid = get_browser_uid()
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
