import datetime
import hashlib
import logging

import pandas as pd
import streamlit as st

from utils.home_data import clear_fundamentus_cache
from utils.identity import get_browser_uid
from utils.market_data import (
    clean_numeric_column,
    get_full_market_data,
    get_sorted_tickers_by_liquidity,
)
from utils.screener import PRESET_FILTERS, filter_stocks, prepare_export
from utils.ui import load_css, loading_overlay, render_page_header, svg_icon
from utils import db as _db

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Screener B3", page_icon="favicon.svg", layout="wide")
load_css()

_screener_uid = get_browser_uid()
ICO_FILTER = svg_icon(
    '<path d="M3 4.5h18l-6.75 8v6.5l-4.5 2v-8.5z" stroke="#00d2ff" stroke-width="1.8" '
    'stroke-linejoin="round" fill="none"/>',
    13,
)

render_page_header(
    "Screener de ações",
    "Explore ações listadas na B3 com critérios claros e ajustáveis. Dados da Fundamentus; consulte a coleta exibida abaixo.",
    "screener",
)


def carregar_dados():
    raw = get_full_market_data()
    renames = {
        "Cotação": "cotacao",
        "P/L": "pl",
        "P/VP": "pvp",
        "PSR": "psr",
        "Div.Yield": "dy",
        "P/Ativo": "pa",
        "P/Cap.Giro": "pcg",
        "P/EBIT": "pebit",
        "P/Ativ Circ.Liq": "pacl",
        "EV/EBIT": "evebit",
        "EV/EBITDA": "evebitda",
        "Mrg Ebit": "mrgebit",
        "Mrg. Líq.": "mrgliq",
        "ROIC": "roic",
        "ROE": "roe",
        "Liq. Corr.": "liqc",
        "Liq.2meses": "liq2m",
        "Patrim. Líq": "patrliq",
        "Dív.Líq/ Patrim.": "divbpatr",
        "Cresc. Rec.5a": "c5y",
    }
    # Debt labels can vary slightly between Fundamentus table versions.
    for col in raw.columns:
        normalized = col.lower().replace(" ", "").replace(".", "")
        if "patrim" in normalized and ("brut" in normalized or "líq" in normalized or "liq" in normalized):
            renames[col] = "divbpatr"
    renamed = raw.rename(columns={source: target for source, target in renames.items() if source in raw.columns})
    renamed.attrs.update(raw.attrs)
    return renamed


if st.sidebar.button(
    "Atualizar dados Fundamentus",
    help="Busca uma nova fotografia do mercado e invalida o cache de dados.",
):
    clear_fundamentus_cache()
    get_sorted_tickers_by_liquidity.clear()
    st.session_state["fund_refresh_requested"] = True
    st.rerun()

st.session_state.pop("fund_refresh_requested", None)

try:
    with loading_overlay("Carregando dados da B3…"):
        df_raw = carregar_dados()
except Exception as exc:
    logger.warning("carregar_dados failed: %s", exc)
    logger.debug("carregar_dados failure details", exc_info=True)
    st.error(f"Não foi possível carregar os dados da Fundamentus: {exc}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.error("A Fundamentus não retornou dados; isso é diferente de uma busca sem resultados.")
    st.stop()

fetched_at = df_raw.attrs.get("fetched_at")
df = df_raw.copy()
for col in df.columns:
    df[col] = clean_numeric_column(df[col])


_PRESET_DESCRIPTIONS = {
    "Explorar B3": "Liquidez média em 2 meses de pelo menos R$ 1 milhão.",
    "Lucro a preço moderado": "P/L > 0 e ≤ 15, ROE ≥ 12% e liquidez ≥ R$ 1 milhão.",
    "Renda atual": "Dividend yield ≥ 5%, ROE ≥ 10% e liquidez ≥ R$ 1 milhão. DY passado não garante dividendos futuros.",
    "Personalizado": "Escolha explicitamente quais critérios aplicar.",
}
_FILTER_WIDGETS = {
    "filter_pl_enabled": ("pl_min", "pl_max"),
    "filter_roe_enabled": ("roe_min",),
    "filter_dy_enabled": ("dy_min",),
    "filter_liq2m_enabled": ("liq2m_min",),
}
for key, value in {
    "preset_select": "Explorar B3",
    "filter_pl_enabled": False,
    "filter_roe_enabled": False,
    "filter_dy_enabled": False,
    "filter_liq2m_enabled": True,
    "pl_min": 0.0,
    "pl_max": 15.0,
    "pl_min_exclusive": False,
    "roe_min": 12.0,
    "dy_min": 5.0,
    "liq2m_min": 1_000_000,
    "sort_by": "Liquidez 2m",
}.items():
    st.session_state.setdefault(key, value)
if st.session_state["preset_select"] not in PRESET_FILTERS:
    st.session_state["preset_select"] = "Explorar B3"


def _mark_custom(changed_key=None):
    st.session_state["preset_select"] = "Personalizado"
    if changed_key == "pl_min":
        st.session_state["pl_min_exclusive"] = False


def _apply_preset():
    preset = st.session_state["preset_select"]
    criteria = PRESET_FILTERS.get(preset)
    if criteria is None or preset == "Personalizado":
        return
    for enabled_key in _FILTER_WIDGETS:
        st.session_state[enabled_key] = False
    for key, value in {"pl_min": 0.0, "pl_max": 15.0, "roe_min": 12.0, "dy_min": 5.0, "liq2m_min": 1_000_000}.items():
        st.session_state[key] = value
    st.session_state["pl_min_exclusive"] = criteria.get("pl_min_exclusive", False)
    for criterion, value in criteria.items():
        if criterion == "pl_min_exclusive":
            continue
        if criterion == "pl_min":
            st.session_state["filter_pl_enabled"] = True
            st.session_state["pl_min"] = value
        elif criterion == "pl_max":
            st.session_state["filter_pl_enabled"] = True
            st.session_state["pl_max"] = value
        elif criterion == "roe_min":
            st.session_state["filter_roe_enabled"] = True
            st.session_state["roe_min"] = value * 100
        elif criterion == "dy_min":
            st.session_state["filter_dy_enabled"] = True
            st.session_state["dy_min"] = value * 100
        elif criterion == "liq2m_min":
            st.session_state["filter_liq2m_enabled"] = True
            st.session_state["liq2m_min"] = value


def _reset_filters():
    st.session_state["preset_select"] = "Explorar B3"
    _apply_preset()


st.sidebar.markdown(
    f'<div class="sidebar-section-label">{ICO_FILTER} Filtros</div>',
    unsafe_allow_html=True,
)
st.sidebar.selectbox(
    "Ponto de partida",
    list(PRESET_FILTERS),
    key="preset_select",
    on_change=_apply_preset,
    help="Cada preset substitui os filtros anteriores. Edite qualquer critério para criar um filtro personalizado.",
)
st.sidebar.caption(_PRESET_DESCRIPTIONS[st.session_state["preset_select"]])
st.sidebar.button(
    "Redefinir para Explorar B3",
    on_click=_reset_filters,
    use_container_width=True,
)

with st.sidebar.expander("Personalize critérios", expanded=True):
    st.markdown("**Valuation**")
    st.checkbox("Aplicar faixa de P/L", key="filter_pl_enabled", on_change=_mark_custom)
    pl_col_min, pl_col_max = st.columns(2)
    with pl_col_min:
        st.number_input(
            "P/L mínimo", min_value=-50.0, max_value=200.0, step=0.5,
            key="pl_min", disabled=not st.session_state["filter_pl_enabled"],
            on_change=_mark_custom, args=("pl_min",),
        )
    with pl_col_max:
        st.number_input(
            "P/L máximo", min_value=-50.0, max_value=200.0, step=0.5,
            key="pl_max", disabled=not st.session_state["filter_pl_enabled"],
            on_change=_mark_custom,
        )

    st.markdown("**Qualidade**")
    st.checkbox("Aplicar ROE mínimo", key="filter_roe_enabled", on_change=_mark_custom)
    st.number_input(
        "ROE mínimo (%)", min_value=0.0, max_value=80.0, step=1.0,
        key="roe_min", disabled=not st.session_state["filter_roe_enabled"],
        on_change=_mark_custom,
    )

    st.markdown("**Dividendos**")
    st.checkbox("Aplicar dividend yield mínimo", key="filter_dy_enabled", on_change=_mark_custom)
    st.number_input(
        "Dividend yield mínimo (%)", min_value=0.0, max_value=30.0, step=0.5,
        key="dy_min", disabled=not st.session_state["filter_dy_enabled"],
        on_change=_mark_custom,
    )

    st.markdown("**Liquidez**")
    st.checkbox("Aplicar liquidez mínima", key="filter_liq2m_enabled", on_change=_mark_custom)
    st.number_input(
        "Liquidez média 2 meses mínima (R$)", min_value=0, max_value=100_000_000,
        step=100_000, key="liq2m_min", disabled=not st.session_state["filter_liq2m_enabled"],
        on_change=_mark_custom,
    )

st.sidebar.selectbox(
    "Ordenar resultados",
    ["Liquidez 2m", "P/L (menor primeiro)", "ROE (maior primeiro)", "DY (maior primeiro)"],
    key="sort_by",
)

criteria = {}
if st.session_state["filter_pl_enabled"]:
    if st.session_state["pl_min"] > st.session_state["pl_max"]:
        st.error("O P/L mínimo não pode ser maior que o P/L máximo.")
        st.stop()
    criteria.update(
        pl_min=st.session_state["pl_min"],
        pl_min_exclusive=st.session_state["pl_min_exclusive"],
        pl_max=st.session_state["pl_max"],
    )
if st.session_state["filter_roe_enabled"]:
    criteria["roe_min"] = st.session_state["roe_min"] / 100
if st.session_state["filter_dy_enabled"]:
    criteria["dy_min"] = st.session_state["dy_min"] / 100
if st.session_state["filter_liq2m_enabled"]:
    criteria["liq2m_min"] = st.session_state["liq2m_min"]

try:
    df_filtrado = filter_stocks(df, criteria)
except ValueError as exc:
    st.error(f"Os dados recebidos não atendem ao esquema do filtro selecionado: {exc}")
    st.stop()

sort_options = {
    "Liquidez 2m": ("liq2m", False),
    "P/L (menor primeiro)": ("pl", True),
    "ROE (maior primeiro)": ("roe", False),
    "DY (maior primeiro)": ("dy", False),
}
sort_column, sort_ascending = sort_options[st.session_state["sort_by"]]
if sort_column not in df_filtrado.columns:
    st.error(f"Não é possível ordenar: a coluna {sort_column} não está disponível.")
    st.stop()
df_filtrado = df_filtrado.sort_values(
    sort_column, ascending=sort_ascending, na_position="last", kind="stable"
)

active_filters = []
if "liq2m_min" in criteria:
    active_filters.append(f"Liquidez 2m ≥ R$ {criteria['liq2m_min']:,.0f}")
if "pl_min" in criteria:
    low_op = ">" if criteria.get("pl_min_exclusive") else "≥"
    active_filters.append(f"P/L {low_op} {criteria['pl_min']:g} e ≤ {criteria['pl_max']:g}")
if "roe_min" in criteria:
    active_filters.append(f"ROE ≥ {criteria['roe_min'] * 100:g}%")
if "dy_min" in criteria:
    active_filters.append(f"DY ≥ {criteria['dy_min'] * 100:g}%")

st.markdown("### Resultados")
st.caption(
    f"{len(df_filtrado)} de {len(df)} ativos correspondem aos filtros · "
    + (" · ".join(active_filters) if active_filters else "sem critérios ativos")
)
if fetched_at:
    try:
        fetched_time = datetime.datetime.fromisoformat(fetched_at)
        local_time = fetched_time.astimezone()
        age = datetime.datetime.now(datetime.timezone.utc) - fetched_time.astimezone(datetime.timezone.utc)
        age_minutes = max(0, int(age.total_seconds() // 60))
        st.caption(f"Fundamentus coletado em {local_time:%d/%m/%Y %H:%M %Z} · há {age_minutes} min")
    except (TypeError, ValueError):
        st.caption("Horário da coleta indisponível; a idade deste dado não pode ser confirmada.")
else:
    st.caption("Horário da coleta indisponível; a idade deste dado não pode ser confirmada.")

avg_dy = df_filtrado["dy"].mean() * 100 if "dy" in df_filtrado.columns and not df_filtrado.empty else None
avg_roe = df_filtrado["roe"].mean() * 100 if "roe" in df_filtrado.columns and not df_filtrado.empty else None
col_count, col_dy, col_roe = st.columns(3)
col_count.metric("Ações correspondentes", len(df_filtrado))
col_dy.metric("DY médio", f"{avg_dy:.2f}%" if avg_dy is not None and pd.notna(avg_dy) else "—")
col_roe.metric("ROE médio", f"{avg_roe:.2f}%" if avg_roe is not None and pd.notna(avg_roe) else "—")

if df_filtrado.empty:
    st.warning("Nenhuma ação encontrada. Reduza os limites ou desative critérios; dados ausentes não são considerados aprovação.")
    st.stop()

col_map = {
    "cotacao": "Cotação (R$)",
    "pl": "P/L",
    "pvp": "P/VP",
    "dy": "Div. Yield (%)",
    "roe": "ROE (%)",
    "roic": "ROIC (%)",
    "evebitda": "EV/EBITDA",
    "evebit": "EV/EBIT",
    "mrgebit": "Mrg. EBIT (%)",
    "mrgliq": "Mrg. Líq. (%)",
    "liqc": "Liq. Corrente",
    "liq2m": "Liq. 2m (R$)",
    "divbpatr": "Dív. Líq./Patrim.",
    "c5y": "Cresc. Rec. 5a (%)",
    "patrliq": "Patrim. Líq. (R$)",
    "setor": "Setor",
}
columns = [column for column in col_map if column in df_filtrado.columns]
display = df_filtrado[columns].rename(columns=col_map).copy()
display.index.name = "Papel"

for column in ("Div. Yield (%)", "ROE (%)", "ROIC (%)", "Mrg. EBIT (%)", "Mrg. Líq. (%)", "Cresc. Rec. 5a (%)"):
    if column in display:
        display[column] = display[column] * 100

# Keep the initial view compact while allowing the full result set to be explored.
default_columns = [
    "Cotação (R$)", "P/L", "P/VP", "Div. Yield (%)", "ROE (%)",
    "ROIC (%)", "EV/EBITDA", "Liq. 2m (R$)",
]
visible_columns = st.multiselect(
    "Colunas exibidas",
    columns,
    default=[column for column in default_columns if column in columns],
    key="screener_visible_columns",
)
# Include optional columns only when selected, without changing ticker membership.
display_columns = [column for column in columns if column in visible_columns]

number_formats = {
    "Cotação (R$)": "R$ %.2f",
    "P/L": "%.2f",
    "P/VP": "%.2f",
    "Div. Yield (%)": "%.2f%%",
    "ROE (%)": "%.2f%%",
    "ROIC (%)": "%.2f%%",
    "EV/EBITDA": "%.2f",
    "EV/EBIT": "%.2f",
    "Mrg. EBIT (%)": "%.2f%%",
    "Mrg. Líq. (%)": "%.2f%%",
    "Liq. Corrente": "%.2f",
    "Liq. 2m (R$)": "R$ %.0f",
    "Dív. Líq./Patrim.": "%.2f",
    "Cresc. Rec. 5a (%)": "%.2f%%",
    "Patrim. Líq. (R$)": "R$ %.0f",
}
column_config = {
    column: st.column_config.NumberColumn(column, format=number_formats[column])
    for column in display_columns
    if column in number_formats
}

st.caption("Selecione uma linha para abrir a análise de preço-alvo ou gerenciar favoritos.")
rows_digest = hashlib.sha1("|".join(map(str, display.index)).encode()).hexdigest()[:12]
table_event = st.dataframe(
    display[display_columns],
    use_container_width=True,
    height=600,
    column_config=column_config,
    key=f"screener_results_{rows_digest}",
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = table_event.selection.rows
if selected_rows:
    ticker = str(display.index[selected_rows[0]])
    st.markdown(f"**Ação selecionada: {ticker}**")
    action_col, favorite_col = st.columns(2)
    with action_col:
        if st.button("Ver preço-alvo", key="screener_selected_valuation"):
            st.session_state["_valuation_handoff_ticker"] = ticker
            st.session_state["valuation_ticker"] = ticker
            st.query_params["uid"] = get_browser_uid()
            st.query_params["valuation_ticker"] = ticker
            st.switch_page("pages/4_Visão_de_mercado.py")
    with favorite_col:
        is_favorited = _db.wl_has(_screener_uid, ticker)
        if st.button(
            "★ Favoritado" if is_favorited else "☆ Favoritar",
            key="screener_selected_watchlist",
            help="Adicionar ou remover este ticker da watchlist da página principal.",
        ):
            if is_favorited:
                _db.wl_remove(_screener_uid, ticker)
            else:
                _db.wl_add(_screener_uid, ticker)
            st.rerun()

# CSV retains every match and uses percentage points for columns labeled (%).
csv_frame = prepare_export(df_filtrado, col_map)
csv_bytes = csv_frame.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    "Exportar todos os resultados como CSV",
    data=csv_bytes,
    file_name=f"screener_b3_{datetime.date.today():%Y%m%d}.csv",
    mime="text/csv",
    help=f"Exporta todas as {len(df_filtrado)} ações correspondentes.",
)

st.markdown("---")
st.caption(
    "Fonte: Fundamentus (fundamentus.com.br). Dados com atraso. Esta triagem não é recomendação de investimento; verifique os indicadores e faça sua própria análise."
)
