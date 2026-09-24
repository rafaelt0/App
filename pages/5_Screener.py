import streamlit as st
import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)

from utils.ui import (
    load_css,
    loading_overlay,
    render_page_header,
    svg_icon,
)
from utils.home_data import clear_fundamentus_cache
from utils.market_data import (
    clean_numeric_column,
    get_full_market_data,
    get_listed_stocks,
    get_sorted_tickers_by_liquidity,
)
from utils.identity import get_browser_uid
from utils import db as _db

st.set_page_config(page_title="Screener B3", page_icon="favicon.svg", layout="wide")

# ─── CSS opcional (dark theme via style.css do projeto) ───────────────────────
load_css()


_screener_uid = get_browser_uid()

# ─── SVG Icon Library (sidebar) ────────────────────────────────────────────────
_svg = svg_icon
ICO_FILTER = _svg(
    '<path d="M3 4.5h18l-6.75 8v6.5l-4.5 2v-8.5z" stroke="#00d2ff" stroke-width="1.8" '
    'stroke-linejoin="round" fill="none"/>',
    13,
)
ICO_PIN = _svg(
    '<path d="M12 21s7-6.1 7-11.5A7 7 0 0 0 5 9.5C5 14.9 12 21 12 21z" stroke="#64748b" '
    'stroke-width="1.6" stroke-linejoin="round" fill="none"/>'
    '<circle cx="12" cy="9.5" r="2.2" stroke="#64748b" stroke-width="1.6"/>',
    12,
)
ICO_FORMULA = _svg(
    '<path d="M6 4h11l-5 8 5 8H6" stroke="#a855f7" stroke-width="1.7" stroke-linejoin="round" '
    'stroke-linecap="round" fill="none"/>',
    13,
)
ICO_SORT = _svg(
    '<path d="M7 4v16M7 4 3.5 7.5M7 4l3.5 3.5" stroke="#ffd600" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M17 20V4M17 20l-3.5-3.5M17 20l3.5-3.5" stroke="#ffd600" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"/>',
    13,
)

# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "Screener de ações",
    "Filtre as ações listadas na B3 por indicadores fundamentalistas e encontre candidatos de investimento. "
    "Os dados são atualizados a cada hora via Fundamentus.",
    "screener",
)


# ─── Carregamento de dados ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados():
    raw = get_full_market_data()

    # Explicit rename map — avoids crashing when fundamentus.com.br changes a column name
    RENAMES = {
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
        "Cresc. Rec.5a": "c5y",
    }
    # Debt/patrimony column name varies — match flexibly
    for col in raw.columns:
        s = col.lower().replace(" ", "").replace(".", "")
        if "brut" in s and "patrim" in s and col not in RENAMES:
            RENAMES[col] = "divbpatr"

    renamed = raw.rename(columns={k: v for k, v in RENAMES.items() if k in raw.columns})
    for required in ("pl", "pvp", "c5y", "dy", "cotacao"):
        if required not in renamed.columns:
            renamed[required] = pd.NA
    return renamed


if st.sidebar.button(
    "Atualizar dados Fundamentus",
    help="Limpa o cache do screener e busca uma nova fotografia do mercado.",
):
    removed = clear_fundamentus_cache()
    carregar_dados.clear()
    get_sorted_tickers_by_liquidity.clear()
    st.session_state["fund_refresh_removed"] = removed
    st.rerun()

_refresh_removed = st.session_state.pop("fund_refresh_removed", None)
if _refresh_removed is not None:
    st.sidebar.success(f"Cache Fundamentus atualizado ({_refresh_removed} entradas removidas).")


try:
    with loading_overlay("Carregando dados da B3…"):
        df_raw = carregar_dados()
except Exception as exc:
    logger.warning("carregar_dados failed: %s", exc)
    logger.debug("carregar_dados failure details", exc_info=True)
    st.error(f"Não foi possível carregar os dados da Fundamentus: {exc}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.error("Não foi possível carregar os dados. Tente novamente em instantes.")
    st.stop()


# ─── Normalização: garante colunas numéricas ──────────────────────────────────
# get_resultado() retorna colunas: pl, pvp, dy, roe, evebitda, liq2m, roic,
# mrgliq, mrgebit, cotacao, psr, liqc, divbpatr, c5y, pa, pcg, pebit, pacl,
# evebit, patrliq
def _num(series):
    return clean_numeric_column(series)


df = df_raw.copy()
for col in df.columns:
    df[col] = _num(df[col])


# ─── Score composto (universo B3 completo) ────────────────────────────────────
def _calcular_score(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ["roic", "roe", "dy", "mrgliq", "c5y"]:
        if col in df.columns:
            parts.append(df[col].rank(pct=True, na_option="keep") * 100)
    for col in ["pl", "pvp", "evebitda"]:
        if col in df.columns:
            s = (
                df[col]
                .where(df[col] > 0)
                .rank(pct=True, ascending=True, na_option="keep")
                * 100
            )
            parts.append(100 - s)
    if not parts:
        return pd.Series(dtype=float, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1).round(1)


df["score"] = _calcular_score(df)

# ─── Setor (para exclusão de financeiras/utilities na Magic Formula) ──────────
try:
    _setores = get_listed_stocks().set_index("Ticker")["Setor"]
    df["setor"] = df.index.map(_setores)
except Exception:
    logger.warning("sector CSV enrichment failed", exc_info=True)
    st.warning("Classificação setorial indisponível; filtros por setor podem ficar limitados.")
    df["setor"] = None

# ─── Colunas calculadas (frameworks) ──────────────────────────────────────────
# Graham Number: produto P/L × P/VP — Graham considera aceitável até 22,5
# (equivalente a P/L≤15 e P/VP≤1,5 combinados, mas permite trade-off entre os dois).
df["graham_number"] = df["pl"] * df["pvp"]
df.loc[(df["pl"] <= 0) | (df["pvp"] <= 0), "graham_number"] = pd.NA

# c5y já vem em decimal (0.15 = 15%) — usamos em pontos percentuais para o PEG.
df["c5y_pct"] = df["c5y"] * 100

# PEG (Lynch) = P/L ÷ crescimento de receita 5a (em pontos percentuais).
df["peg"] = df["pl"] / df["c5y_pct"]
df.loc[(df["pl"] <= 0) | (df["c5y_pct"] <= 0), "peg"] = pd.NA

# Earnings Yield (Greenblatt) = EBIT/EV = inverso do EV/EBIT.
if "evebit" in df.columns:
    df["earnings_yield"] = 1 / df["evebit"]
    df.loc[df["evebit"] <= 0, "earnings_yield"] = pd.NA
else:
    df["earnings_yield"] = pd.NA

# Magic Formula score = média dos percentis de ROIC e Earnings Yield (maior = melhor).
_mf_parts = []
if "roic" in df.columns:
    _mf_parts.append(df["roic"].rank(pct=True, na_option="keep") * 100)
_mf_parts.append(df["earnings_yield"].rank(pct=True, na_option="keep") * 100)
df["magic_score"] = pd.concat(_mf_parts, axis=1).mean(axis=1).round(1)

# ─── Filtros na barra lateral ─────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div class="sidebar-section-label">{ICO_FILTER} Filtros</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(
    "Ajuste os parâmetros para encontrar ações que se encaixam no seu perfil, "
    "ou escolha um filtro pré-definido baseado em frameworks consagrados de stock picking."
)

# ─── Presets de frameworks consagrados ─────────────────────────────────────────
DEFAULTS = {
    "pl_min": 0.0, "pl_max": 30.0,
    "pvp_min": 0.0, "pvp_max": 5.0,
    "dy_min": 0.0,
    "roe_min": 0.0,
    "evebitda_max": 20.0,
    "liq2m_min": 0,
    "divbpatr_max": 5.0,
    "liqc_min": 0.0,
    "c5y_min": -100.0, "c5y_max": 500.0,
    "peg_on": False, "peg_max": 1.0,
    "graham_on": False,
    "magic_exclude_fin": True,
    "bazin_yield": 6.0,
    "ordenar_por": "Score",
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("preset_select", "Personalizado")

PRESETS = {
    "Personalizado": {},
    "Bazin / Barsi (Dividendos)": {
        "dy_min": 6.0, "roe_min": 12.0, "divbpatr_max": 0.6,
        "pl_min": 0.0, "pl_max": 30.0, "pvp_min": 0.0, "pvp_max": 5.0,
        "liqc_min": 0.0, "peg_on": False, "graham_on": False,
        "ordenar_por": "Div. Yield",
    },
    "Graham — Investidor Defensivo": {
        "pl_min": 0.0, "pl_max": 15.0, "pvp_min": 0.0, "pvp_max": 1.5,
        "liqc_min": 2.0, "divbpatr_max": 1.0, "graham_on": True,
        "dy_min": 0.0, "roe_min": 0.0, "peg_on": False,
        "ordenar_por": "Score",
    },
    "Peter Lynch (GARP)": {
        "pl_min": 0.0, "pl_max": 40.0, "pvp_min": 0.0, "pvp_max": 10.0,
        "divbpatr_max": 0.6, "liqc_min": 1.0, "peg_on": True, "peg_max": 1.0,
        "c5y_min": 15.0, "c5y_max": 30.0, "graham_on": False,
        "ordenar_por": "Score",
    },
    "Greenblatt — Magic Formula": {
        "pl_min": -50.0, "pl_max": 100.0, "pvp_min": 0.0, "pvp_max": 20.0,
        "dy_min": 0.0, "roe_min": 0.0, "divbpatr_max": 5.0, "liqc_min": 0.0,
        "peg_on": False, "graham_on": False, "magic_exclude_fin": True,
        "ordenar_por": "Magic Formula (Greenblatt)",
    },
}
PRESET_DESC = {
    "Personalizado": "Ajuste livre dos parâmetros abaixo.",
    "Bazin / Barsi (Dividendos)": "DY ≥ 6%, ROE ≥ 12%, baixo endividamento — dividendos consistentes.",
    "Graham — Investidor Defensivo": "P/L ≤ 15, P/VP ≤ 1,5, liquidez corrente ≥ 2, Graham Number ≤ 22,5.",
    "Peter Lynch (GARP)": "PEG ≤ 1, crescimento de receita 15–30% a.a., dívida/patrimônio ≤ 0,6.",
    "Greenblatt — Magic Formula": "Ranking por ROIC + Earnings Yield; exclui financeiras/utilities.",
}




def _reset_filters():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["preset_select"] = "Personalizado"

def _apply_preset_name(preset_name):
    st.session_state["preset_select"] = preset_name
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key, value in PRESETS.get(preset_name, {}).items():
        st.session_state[key] = value


with st.expander("Filtros rápidos", expanded=False):
    st.caption(
        "Aplique um ponto de partida sem abrir a barra lateral. "
        "Os limites continuam editáveis depois."
    )
    _quick_preset_options = (
        ("Bazin / Barsi", "Bazin / Barsi (Dividendos)"),
        ("Graham", "Graham — Investidor Defensivo"),
        ("Peter Lynch", "Peter Lynch (GARP)"),
        ("Magic Formula", "Greenblatt — Magic Formula"),
    )
    _quick_preset_cols = st.columns(len(_quick_preset_options))
    for _index, (_label, _preset_name) in enumerate(_quick_preset_options):
        with _quick_preset_cols[_index]:
            st.button(
                _label,
                key=f"screener_quick_preset_{_index}",
                use_container_width=True,
                help=PRESET_DESC[_preset_name],
                on_click=_apply_preset_name,
                args=(_preset_name,),
            )


def _apply_preset():
    _apply_preset_name(st.session_state["preset_select"])


st.sidebar.selectbox(
    "Filtro pré-definido",
    list(PRESETS.keys()),
    key="preset_select",
    on_change=_apply_preset,
    help="Aplica automaticamente os parâmetros do framework escolhido — "
    "você pode ajustar cada campo individualmente depois.",
)
st.sidebar.caption(
    f"{ICO_PIN} {PRESET_DESC.get(st.session_state['preset_select'], '')}",
    unsafe_allow_html=True,
)
st.sidebar.button(
    "Limpar filtros",
    on_click=_reset_filters,
    help="Restaura os limites padrão e volta à ordenação por Score.",
    use_container_width=True,
)
st.sidebar.markdown("---")


def _range_input(label, key_min, key_max, bounds, step, help_text=""):
    st.sidebar.markdown(
        f'<div class="sidebar-filter-label">{label}</div>', unsafe_allow_html=True
    )
    c1, c2 = st.sidebar.columns(2)
    with c1:
        vmin = st.number_input(
            "Mín.", bounds[0], bounds[1], step=step, key=key_min, help=help_text
        )
    with c2:
        vmax = st.number_input("Máx.", bounds[0], bounds[1], step=step, key=key_max)
    return vmin, vmax


# P/L
pl_range = _range_input(
    "P/L (Preço / Lucro)",
    "pl_min",
    "pl_max",
    (-50.0, 200.0),
    0.5,
    "Exclui empresas fora da faixa. Graham: ≤15. Lynch tolera P/L mais alto se o "
    "crescimento justificar (ver PEG).",
)

# P/VP
pvp_range = _range_input(
    "P/VP (Preço / Valor Patrimonial)",
    "pvp_min",
    "pvp_max",
    (0.0, 30.0),
    0.1,
    "Graham: ≤1,5. P/VP < 1 pode indicar desconto sobre o patrimônio.",
)

# Dividend Yield
dy_min = st.sidebar.number_input(
    "Dividend Yield mínimo (%)",
    0.0,
    30.0,
    step=0.5,
    key="dy_min",
    help="Bazin: ≥6% a.a. — o 'número mágico'. Selic atual ~10%.",
)

# ROE
roe_min = st.sidebar.number_input(
    "ROE mínimo (%)",
    0.0,
    80.0,
    step=1.0,
    key="roe_min",
    help="Return on Equity. Acima de 15% é considerado bom; abaixo de 5% é fraco.",
)

# EV/EBITDA
evebitda_max = st.sidebar.number_input(
    "EV/EBITDA máximo",
    0.0,
    100.0,
    step=0.5,
    key="evebitda_max",
    help="Múltiplo de valuation que considera dívida. Menor = mais barato. Referência setorial: 6–12×.",
)

# Liquidez 2 meses
liq2m_min = st.sidebar.number_input(
    "Liquidez 2 meses mínima (R$)",
    0,
    100_000_000,
    step=100_000,
    key="liq2m_min",
    help="Volume médio negociado nos últimos 2 meses. Filtra ações ilíquidas difíceis de comprar/vender.",
)

# Dívida/Patrimônio
divbpatr_max = st.sidebar.number_input(
    "Dívida/Patrimônio máximo",
    0.0,
    10.0,
    step=0.1,
    key="divbpatr_max",
    help="Lynch: <0,6 (idealmente <0,25). Graham: dívida de LP baixa em relação ao patrimônio.",
)

# Liquidez Corrente
liqc_min = st.sidebar.number_input(
    "Liquidez Corrente mínima",
    0.0,
    20.0,
    step=0.1,
    key="liqc_min",
    help="Ativo circulante ÷ passivo circulante. Graham: ≥2. Lynch: ≥1.",
)

# Crescimento de receita 5a — faixa
c5y_range = _range_input(
    "Cresc. Receita 5a (%) — faixa",
    "c5y_min",
    "c5y_max",
    (-100.0, 500.0),
    1.0,
    "Lynch: 15–30% a.a. é a faixa saudável — acima de 30% raramente é sustentável.",
)

st.sidebar.markdown(
    f'<div class="sidebar-section-label">{ICO_FORMULA} Filtros calculados (frameworks)</div>',
    unsafe_allow_html=True,
)
graham_on = st.sidebar.checkbox(
    "Graham Number: P/L × P/VP ≤ 22,5",
    key="graham_on",
    help="Limite combinado de Graham — permite trade-off entre P/L e P/VP em vez "
    "de exigir os dois limites simultaneamente.",
)
peg_on = st.sidebar.checkbox(
    "Aplicar filtro PEG (Lynch)",
    key="peg_on",
    help="PEG = P/L ÷ crescimento de receita 5a (%). Lynch: <1 (idealmente <0,8).",
)
peg_max = st.sidebar.number_input(
    "PEG máximo", 0.0, 20.0, step=0.1, key="peg_max", disabled=not peg_on
)
bazin_yield = st.sidebar.number_input(
    "Yield desejado p/ Preço-teto Bazin (%)",
    1.0,
    20.0,
    step=0.5,
    key="bazin_yield",
    help="Preço-teto Bazin = Dividendo/ação (12m) ÷ yield desejado. Padrão do método: 6%.",
)

st.sidebar.markdown(
    f'<div class="sidebar-section-label">{ICO_SORT} Ordenação</div>',
    unsafe_allow_html=True,
)
ordenar_por = st.sidebar.radio(
    "Ordenar por",
    ["Score", "Liquidez 2m", "Div. Yield", "Magic Formula (Greenblatt)"],
    key="ordenar_por",
    help="Score: ranking composto de qualidade e valuation relativo ao universo B3. "
    "Magic Formula: ROIC + Earnings Yield (Greenblatt).",
)
magic_exclude_fin = st.sidebar.checkbox(
    "Excluir financeiras/utilities (regra Greenblatt)",
    key="magic_exclude_fin",
    help="Só se aplica quando 'Ordenar por' = Magic Formula. Greenblatt exclui "
    "esses setores por terem contabilidade não comparável às demais empresas.",
)

st.sidebar.caption(
    f"Dados: Fundamentus · Atualização: {datetime.date.today().strftime('%d/%m/%Y')}"
)

# Preço-teto Bazin — depende do yield desejado escolhido acima.
df["bazin_teto"] = (df["dy"] * df["cotacao"]) / (bazin_yield / 100.0)

# ─── Aplicação dos filtros ────────────────────────────────────────────────────
mask = pd.Series(True, index=df.index)

# P/L: inclui apenas linhas com valor não-nulo dentro do range
if "pl" in df.columns:
    pl_valid = df["pl"].notna() & (df["pl"] >= pl_range[0]) & (df["pl"] <= pl_range[1])
    mask = mask & pl_valid

# P/VP
if "pvp" in df.columns:
    pvp_valid = (
        df["pvp"].notna() & (df["pvp"] >= pvp_range[0]) & (df["pvp"] <= pvp_range[1])
    )
    mask = mask & pvp_valid

# Dividend Yield — dy em formato decimal (0.06 = 6%) no get_resultado
if "dy" in df.columns:
    dy_min_dec = dy_min / 100.0
    dy_valid = df["dy"].notna() & (df["dy"] >= dy_min_dec)
    mask = mask & dy_valid

# ROE — em formato decimal
if "roe" in df.columns:
    roe_min_dec = roe_min / 100.0
    roe_valid = df["roe"].notna() & (df["roe"] >= roe_min_dec)
    mask = mask & roe_valid

# EV/EBITDA
if "evebitda" in df.columns:
    ev_valid = (
        df["evebitda"].notna() & (df["evebitda"] > 0) & (df["evebitda"] <= evebitda_max)
    )
    mask = mask & ev_valid

# Liquidez 2 meses
if "liq2m" in df.columns:
    liq_valid = df["liq2m"].notna() & (df["liq2m"] >= liq2m_min)
    mask = mask & liq_valid

# Dívida/Patrimônio e Liquidez Corrente — filtros opcionais: linhas sem o dado
# passam (benefício da dúvida), só filtram quando o valor está disponível.
if "divbpatr" in df.columns:
    dp_valid = df["divbpatr"].isna() | (df["divbpatr"] <= divbpatr_max)
    mask = mask & dp_valid

if "liqc" in df.columns:
    liqc_valid = df["liqc"].isna() | (df["liqc"] >= liqc_min)
    mask = mask & liqc_valid

# Crescimento de receita 5a — faixa
if "c5y_pct" in df.columns:
    c5y_valid = df["c5y_pct"].isna() | (
        (df["c5y_pct"] >= c5y_range[0]) & (df["c5y_pct"] <= c5y_range[1])
    )
    mask = mask & c5y_valid

# Graham Number (P/L × P/VP ≤ 22,5)
if graham_on:
    gn_valid = df["graham_number"].isna() | (df["graham_number"] <= 22.5)
    mask = mask & gn_valid

# PEG (Lynch)
if peg_on:
    peg_valid = df["peg"].isna() | (df["peg"] <= peg_max)
    mask = mask & peg_valid

if ordenar_por == "Magic Formula (Greenblatt)":
    # Greenblatt requires both positive profitability and earnings yield.
    # Do not rank incomplete rows or let a single available metric carry them.
    if "roic" in df.columns:
        mask = mask & df["roic"].notna() & (df["roic"] > 0)
    else:
        mask = mask & pd.Series(False, index=df.index)
    mask = mask & df["earnings_yield"].notna() & (df["earnings_yield"] > 0)

    # Exclude financials/utilities (accounting is not comparable)
    # when the option is enabled.
    if magic_exclude_fin and "setor" in df.columns:
        EXCLUDED_SECTORS_MAGIC = {
            "Intermediários Financeiros",
            "Previdência e Seguros",
            "Serviços Financeiros Diversos",
            "Energia Elétrica",
            "Água e Saneamento",
            "Gás",
        }
        mask = mask & (~df["setor"].isin(EXCLUDED_SECTORS_MAGIC))

df_filtrado = df[mask].copy()

if ordenar_por == "Score" and "score" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values("score", ascending=False)
elif ordenar_por == "Div. Yield" and "dy" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values("dy", ascending=False)
elif (
    ordenar_por == "Magic Formula (Greenblatt)" and "magic_score" in df_filtrado.columns
):
    df_filtrado = df_filtrado.sort_values("magic_score", ascending=False)
elif "liq2m" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values("liq2m", ascending=False)

df_filtrado = df_filtrado.head(200)

# ─── Métricas resumo ─────────────────────────────────────────────────────────
total_passaram = int(mask.sum())


def _safe_mean_pct(df, col):
    if col not in df.columns or df.empty:
        return None
    v = df[col].mean()
    return None if pd.isna(v) else v * 100


avg_dy = _safe_mean_pct(df_filtrado, "dy")
avg_roe = _safe_mean_pct(df_filtrado, "roe")
top_score = (
    round(df_filtrado["score"].max(), 1)
    if "score" in df_filtrado.columns and not df_filtrado.empty
    else None
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(
        label="Ações que passaram nos filtros",
        value=f"{total_passaram}",
        help="Total de ações da B3 que atendem a todos os critérios definidos.",
    )
with col_m2:
    st.metric(
        label="Div. Yield médio",
        value=f"{avg_dy:.2f}%" if avg_dy is not None else "—",
        help="Dividend Yield médio das ações filtradas (exibindo até 200).",
    )
with col_m3:
    st.metric(
        label="ROE médio",
        value=f"{avg_roe:.2f}%" if avg_roe is not None else "—",
        help="Return on Equity médio das ações filtradas (exibindo até 200).",
    )
with col_m4:
    st.metric(
        label="Maior Score",
        value=f"{top_score:.0f} / 100" if top_score is not None else "—",
        help="Score composto mais alto entre as ações filtradas. Média dos percentis de ROIC, ROE, DY, Mrg.Líq., Cresc.5a (maiores = melhor) e P/L, P/VP, EV/EBITDA invertidos (menores = melhor).",
    )

_filter_summary = []
if st.session_state["preset_select"] != "Personalizado":
    _filter_summary.append(f"Preset: {st.session_state['preset_select']}")
if pl_range != (DEFAULTS["pl_min"], DEFAULTS["pl_max"]):
    _filter_summary.append(f"P/L {pl_range[0]:g}–{pl_range[1]:g}")
if pvp_range != (DEFAULTS["pvp_min"], DEFAULTS["pvp_max"]):
    _filter_summary.append(f"P/VP {pvp_range[0]:g}–{pvp_range[1]:g}")
if dy_min:
    _filter_summary.append(f"DY ≥ {dy_min:g}%")
if roe_min:
    _filter_summary.append(f"ROE ≥ {roe_min:g}%")
if evebitda_max != DEFAULTS["evebitda_max"]:
    _filter_summary.append(f"EV/EBITDA ≤ {evebitda_max:g}")
if liq2m_min:
    _filter_summary.append(f"Liquidez ≥ R$ {liq2m_min:,.0f}")
if divbpatr_max != DEFAULTS["divbpatr_max"]:
    _filter_summary.append(f"Dív./PL ≤ {divbpatr_max:g}")
if liqc_min:
    _filter_summary.append(f"Liq. corrente ≥ {liqc_min:g}")
if c5y_range != (DEFAULTS["c5y_min"], DEFAULTS["c5y_max"]):
    _filter_summary.append(f"Cresc. 5a {c5y_range[0]:g}–{c5y_range[1]:g}%")
if graham_on:
    _filter_summary.append("Graham Number")
if peg_on:
    _filter_summary.append(f"PEG ≤ {peg_max:g}")
if ordenar_por != DEFAULTS["ordenar_por"]:
    _filter_summary.append(f"ordenação: {ordenar_por}")
if not _filter_summary:
    _filter_summary.append("configuração padrão")

_coverage = total_passaram / len(df) * 100 if len(df) else 0
st.caption(
    "Filtros ativos · "
    + " · ".join(_filter_summary)
    + f" · {_coverage:.0f}% do universo"
)

if _filter_summary != ["configuração padrão"]:
    st.button(
        "Limpar filtros ativos",
        key="screener_inline_reset",
        on_click=_reset_filters,
        use_container_width=True,
        help="Restaura os limites padrão e remove os filtros calculados.",
    )

st.markdown("---")

# ─── Tabela de resultados ─────────────────────────────────────────────────────
if df_filtrado.empty:
    st.warning(
        "Nenhuma ação encontrada com os filtros atuais. "
        "Tente ampliar os intervalos ou reduzir os mínimos."
    )
else:
    n_exib = min(len(df_filtrado), 200)
    sort_label = {
        "Score": "score composto",
        "Div. Yield": "dividend yield",
        "Magic Formula (Greenblatt)": "Magic Formula (ROIC + Earnings Yield)",
    }.get(ordenar_por, "liquidez (2 meses)")
    st.caption(
        f"Exibindo **{n_exib}** ação(ões) ordenadas por {sort_label}. "
        f"Clique no cabeçalho da coluna para reordenar."
    )

    st.markdown("#### Destaques da triagem")
    st.caption("As primeiras posições seguem a ordenação selecionada.")
    _top_picks = df_filtrado.head(3)
    _pick_cols = st.columns(len(_top_picks))

    def _pick_value(value, formatter):
        return "—" if pd.isna(value) else formatter(value)

    for _rank, (_ticker, _pick_row) in enumerate(_top_picks.iterrows(), start=1):
        with _pick_cols[_rank - 1]:
            st.markdown(f"**#{_rank} · {_ticker}**")
            st.metric(
                "Score",
                _pick_value(_pick_row.get("score"), lambda value: f"{float(value):.0f} / 100"),
            )
            st.caption(
                f"Cotação {_pick_value(_pick_row.get('cotacao'), lambda value: f'R$ {float(value):.2f}')}"
                f" · DY {_pick_value(_pick_row.get('dy'), lambda value: f'{float(value) * 100:.2f}%')}"
                f" · P/L {_pick_value(_pick_row.get('pl'), lambda value: f'{float(value):.2f}')}"
            )
            if st.button(
                "Abrir valuation",
                key=f"screener_valuation_{_ticker}",
                use_container_width=True,
                help="Abre o DCF já com este ticker selecionado.",
            ):
                st.session_state["_valuation_handoff_ticker"] = str(_ticker)
                st.session_state["valuation_ticker"] = str(_ticker)
                st.query_params["uid"] = get_browser_uid()
                st.query_params["valuation_ticker"] = str(_ticker)
                st.switch_page("pages/4_Valuation.py")
            _ticker_label = str(_ticker)
            _is_favorited = _db.wl_has(_screener_uid, _ticker_label)
            if st.button(
                "★ Favoritado" if _is_favorited else "☆ Favoritar",
                key=f"screener_watchlist_{_ticker_label}",
                use_container_width=True,
                help=(
                    "Remover dos favoritos"
                    if _is_favorited
                    else "Salvar este ticker na watchlist da página principal."
                ),
            ):
                if _is_favorited:
                    _db.wl_remove(_screener_uid, _ticker_label)
                else:
                    _db.wl_add(_screener_uid, _ticker_label)
                st.rerun()

    # Seleciona e renomeia colunas relevantes para exibição
    col_map = {
        "score": "Score",
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
        "divbpatr": "Dív./Patrim.",
        "c5y": "Cresc. Rec. 5a (%)",
        "patrliq": "Patrim. Líq. (R$)",
        "graham_number": "Graham Nº",
        "peg": "PEG",
        "magic_score": "Magic Score",
        "bazin_teto": "Preço-teto Bazin (R$)",
        "setor": "Setor",
    }

    cols_existentes = [c for c in col_map if c in df_filtrado.columns]
    df_exib = df_filtrado[cols_existentes].rename(columns=col_map).copy()
    df_exib.index.name = "Papel"

    # Converte colunas de percentual (decimal → %)
    pct_cols_dec = [
        "Div. Yield (%)",
        "ROE (%)",
        "ROIC (%)",
        "Mrg. EBIT (%)",
        "Mrg. Líq. (%)",
        "Cresc. Rec. 5a (%)",
    ]
    for col in pct_cols_dec:
        if col in df_exib.columns:
            df_exib[col] = df_exib[col] * 100

    # CSV mantém os valores numéricos crus (útil para análise externa) — a
    # formatação em texto abaixo é só para a tabela em tela.
    csv_bytes = df_exib.to_csv(index=True).encode("utf-8-sig")

    # Formatação de exibição. Score e Magic Score ficam numéricos (para o
    # gradiente de cor); as demais colunas viram texto já formatado — o
    # st.dataframe() não aplica o na_rep do Styler em células nulas (mostra o
    # literal "None"), então tratamos o "—" na própria célula em vez de confiar
    # no Styler para isso.
    _fmt_specs = {
        "P/L": "{:.2f}",
        "P/VP": "{:.2f}",
        "EV/EBITDA": "{:.2f}",
        "EV/EBIT": "{:.2f}",
        "Liq. Corrente": "{:.2f}",
        "Dív./Patrim.": "{:.2f}",
        "Graham Nº": "{:.2f}",
        "PEG": "{:.2f}",
        "Cotação (R$)": "R$ {:.2f}",
        "Preço-teto Bazin (R$)": "R$ {:.2f}",
        "Liq. 2m (R$)": "{:,.0f}",
        "Patrim. Líq. (R$)": "{:,.0f}",
    }
    for col in pct_cols_dec:
        _fmt_specs[col] = "{:.2f}%"

    for col, spec in _fmt_specs.items():
        if col in df_exib.columns:
            df_exib[col] = df_exib[col].apply(
                lambda v, s=spec: "—" if pd.isna(v) else s.format(v)
            )
    if "Setor" in df_exib.columns:
        df_exib["Setor"] = df_exib["Setor"].fillna("—")

    styled = df_exib.style
    if "Score" in df_exib.columns:
        styled = styled.background_gradient(
            subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100
        ).format({"Score": "{:.0f}"}, subset=["Score"], na_rep="—")
    if "Magic Score" in df_exib.columns:
        styled = styled.background_gradient(
            subset=["Magic Score"], cmap="RdYlGn", vmin=0, vmax=100
        ).format({"Magic Score": "{:.0f}"}, subset=["Magic Score"], na_rep="—")
    st.dataframe(styled, use_container_width=True, height=500)

    # ─── Botão de download ────────────────────────────────────────────────────
    st.download_button(
        label="Exportar resultados como CSV",
        data=csv_bytes,
        file_name=f"screener_b3_{datetime.date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Baixa as ações filtradas em formato CSV para análise externa.",
    )

# ─── Nota de rodapé ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Fonte: Fundamentus (fundamentus.com.br). Dados com atraso. "
    "Este screener é apenas uma ferramenta de triagem — realize sua própria análise antes de investir. "
    "EV/EBITDA negativo é excluído automaticamente (EBITDA negativo ou empresa pré-operacional)."
)
