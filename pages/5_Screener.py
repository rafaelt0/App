import os
import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Screener B3", page_icon="🔍", layout="wide")

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ─── Dados ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados():
    import fundamentus.resultado as fzr
    raw = fzr.get_resultado_raw()
    RENAMES = {
        'Cotação': 'cotacao', 'P/L': 'pl', 'P/VP': 'pvp', 'PSR': 'psr',
        'Div.Yield': 'dy', 'P/Ativo': 'pa', 'P/Cap.Giro': 'pcg',
        'P/EBIT': 'pebit', 'P/Ativ Circ.Liq': 'pacl', 'EV/EBIT': 'evebit',
        'EV/EBITDA': 'evebitda', 'Mrg Ebit': 'mrgebit', 'Mrg. Líq.': 'mrgliq',
        'ROIC': 'roic', 'ROE': 'roe', 'Liq. Corr.': 'liqc',
        'Liq.2meses': 'liq2m', 'Patrim. Líq': 'patrliq', 'Cresc. Rec.5a': 'c5y',
    }
    for col in raw.columns:
        s = col.lower().replace(' ', '').replace('.', '')
        if 'brut' in s and 'patrim' in s and col not in RENAMES:
            RENAMES[col] = 'divbpatr'
    return raw.rename(columns={k: v for k, v in RENAMES.items() if k in raw.columns})


try:
    with st.spinner("Carregando dados da B3..."):
        df_raw = carregar_dados()
except Exception as e:
    st.error(f"Não foi possível carregar os dados da Fundamentus: {e}")
    st.stop()

if df_raw is None or df_raw.empty:
    st.error("Não foi possível carregar os dados. Tente novamente em instantes.")
    st.stop()

df_base = df_raw.copy()
for col in df_base.columns:
    df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

# ─── Setor (sidebar) ──────────────────────────────────────────────────────────
_SECTOR_CSV = os.path.join(os.path.dirname(__file__), '..', 'acoes-listadas-b3.csv')
_sector_df = None
_sector_options = []
if os.path.exists(_SECTOR_CSV):
    try:
        _sector_df = pd.read_csv(_SECTOR_CSV, usecols=["Ticker", "Setor"])
        _sector_options = sorted(_sector_df["Setor"].dropna().unique().tolist())
    except Exception:
        pass

st.sidebar.header("Filtros")
ticker_search = st.sidebar.text_input("Buscar ticker", placeholder="Ex: WEGE3")

setor_selecionado = []
if _sector_options:
    setor_selecionado = st.sidebar.multiselect(
        "Setor", options=_sector_options,
        help="Filtra por setor (fonte: B3)."
    )

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Resetar filtros", use_container_width=True):
    for k in ["preset", "pl_min", "pl_max", "pvp_min", "pvp_max",
              "dy_min", "roe_min", "evebitda_max", "liq2m_min"]:
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Dados: Fundamentus · {datetime.date.today().strftime('%d/%m/%Y')}")

# ─── Cabeçalho ────────────────────────────────────────────────────────────────
st.title("🔍 Screener B3")
st.caption("Filtre as ações da B3 por indicadores fundamentalistas. Escolha uma estratégia ou ajuste os filtros manualmente.")

# ─── Estratégias predefinidas ─────────────────────────────────────────────────
PRESETS = {
    "Valor":       dict(pl_min=0.0,  pl_max=15.0, pvp_min=0.0, pvp_max=2.0,  dy_min=0.0,  roe_min=8.0,  evebitda_max=10.0, liq2m_min=500_000),
    "Dividendos":  dict(pl_min=0.0,  pl_max=25.0, pvp_min=0.0, pvp_max=5.0,  dy_min=5.0,  roe_min=5.0,  evebitda_max=20.0, liq2m_min=500_000),
    "Crescimento": dict(pl_min=0.0,  pl_max=40.0, pvp_min=0.0, pvp_max=8.0,  dy_min=0.0,  roe_min=15.0, evebitda_max=25.0, liq2m_min=500_000),
    "Qualidade":   dict(pl_min=0.0,  pl_max=30.0, pvp_min=0.0, pvp_max=5.0,  dy_min=0.0,  roe_min=15.0, evebitda_max=15.0, liq2m_min=1_000_000),
    "Sem filtros": dict(pl_min=-50.0, pl_max=200.0, pvp_min=0.0, pvp_max=50.0, dy_min=0.0, roe_min=0.0, evebitda_max=100.0, liq2m_min=0),
}

PRESET_ICONS = {"Valor": "🏛️", "Dividendos": "💰", "Crescimento": "🚀", "Qualidade": "💎", "Sem filtros": "🔓"}

active_preset = st.session_state.get("preset", None)

btn_cols = st.columns(len(PRESETS))
for i, (name, vals) in enumerate(PRESETS.items()):
    with btn_cols[i]:
        label = f"{PRESET_ICONS[name]} {name}"
        is_active = active_preset == name
        if st.button(label, key=f"preset_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state["preset"] = name
            for k, v in vals.items():
                st.session_state[k] = v
            st.rerun()

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

# ─── Filtros manuais (number inputs) ─────────────────────────────────────────
with st.expander("⚙️ Ajustar filtros", expanded=(active_preset is None)):
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        pl_min = st.number_input("P/L mínimo", value=float(st.session_state.get("pl_min", 0.0)),
                                 step=1.0, format="%.1f", help="0 exclui empresas com prejuízo")
    with r1c2:
        pl_max = st.number_input("P/L máximo", value=float(st.session_state.get("pl_max", 30.0)),
                                 step=1.0, format="%.1f")
    with r1c3:
        pvp_min = st.number_input("P/VP mínimo", value=float(st.session_state.get("pvp_min", 0.0)),
                                  step=0.1, format="%.1f")
    with r1c4:
        pvp_max = st.number_input("P/VP máximo", value=float(st.session_state.get("pvp_max", 5.0)),
                                  step=0.1, format="%.1f")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        dy_min = st.number_input("DY mínimo (%)", value=float(st.session_state.get("dy_min", 0.0)),
                                 min_value=0.0, max_value=50.0, step=0.5, format="%.1f",
                                 help="Dividend Yield anual mínimo")
    with r2c2:
        roe_min = st.number_input("ROE mínimo (%)", value=float(st.session_state.get("roe_min", 0.0)),
                                  min_value=0.0, max_value=100.0, step=1.0, format="%.0f")
    with r2c3:
        evebitda_max = st.number_input("EV/EBITDA máximo", value=float(st.session_state.get("evebitda_max", 20.0)),
                                       min_value=0.0, step=1.0, format="%.0f",
                                       help="Valores negativos excluídos automaticamente")
    with r2c4:
        liq2m_min = st.number_input("Liquidez 2m mín (R$)", value=float(st.session_state.get("liq2m_min", 0.0)),
                                    min_value=0.0, step=100_000.0, format="%.0f",
                                    help="Volume médio negociado nos últimos 2 meses")

# ─── Aplicação dos filtros ────────────────────────────────────────────────────
df = df_base.copy()

if ticker_search:
    df = df[df.index.str.contains(ticker_search.upper(), na=False)]

if setor_selecionado and _sector_df is not None:
    tickers_no_setor = _sector_df[_sector_df["Setor"].isin(setor_selecionado)]["Ticker"].str.upper()
    df = df[df.index.isin(tickers_no_setor)]

mask = pd.Series(True, index=df.index)

if "pl" in df.columns:
    mask &= df["pl"].notna() & (df["pl"] >= pl_min) & (df["pl"] <= pl_max)
if "pvp" in df.columns:
    mask &= df["pvp"].notna() & (df["pvp"] >= pvp_min) & (df["pvp"] <= pvp_max)
if "dy" in df.columns:
    mask &= df["dy"].notna() & (df["dy"] >= dy_min / 100.0)
if "roe" in df.columns:
    mask &= df["roe"].notna() & (df["roe"] >= roe_min / 100.0)
if "evebitda" in df.columns:
    mask &= df["evebitda"].notna() & (df["evebitda"] > 0) & (df["evebitda"] <= evebitda_max)
if "liq2m" in df.columns:
    mask &= df["liq2m"].notna() & (df["liq2m"] >= liq2m_min)

df_filtrado = df[mask].copy()
if "liq2m" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values("liq2m", ascending=False)
df_filtrado = df_filtrado.head(200)

# ─── Métricas resumo ─────────────────────────────────────────────────────────
def _safe_pct(frame, col):
    if col not in frame.columns or frame.empty:
        return None
    v = frame[col].mean()
    return None if pd.isna(v) else v * 100

total = int(mask.sum())
avg_dy  = _safe_pct(df_filtrado, "dy")
avg_roe = _safe_pct(df_filtrado, "roe")
avg_pl  = df_filtrado["pl"].mean() if "pl" in df_filtrado.columns and not df_filtrado.empty else None
avg_pl  = None if avg_pl is not None and pd.isna(avg_pl) else avg_pl

st.markdown("---")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Ações encontradas", str(total))
mc2.metric("DY médio", f"{avg_dy:.1f}%" if avg_dy is not None else "—")
mc3.metric("ROE médio", f"{avg_roe:.1f}%" if avg_roe is not None else "—")
mc4.metric("P/L médio", f"{avg_pl:.1f}x" if avg_pl is not None else "—")

# ─── Tabela de resultados ─────────────────────────────────────────────────────
if df_filtrado.empty:
    st.warning("Nenhuma ação encontrada com os filtros atuais. Tente ampliar os intervalos ou usar 'Sem filtros'.")
else:
    st.caption(
        f"Exibindo **{min(len(df_filtrado), 200)}** ação(ões) ordenadas por liquidez. "
        "Clique no cabeçalho da coluna para reordenar."
    )

    col_map = {
        "cotacao": "Cotação (R$)", "pl": "P/L", "pvp": "P/VP",
        "dy": "DY (%)", "roe": "ROE (%)", "roic": "ROIC (%)",
        "evebitda": "EV/EBITDA", "mrgebit": "Mrg. EBIT (%)", "mrgliq": "Mrg. Líq. (%)",
        "liq2m": "Liq. 2m (R$)", "divbpatr": "Dív./Patrim.", "c5y": "Cresc. 5a (%)",
    }
    pct_cols = ["DY (%)", "ROE (%)", "ROIC (%)", "Mrg. EBIT (%)", "Mrg. Líq. (%)", "Cresc. 5a (%)"]

    cols_ok = [c for c in col_map if c in df_filtrado.columns]
    df_exib = df_filtrado[cols_ok].rename(columns=col_map).copy()
    for col in pct_cols:
        if col in df_exib.columns:
            df_exib[col] = df_exib[col] * 100

    fmt = {}
    for col in df_exib.columns:
        if col in pct_cols:            fmt[col] = "{:.2f}%"
        elif col == "Cotação (R$)":    fmt[col] = "R$ {:.2f}"
        elif col == "Liq. 2m (R$)":   fmt[col] = "{:,.0f}"
        elif col in ("P/L", "P/VP", "EV/EBITDA", "Dív./Patrim."): fmt[col] = "{:.2f}"

    st.dataframe(df_exib.style.format(fmt, na_rep="—"), use_container_width=True, height=480)

    col_dl, col_nav = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "⬇ Exportar CSV",
            data=df_exib.to_csv(index=True).encode("utf-8"),
            file_name=f"screener_b3_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_nav:
        ticker_abrir = st.selectbox(
            "Abrir no B3 Explorer →",
            [""] + list(df_filtrado.index),
            help="Selecione uma ação para abrir a análise completa.",
        )
        if ticker_abrir:
            st.session_state["selected_tickers"] = [ticker_abrir]
            st.switch_page("Main_Page.py")

st.markdown("---")
st.caption(
    "Fonte: Fundamentus (fundamentus.com.br). Dados com atraso. "
    "Este screener é uma ferramenta de triagem — realize sua própria análise antes de investir."
)
