import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Screener B3", page_icon="🔍", layout="wide")

# ─── CSS opcional (dark theme via style.css do projeto) ───────────────────────
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ─── Cabeçalho ────────────────────────────────────────────────────────────────
st.title("🔍 Screener B3")
st.markdown(
    "Filtre todas as ações listadas na B3 por indicadores fundamentalistas e encontre "
    "candidatos de investimento. Os dados são atualizados a cada hora via Fundamentus."
)


# ─── Carregamento de dados ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados():
    import fundamentus.resultado as fzr

    raw = fzr.get_resultado_raw()

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


# ─── Normalização: garante colunas numéricas ──────────────────────────────────
# get_resultado() retorna colunas: pl, pvp, dy, roe, evebitda, liq2m, roic,
# mrgliq, mrgebit, cotacao, psr, liqc, divbpatr, c5y, pa, pcg, pebit, pacl,
# evebit, patrliq
def _num(series):
    return pd.to_numeric(series, errors="coerce")


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

# ─── Filtros na barra lateral ─────────────────────────────────────────────────
st.sidebar.header("Filtros")
st.sidebar.caption(
    "Ajuste os parâmetros para encontrar ações que se encaixam no seu perfil."
)

# P/L
pl_range = st.sidebar.slider(
    "P/L (Preço / Lucro)",
    min_value=-50.0,
    max_value=100.0,
    value=(0.0, 30.0),
    step=0.5,
    help="Exclui empresas com P/L negativo ou muito elevado. Faixas comuns: 0–15 (valor), 15–30 (crescimento).",
)

# P/VP
pvp_range = st.sidebar.slider(
    "P/VP (Preço / Valor Patrimonial)",
    min_value=0.0,
    max_value=20.0,
    value=(0.0, 5.0),
    step=0.1,
    help="P/VP < 1 pode indicar desconto sobre o patrimônio. Acima de 3–5 exige alto crescimento.",
)

# Dividend Yield
dy_min = st.sidebar.slider(
    "Dividend Yield mínimo (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.5,
    help="Percentual mínimo de retorno em dividendos ao ano. Selic atual ~10%.",
)

# ROE
roe_min = st.sidebar.slider(
    "ROE mínimo (%)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=1.0,
    help="Return on Equity. Acima de 15% é considerado bom; abaixo de 5% é fraco.",
)

# EV/EBITDA
evebitda_max = st.sidebar.slider(
    "EV/EBITDA máximo",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=0.5,
    help="Múltiplo de valuation que considera dívida. Menor = mais barato. Referência setorial: 6–12×.",
)

# Liquidez 2 meses
liq2m_min = st.sidebar.slider(
    "Liquidez 2 meses mínima (R$)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=100_000,
    format="%d",
    help="Volume médio negociado nos últimos 2 meses. Filtra ações ilíquidas difíceis de comprar/vender.",
)

st.sidebar.markdown("---")
ordenar_por = st.sidebar.radio(
    "Ordenar por",
    ["Score", "Liquidez 2m"],
    help="Score: ranking composto de qualidade e valuation relativo ao universo B3. Liquidez: volume médio 2 meses.",
)

st.sidebar.caption(
    f"Dados: Fundamentus · Atualização: {datetime.date.today().strftime('%d/%m/%Y')}"
)

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

df_filtrado = df[mask].copy()

if ordenar_por == "Score" and "score" in df_filtrado.columns:
    df_filtrado = df_filtrado.sort_values("score", ascending=False)
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

st.markdown("---")

# ─── Tabela de resultados ─────────────────────────────────────────────────────
if df_filtrado.empty:
    st.warning(
        "Nenhuma ação encontrada com os filtros atuais. "
        "Tente ampliar os intervalos ou reduzir os mínimos."
    )
else:
    n_exib = min(len(df_filtrado), 200)
    sort_label = "score composto" if ordenar_por == "Score" else "liquidez (2 meses)"
    st.caption(
        f"Exibindo **{n_exib}** ação(ões) ordenadas por {sort_label}. "
        f"Clique no cabeçalho da coluna para reordenar."
    )

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
    }

    cols_existentes = [c for c in col_map if c in df_filtrado.columns]
    df_exib = df_filtrado[cols_existentes].rename(columns=col_map).copy()

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

    # Formatação de exibição
    fmt = {}
    for col in df_exib.columns:
        if col == "Score":
            fmt[col] = "{:.0f}"
        elif col in pct_cols_dec:
            fmt[col] = "{:.2f}%"
        elif col in (
            "P/L",
            "P/VP",
            "EV/EBITDA",
            "EV/EBIT",
            "Liq. Corrente",
            "Dív./Patrim.",
        ):
            fmt[col] = "{:.2f}"
        elif col == "Cotação (R$)":
            fmt[col] = "R$ {:.2f}"
        elif col in ("Liq. 2m (R$)", "Patrim. Líq. (R$)"):
            fmt[col] = "{:,.0f}"

    styled = df_exib.style.format(fmt, na_rep="—")
    if "Score" in df_exib.columns:
        styled = styled.background_gradient(
            subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100
        )
    st.dataframe(styled, use_container_width=True, height=500)

    # ─── Botão de download ────────────────────────────────────────────────────
    csv_bytes = df_exib.to_csv(index=True).encode("utf-8")
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
