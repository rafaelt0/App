import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import hashlib
import logging
import plotly.express as px
import plotly.graph_objects as go

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.identity import get_browser_uid
from utils.portfolio_data import get_portfolio_prices
from utils.simulation import simulate_portfolio
from utils.ui import (
    analyst_synthesis_header,
    empty_state_card,
    load_css,
    loading_overlay,
    next_step_card,
    render_page_header,
    svg_icon,
)

logger = logging.getLogger(__name__)



# CSS customizado
load_css()



# ─── SVG Icon Library ─────────────────────────────────────────────────────────
_svg = svg_icon

ICO_CHART = _svg(
    '<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
    '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
    '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>',
    16,
)
ICO_SIGNAL = _svg(
    '<path d="M2 12 Q6 4 12 12 Q18 20 22 12" stroke="#00d2ff" stroke-width="2" '
    'stroke-linecap="round" fill="none"/>'
    '<circle cx="12" cy="12" r="2" fill="#ffd600"/>',
    16,
)
ICO_FRONTIER = _svg(
    '<path d="M3 20 Q8 8 14 10 Q18 12 21 4" stroke="#00ff87" stroke-width="2" stroke-linecap="round" fill="none"/>'
    '<circle cx="18" cy="6" r="2.5" fill="#ff3d5a"/>'
    '<circle cx="10" cy="17" r="2" fill="#ffd600"/>',
    16,
)
ICO_METRICS = _svg(
    '<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
    '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>',
    16,
)


def section_header(icon_svg, text, tag="h2"):
    st.markdown(
        f'<{tag} class="ui-section-heading">{icon_svg}<span>{text}</span></{tag}>',
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

# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "Simulação de portfólio",
    "Projete faixas de retorno e risco a partir dos pesos definidos na carteira.",
    "simulation",
)

_session_uid = get_browser_uid()

def _restore_saved_portfolio_context() -> None:
    """Restore a saved portfolio when this page opens in a fresh session."""
    if "selected_tickers" in st.session_state:
        return

    saved_tickers, saved_weights = _db.portfolio_get(_session_uid)
    tickers = [str(ticker).replace(".SA", "") for ticker in saved_tickers]
    if len(tickers) < 2:
        return

    try:
        start_date = datetime.date.today() - datetime.timedelta(days=365 * 2)
        with loading_overlay("Restaurando carteira salva…", tickers=tickers):
            prices = get_portfolio_prices(
                [f"{ticker}.SA" for ticker in tickers],
                start_date,
            )
    except Exception:
        logger.warning("simulation saved portfolio restore failed", exc_info=True)
        st.session_state["_simulation_restore_error"] = True
        return

    if prices is None or prices.empty:
        st.session_state["_simulation_restore_error"] = True
        return
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = ["_".join(col).strip() for col in prices.columns.values]

    expected_columns = {f"{ticker}.SA" for ticker in tickers}
    missing_tickers = sorted(expected_columns.difference(map(str, prices.columns)))
    if missing_tickers:
        logger.warning(
            "simulation saved portfolio missing price history for %s", missing_tickers
        )
        st.session_state["_simulation_restore_error"] = True
        return

    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 30:
        logger.warning(
            "simulation saved portfolio has only %d complete return rows", len(returns)
        )
        st.session_state["_simulation_restore_error"] = True
        return

    raw_weights = {}
    for ticker, weight in (saved_weights or {}).items():
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0:
            raw_weights[str(ticker).replace(".SA", "")] = value

    default_weight = 1.0 / len(tickers)
    weights = {
        ticker: raw_weights[ticker] if ticker in raw_weights else default_weight
        for ticker in tickers
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        weights = {ticker: default_weight for ticker in tickers}
    else:
        weights = {
            ticker: weight / weight_total for ticker, weight in weights.items()
        }

    st.session_state.update(
        {
            "selected_tickers": tickers,
            "portfolio_loaded_tickers": tickers,
            "portfolio_analysis_tickers": tickers,
            "portfolio_loaded": True,
            "modo": "Otimização de Markowitz (restaurada)",
            "returns": returns,
            "pesos_manuais": {
                f"{ticker}.SA": weight for ticker, weight in weights.items()
            },
            "peso_manual_df": pd.DataFrame(
                {"Peso": [weights[ticker] for ticker in tickers]},
                index=tickers,
            ),
            "_simulation_restore_notice": (
                "Carteira salva restaurada. Confira os ativos e pesos antes de "
                "rodar a simulação."
            ),
        }
    )


_restore_saved_portfolio_context()
_simulation_restore_error = st.session_state.pop(
    "_simulation_restore_error", False
)
_simulation_restore_notice = st.session_state.pop(
    "_simulation_restore_notice", None
)



# Verifica se o portfólio atual foi carregado e analisado nesta sessão.
required_keys = ["modo", "returns", "peso_manual_df"]
_current_tickers = list(st.session_state.get("selected_tickers", []))
_loaded_tickers = list(st.session_state.get("portfolio_loaded_tickers", []))
_analyzed_tickers = list(st.session_state.get("portfolio_analysis_tickers", []))
_has_analysis_state = all(key in st.session_state for key in required_keys)
_portfolio_ready = (
    _has_analysis_state
    and bool(st.session_state.get("portfolio_loaded"))
    and bool(_current_tickers)
    and _current_tickers == _loaded_tickers == _analyzed_tickers
)

if _portfolio_ready and _simulation_restore_notice:
    st.info(_simulation_restore_notice)

if not _portfolio_ready:
    _selection_changed = bool(_current_tickers) and (
        _current_tickers != _loaded_tickers
        or _current_tickers != _analyzed_tickers
    )
    if _selection_changed:
        _empty_title = "Atualize o portfólio"
        _empty_message = (
            "A seleção de ativos mudou desde a última análise. "
            "Volte para <strong style=\"color:#61d4c6\">Portfolio</strong> e clique em "
            "<strong>Carregar portfólio</strong> antes de rodar a simulação."
        )
    elif _simulation_restore_error:
        _empty_title = "Não foi possível restaurar a carteira"
        _empty_message = (
            "A carteira salva foi encontrada, mas as cotações históricas não "
            "puderam ser carregadas. Abra <strong style=\"color:#61d4c6\">Portfolio</strong> "
            "e carregue a análise novamente."
        )
    else:
        _empty_title = "Portfólio não configurado"
        _empty_message = (
            "Configure seu portfólio na página <strong style=\"color:#61d4c6\">Portfolio</strong> "
            "e carregue a análise para liberar a Simulação Monte Carlo."
        )
    empty_state_card(
        icon_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
            <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
        </svg>""",
        title=_empty_title,
        message=_empty_message,
        cta_label="Ir para Portfolio",
        cta_page="pages/1_Portfolio.py",
    )
    st.stop()

# Recupera as variáveis da aba 1
modo = st.session_state["modo"]
returns = st.session_state["returns"]
peso_manual_df = st.session_state["peso_manual_df"]

section_header(ICO_METRICS, "Alocação usada na simulação", "h2")
_weight_preview = {
    str(index).replace(".SA", ""): f"{float(weight):.1%}"
    for index, weight in peso_manual_df["Peso"].items()
}
render_cards_grid(_weight_preview)
st.caption(
    "Pesos normalizados a partir da carteira carregada; ativos com peso zero "
    "permanecem zerados."
)


_SIMULATION_PRESETS = (
    (
        "Rápida",
        {
            "sim_n_simulations_input": 100,
            "sim_valor_input": 10_000,
            "sim_years_input": 1,
        },
        "100 trajetórias · 1 ano",
    ),
    (
        "Padrão",
        {
            "sim_n_simulations_input": 200,
            "sim_valor_input": 10_000,
            "sim_years_input": 1,
        },
        "200 trajetórias · 1 ano",
    ),
    (
        "Longo prazo",
        {
            "sim_n_simulations_input": 300,
            "sim_valor_input": 10_000,
            "sim_years_input": 5,
        },
        "300 trajetórias · 5 anos",
    ),
)


def _apply_simulation_preset(values: dict[str, int]) -> None:
    for key, value in values.items():
        st.session_state[key] = value


st.caption("Escolha um ponto de partida; você pode ajustar os valores abaixo.")
_preset_cols = st.columns(3)
for _index, (_label, _values, _description) in enumerate(_SIMULATION_PRESETS):
    with _preset_cols[_index]:
        st.button(
            _label,
            key=f"simulation_preset_{_label}",
            use_container_width=True,
            help=_description,
            on_click=_apply_simulation_preset,
            args=(_values,),
        )


with st.form("form_simulacao"):
    n_simulations = st.number_input(
        "Número de Simulações",
        min_value=10,
        max_value=500,
        value=200,
        help="Quantidade de trajetórias simuladas para o portfólio.",
        key="sim_n_simulations_input",
    )
    valor = st.number_input(
        "Capital Inicial (R$)",
        min_value=100,
        value=10_000,
        help="Valor inicial investido no portfólio.",
        key="sim_valor_input",
    )
    years = int(
        st.number_input(
            "Anos",
            min_value=1,
            max_value=10,
            value=1,
            help=(
                "Horizonte da simulação em anos. Limitado a 10 anos para "
                "manter o consumo de memória previsível."
            ),
            key="sim_years_input",
        )
    )

    submitted = st.form_submit_button(
        "Rodar Simulação", type="primary", use_container_width=True
    )

_simulation_fingerprint = hashlib.sha256(
    repr((
        int(n_simulations), int(valor), years, modo,
        peso_manual_df.to_json(), returns.to_json(),
    )).encode()
).hexdigest()
if submitted:
    st.session_state["_simulation_fingerprint"] = _simulation_fingerprint
elif st.session_state.get("_simulation_fingerprint") != _simulation_fingerprint:
    st.info(
        "Configure os parâmetros acima e clique em 'Rodar Simulação' para ver os resultados."
    )
    st.stop()

loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        """
    <div class="loading-container" role="status" aria-live="polite">
        <div class="loading-spinner"></div>
        <div class="loading-text">Rodando simulações Monte Carlo multivariadas…</div>
        <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

n_dias = years * 252  # 252 dias úteis no ano
valor_inicial = valor

# The displayed analyzed allocation is canonical for every portfolio mode.
weight_index = pd.Index([str(ticker).removesuffix(".SA") for ticker in peso_manual_df.index])
weights = pd.Series(peso_manual_df["Peso"].to_numpy(), index=weight_index, dtype=float)
weights.index = weights.index + ".SA"
if (
    not np.isfinite(weights.to_numpy()).all()
    or (weights < 0).any()
    or weights.sum() <= 0
):
    st.error("Os pesos analisados devem ser finitos, não negativos e somar um valor positivo.")
    st.stop()
weights = weights / weights.sum()
weights = weights[weights > 0]
missing_weight_tickers = weights.index.difference(returns.columns)
if len(missing_weight_tickers):
    st.error("Não há retornos disponíveis para todos os ativos com peso positivo.")
    st.stop()
aligned_returns = returns.loc[:, weights.index]
if aligned_returns.empty or aligned_returns.isna().any().any() or not np.isfinite(aligned_returns.to_numpy()).all() or (aligned_returns <= -1).any().any():
    st.error("Retornos históricos inválidos: verifique preços ausentes ou retornos de -100% ou menos.")
    st.stop()

log_returns = np.log1p(aligned_returns)
mu = log_returns.mean().values
cov = log_returns.cov().values
if not np.isfinite(mu).all() or not np.isfinite(cov).all():
    st.error("Não foi possível estimar retornos e covariância finitos com este histórico.")
    st.stop()

sim_df = simulate_portfolio(
    tuple(mu), tuple(map(tuple, cov)), tuple(weights.to_numpy()), n_dias, int(n_simulations),
    float(valor_inicial), datetime.date.today().isoformat(),
)

# Estatísticas finais da simulação
valores_finais = sim_df.iloc[-1]
valor_esperado = valores_finais.mean()
var_5 = np.percentile(valores_finais, 5)
cvar_5 = valores_finais[valores_finais <= var_5].mean()
pior_cenario = valores_finais.min()
melhor_cenario = valores_finais.max()
prob_ganho = (valores_finais > valor_inicial).mean() * 100

# Retorno anualizado esperado
ret_esperado_pct = (valor_esperado / valor_inicial) ** (1 / years) - 1
ret_otimista_pct = (np.percentile(valores_finais, 75) / valor_inicial) ** (
    1 / years
) - 1
ret_pessimista_pct = (np.percentile(valores_finais, 25) / valor_inicial) ** (
    1 / years
) - 1

sim_stats_dict = {
    "Valor Esperado Final": f"R$ {valor_esperado:,.2f}",
    "Probabilidade de Ganho": f"{prob_ganho:.1f}%",
    "Retorno Anual Esperado": f"{ret_esperado_pct * 100:.1f}% a.a.",
    "Retorno Anual Q75": f"{ret_otimista_pct * 100:.1f}% a.a.",
    "Valor final P5": f"R$ {var_5:,.2f}",
    "Retorno final P5": f"{(var_5 / valor_inicial - 1) * 100:.1f}%",
    "Média dos 5% menores valores": f"R$ {cvar_5:,.2f}",
    "Pior Cenário": f"R$ {pior_cenario:,.2f}",
    "Melhor Cenário": f"R$ {melhor_cenario:,.2f}",
}

section_header(ICO_CHART, "Estatísticas da Simulação Monte Carlo", "h2")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    ret_str = f"{ret_esperado_pct * 100:.1f}% a.a."
    st.metric(
        "Retorno Anual Esperado",
        ret_str,
        delta="Positivo" if ret_esperado_pct > 0 else "Negativo",
        delta_color="normal" if ret_esperado_pct > 0 else "inverse",
        help="Retorno anual composto implícito no valor esperado médio.",
    )
with col_s2:
    st.metric(
        "Probabilidade de Ganho",
        f"{prob_ganho:.1f}%",
        delta="Alta"
        if prob_ganho > 70
        else ("Moderada" if prob_ganho > 50 else "Baixa"),
        delta_color="normal"
        if prob_ganho > 60
        else ("off" if prob_ganho > 40 else "inverse"),
        help="Percentual de simulações que terminaram acima do capital inicial.",
    )
with col_s3:
    perda_var = (var_5 / valor_inicial - 1) * 100
    st.metric(
        "Retorno final P5",
        f"{perda_var:.1f}%",
        delta="Controlado" if perda_var > -30 else "Elevado",
        delta_color="normal" if perda_var > -30 else "inverse",
        help=(
            "Retorno no percentil 5 dos valores finais simulados; não é uma "
            "estimativa de VaR nem representa perda máxima."
        ),
    )

render_cards_grid(
    {
        label: value
        for label, value in sim_stats_dict.items()
        if label not in {
            "Probabilidade de Ganho",
            "Retorno Anual Esperado",
            "Retorno final P5",
        }
    }
)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.markdown(
        """
    <div style="background:rgba(0,210,255,0.06);border:1px solid rgba(0,210,255,0.2);border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#b8eeff;">
    <b>Valor final P5:</b> percentil 5 dos valores finais simulados; não é uma medida VaR de perda.
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_exp2:
    st.markdown(
        """
    <div style="background:rgba(255,214,0,0.06);border:1px solid rgba(255,214,0,0.2);border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#fff3b0;">
    <b>Média dos 5% menores valores:</b> média dos resultados finais na cauda inferior simulada.
    </div>
    """,
        unsafe_allow_html=True,
    )

# Gráfico com algumas trajetórias individuais para ilustrar a dispersão
section_header(ICO_SIGNAL, "Trajetórias Individuais das Simulações", "h2")
n_plot_max = min(50, n_simulations)
n_plot = st.number_input(
    "Número de trajetórias exibidas",
    min_value=5,
    max_value=n_plot_max,
    value=min(20, n_plot_max),
    step=5,
)

fig_individual = go.Figure()

for i in range(n_plot):
    fig_individual.add_trace(
        go.Scatter(
            x=sim_df.index,
            y=sim_df.iloc[:, i],
            mode="lines",
            name=f"Simulação {i + 1}",
            line=dict(width=1),
            opacity=0.4,
        )
    )
fig_individual.update_layout(
    title="Exemplos de Trajetórias Simuladas do Valor do Portfólio",
    xaxis_title="Data",
    yaxis_title="Valor do Portfólio (R$)",
)
apply_plotly_theme(fig_individual)
st.plotly_chart(fig_individual, use_container_width=True)

# Fan chart com percentis
percentis = [5, 25, 50, 75, 95]
fan_chart = sim_df.quantile(q=np.array(percentis) / 100, axis=1).T
fan_chart.columns = [f"P{p}" for p in percentis]

fig_fan = go.Figure()
fig_fan.add_trace(
    go.Scatter(
        x=fan_chart.index,
        y=fan_chart["P95"],
        line=dict(color="rgba(0, 210, 255, 0.05)"),
        showlegend=False,
    )
)
fig_fan.add_trace(
    go.Scatter(
        x=fan_chart.index,
        y=fan_chart["P5"],
        fill="tonexty",
        fillcolor="rgba(0, 210, 255, 0.1)",
        line=dict(color="rgba(0, 210, 255, 0.05)"),
        name="Faixa 5%-95%",
    )
)
fig_fan.add_trace(
    go.Scatter(
        x=fan_chart.index,
        y=fan_chart["P75"],
        line=dict(color="rgba(0, 210, 255, 0.1)"),
        showlegend=False,
    )
)
fig_fan.add_trace(
    go.Scatter(
        x=fan_chart.index,
        y=fan_chart["P25"],
        fill="tonexty",
        fillcolor="rgba(0, 210, 255, 0.25)",
        line=dict(color="rgba(0, 210, 255, 0.1)"),
        name="Faixa 25%-75%",
    )
)
fig_fan.add_trace(
    go.Scatter(
        x=fan_chart.index,
        y=fan_chart["P50"],
        line=dict(color="#00ff87", width=2.5),
        name="Mediana",
    )
)
fig_fan.update_layout(
    title="Simulação Monte Carlo por Ativos - Fan Chart com Faixas de Percentis",
    xaxis_title="Data",
    yaxis_title="Valor do Portfólio (R$)",
)
apply_plotly_theme(fig_fan)
fig_fan.add_hline(
    y=valor_inicial,
    line_dash="dash",
    line_color="#94a3b8",
    line_width=1.5,
    annotation_text=f"Capital Inicial: R$ {valor_inicial:,.0f}",
    annotation_position="bottom right",
    annotation_font=dict(color="#94a3b8", size=11),
)
st.plotly_chart(fig_fan, use_container_width=True)

# Download simulation summary

sim_summary = pd.DataFrame(
    [{"Métrica": k, "Valor": v} for k, v in sim_stats_dict.items()]
)
sim_summary_csv = sim_summary.to_csv(index=False).encode("utf-8-sig")

# Download dos percentis por data
fan_export = fan_chart.copy()
fan_export.index = fan_export.index.strftime("%Y-%m-%d")
fan_export_csv = fan_export.to_csv().encode("utf-8-sig")

col_dl1, col_dl2, _ = st.columns([1, 1, 1])
with col_dl1:
    st.download_button(
        label="⬇ Resumo (CSV)",
        data=sim_summary_csv,
        file_name=f"monte_carlo_resumo_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_dl2:
    st.download_button(
        label="⬇ Percentis por Data (CSV)",
        data=fan_export_csv,
        file_name=f"monte_carlo_percentis_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Histograma valor final
q1 = valores_finais.quantile(0.25)
q2 = valores_finais.quantile(0.50)
q3 = valores_finais.quantile(0.75)

section_header(ICO_FRONTIER, "Distribuição do Valor Final do Portfólio", "h2")
fig_hist = px.histogram(
    x=valores_finais,
    nbins=30,
    title="Distribuição dos Valores Finais da Simulação Monte Carlo",
    labels={"x": "Valor Final do Portfólio (R$)", "y": "Frequência"},
    color_discrete_sequence=["#00d2ff"],
)
fig_hist.update_layout(
    xaxis_title="Valor Final do Portfólio (R$)", yaxis_title="Frequência", bargap=0.05
)

fig_hist.add_vline(
    x=q1,
    line_width=2,
    line_dash="dash",
    line_color="#ff1744",
    annotation_text="Q1 (25%)",
    annotation_position="top left",
)
fig_hist.add_vline(
    x=q2,
    line_width=2.5,
    line_color="#00ff87",
    annotation_text="Mediana (50%)",
    annotation_position="top left",
)
fig_hist.add_vline(
    x=q3,
    line_width=2,
    line_dash="dash",
    line_color="#ffd600",
    annotation_text="Q3 (75%)",
    annotation_position="top left",
)

apply_plotly_theme(fig_hist)
st.plotly_chart(fig_hist, use_container_width=True)

# Estatísticas da distribuição final
estatisticas = {
    "Mínimo": valores_finais.min(),
    "Q1 (25%)": q1,
    "Mediana (50%)": q2,
    "Q3 (75%)": q3,
    "Máximo": valores_finais.max(),
    "Média": valores_finais.mean(),
    "Desvio Padrão": valores_finais.std(),
}
estatisticas_dict = {
    "Mínimo": f"R$ {estatisticas['Mínimo']:,.2f}",
    "Q1 (25%)": f"R$ {estatisticas['Q1 (25%)']:,.2f}",
    "Mediana (50%)": f"R$ {estatisticas['Mediana (50%)']:,.2f}",
    "Q3 (75%)": f"R$ {estatisticas['Q3 (75%)']:,.2f}",
    "Máximo": f"R$ {estatisticas['Máximo']:,.2f}",
    "Média": f"R$ {estatisticas['Média']:,.2f}",
    "Desvio Padrão": f"R$ {estatisticas['Desvio Padrão']:,.2f}",
}
section_header(ICO_METRICS, "Estatísticas da Distribuição Final", "h2")
render_cards_grid(estatisticas_dict)
loading_placeholder.empty()

# Salvar estatísticas da simulação em session_state para uso no relatório
st.session_state["simulation_run"] = True
st.session_state["sim_n_simulations"] = n_simulations
st.session_state["sim_valor_inicial"] = valor_inicial
st.session_state["sim_years"] = years
st.session_state["sim_valor_esperado"] = valor_esperado
st.session_state["sim_var_5"] = var_5
st.session_state["sim_cvar_5"] = cvar_5
st.session_state["sim_pior_cenario"] = pior_cenario
st.session_state["sim_melhor_cenario"] = melhor_cenario
# ── Síntese do Analista (Simulação) ─────────────────────────────────────
st.markdown("---")
analyst_synthesis_header()


# Gera uma leitura descritiva dos cenários, sem recomendação de investimento.
risco_label = (
    "controlado" if perda_var > -30 else ("elevado" if perda_var < -50 else "moderado")
)
ret_label = "positivo" if ret_esperado_pct > 0 else "negativo"

sintese_sim_items = []
if prob_ganho > 70:
    sintese_sim_items.append(
        f'<li style="color:#00ff87;margin-bottom:3px;">Mais de 70% dos cenários simulados terminaram acima do capital inicial ({prob_ganho:.1f}%).</li>'
    )
elif prob_ganho > 50:
    sintese_sim_items.append(
        f'<li style="color:#ffd600;margin-bottom:3px;">Entre 50% e 70% dos cenários simulados terminaram acima do capital inicial ({prob_ganho:.1f}%).</li>'
    )
else:
    sintese_sim_items.append(
        f'<li style="color:#ff3d5a;margin-bottom:3px;">Metade ou menos dos cenários simulados terminou acima do capital inicial ({prob_ganho:.1f}%).</li>'
    )

sintese_sim_items.append(
    f'<li style="color:{"#00ff87" if ret_esperado_pct > 0.05 else "#ffd600" if ret_esperado_pct > 0 else "#ff3d5a"};margin-bottom:3px;">Retorno anual esperado {ret_label}: {ret_esperado_pct * 100:.1f}% a.a. ao longo de {years} ano(s)</li>'
)
sintese_sim_items.append(
    f'<li style="color:{"#00ff87" if risco_label == "controlado" else "#ffd600" if risco_label == "moderado" else "#ff3d5a"};margin-bottom:3px;">Retorno no percentil 5 dos valores finais: {perda_var:.1f}%</li>'
)

amplitude = (melhor_cenario - pior_cenario) / valor_inicial * 100
if amplitude > 200:
    sintese_sim_items.append(
        f'<li style="color:#ffd600;margin-bottom:3px;">Alta dispersão de cenários ({amplitude:.0f}% de amplitude) — portfólio com volatilidade significativa</li>'
    )
else:
    sintese_sim_items.append(
        f'<li style="color:#00ff87;margin-bottom:3px;">Dispersão de cenários controlada ({amplitude:.0f}% de amplitude)</li>'
    )

st.markdown(
    f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;flex-wrap:wrap;gap:0.5rem;">
    <span style="font-weight:700;color:#f8fafc;font-size:0.95rem;">Projeção {years} ano(s) — {n_simulations} simulações</span>
  </div>
  <ul style="margin:0;padding-left:1.1rem;font-size:0.82rem;line-height:1.9;list-style:disc;">
    {"".join(sintese_sim_items)}
  </ul>
  <p style="font-size:0.75rem;color:#94a3b8;margin:0.75rem 0 0;">Modelo normal de retornos logarítmicos multivariados, estimados sobre a janela histórica disponível ({len(aligned_returns)} observações); pesos rebalanceados diariamente. Não inclui taxas nem impostos. Resultados são cenários, não previsões.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Próximo Passo ────────────────────────────────────────────────────────
cenario_desc = (
    f"{prob_ganho:.0f}% dos cenários simulados terminaram acima do capital inicial"
)
next_step_card(
    message=f"Complemente a leitura da projeção com as notícias das empresas — {cenario_desc}.",
    accent="var(--brand-info)",
    cta_label="Abrir Notícias",
    cta_page="pages/3_Notícias.py",
)

st.session_state["sim_estatisticas"] = estatisticas
