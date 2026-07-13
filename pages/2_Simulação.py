import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.charts import apply_plotly_theme
from utils.ui import empty_state_card, load_css, render_flow_sidebar, svg_icon


# CSS customizado
load_css()

render_flow_sidebar(active_step=3, pending_opacities=[0.35, 0.22, 0.12])


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


def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f"{icon_svg}<span>{text}</span></{tag}>",
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
plt.rcParams["figure.facecolor"] = "#080c14"
plt.rcParams["axes.facecolor"] = "#0e1524"
plt.rcParams["text.color"] = "#f8fafc"
plt.rcParams["axes.labelcolor"] = "#94a3b8"
plt.rcParams["xtick.color"] = "#94a3b8"
plt.rcParams["ytick.color"] = "#94a3b8"
plt.rcParams["grid.color"] = "#1e293b"
plt.rcParams["font.family"] = "sans-serif"

# Customização do Plotly para o tema Obsidian Neo-Financial
# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- Background grid lines -->
          <line x1="4" y1="56" x2="56" y2="56" stroke="#1e293b" stroke-width="1"/>
          <line x1="4" y1="42" x2="56" y2="42" stroke="#1e293b" stroke-width="0.5" stroke-dasharray="3 3"/>
          <line x1="4" y1="28" x2="56" y2="28" stroke="#1e293b" stroke-width="0.5" stroke-dasharray="3 3"/>
          <!-- Stochastic path — upper scenario (cyan, faint) -->
          <path d="M8 48 C14 38 18 32 24 38 S34 46 44 22 50 10 56 14"
                stroke="#00d2ff" stroke-width="1.5" stroke-linecap="round" fill="none" opacity="0.45"/>
          <!-- Stochastic path — lower scenario (green, faint) -->
          <path d="M8 50 C16 44 20 40 26 44 S38 50 46 30 52 20 56 24"
                stroke="#00ff87" stroke-width="1.5" stroke-linecap="round" fill="none" opacity="0.45"/>
          <!-- Median trajectory (gold, prominent) -->
          <path d="M8 49 C14 40 20 36 26 41 S38 48 46 26 52 14 56 18"
                stroke="#ffd600" stroke-width="2.5" stroke-linecap="round" fill="none"/>
          <!-- Origin node -->
          <circle cx="8" cy="49" r="3.5" fill="#ffd600" opacity="0.9"/>
          <!-- End nodes -->
          <circle cx="56" cy="14" r="2.5" fill="#00d2ff" opacity="0.7"/>
          <circle cx="56" cy="24" r="2.5" fill="#00ff87" opacity="0.7"/>
          <circle cx="56" cy="18" r="3.5" fill="#ffd600" opacity="0.9"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">Simulação Monte Carlo</h1>
        <p class="page-hero-subtitle">Projete trajetórias de retorno com 1.000+ simulações estocásticas e avalie o espectro de cenários para o seu portfólio.</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        empty_state_card(
            icon_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
                <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
                <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
                <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            </svg>""",
            title="Portfólio não configurado",
            message='Configure primeiro seu portfólio na página <strong style="color:#00ff87">Portfolio</strong> para liberar a Simulação Monte Carlo.',
            cta_label="Ir para Portfolio",
            cta_page="pages/1_Portfolio.py",
        )
        st.stop()

# Recupera as variáveis da aba 1
modo = st.session_state["modo"]
returns = st.session_state["returns"]
pesos_manuais = st.session_state["pesos_manuais"]
peso_manual_df = st.session_state["peso_manual_df"]


with st.form("form_simulacao"):
    n_simulations = st.slider(
        "Número de Simulações",
        10,
        500,
        200,
        help="Quantidade de trajetórias simuladas para o portfólio.",
    )
    valor = st.number_input(
        "Capital Inicial (R$)",
        min_value=100,
        help="Valor inicial investido no portfólio.",
    )
    years = int(
        st.number_input("Anos", min_value=1, help="Horizonte da simulação em anos.")
    )

    submitted = st.form_submit_button(
        "Rodar Simulação", type="primary", use_container_width=True
    )

if not submitted:
    st.info(
        "Configure os parâmetros acima e clique em 'Rodar Simulação' para ver os resultados."
    )
    st.stop()

loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown(
        """
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Rodando simulações Monte Carlo multivariadas...</div>
        <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

n_dias = years * 252  # 252 dias úteis no ano
valor_inicial = valor

# Garante que temos um dicionário de pesos, independente do modo escolhido
if modo == "Alocação Manual":
    pesos_dict = pesos_manuais
else:
    pesos_dict = dict(zip(peso_manual_df.index + ".SA", peso_manual_df["Peso"].values))

# Remove ativos com peso zero (se houver)
pesos_dict = {k: v for k, v in pesos_dict.items() if v > 1e-6}

aligned_returns = returns.loc[:, pesos_dict.keys()].dropna()

pesos = np.array(list(pesos_dict.values()))

log_returns = np.log(1 + aligned_returns)
mu = log_returns.mean().values
cov = log_returns.cov().values

np.random.seed(42)  # para reprodutibilidade

# Simular retornos multivariados normais correlacionados
retornos_simulados = np.random.multivariate_normal(
    mu, cov, size=(n_dias, n_simulations)
)

# Calcular trajetórias para cada ativo em cada simulação
precos_simulados = np.exp(retornos_simulados.cumsum(axis=0))

# Calcular valor do portfólio: soma ponderada dos ativos para cada dia e simulação
valor_portfolio = (precos_simulados * pesos).sum(axis=2) * valor_inicial

# Criar DataFrame para facilitar manipulação e plotagem
datas = pd.date_range(start=datetime.date.today(), periods=n_dias + 1, freq="B")
valor_portfolio = np.vstack([np.ones(n_simulations) * valor_inicial, valor_portfolio])
sim_df = pd.DataFrame(valor_portfolio, index=datas)

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
    "VaR 5%": f"R$ {var_5:,.2f}",
    "CVaR 5%": f"R$ {cvar_5:,.2f}",
    "Pior Cenário": f"R$ {pior_cenario:,.2f}",
    "Melhor Cenário": f"R$ {melhor_cenario:,.2f}",
}

section_header(ICO_CHART, "Estatísticas da Simulação Monte Carlo", "h3")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    ret_str = f"{ret_esperado_pct * 100:.1f}% a.a."
    st.metric(
        "Retorno Anual Esperado",
        ret_str,
        delta="Positivo" if ret_esperado_pct > 0 else "Negativo",
        delta_color="normal" if ret_esperado_pct > 0 else "inverse",
        help="Retorno anual composto implícito no valor esperado mediano.",
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
        "VaR 5% (perda máx.)",
        f"{perda_var:.1f}%",
        delta="Controlado" if perda_var > -30 else "Elevado",
        delta_color="normal" if perda_var > -30 else "inverse",
        help="Em 95% dos cenários, a perda máxima é este percentual.",
    )

render_cards_grid(sim_stats_dict)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    st.markdown(
        """
    <div style="background:rgba(0,210,255,0.06);border:1px solid rgba(0,210,255,0.2);border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#b8eeff;">
    <b>VaR 5%:</b> Valor mínimo do portfólio em 95% dos cenários simulados. Abaixo disso está a zona de risco extremo.
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_exp2:
    st.markdown(
        """
    <div style="background:rgba(255,214,0,0.06);border:1px solid rgba(255,214,0,0.2);border-radius:8px;padding:0.75rem 1rem;font-size:0.85rem;color:#fff3b0;">
    <b>CVaR 5%:</b> Média das perdas nos piores 5% dos cenários — representa o risco em cauda extrema.
    </div>
    """,
        unsafe_allow_html=True,
    )

# Gráfico com algumas trajetórias individuais para ilustrar a dispersão
section_header(ICO_SIGNAL, "Trajetórias Individuais das Simulações", "h3")
n_plot_max = min(50, n_simulations)
n_plot = st.slider(
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
    title="Simulação Monte Carlo por Ativos - Fan Chart com Faixas de Confiança",
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
sim_summary_csv = sim_summary.to_csv(index=False).encode("utf-8")

# Download dos percentis por data
fan_export = fan_chart.copy()
fan_export.index = fan_export.index.strftime("%Y-%m-%d")
fan_export_csv = fan_export.to_csv().encode("utf-8")

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

section_header(ICO_FRONTIER, "Distribuição do Valor Final do Portfólio", "h3")
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
section_header(ICO_METRICS, "Estatísticas da Distribuição Final", "h3")
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

# Gera análise textual dos resultados da simulação
cenario_label = (
    "otimista" if prob_ganho > 70 else ("equilibrado" if prob_ganho > 50 else "adverso")
)
risco_label = (
    "controlado" if perda_var > -30 else ("elevado" if perda_var < -50 else "moderado")
)
ret_label = "positivo" if ret_esperado_pct > 0 else "negativo"

sintese_sim_items = []
if prob_ganho > 70:
    sintese_sim_items.append(
        f'<li style="color:#00ff87;margin-bottom:3px;">Alta probabilidade de ganho ({prob_ganho:.1f}%) — cenário {cenario_label}</li>'
    )
elif prob_ganho > 50:
    sintese_sim_items.append(
        f'<li style="color:#ffd600;margin-bottom:3px;">Probabilidade de ganho moderada ({prob_ganho:.1f}%) — cenário {cenario_label}</li>'
    )
else:
    sintese_sim_items.append(
        f'<li style="color:#ff3d5a;margin-bottom:3px;">Baixa probabilidade de ganho ({prob_ganho:.1f}%) — revise a alocação</li>'
    )

sintese_sim_items.append(
    f'<li style="color:{"#00ff87" if ret_esperado_pct > 0.05 else "#ffd600" if ret_esperado_pct > 0 else "#ff3d5a"};margin-bottom:3px;">Retorno anual esperado {ret_label}: {ret_esperado_pct * 100:.1f}% a.a. ao longo de {years} ano(s)</li>'
)
sintese_sim_items.append(
    f'<li style="color:{"#00ff87" if risco_label == "controlado" else "#ffd600" if risco_label == "moderado" else "#ff3d5a"};margin-bottom:3px;">Risco de cauda {risco_label}: perda máxima esperada de {perda_var:.1f}% no VaR 5%</li>'
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

veredicto_sim = (
    ("FAVORÁVEL", "#00ff87")
    if prob_ganho > 65 and ret_esperado_pct > 0.05
    else ("NEUTRO", "#ffd600")
    if prob_ganho > 45
    else ("DESFAVORÁVEL", "#ff3d5a")
)

st.markdown(
    f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;flex-wrap:wrap;gap:0.5rem;">
    <span style="font-weight:700;color:#f8fafc;font-size:0.95rem;">Projeção {years} ano(s) — {n_simulations} simulações</span>
    <span style="background:rgba(0,0,0,0.3);border:1px solid {veredicto_sim[1]}40;border-radius:6px;padding:0.2rem 0.75rem;font-size:0.72rem;font-weight:800;color:{veredicto_sim[1]};letter-spacing:0.08em;">{veredicto_sim[0]}</span>
  </div>
  <ul style="margin:0;padding-left:1.1rem;font-size:0.82rem;line-height:1.9;list-style:disc;">
    {"".join(sintese_sim_items)}
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

# ── Próximo Passo ────────────────────────────────────────────────────────
cenario_desc = (
    f"cenário {cenario_label} com {prob_ganho:.0f}% de probabilidade de ganho"
)
st.markdown(
    f"""
<div style="background:linear-gradient(135deg,rgba(168,85,247,0.06),rgba(0,210,255,0.03));border:1px solid rgba(168,85,247,0.25);border-radius:14px;padding:1.2rem 1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:0.5rem;">
  <div>
    <div style="font-size:0.72rem;color:#64748b;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">Próximo Passo</div>
    <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;">Abra <span style="color:#a855f7">Notícias</span> na barra lateral</div>
    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">Monitore o sentimento qualitativo — {cenario_desc}</div>
  </div>
  <div style="font-size:1.8rem;opacity:0.6;">→</div>
</div>
""",
    unsafe_allow_html=True,
)

st.session_state["sim_estatisticas"] = estatisticas
