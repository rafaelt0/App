import streamlit as st
import pandas as pd
import quantstats as qs
import base64

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
import tempfile
# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- Document body -->
          <rect x="10" y="4" width="40" height="52" rx="5" fill="#0e1524" stroke="#1e293b" stroke-width="1.5"/>
          <!-- Fold corner -->
          <path d="M38 4 L50 16 L38 16 Z" fill="#080c14" stroke="#1e293b" stroke-width="1"/>
          <!-- Header line (green accent) -->
          <line x1="16" y1="24" x2="44" y2="24" stroke="#00ff87" stroke-width="2.5" stroke-linecap="round"/>
          <!-- Body text lines -->
          <line x1="16" y1="31" x2="40" y2="31" stroke="#334155" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="16" y1="36" x2="36" y2="36" stroke="#334155" stroke-width="1.5" stroke-linecap="round"/>
          <!-- Mini bar chart (bottom of doc) -->
          <rect x="16" y="47" width="5" height="7"  rx="1.5" fill="#00ff87"/>
          <rect x="24" y="43" width="5" height="11" rx="1.5" fill="#00d2ff"/>
          <rect x="32" y="45" width="5" height="9"  rx="1.5" fill="#ffd600"/>
          <rect x="40" y="40" width="5" height="14" rx="1.5" fill="#a855f7"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">Relatório Completo do Portfólio</h1>
        <p class="page-hero-subtitle">Exporte uma análise quantitativa completa via QuantStats — com Sharpe, Drawdowns, Retornos Anuais e muito mais.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.warning("Configure primeiro seu portfólio na aba Portfolio para liberar a geração do relatório.")
        st.stop()

modo = st.session_state["modo"]
returns = st.session_state["returns"]
pesos_manuais = st.session_state["pesos_manuais"]
peso_manual_df = st.session_state["peso_manual_df"]
portfolio_returns = st.session_state["portfolio_returns"]
retorno_bench = st.session_state["retorno_bench"]

          
# Converte para formato aceito pelo QuantStats
portfolio_returns.index = pd.to_datetime(portfolio_returns.index, errors='coerce')
portfolio_returns = portfolio_returns.tz_localize(None)  # Remove timezone

# Cria arquivo temporário
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
    tmp_path = tmpfile.name

# Carregamento página
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Gerando relatório estatístico detalhado (QuantStats)... Isso pode levar alguns segundos.</div>
    </div>
    """, unsafe_allow_html=True)

qs.reports.html(
    portfolio_returns,
    benchmark=retorno_bench,
    output=tmp_path,
    title="Relatório Completo do Portfólio",
    download_filename="relatorio_portfolio.html")

loading_placeholder.empty()
    
# Botão para download HTML
with open(tmp_path, "rb") as f:
    html_data = f.read()

# Gera PDF resumo
from fpdf import FPDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Relatorio Resumo do Portfolio', 0, 1, 'C')
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Pesos da Alocacao', 0, 1)
pdf.set_font('Arial', '', 10)
for idx, row in peso_manual_df.iterrows():
    val = row.iloc[0] if isinstance(row, pd.Series) else row
    # Se o valor for a proporção, x100, se for %
    if "Pesos Otimizados" in peso_manual_df.columns:
        val = row["Pesos Otimizados"]
    elif "Pesos" in peso_manual_df.columns:
        val = row["Pesos"]
    if isinstance(val, (int, float)):
        pdf.cell(0, 6, f"{idx}: {val*100:.2f}%", 0, 1)
    else:
        pdf.cell(0, 6, f"{idx}: {val}", 0, 1)

pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Metricas de Performance', 0, 1)
pdf.set_font('Arial', '', 10)

metrics_df = qs.reports.metrics(portfolio_returns, benchmark=retorno_bench, display=False)
if isinstance(metrics_df, pd.DataFrame):
    # O dataframe tem colunas Benchmark e Strategy
    pdf.cell(60, 6, "Metrica", border=1)
    if "Benchmark" in metrics_df.columns:
        pdf.cell(40, 6, "Benchmark", border=1)
    if "Strategy" in metrics_df.columns:
        pdf.cell(40, 6, "Strategy", border=1)
    pdf.ln()
    for idx, row in metrics_df.iterrows():
        idx_name = str(idx).replace('﹪', '%').encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(60, 6, idx_name, border=1)
        if "Benchmark" in metrics_df.columns:
            bench_val = str(row["Benchmark"]).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(40, 6, bench_val, border=1)
        if "Strategy" in metrics_df.columns:
            strat_val = str(row["Strategy"]).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(40, 6, strat_val, border=1)
        pdf.ln()

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
    pdf_path = tmp_pdf.name
pdf.output(pdf_path)

with open(pdf_path, "rb") as f:
    pdf_data = f.read()

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Baixar Relatório Completo (HTML)",
        data=html_data,
        file_name="relatorio_portfolio.html",
        mime="text/html"
    )
with col2:
    st.download_button(
        label="Baixar Resumo (PDF)",
        data=pdf_data,
        file_name="resumo_portfolio.pdf",
        mime="application/pdf"
    )
