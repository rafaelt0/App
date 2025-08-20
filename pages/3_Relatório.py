import streamlit as st
import pandas as pd
import quantstats as qs

# Botão para gerar PDF via quantstats
import tempfile
st.subheader("Baixar Relatório Completo (QuantStats)")

# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.warning("⚠️ Configure primeiro seu portfólio na aba 1 para liberar a geração do relatório.")
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


with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
      st.download_button(
      label="Baixar Relatório HTML Completo (QuantStats)",
      data=open(tmpfile.name, "rb").read(),
      file_name="relatorio_portfolio.html",
      mime="text/html")
    
      qs.reports.html(
      portfolio_returns,
      benchmark= retorno_bench,
      output=tmpfile.name,
      title="Relatório Completo do Portfólio",
      download_filename="relatorio_portfolio.html")
    
