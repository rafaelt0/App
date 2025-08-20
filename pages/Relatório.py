import streamlit as st
import pandas as pd
import quantstats as qs

# Botão para gerar PDF via quantstats
import tempfile
st.subheader("Baixar Relatório Completo (QuantStats)")

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
      qs.reports.html(
        portfolio_returns,
        benchmark= retorno_bench,
        output=tmpfile.name,
        title="Relatório Completo do Portfólio",
        download_filename="relatorio_portfolio.html"
    )
   
     st.download_button(
        label="Baixar Relatório HTML Completo (QuantStats)",
        data=open(tmpfile.name, "rb").read(),
        file_name="relatorio_portfolio.html",
        mime="text/html"
    )
