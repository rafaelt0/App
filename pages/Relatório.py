import streamlit as st
import pandas as pd

# Botão para gerar PDF via quantstats
import tempfile
st.subheader("Baixar Relatório Completo (QuantStats)")

# Converte para formato aceito pelo QuantStats
portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
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
