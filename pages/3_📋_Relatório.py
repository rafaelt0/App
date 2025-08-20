import streamlit as st
import pandas as pd
import quantstats as qs
import base64

# Botão para gerar PDF via quantstats
import tempfile
# Título da página
st.markdown("<h1 style='text-align: center;'>Baixar Relatório do Portfolio (Quantstats) 📝</h1>", unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# converte a imagem para base64
img_base64 = get_base64_of_bin_file("Relatorio.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

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

# Cria arquivo temporário
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
    tmp_path = tmpfile.name

# Carregamento página
with st.spinner("Gerando relatório... Isso pode levar alguns segundos. ⏳"):
    qs.reports.html(
    portfolio_returns,
    benchmark=retorno_bench,
    output=tmp_path,
    title="Relatório Completo do Portfólio",
    download_filename="relatorio_portfolio.html")
    
# Botão para download
with open(tmp_path, "rb") as f:
    st.download_button(
    label="Baixar Relatório HTML Completo (QuantStats)",
    data=f.read(),
    file_name="relatorio_portfolio.html",
    mime="text/html")

    
