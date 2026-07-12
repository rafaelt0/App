"""Identidade anônima por navegador, usada para isolar dados persistidos
(portfólio, watchlist) por visitante em vez de compartilhá-los globalmente.

Sem login: o id vive na query string da URL. Enquanto a aba/favorito do
navegador mantiver esse parâmetro, o mesmo visitante mantém seus dados;
abrir a URL "limpa" (sem o parâmetro) começa uma identidade nova.
"""
import uuid

import streamlit as st


def get_browser_uid() -> str:
    if "_browser_uid" in st.session_state:
        return st.session_state["_browser_uid"]

    uid = st.query_params.get("uid")
    if not uid:
        uid = uuid.uuid4().hex
        st.query_params["uid"] = uid

    st.session_state["_browser_uid"] = uid
    return uid
