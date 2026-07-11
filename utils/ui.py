"""Shared UI helpers for consistent look & feel across pages."""

from contextlib import contextmanager

import streamlit as st


def load_css(path: str = "style.css") -> None:
    """Load a CSS file and inject it into the page via `st.markdown`.

    Silently no-ops if the file is missing, so callers don't need their
    own try/except boilerplate.
    """
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


@contextmanager
def loading_overlay(text: str, tickers=None):
    """Glassmorphic loading animation — drop-in replacement for `st.spinner`.

    Usage: `with loading_overlay("Carregando..."):` instead of
    `with st.spinner("Carregando..."):`.
    """
    placeholder = st.empty()
    chips_html = ""
    if tickers:
        chips = "".join(f'<span class="loading-ticker-chip">{t}</span>' for t in tickers)
        chips_html = f'<div class="loading-tickers">{chips}</div>'
    with placeholder.container():
        st.markdown(
            f"""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">{text}</div>
                {chips_html}
                <div class="loading-bar-track"><div class="loading-bar-fill"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    try:
        yield
    finally:
        placeholder.empty()
