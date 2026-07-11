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


def svg_icon(body, size=14):
    """Wrap raw SVG shape markup into a small inline icon."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
        f"{body}</svg>"
    )


def section_header(icon_svg, text, tag="h3"):
    """Render an icon + label heading, e.g. `<h3>` with an inline SVG."""
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f"{icon_svg}<span>{text}</span></{tag}>",
        unsafe_allow_html=True,
    )


def diag_row(icon_svg, text, color):
    """Render a single icon + colored-text diagnostic line."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
        f'color:{color};font-size:0.88rem">{icon_svg}{text}</div>',
        unsafe_allow_html=True,
    )


def render_cards_grid(data_dict, colors_sequence=None):
    """Render a metric grid of `{label: value}` pairs as colored cards."""
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
