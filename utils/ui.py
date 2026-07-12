"""Shared UI helpers for consistent look & feel across pages."""

from contextlib import contextmanager

import streamlit as st


def svg_icon(body: str, size: int = 14) -> str:
    """Wrap raw SVG path/shape markup in a small inline `<svg>` icon.

    `body` is the inner SVG markup (paths, circles, etc.); `size` sets the
    icon's width/height in pixels.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
        f"{body}</svg>"
    )


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


_FLOW_STEPS = [
    {"label": "Análise Fundamentalista", "color": "#00ff87", "shadow": "0,255,135", "text_color": "#080c14"},
    {"label": "Portfolio", "color": "#00d2ff", "shadow": "0,210,255", "text_color": "#080c14"},
    {"label": "Simulação", "color": "#ffd600", "shadow": "255,214,0", "text_color": "#080c14"},
    {"label": "Notícias", "color": "#a855f7", "shadow": "168,85,247", "text_color": "#fff"},
]

_FLOW_DIVIDER = '    <div style="width:1px;height:12px;background:#1e293b;margin-left:11px;"></div>\n'

_ICO_FLOW = svg_icon(
    '<circle cx="5" cy="6" r="2.2" stroke="#64748b" stroke-width="1.6"/>'
    '<circle cx="19" cy="18" r="2.2" stroke="#64748b" stroke-width="1.6"/>'
    '<path d="M7 7.2c0 4 3 4.6 5 5.8s5 1.8 5 5" stroke="#64748b" stroke-width="1.6" '
    'fill="none" stroke-linecap="round"/>',
    12,
)


def _flow_done_step_html(step):
    return (
        '    <div style="display:flex;align-items:center;gap:0.6rem;">\n'
        '      <div style="width:22px;height:22px;border-radius:50%;background:#1e293b;border:1.5px solid #00ff87;display:flex;align-items:center;justify-content:center;flex-shrink:0;">\n'
        '        <span style="font-size:0.7rem;color:#00ff87;">✓</span>\n'
        '      </div>\n'
        f'      <span style="font-size:0.8rem;font-weight:600;color:#475569;">{step["label"]}</span>\n'
        '    </div>\n'
    )


def _flow_active_step_html(step, num):
    return (
        '    <div style="display:flex;align-items:center;gap:0.6rem;">\n'
        f'      <div style="width:22px;height:22px;border-radius:50%;background:{step["color"]};display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 0 8px rgba({step["shadow"]},0.4);">\n'
        f'        <span style="font-size:0.65rem;font-weight:800;color:{step["text_color"]};">{num}</span>\n'
        '      </div>\n'
        f'      <span style="font-size:0.8rem;font-weight:700;color:{step["color"]};">{step["label"]}</span>\n'
        '    </div>\n'
    )


def _flow_pending_step_html(step, num, opacity):
    return (
        f'    <div style="display:flex;align-items:center;gap:0.6rem;opacity:{opacity};">\n'
        '      <div style="width:22px;height:22px;border-radius:50%;background:#1e293b;border:1.5px solid #334155;display:flex;align-items:center;justify-content:center;flex-shrink:0;">\n'
        f'        <span style="font-size:0.65rem;font-weight:700;color:#64748b;">{num}</span>\n'
        '      </div>\n'
        f'      <span style="font-size:0.8rem;font-weight:600;color:#64748b;">{step["label"]}</span>\n'
        '    </div>\n'
    )


def render_flow_sidebar(active_step: int, pending_opacities=None) -> None:
    """Render the "Fluxo de Análise" step-tracker in the sidebar.

    `active_step` is 1-indexed (1=Análise Fundamentalista, 2=Portfolio,
    3=Simulação, 4=Notícias). Steps before it render as done (checkmark),
    the active step is highlighted, and steps after it render as pending,
    using the opacities in `pending_opacities` (one value per pending step,
    in order) to match each page's original fade-out styling.
    """
    opacities = list(pending_opacities or [])
    parts = []
    for i, step in enumerate(_FLOW_STEPS, start=1):
        if i > 1:
            parts.append(_FLOW_DIVIDER)
        if i < active_step:
            parts.append(_flow_done_step_html(step))
        elif i == active_step:
            parts.append(_flow_active_step_html(step, i))
        else:
            parts.append(_flow_pending_step_html(step, i, opacities.pop(0)))
    body = "".join(parts)
    html = (
        '<div style="padding:1rem 0 0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem;">\n'
        f'  <div style="display:flex;align-items:center;gap:6px;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;margin-bottom:0.75rem;">{_ICO_FLOW} Fluxo de Análise</div>\n'
        '  <div style="display:flex;flex-direction:column;gap:0.35rem;">\n'
        f'{body}'
        '  </div>\n'
        '</div>\n'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


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
        # Built as a single unindented line — an indented multi-line f-string
        # can leave a whitespace-only line where chips_html is empty, which
        # breaks CommonMark's HTML-block detection and makes markdown render
        # the remaining tags as literal text instead of passing them through.
        html = (
            '<div class="loading-container">'
            '<div class="loading-spinner"></div>'
            f'<div class="loading-text">{text}</div>'
            f"{chips_html}"
            '<div class="loading-bar-track"><div class="loading-bar-fill"></div></div>'
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()
