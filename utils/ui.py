"""Shared UI helpers for consistent look & feel across pages."""

from contextlib import contextmanager
from html import escape
from pathlib import Path

import streamlit as st


def svg_icon(body: str, size: int = 14) -> str:
    """Wrap raw SVG path/shape markup in a small inline `<svg>` icon.

    `body` is the inner SVG markup (paths, circles, etc.); `size` sets the
    icon's width/height in pixels.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false" '
        f'style="vertical-align:-2px;margin-right:5px">'
        f"{body}</svg>"
    )


def section_header(icon_svg: str, text: str, tag: str = "h2") -> None:
    """Render a section title with a leading inline SVG icon."""
    st.markdown(
        f'<{tag} class="ui-section-heading">{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True,
    )


def diag_row(icon_svg: str, text: str, color: str) -> None:
    """Render a one-line diagnostic message with a leading icon."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
        f'color:{color};font-size:0.88rem">{icon_svg}{text}</div>',
        unsafe_allow_html=True,
    )


_CARD_GRID_COLORS = [
    "#38bdf8",
    "#4ade80",
    "#fbbf24",
    "#fb7185",
    "#c084fc",
    "#f472b6",
    "#34d399",
    "#60a5fa",
]


def render_cards_grid(data_dict: dict, colors_sequence=None) -> None:
    """Render a `label -> value` dict as a grid of `.mcard` divs."""
    colors_sequence = colors_sequence or _CARD_GRID_COLORS
    items = list(data_dict.items())
    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{escape(str(lbl))}</div>'
        f'<div class="mcard-value" style="color:{colors_sequence[i % len(colors_sequence)]}">{escape(str(val))}</div></div>'
        for i, (lbl, val) in enumerate(items)
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)


def empty_state_card(icon_svg: str, title: str, message: str, cta_label: str, cta_page: str) -> None:
    """Render a restrained empty state with a separate navigation action."""
    st.markdown(
        f"""
<div class="empty-state-card">
  {icon_svg}
  <div class="empty-state-title">{title}</div>
  <div class="empty-state-message">{message}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.page_link(cta_page, label=cta_label)


def load_css(path: str = "style.css") -> None:
    """Load a CSS file and inject it into the page via `st.markdown`.

    Silently no-ops if the file is missing, so callers don't need their
    own try/except boilerplate.
    """
    try:
        css_path = Path(path)
        if not css_path.is_absolute():
            css_path = Path(__file__).resolve().parent.parent / css_path
        with css_path.open(encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


_FLOW_STEPS = [
    "Análise Fundamentalista",
    "Portfolio",
    "Simulação",
    "Notícias",
    "Valuation",
    "Screener",
]

_ICO_FLOW = svg_icon(
    '<circle cx="5" cy="6" r="2.2" stroke="#829196" stroke-width="1.6"/>'
    '<circle cx="19" cy="18" r="2.2" stroke="#829196" stroke-width="1.6"/>'
    '<path d="M7 7.2c0 4 3 4.6 5 5.8s5 1.8 5 5" stroke="#829196" stroke-width="1.6" '
    'fill="none" stroke-linecap="round"/>',
    12,
)


def _flow_done_step_html(label: str) -> str:
    return (
        '<div class="flow-step flow-step-done">'
        '<span class="flow-step-marker">✓</span>'
        f"<span>{label}</span>"
        "</div>"
    )


def _flow_active_step_html(label: str, num: int) -> str:
    return (
        '<div class="flow-step flow-step-current">'
        f'<span class="flow-step-marker">{num}</span>'
        f"<span>{label}</span>"
        "</div>"
    )


def _flow_pending_step_html(label: str, num: int, opacity: float) -> str:
    del opacity
    return (
        '<div class="flow-step flow-step-pending">'
        f'<span class="flow-step-marker">{num}</span>'
        f"<span>{label}</span>"
        "</div>"
    )


def render_flow_sidebar(active_step: int, pending_opacities=None) -> None:
    """Render the compact analysis flow in the sidebar."""
    opacities = list(pending_opacities or [])
    parts = []
    for i, label in enumerate(_FLOW_STEPS, start=1):
        if i < active_step:
            parts.append(_flow_done_step_html(label))
        elif i == active_step:
            parts.append(_flow_active_step_html(label, i))
        else:
            parts.append(_flow_pending_step_html(label, i, opacities.pop(0) if opacities else 1))
    html = (
        '<div class="flow-sidebar">'
        f'<div class="flow-sidebar-label">{_ICO_FLOW}<span>Fluxo de análise</span></div>'
        f'<div class="flow-steps">{"".join(parts)}</div>'
        "</div>"
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)




@contextmanager
def loading_overlay(text: str, tickers=None):
    """Explicit loading feedback for network and compute work.

    Usage: `with loading_overlay("Carregando…"):` instead of
    `with st.spinner("Carregando…"):`.
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
            '<div class="loading-container" role="status" aria-live="polite">'
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
