"""Shared UI helpers for consistent look & feel across pages."""

from contextlib import contextmanager
from html import escape
from pathlib import Path

from urllib.parse import parse_qsl, urlsplit

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


_PAGE_HEADER_CONFIG = {
    "home": ("favicon.svg", "Main_Page"),
    "portfolio": ("icons/portfolio.svg", "Portfolio"),
    "simulation": ("icons/simulation.svg", "Simulação"),
    "news": ("icons/news.svg", "Notícias"),
    "valuation": ("icons/valuation.svg", "Visão_de_mercado"),
    "screener": ("icons/screener.svg", "Screener"),
}


def render_page_header(title: str, subtitle: str, icon: str) -> None:
    """Render a shared page heading and mark its sidebar navigation link."""
    icon_file, page_path = _PAGE_HEADER_CONFIG[icon]
    icon_path = Path(__file__).resolve().parent.parent / icon_file
    icon_svg = icon_path.read_text(encoding="utf-8")
    st.markdown(
        f"""
<style>
[data-testid="stSidebarNavLink"][href$="/{page_path}"] {{
  border-left-color: var(--brand-primary) !important;
  background: rgba(97, 212, 198, 0.12) !important;
}}
[data-testid="stSidebarNavLink"][href$="/{page_path}"] span {{
  color: var(--brand-primary) !important;
  font-weight: 650 !important;
}}
</style>
<header class="page-hero" data-page="{escape(icon)}" aria-labelledby="page-title">
  <span class="page-hero-icon" aria-hidden="true">{icon_svg}</span>
  <div class="page-hero-content">
    <h1 class="page-hero-title" id="page-title">{escape(title)}</h1>
    <p class="page-hero-subtitle">{escape(subtitle)}</p>
  </div>
</header>
""",
        unsafe_allow_html=True,
    )



def analyst_synthesis_header() -> None:
    """Render the shared analyst-synthesis heading without fragile inline CSS."""
    st.markdown(
        """
<h3 class="analyst-synthesis-heading">
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
       fill="none" aria-hidden="true" focusable="false">
    <circle cx="12" cy="12" r="10" stroke="#a855f7" stroke-width="1.8"/>
    <path d="M12 8v4l3 3" stroke="#a855f7" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span class="analyst-synthesis-title">Síntese do Analista</span>
</h3>
""",
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

def next_step_card(
    message: str,
    accent: str,
    cta_label: str,
    cta_page: str,
    cta_url: str | None = None,
) -> None:
    """Render a compact, clickable hand-off to the next analysis page."""
    st.markdown(
        f"""
<div class="next-step-card" style="--next-step-accent:{escape(accent)}">
  <div class="next-step-eyebrow">Próximo passo</div>
  <div class="next-step-title">Continue sua análise</div>
  <div class="next-step-message">{escape(message)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if cta_url:
        cta_target = urlsplit(cta_url)
        hidden_inputs = "".join(
            f'<input type="hidden" name="{escape(name)}" value="{escape(value)}">'
            for name, value in parse_qsl(cta_target.query, keep_blank_values=True)
        )
        st.html(
            f'<form class="next-step-form" action="{escape(cta_target.path)}" method="get">'
            f"{hidden_inputs}"
            f'<button class="next-step-link" type="submit">{escape(cta_label)}</button>'
            "</form>"
        )
    else:
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


@contextmanager
def loading_overlay(text: str, tickers=None):
    """Explicit loading feedback for network and compute work.

    Usage: `with loading_overlay("Carregando…"):` instead of
    `with st.spinner("Carregando…"):`.
    """
    placeholder = st.empty()
    chips_html = ""
    if tickers:
        chips = "".join(
            f'<span class="loading-ticker-chip">{escape(str(t))}</span>' for t in tickers
        )
        chips_html = f'<div class="loading-tickers">{chips}</div>'
    with placeholder.container():
        # Built as a single unindented line — an indented multi-line f-string
        # can leave a whitespace-only line where chips_html is empty, which
        # breaks CommonMark's HTML-block detection and makes markdown render
        # the remaining tags as literal text instead of passing them through.
        html = (
            '<div class="loading-container" role="status" aria-live="polite">'
            '<div class="loading-spinner"></div>'
            f'<div class="loading-text">{escape(str(text))}</div>'
            f"{chips_html}"
            '<div class="loading-bar-track"><div class="loading-bar-fill"></div></div>'
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()
