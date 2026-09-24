# B3Lab — Brand Guide

## Identity

**Name:** B3Lab  
**Tagline:** Análise Quantitativa · B3  
**Concept:** A research laboratory for the Brazilian equity market (B3). The flask icon with candlestick bars captures "scientific analysis of financial data."

## Logo Files

| File | Use |
|------|-----|
| `logo.svg` | Sidebar header (300×72px) |
| `favicon.svg` | Browser tab / `st.set_page_config(page_icon=...)` |
| `icons/*.svg` | Page navigation icons (24×24, currentColor) |

## Color System

The interface uses a sober dusk palette: navy graphite surfaces with teal,
cornflower blue, amber, coral, and lavender reserved for hierarchy and
financial semantics.

| Token | Hex | Use |
|-------|-----|-----|
| `--brand-primary` | `#61d4c6` | Primary action, active state, positive signal |
| `--brand-secondary` | `#8cb4f2` | Supporting data, links, secondary emphasis |
| `--brand-accent` | `#e7b96b` | Warnings, yield, return metrics |
| `--brand-danger` | `#e58a93` | Negative signal, errors, risk |
| `--brand-info` | `#b7a2e6` | Secondary guidance and exploratory emphasis |
| `--bg-color` | `#0b111a` | App background |
| `--panel-bg` | `#151d2a` | Card and sidebar surface |
| `--panel-raised` | `#1d2938` | Raised controls and selected surfaces |
| `--panel-border` | `#34465b` | Borders and dividers |
| `--text-main` | `#f0f4f8` | Primary text |
| `--text-muted` | `#aebaca` | Secondary and label text |
| `--text-faint` | `#748297` | Metadata and low-priority text |

Color communicates hierarchy and state, never decoration alone. Surfaces remain
flat: no gradients, glow, or color-only status indicators.

Page headers reuse these accents for wayfinding: mint for Home, blue for Portfolio,
lavender for Simulation, amber for News, coral for Valuation, and blue-violet for
Screener. Color stays on the title, icon badge, and divider; surfaces remain neutral.

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Headings | System UI stack | 600–700 | 20–32px |
| Body | System UI stack | 400 | 14–16px |
| Code / metrics | System monospace stack | 400–600 | 12–14px |
| Labels | System UI stack | 500 | 11–12px, sentence case |

## Icons (page set)

| Page | File | Icon concept |
|------|------|-------------|
| Portfólio | `icons/portfolio.svg` | Briefcase + bar chart |
| Simulação | `icons/simulation.svg` | Normal distribution bell curve |
| Notícias | `icons/news.svg` | Document / newspaper |
| Valuation | `icons/valuation.svg` | Price tag + dollar |
| Screener | `icons/screener.svg` | Funnel + search circle |

Icons are decorative unless they carry meaning not already present in adjacent
text. Decorative inline SVGs use `aria-hidden="true"` and
`focusable="false"`; interactive controls still need an accessible label.

## Shared UI Helpers

| Helper | Purpose |
|--------|---------|
| `svg_icon` | Safe inline decorative icon wrapper |
| `render_page_header` | Consistent page title, description, and existing B3Lab icon |
| `section_header` | Consistent semantic section heading |
| `empty_state_card` | Actionable empty and error state |
| `loading_overlay` | Explicit, live loading feedback |

## Integration — Streamlit

```python
st.set_page_config(
    page_title="B3Lab",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# Sidebar logo
with st.sidebar:
    st.image("logo.svg", use_column_width=True)
```

## Dos and Don'ts

- **Do** reserve mint for primary actions, active states, and positive signals
- **Do** keep gold/red semantic: returns and warnings / negative and risk states
- **Do** preserve visible focus, responsive touch targets, and reduced-motion support
- **Do** use sentence-case labels and plain user-facing copy
- **Don't** use gradients, glow, all-caps eyebrows, or decorative card noise
- **Don't** rely on color alone to communicate a state
- **Don't** force the sidebar open on narrow screens
