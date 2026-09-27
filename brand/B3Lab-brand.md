# B3Lab — Brand Guide

## Identity

**Name:** B3Lab  
**Tagline:** Análise Quantitativa · B3  
**Concept:** A research laboratory for the Brazilian equity market (B3). The flask icon with candlestick bars captures "scientific analysis of financial data."

`../DESIGN.md` defines the application’s surface tokens, typography, components,
and responsive behavior; this guide preserves the B3Lab identity and assets.

## Logo Files

| File | Use |
|------|-----|
| `logo.svg` | Sidebar header (300×72px); teal flask mark and light wordmark |
| `favicon.svg` | Browser tab / `st.set_page_config(page_icon=...)` |
| `icons/*.svg` | Page navigation icons (24×24, currentColor) |

## Color System

The dark-first interface uses blue-green charcoal surfaces and light text.
Teal, blue, amber, coral, and violet remain reserved for hierarchy and
financial semantics.

| Token | Hex | Use |
|-------|-----|-----|
| `--brand-primary` | `#76d5c5` | Primary action, active state, positive signal |
| `--brand-secondary` | `#91b9e7` | Supporting data, links, secondary emphasis |
| `--brand-accent` | `#e7bd69` | Warnings, yield, return metrics |
| `--brand-danger` | `#ef969d` | Negative signal, errors, risk |
| `--brand-info` | `#b7a3e2` | Secondary guidance and exploratory emphasis |
| `--surface-canvas` | `#0c1518` | Main application background |
| `--surface-panel` | `#142126` | Cards, controls, and chart surfaces |
| `--surface-muted` | `#1c2b30` | Grouped and selected surfaces |
| `--border-default` | `#33464a` | Borders and dividers |
| `--text-primary` | `#eef4f2` | Primary text |
| `--text-secondary` | `#c4d1cf` | Supporting copy and labels |
| `--text-muted` | `#9cafae` | Captions and metadata |
| `--chart-grid` | `#63787c` | Visible chart grid and axes |

Color communicates hierarchy and state, never decoration alone. Surface depth
comes from adjacent dark neutrals and visible borders. Avoid gradients, glow,
shadows, and color-only status indicators.

Page headers reuse these accents for wayfinding: teal for Home, blue for Portfolio,
violet for Simulation, amber for News, coral for Valuation, and blue-violet for
Screener. Color stays on the title, icon, and divider; surfaces remain neutral.

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Page headings | Georgia with system serif fallback | 600–700 | 28–32px |
| Section headings | System UI stack | 600–700 | 18–22px |
| Body | System UI stack | 400 | 14–16px |
| Code / metrics | System monospace stack with tabular numerals | 400–700 | 12–24px |
| Labels | System UI stack | 500–600 | 12–14px, sentence case |

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

- **Do** reserve bright teal for primary actions, active states, and positive signals
- **Do** keep gold/red semantic: returns and warnings / negative and risk states
- **Do** preserve visible focus, responsive touch targets, and reduced-motion support
- **Do** use sentence case for new labels while preserving existing PT-BR copy
- **Don't** use gradients, glow, all-caps eyebrows, or decorative card noise
- **Don't** rely on color alone to communicate a state
- **Don't** force the sidebar open on narrow screens
