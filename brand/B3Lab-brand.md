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

| Token | Hex | Use |
|-------|-----|-----|
| `--brand-primary` | `#00ff87` | Bullish signal, CTA, active state |
| `--brand-secondary` | `#00d2ff` | Data highlights, links |
| `--brand-accent` | `#ffd600` | Warnings, gold/return metrics |
| `--brand-danger` | `#ff3d5a` | Bearish signal, errors |
| `--bg-color` | `#080c14` | App background (deep space) |
| `--panel-bg` | `#0e1524` | Card / sidebar background |
| `--text-main` | `#f8fafc` | Primary text |
| `--text-muted` | `#94a3b8` | Secondary / label text |

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Headings | Space Grotesk | 700 | 20–32px |
| Body | Space Grotesk | 400 | 14–16px |
| Code / metrics | JetBrains Mono | 400–500 | 12–14px |
| Labels | Space Grotesk | 500 | 11–12px (caps, 0.08em tracking) |

## Icons (page set)

| Page | File | Icon concept |
|------|------|-------------|
| Portfólio | `icons/portfolio.svg` | Briefcase + bar chart |
| Simulação | `icons/simulation.svg` | Normal distribution bell curve |
| Notícias | `icons/news.svg` | Document / newspaper |
| Valuation | `icons/valuation.svg` | Price tag + dollar |
| Screener | `icons/screener.svg` | Funnel + search circle |

All icons are 24×24, `stroke="currentColor"`, no fill. Color them via CSS:
```css
color: var(--brand-primary);  /* neon green */
```

## Utility Classes (style.css)

| Class | Purpose |
|-------|---------|
| `.b3lab-gradient-text` | Gradient text (green → cyan) |
| `.b3lab-badge` | Green pill badge |
| `.b3lab-badge-blue / -gold / -red` | Color variants |
| `.b3lab-stat` | Metric card with top gradient accent bar |

## Integration — Streamlit

```python
# Main_Page.py — set_page_config
st.set_page_config(
    page_title="B3Lab",
    page_icon="favicon.svg",
    layout="wide",
)

# Sidebar logo
with st.sidebar:
    st.image("logo.svg", use_column_width=True)
```

## Dos and Don'ts

- **Do** use neon green (`#00ff87`) only for positive/buy signals and active states
- **Do** use red (`#ff3d5a`) only for negative/sell signals and errors  
- **Do** use gold (`#ffd600`) for returns, yield, alerts
- **Don't** put the logo on a light background
- **Don't** change font weights — Space Grotesk 700 for headings, 400 for body
