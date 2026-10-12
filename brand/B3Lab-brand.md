# B3Lab brand guide

## Identity

**Name:** B3Lab  
**Tagline:** Análise quantitativa · B3  
**Concept:** A research laboratory for the Brazilian equity market (B3). The flask icon with candlestick bars captures "scientific analysis of financial data."

`../DESIGN.md` defines the application’s surface tokens, typography, components,
and responsive behavior; this guide preserves the B3Lab identity and assets.

## Logo Files

| File | Use |
|------|-----|
| `frontend/public/logo.svg` | Sidebar header; indigo flask, green/coral candles, light wordmark |
| `frontend/public/favicon.svg` | Browser tab and Home identity mark |
| `frontend/public/icons/*.svg` | Page icons, 24px, currentColor |

## Color System

The dark-first interface uses cool neutral dark surfaces and light text. A
single indigo accent marks brand, primary actions, and active state; green,
coral, and amber are reserved for financial semantics.

| Token | Hex | Use |
|-------|-----|-----|
| `--brand-primary` | `#8b93ff` | Primary action, active state, focus, first series |
| `--brand-positive` | `#3ecf8e` | Gains, favorable and healthy states |
| `--brand-secondary` | `#4cc9f0` | Supporting data, benchmarks, info |
| `--brand-accent` | `#f5b544` | Warnings, yield, return metrics |
| `--brand-danger` | `#f2757a` | Negative signal, errors, risk |
| `--brand-info` | `#e879c6` | Categorical and exploratory emphasis |
| `--surface-app` | `#0a0b0f` | Shell behind the inset frame |
| `--surface-canvas` | `#0f1116` | Inset content frame |
| `--sidebar-bg` | `#0a0b0f` | Sidebar, mobile header and menu |
| `--surface-panel` | `#15171d` | Tiles, ticker cards, table headers |
| `--surface-muted` | `#1b1e25` | Buttons, chips, badges |
| `--surface-raised` | `#23262f` | Menus, palette, tooltips, hover |
| `--border-default` | `rgb(255 255 255 / .085)` | Borders and dividers |
| `--border-strong` | `#5e687e` | Input and control outlines |
| `--text-primary` | `#f3f5f9` | Primary text |
| `--text-secondary` | `#c5cad6` | Supporting copy and labels |
| `--text-muted` | `#9aa2b4` | Captions and metadata |
| `--chart-grid` | `#646e88` | Visible chart grid and axes |
| `--text-on-primary` | `#0e1015` | Text on indigo-filled actions |

Color communicates hierarchy and state, never decoration alone. The shell is an
inset frame on the darkest surface; headline tiles and ticker cards sit on
layered surfaces with hairline borders, while prose, tables, and charts stay on
the canvas. Avoid gradients, glow, drop shadows, and color-only status
indicators. `DESIGN.md` lists the full surface ladder.

All pages share the indigo accent; wayfinding comes from icon, title, and the active navigation link.

The indigo ramp uses `--indigo-50` through `--indigo-950`, with the original
`#8b93ff` at `--indigo-400`. Primary actions use `--brand-primary`,
`--brand-primary-hover`, and `--brand-primary-active`. Selected backgrounds
use `--brand-primary-surface`; decorative icon borders use
`--brand-primary-border`. The [indigo ramp reference](../DESIGN.md#indigo-ramp)
lists every shade and its contrast constraints.

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Page headings | IBM Plex Sans | 500 | clamp(26–32px) |
| Section headings | IBM Plex Sans | 500 | 18–20px |
| Body | IBM Plex Sans | 400 | 14–16px |
| Financial values | IBM Plex Sans with tabular numerals | 500–650 | 14–30px |
| Tickers / kbd / code | IBM Plex Mono | 400–600 | 11–13px |
| Labels | IBM Plex Sans | 500–600 | 13px, sentence case |

IBM Plex Sans and IBM Plex Mono are bundled in `frontend/app/fonts/` and served
locally through `next/font/local`. Charts inherit the interface font. Numbers
stay in Plex Sans with tabular numerals; Plex Mono sets tickers, keyboard
hints, formulas, and chart value tags.

## Icons (page set)

| Page | File | Icon concept |
|------|------|-------------|
| Portfólio | `frontend/public/icons/portfolio.svg` | Briefcase and bar chart |
| Simulação | `frontend/public/icons/simulation.svg` | Normal distribution bell curve |
| Notícias | `frontend/public/icons/news.svg` | Newspaper |
| Valuation | `frontend/public/icons/valuation.svg` | Price tag and dollar |
| Screener | `frontend/public/icons/screener.svg` | Funnel and search circle |

Icons are decorative unless they carry meaning not already present in adjacent
text. Decorative inline SVGs use `aria-hidden="true"` and
`focusable="false"`; interactive controls still need an accessible label.

## Shared React components

| Component | Purpose |
|---|---|
| `Icon` | Decorative SVG assets and shared glyphs |
| `PageHeader` | Page title, description, visible identity icon |
| `Section` | Semantic section heading |
| `EmptyState` | Actionable empty and error state |
| `Loading` | Actual operation label and accessible status |
| `Metrics`, `Table` | Neutral values, explicit units, semantic assessments |

## Next.js integration

`frontend/app/layout.tsx` owns metadata, language, and the shared application
shell. `frontend/components/ui.tsx` owns the presentation components.
Page icons retain their existing SVG geometry and inherit the brand color.
Financial classifications come from the Python API.

```tsx
<PageHeader
  title="Otimização de portfólio"
  description="Monte uma carteira e compare risco e retorno."
  icon="portfolio"
/>
```

## Dos and Don'ts

- **Do** use indigo for primary actions and active state, never gains.
- **Do** use green for gains, amber for warnings, and coral for losses and risk.
- **Do** preserve visible focus, responsive touch targets, and reduced-motion support
- **Do** use sentence case for new labels while preserving existing PT-BR copy
- **Don't** use gradients, glow, all-caps eyebrows, or decorative card noise
- **Don't** rely on color alone to communicate a state
- **Don't** force the sidebar open on narrow screens
