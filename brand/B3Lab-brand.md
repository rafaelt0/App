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
| `--surface-canvas` | `#0e1015` | Main application background |
| `--sidebar-bg` | `#12151c` | Navigation rail |
| `--surface-panel` | `#161a22` | Cards, controls, and chart surfaces |
| `--surface-muted` | `#1d222c` | Grouped and selected surfaces |
| `--surface-raised` | `#252b37` | Menus, tooltips, hover |
| `--border-default` | `#2b313d` | Borders and dividers |
| `--border-strong` | `#5e687e` | Input and control outlines |
| `--text-primary` | `#f3f5f9` | Primary text |
| `--text-secondary` | `#c5cad6` | Supporting copy and labels |
| `--text-muted` | `#9aa2b4` | Captions and metadata |
| `--chart-grid` | `#646e88` | Visible chart grid and axes |
| `--text-on-primary` | `#0e1015` | Text on indigo-filled actions |

Color communicates hierarchy and state, never decoration alone. Surface depth
comes from adjacent dark neutrals and visible borders. Avoid gradients, glow,
shadows, and color-only status indicators.

All pages share the indigo accent; wayfinding comes from icon, title, and the active navigation link.

## Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Page headings | System sans stack | 700 | clamp(26–30px) |
| Section headings | System sans stack | 650 | 18–20px |
| Body | System sans stack | 400 | 14–16px |
| Financial values | System sans stack with tabular numerals | 600–700 | 14–24px |
| Chips / code | System monospace stack | 400–600 | 12px |
| Labels | System sans stack | 500–600 | 13px, sentence case |

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
