# B3Lab fintech desk

## Atmosphere and identity

B3Lab is a dark-first research terminal for Brazilian equities. It borrows its
structure from professional terminals (Koyfin, TradingView, Bloomberg) and its
finish from crafted product UIs (Linear, Vercel, Raycast): layered cool-neutral
surfaces, hairline borders, one vivid indigo accent, IBM Plex Sans with precise
tabular numerals, and IBM Plex Mono for tickers. The result should read as a
serious analysis tool with a recognizable identity, not a marketing page.

The app shell is an inset frame: the sidebar sits on the darkest surface and
the page content sits in a slightly lighter, rounded panel with a 1px border.
Headline numbers, ticker cards, criteria, and article cards sit on tile
surfaces; prose, tables, and the primary task form stay on the canvas. The
brand color marks primary actions, active state, focus, selection, the first
chart series, and the activity line. Financial semantics use their own
colors: green for gains, coral red for losses, and amber for warnings, always
paired with a sign, glyph, or label.

Keep the B3Lab name, flask-and-candlestick mark, Brazilian Portuguese copy, and
current financial behavior. Bundle fonts locally. Do not add gradients, glow,
blur, drop shadows, decorative texture, remote font services, fake live data
(invented quotes, flashing prices), or color-only status.

The market strip under the workspace bar is the one scrolling element. It shows
only provider values from `/api/market/strip`, focused on B3 and the Brazilian
market: delayed Yahoo Finance closes for B3 indices, the dollar, and liquid B3
stocks with a signed daily change and a ▲ or ▼ glyph, then BCB SGS macro rates
in percent. Each item's title names its source and reference date. A failed source is listed
as unavailable, never filled in. The strip pauses on hover, focus, and its
pause button, and becomes a static, horizontally scrollable row under reduced
motion.

The Next.js interface defaults to dark. Tokens in `frontend/app/globals.css`,
the chart theme in `frontend/components/Chart.tsx`, and the `CHART_*` constants
in `utils/charts.py` use the same palette. Verify rendered colors on the live
surface.

## Color tokens

| Role | Token | Value | Use |
|---|---|---|---|
| App background | `--surface-app` | `#0A0B0F` | Shell and sidebar behind the inset frame |
| Canvas | `--surface-canvas` | `#0F1116` | Inset content frame, tables, charts |
| Input well | `--surface-input` | `#0B0C10` | Text inputs and selects |
| Mobile navigation | `--sidebar-bg` | `#0A0B0F` | Sidebar and mobile header |
| Tile | `--surface-panel` | `#15171D` | Metric tiles, ticker cards, criteria, table headers |
| Grouped surface | `--surface-muted` | `#1B1E25` | Buttons, chips, badges |
| Raised surface | `--surface-raised` | `#23262F` | Menus, palette, tooltips, button hover, meter tracks |
| Primary text | `--text-primary` | `#F3F5F9` | Headings, body copy, and values |
| Supporting text | `--text-secondary` | `#C5CAD6` | Labels and supporting copy |
| Muted text | `--text-muted` | `#9AA2B4` | Captions, metadata, tile labels |
| Subtle rule | `--border-subtle` | `rgb(255 255 255 / .055)` | Table rows and in-tile dividers |
| Default border | `--border-default` | `rgb(255 255 255 / .085)` | Tile, table, and frame borders |
| Hover border | `--border-hover` | `rgb(255 255 255 / .14)` | Hovered tiles, secondary buttons |
| Strong border | `--border-strong` | `#5E687E` | Input and control outlines (3:1) |
| Hairline highlight | `--highlight` | `inset 0 1px 0 rgb(255 255 255 / .04)` | Top edge of tiles; `--highlight-strong` for overlays |
| Brand | `--brand-primary` | `#8B93FF` | Primary actions, active state, focus, first series |
| Brand tint | `--brand-tint` | `rgb(139 147 255 / .13)` | Selected rows, active criteria, chips |
| Supporting data | `--brand-secondary` | `#4CC9F0` | Benchmarks, info alerts, neutral sentiment |
| Warning / return | `--brand-accent` | `#F5B544` | Caution and return semantics, favorite star |
| Negative / risk | `--brand-danger` | `#F2757A` | Loss, risk, and errors |
| Guidance | `--brand-info` | `#E879C6` | Categorical and exploratory emphasis |
| Positive | `--brand-positive` | `#3ECF8E` | Gains, favorable and healthy states |
| Semantic tints | `--positive-tint`, `--danger-tint`, `--accent-tint`, `--secondary-tint` | hue at 12% | Status pills and delta chips |
| Text on primary | `--text-on-primary` | `#0E1015` | Text on indigo-filled actions |

Separators are translucent white so they read the same on every surface; any
outline that identifies a control keeps the opaque `--border-strong`. Muted
text stays at least 5.9:1 on the raised surface and every accent is at least
5.1:1 on it. Status colors are always paired with a sign, glyph, or label.
Never use `--brand-primary` to mean "positive"; use `--brand-positive` /
`CHART_POSITIVE`. Green and coral are hard to separate under deuteranopia, so
gains and losses always print their sign and an arrow or check.

### Indigo ramp

The indigo ramp in `frontend/app/globals.css` runs from light `50` to dark
`950`. The existing brand color is `--indigo-400`. Use semantic tokens in
components so primary states and selected surfaces share the same shades.

| Token | Value | Current role |
|---|---|---|
| `--indigo-50` | `#F1F2FF` | Lightest shade |
| `--indigo-100` | `#E1E3FF` | Light shade |
| `--indigo-200` | `#BEC3FF` | Light shade |
| `--indigo-300` | `#A8AFFF` | `--brand-primary-hover` |
| `--indigo-400` | `#8B93FF` | `--brand-primary` |
| `--indigo-500` | `#7078E8` | `--brand-primary-active`, pressed primary actions |
| `--indigo-600` | `#565FC8` | Dark shade |
| `--indigo-700` | `#444CA3` | `--brand-primary-border`, decorative icon borders |
| `--indigo-800` | `#343B78` | Dark shade |
| `--indigo-900` | `#282E57` | Dark shade |
| `--indigo-950` | `#252A41` | `--brand-primary-surface`, selected control backgrounds |

Primary buttons and links use `--text-on-primary` in their default, hover,
and pressed states. Their text contrast is 6.99:1, 9.28:1, and 5.02:1,
respectively. Indigo icons on the selected surface have 5.19:1 contrast.
Keep input outlines and focus rings on `--brand-primary`. The darker ramp
shades are fills and decorative borders; check contrast before using them
for text or control outlines. Keep selection labels, borders, and
`aria-current` or `aria-pressed` alongside the selected background.

## Typography

| Role | Font | Size | Weight | Line height |
|---|---|---|---:|---:|
| Page title | IBM Plex Sans | `clamp(26px–32px)`, −0.024em | 600 | 1.15 |
| Section heading | IBM Plex Sans | 17–19px, −0.012em | 550 | 1.3 |
| Body | IBM Plex Sans | 14–16px | 400 | 1.5 |
| Labels | IBM Plex Sans | 12.5–13px | 500 | 1.4 |
| Headline values | IBM Plex Sans, `--numeric` | 26–36px, −0.022em | 500 | 1.1 |
| Table values | IBM Plex Sans, `--numeric` | 13.5px | 400 | 1.4 |
| Tickers, kbd, formulas | IBM Plex Mono | 11–13px, +0.02em | 500–600 | 1 |

IBM Plex Sans (variable, 400–700) and IBM Plex Mono (400, 500, 600) live in
`frontend/app/fonts/` and load through `next/font/local`. Numbers use
`--numeric` (`tabular-nums lining-nums slashed-zero`), never monospace, so
tables stay dense. Mono is reserved for tickers, keyboard hints, formulas, and
chart value tags. Units are demoted: `ValueText` sets `R$`, `%`, and `×` at
0.62em in muted text so the figure reads first while the unit stays explicit.
Preserve existing Portuguese strings. New labels use sentence case; no
all-caps eyebrows.

## Spacing and layout

Use a 4px base unit: 4, 8, 12, 16, 24, 32, and 40px. Keep desktop content
within 1360px inside the inset frame (8px gutter, 12px radius, 1px border). The document owns vertical scrolling. Navigation is collapsed
behind an explicit Menu control on narrow screens. At desktop widths, the
sidebar reserves its own column instead of obscuring the page. At 375px,
cards and columns reflow without horizontal page scrolling. Wide tables may
scroll inside their own container.

Keep the page sequence clear: page identity, task controls, key metrics, then
evidence and next steps. Use spacing and dividers to group analytical tasks.
Avoid enclosing forms, metrics, charts, or article lists in cards.

The desktop navigation reserves 232px, or 64px when collapsed. It holds the
logo, a search trigger with its keyboard hint, the five pages, and the
favorites rail. A sticky 48px workspace bar at the top of the frame shows the
breadcrumb, the selection count, and the market badge; a 2px indigo activity
line runs along its bottom edge while any API request is in flight. Hide this
bar below 769px to bring the task controls into view sooner. Below 1024px the
frame flattens into the full-width canvas.
Each page uses the same shell:

- Fundamentos opens on a landing state while the selection is empty: a compact
  hero (page icon, a 32–52px headline, one subtitle line) keeps the search above
  the fold at 375×667 and 1280×720. Quick selections are logo tiles. On wide
  workspaces the search sits beside a sector map, which is a 100% strip of the
  largest sectors in the local B3 list with a counted legend whose rows select
  that sector's most liquid stocks. Tiles for the other four pages follow. Once
  anything is selected the standard page header returns and the search stays
  mounted in place.
- Fundamentos puts search and quick selections before source metadata. The task
  uses the available width. Loaded results lead with ticker summary cards:
  sector and subsector, quote, P/L, ROE, Dividend Yield, the sector synthesis,
  and a star favorite toggle. One tab set follows with Visão geral (market data
  and indicators), Saúde financeira, Comparação when more than one ticker is
  analyzed, Preço-alvo, and Síntese. Results appear before the collapsed Setores
  e favoritos and Como funciona disclosures.
- Portfólio puts asset selection beside parameters on wide screens, then groups
  headline metrics with the allocation before the historical evidence.
- Simulação puts parameters before the allocation in reading order and beside
  it on wide screens. Results begin with the submitted run and statistics.
- Notícias groups the loaded portfolio and update action above the sample
  summary, filters, articles, and source evidence.
- Screener puts filters beside results on wide screens. Both stay in the
  document's vertical scroll; table containers own horizontal scrolling only.

## Radii

| Token | Value | Use |
|---|---:|---|
| `--radius-sm` | 6px | Chips, badges, status pills, code |
| `--radius-md` | 8px | Buttons, inputs, nav items, notices |
| `--radius-lg` | 10px | Tiles, tables, chart panels, criteria |
| `--radius-xl` | 12px | Shell frame, ticker cards, palette, next step |
| `--radius-pill` | 999px | Meters, removable filter chips, selection pill |

## Components

### Page header

One page-level heading with the page icon in a 44px tile (panel surface,
hairline highlight, indigo icon), a short muted subtitle, and a subtle bottom
rule. The active navigation item uses a raised tint, primary text, an indigo
icon, a 3px indigo edge marker, and `aria-current`. Decorative SVGs remain
hidden from assistive technology.

### Command palette

`Ctrl K` / `⌘K` or `/` opens a native `<dialog>` palette (640px, raised
surface, no backdrop blur). Ativos rank by exact ticker, ticker prefix,
company prefix, word start, then substring; choosing one adds it to the
Fundamentos selection (never beyond 20) and opens Fundamentos. Páginas
navigate. The input is a combobox with `aria-activedescendant`; arrows move,
Enter runs, Escape closes. The sidebar favorites rail uses the same action.

### Section headings

Sans, 18–20px, weight 500, primary text, with the icon in the brand color. No
underline or uppercase.

### Metrics and cards

`Metrics` renders tiles: panel surface, 1px default border, 10px radius,
hairline highlight, 14–16px padding, 12px grid gap. The label row is 12.5px
muted text; when an assessment is short (24 characters or fewer, no period) it
becomes a status pill with a glyph (✓, ✕, !, i, •) on the semantic tint;
longer help stays as a caption below. Values use `ValueText`. Optional
`trend` draws a decorative sparkline from a series the API already returned
(the Portfólio value series feeds Retorno total), and optional `meter` draws a
0–1 bar (Simulação probability of gain). Passive tiles never animate on hover.

Fundamentos ticker cards are 12px-radius tiles: ticker badge, favorite star
(amber when saved, filled vs outline shape), company and sector, a 30px quote,
P/L, ROE, and Dividend Yield, the sector synthesis as a segmented meter with
its count in text, and the 52-week range bar labeled with source and date.
The Preço-alvo tab leads with a target ladder: per ticker, the analyst
low-to-high band, the mean target (diamond), the source's own reference quote
(white tick), an upside delta chip with arrow and sign, and opinion count.
The Fundamentus quote and the Yahoo reference quote are never placed on the
same scale. The full consensus table follows.

Run summaries (Portfólio and Simulação) are single panels split by subtle
vertical rules. The Portfólio allocation shows a 100% strip of indigo-ramp
segments above the ranked weight list; each row repeats its swatch so the
strip is never the only encoding.

### Controls

Buttons: 44px minimum height, 8px radius, grouped surface with a hover-alpha
border and the hairline highlight; hover raises the surface and strengthens
the border. Primary buttons are indigo-filled with `--text-on-primary`, weight
600, and an inset top highlight. Inputs use the darker input well with a
strong border; hover lightens the border and focus turns it indigo. Tabs use an indigo 2px highlight. Keep
visible labels, keyboard operation, and clear default, hover, focus, disabled,
and error states.

Tabs follow the ARIA tabs pattern: `role="tablist"`, `tab`, and `tabpanel`,
`aria-selected`, and a roving tabindex. Left and Right move between tabs and
wrap; Home and End jump to the ends. The selected tab uses indigo text, weight
600, and the 2px bottom rule. Tabs wrap onto a second row on narrow screens
instead of scrolling sideways. One typed list decides which tabs are available.

Icon-only toggles, such as the favorite star, keep a 44px target, an
`aria-label`, and `aria-pressed`. Their state changes shape, outline or filled,
not only color.

Quick-selection shortcuts use compact ticker rows with a bottom rule. Their hover and
focus states keep the indigo border or outline. Primary task actions have 24px
horizontal padding. Active screener criteria use a subdued indigo top rule;
their checkbox and textual state identify which filters are enabled. Inactive
criteria show only their checkbox; inputs appear when enabled. Active criteria
also appear above the results as removable chips with the live match count.

Disabled inputs and selects drop to 50% opacity with a not-allowed cursor, the
same as disabled buttons. A page's primary task form sits in one `.task-panel`
with no enclosing border, background, or padding. The order inside it is the
selection first, then parameters in a `.controls-grid`, then one `.actions`
row that leads with the primary button. Multi-choice options with up to about
15 entries, such as table columns, use a `.checkbox-group` of 44px checkbox
chips, not a native multi-select. The checked chip takes the indigo border.

Long result lists render 25 items at a time. A `Ver mais N … (M restantes)`
button reveals the next 25. Row actions render above the list, never below it.
Exports always cover the full result.

Screener column options sit in a collapsed disclosure above the table. Its
summary shows the selected count; opening it reveals the checkbox chips.

Shared native buttons and form controls live in `frontend/components/ui.tsx`
and inherit the global state rules. Use the same scope for base, hover, focus,
and disabled states; do not depend on framework-generated wrapper markup.

### Alerts and empty states

Alerts use a 3px semantic left rule, a 6% semantic tint, a 22% semantic
border, an icon, and 8px radius. Empty states are dashed 12px-radius regions
with a 44px icon tile, a message, and a recovery action. Loading feedback uses
an inline spinner and operation label, with `role="status"` and
`aria-live="polite"`.

### Diagnostics, next step, and loading

Financial diagnostics pair their semantic color with a textual assessment.
Next steps are a panel with a 3px indigo edge marker above the action. Loading uses an indigo spinner
with an actual operation label and an accessible status region.

React consumes calculated financial states from the Python API. Keep fixed
colors in stylesheet tokens. Page icons reuse the SVG assets in
`frontend/public/icons`; the flask-and-candlestick mark keeps its geometry.

### Charts and tables

`Chart.tsx` applies one presentation theme to every server figure, following
terminal conventions: transparent background inside a bordered chart panel,
caption above the plot (the figure title moves into the `figcaption`),
legend on top-left, no vertical grid, solid 6%-white horizontal grid, axis
labels in 11px muted text, pt-BR separators, a raised hover label, and
crosshair spikes on the x axis. Time-series figures put the y axis on the
right, use `x unified` hover, and tag up to three line series with their last
value in a colored mono label on the axis. Python keeps series colors:
indigo for the focal series, cyan for IBOVESPA, indigo bands for Monte Carlo
percentiles (10% and 22% alpha), monochrome indigo trajectories, and the
indigo sequential scale (`CHART_SEQUENTIAL`) instead of Viridis.

Tables live in a bordered 10px-radius container that owns horizontal
scrolling, with panel-colored headers in 12.5px muted text, 13.5px tabular
values, subtle row rules, a white-tint hover, and selected rows in the brand
tint with a 2px indigo edge. The Portfólio monthly returns table is a heatmap:
cell tint scales with the move (green or coral, up to 38%) while the text
stays primary and signed. Screener tickers use the mono face.

Chart toolbar actions stay local. Do not expose cloud sharing of session
analyses. CSV exports come from the owned Python result, not visible table cells.

## Motion and interaction

Shared motion lives in `frontend/app/motion.css`. Page identity and task panels
enter with an 8px translation over 480ms, with short staggered delays. Results
fade in, allocation meters draw from the left, and disclosure arrows rotate
when opened. The palette enters with a 6px drop and 98.5% scale over 200ms.
The activity line slides only while a request is in flight and appears after
180ms, so fast requests do not flash. Keep financial values at their actual values throughout; never
count them up or animate chart series. Keep card hover stationary and animate
only its border. Controls use 150ms color feedback and a small press response.
Do not animate layout or add looping decorative movement; the market strip's
slow linear scroll is the only loop. Focus shows a 2px
indigo outline. Under `prefers-reduced-motion: reduce`, show everything in its
final state and disable transitions, animations, and smooth scrolling.

## Surface treatment

Elevation comes from lightness steps and hairlines, never from shadows: tiles
use `--surface-panel`, a translucent border, and the 1px top highlight;
overlays use `--surface-raised` and `--highlight-strong`. Primary buttons add
a subtle inner bevel (inset highlights only). Do not use drop shadows,
gradients, glow, blur, or texture. Prose, tables, charts, and task forms stay
on the canvas.

## Accessibility and responsive checks

- Normal text contrast is at least 4.5:1. Large text and chart graphics meet
  3:1 where the contrast criterion applies.
- Every interactive control has visible keyboard focus.
- Status never relies on color alone.
- Primary controls use a minimum 44px target.
- Check 375, 768, 1024, and 1440px widths.
- Honor `prefers-reduced-motion`.
- Run `scripts/check_ui_layout.js` in the browser console on each loaded page.
  It checks document overflow, canvas-only containers, blurred shadows,
  primary targets, and titles.
- Keep chart grid lines distinct from the canvas.

## Accepted debt

None. Do not add new design debt during this visual migration.
