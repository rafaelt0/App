# B3Lab fintech desk

## Atmosphere and identity

B3Lab is a dark-first workspace for researching Brazilian equities. Cool neutral
dark grays, a single vivid indigo brand accent, sans-serif typography, and
precise tabular numerals keep long analysis sessions calm and readable. The
brand color marks primary actions, active state, and section identity only.
Financial semantics use their own colors: green for gains, coral red for
losses, and amber for warnings. Every page shares the same indigo accent;
wayfinding comes from icon, title, and the active navigation link.

Keep the B3Lab name, flask-and-candlestick mark, Brazilian Portuguese copy, and
current financial behavior. This is a data-dense dashboard, not a marketing
page. Bundle IBM Plex Sans locally so typography works without external font
requests. Do not add gradients, glow, blur, shadows, decorative texture,
remote font services, or color-only status.

The Next.js interface defaults to dark. Tokens in
`frontend/app/globals.css` and the `CHART_*` constants in `utils/charts.py`
use the same palette. Verify their rendered colors on the live surface.

## Color tokens

| Role | Token | Value | Use |
|---|---|---|---|
| Canvas | `--surface-canvas` | `#0E1015` | Main application background, inputs |
| Sidebar | `--sidebar-bg` | `#12151C` | Navigation rail |
| Panel | `--surface-panel` | `#161A22` | Cards, charts, tables, forms |
| Grouped surface | `--surface-muted` | `#1D222C` | Buttons, table headers, hover rows, chips |
| Raised surface | `--surface-raised` | `#252B37` | Menus, tooltips, button hover |
| Primary text | `--text-primary` | `#F3F5F9` | Headings, body copy, and values |
| Supporting text | `--text-secondary` | `#C5CAD6` | Labels and supporting copy |
| Muted text | `--text-muted` | `#9AA2B4` | Captions and metadata |
| Default border | `--border-default` | `#2B313D` | Cards, tables, and separators |
| Strong border | `--border-strong` | `#5E687E` | Inputs and control outlines |
| Chart grid | `--chart-grid` | `#646E88` | Plot grid and axes |
| Brand | `--brand-primary` | `#8B93FF` | Primary actions, active state, focus, first series |
| Supporting data | `--brand-secondary` | `#4CC9F0` | Benchmarks, info alerts, neutral sentiment |
| Warning / return | `--brand-accent` | `#F5B544` | Caution and return semantics |
| Negative / risk | `--brand-danger` | `#F2757A` | Loss, risk, and errors |
| Guidance | `--brand-info` | `#E879C6` | Categorical and exploratory emphasis |
| Positive | `--brand-positive` | `#3ECF8E` | Gains, favorable and healthy states |
| Text on primary | `--text-on-primary` | `#0E1015` | Text on indigo-filled actions |

Measured contrast: muted text is 6.81:1 on panel; every accent is at least
5.1:1 on the raised surface; the chart grid is 3.43:1 on panel;
`--border-strong` is 3.12:1 on panel; text on primary is 6.99:1. Keep status
colors paired with a sign, label, or icon. Never use `--brand-primary` to mean
"positive"; use `--brand-positive` / `CHART_POSITIVE`.

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
| `--indigo-950` | `#252A41` | `--brand-primary-surface`, selected backgrounds and icon tiles |

Primary buttons and links use `--text-on-primary` in their default, hover,
and pressed states. Their text contrast is 6.99:1, 9.28:1, and 5.02:1,
respectively. Indigo icons on the selected surface have 5.19:1 contrast.
Keep input outlines and focus rings on `--brand-primary`. The darker ramp
shades are fills and decorative borders; check contrast before using them
for text or control outlines. Keep selection labels, borders, and
`aria-current` or `aria-pressed` alongside the selected background.

## Typography

| Role | Font | Size | Weight | Line height |
|---|---|---:|---:|---:|
| Page title | IBM Plex Sans | `clamp(26px–30px)` | 700 | 1.2 |
| Section heading | IBM Plex Sans | 18–20px | 650 | 1.3 |
| Body | IBM Plex Sans | 14–16px | 400 | 1.5 |
| Labels | IBM Plex Sans | 13px | 500–600 | 1.4 |
| Financial values | IBM Plex Sans with tabular numerals | 14–24px | 600–700 | 1.3 |
| Chips and code | System monospace | 12px | 400–600 | 1.4 |

Interface text uses locally bundled IBM Plex Sans with a system sans fallback
(`--font-display`). The variable font in `frontend/app/fonts/` supports weights
400–700 and Brazilian Portuguese characters. Plotly charts inherit the
interface font. Financial values use `font-variant-numeric: tabular-nums`.
Monospace (`--font-mono`) is reserved for ticker chips, formulas, and code.
Preserve existing Portuguese strings. New labels use sentence case; no
all-caps eyebrows. Body copy stays at least 14px where the component permits it.

## Spacing and layout

Use a 4px base unit: 4, 8, 12, 16, 24, 32, and 40px. Keep desktop content
within 1360px. The document owns vertical scrolling. Navigation is collapsed
behind an explicit Menu control on narrow screens. At desktop widths, the
sidebar reserves its own column instead of obscuring the page. At 375px,
cards and columns reflow without horizontal page scrolling. Wide tables may
scroll inside their own container.

Keep the page sequence clear: page identity, task controls, key metrics, then
evidence and next steps. Use cards only when they group one analytical task.

The desktop navigation reserves 232px, or 72px when collapsed. A slim workspace
bar names the current page above its header without adding a second divider.
Each page uses the same shell:

- Fundamentos puts search and quick selections before source metadata and pairs
  the empty selection with a short guide. Once an asset is
  selected, the task fills the available width. Loaded results lead with ticker
  summaries, followed by section navigation and evidence.
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
| `--radius-sm` | 6px | Chips, tags, code |
| `--radius-md` | 8px | Buttons, inputs, menus, nav links |
| `--radius-lg` | 12px | Cards, metrics, charts, tables, alerts |
| `--radius-xl` | 16px | Large containers: onboarding, empty state, analyst synthesis |
| `--radius-pill` | 999px | Badges, meters, progress tracks |

## Components

### Page header

One page-level heading with its icon in a 44px indigo-tinted tile, a short
subtitle, and a bottom border. Title uses primary text. Decorative SVGs remain
hidden from assistive technology. The active navigation link uses a 3px indigo
left border, an indigo-tinted background, and `aria-current`.

Keep the icon visible at every viewport width. At widths up to 768px, use a
40px tile and keep the header flex layout so long titles and subtitles wrap.

### Section headings

Sans, 18–20px, weight 650, primary text, with the icon in the brand color. No
underline or uppercase.

### Metrics and cards

Panel surface, 1px default border, 12px radius, 20px padding. Values are
neutral primary text with tabular numerals; metric deltas use positive/danger
colors and keep their arrow icon. Hover changes the border only; do not lift
cards.

Supporting guides and allocation context use the canvas with a single divider
and 16px secondary headings. Primary tasks keep the panel surface and 20px
headings. On narrow screens, the supporting divider moves above the content.

### Controls

Buttons: 44px minimum height, 8px radius, muted surface with strong border;
hover raises the surface and turns the border indigo. Primary buttons are
indigo-filled with `--text-on-primary`. Inputs use the canvas surface with a
strong border; focus turns it indigo. Tabs use an indigo 2px highlight. Keep
visible labels, keyboard operation, and clear default, hover, focus, disabled,
and error states.

Quick-selection shortcuts use the canvas and default border. Their hover and
focus states keep the indigo border or outline. Primary task actions have 24px
horizontal padding. Active screener criteria use a subdued indigo border;
their checkbox and textual state identify which filters are enabled.

Disabled inputs and selects drop to 50% opacity with a not-allowed cursor, the
same as disabled buttons. A page's primary task form sits in one `.task-panel`
(panel surface, 12px radius, 24px padding). The order inside it is the
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

Alerts tint the panel with the semantic color (info cyan, success green, warning
amber, error coral) and color the icon. Empty states are centered panels with an
indigo icon tile, a title, a message, and a primary recovery action. Loading feedback keeps
`role="status"` and `aria-live="polite"`.

### Diagnostics, next step, and loading

Financial diagnostics pair their semantic color with a textual assessment.
Next-step cards use a 3px indigo left border. Loading uses an indigo spinner
with an actual operation label and an accessible status region.

React consumes calculated financial states from the Python API. Keep fixed
colors in stylesheet tokens. Page icons reuse the SVG assets in
`frontend/public/icons`; the flask-and-candlestick mark keeps its geometry.

### Charts and tables

Plotly charts sit on a bordered panel with 12px radius. Keep readable axes,
visible grid lines, semantic series colors, tooltips, and the existing table
alternative where available. Tables use muted headers, row borders, and a muted
hover row. Data must be readable before any animation.

Chart toolbar actions stay local. Do not expose cloud sharing of session
analyses. CSV exports come from the owned Python result, not visible table cells.

## Motion and interaction

Shared motion lives in `frontend/app/motion.css`. Page identity and task panels
enter with an 8px translation over 480ms, with short staggered delays. Results
fade in, allocation meters draw from the left, and disclosure arrows rotate
when opened. Keep financial values at their actual values throughout; never
count them up or animate chart series. Keep card hover stationary and animate
only its border. Controls use 150ms color feedback and a small press response.
Do not animate layout or add looping decorative movement. Focus shows a 2px
indigo outline. Under `prefers-reduced-motion: reduce`, show everything in its
final state and disable transitions, animations, and smooth scrolling.

## Surface treatment

Create depth with adjacent neutral dark surface values and clear borders. Do
not use shadows, gradients, glow, blur, or texture.

## Accessibility and responsive checks

- Normal text contrast is at least 4.5:1. Large text and chart graphics meet
  3:1 where the contrast criterion applies.
- Every interactive control has visible keyboard focus.
- Status never relies on color alone.
- Primary controls use a minimum 44px target.
- Check 375, 768, 1024, and 1440px widths.
- Honor `prefers-reduced-motion`.
- Keep chart grid lines distinct from the panel surface.

## Accepted debt

None. Do not add new design debt during this visual migration.
