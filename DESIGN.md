# B3Lab dark research desk

## Atmosphere and identity

B3Lab is a dark-first workspace for researching Brazilian equities. A blue-green
charcoal canvas, quieter panel layers, serif page titles, and precise numeric
typography keep long analysis sessions readable. Teal remains the primary B3Lab
signal. Blue, amber, coral, and violet retain their existing semantic roles.

Keep the B3Lab name, flask-and-candlestick mark, Brazilian Portuguese copy, and
current financial behavior. This is a data-dense dashboard, not a marketing
page. Use system fonts so the app works offline. Do not add gradients, glow,
blur, decorative texture, external fonts, or color-only status.

The Streamlit theme defaults to dark. CSS tokens and Python chart constants
must use the same palette. The light paper design is no longer the default.

## Color tokens

| Role | Token | Value | Use |
|---|---|---|---|
| Canvas | `--surface-canvas` | `#0C1518` | Main application background |
| Sidebar | `--sidebar-bg` | `#111E22` | Navigation rail |
| Panel | `--surface-panel` | `#142126` | Cards, controls, and chart surfaces |
| Grouped surface | `--surface-muted` | `#1C2B30` | Selected rows and grouped controls |
| Raised surface | `--surface-raised` | `#22343A` | Menus and selected component states |
| Primary text | `--text-primary` | `#EEF4F2` | Headings, body copy, and values |
| Supporting text | `--text-secondary` | `#C4D1CF` | Labels and supporting copy |
| Muted text | `--text-muted` | `#9CAFAE` | Captions and metadata |
| Default border | `--border-default` | `#33464A` | Cards, tables, and separators |
| Strong border | `--border-strong` | `#52686D` | Inputs and control outlines |
| Chart grid | `--chart-grid` | `#63787C` | Plot grid and axes |
| Primary / positive | `--brand-primary` | `#76D5C5` | Primary actions and positive signals |
| Supporting data | `--brand-secondary` | `#91B9E7` | Links and supporting series |
| Warning / return | `--brand-accent` | `#E7BD69` | Caution and return semantics |
| Negative / risk | `--brand-danger` | `#EF969D` | Loss, risk, and errors |
| Guidance | `--brand-info` | `#B7A3E2` | Informational and exploratory states |
| Text on primary | `--text-on-primary` | `#10201F` | Text on teal-filled actions |

Text meets WCAG AA contrast on canvas, panel, and grouped surfaces. The measured
muted-text contrast is 7.18:1 on panels. The measured chart-grid contrast is
3.54:1 against chart panels. Keep status colors paired with a sign, label, or
icon. Update `utils/charts.py` and `.streamlit/config.toml` whenever these
values change.

## Typography

| Role | Font | Size | Weight | Line height |
|---|---|---:|---:|---:|
| Page title | Georgia with system serif fallback | 28–32px | 600–700 | 1.2 |
| Section heading | System UI | 18–22px | 600–700 | 1.3 |
| Body | System UI | 14–16px | 400 | 1.5 |
| Labels | System UI | 12–14px | 500–600 | 1.4 |
| Financial values | System monospace with tabular figures | 14–24px | 500–700 | 1.3 |

Preserve existing Portuguese strings. New labels use sentence case. Body copy
stays at least 14px where the component permits it.

## Spacing and layout

Use a 4px base unit: 4, 8, 12, 16, 24, 32, and 40px. Keep desktop content
within 1360px. The document owns vertical scrolling. Keep the sidebar automatic
and collapsed when Streamlit chooses that state on narrow screens. At tablet
widths, an expanded sidebar reserves its own column instead of obscuring the
page. At 375px, cards and columns reflow without horizontal page scrolling.
Wide tables may scroll inside their own container.

Keep the page sequence clear: page identity, task controls, key metrics, then
evidence and next steps. Use cards only when they group one analytical task.

## Components

### Page header

Use one page-level heading, its existing icon, a short description, and a
semantic page accent. Keep the title and icon accent on the neutral dark
surface. Decorative SVGs remain hidden from assistive technology. The active
navigation link uses a muted surface and `aria-current`.

### Metrics and cards

Use a panel surface, thin border, and 8px radius. Use tabular numbers. Positive,
negative, and warning states retain their labels or signs as well as their
colors. Hover changes border or surface only. Do not lift cards.

### Controls

Keep visible labels, minimum 44px targets where Streamlit permits them, and
clear default, hover, focus, disabled, and error states. Inputs use a recessed
dark surface. Focus uses a visible teal outline. Keep keyboard operation and
associated labels.

### Alerts and empty states

Use a semantic icon and sentence-case message. Keep recovery actions beside the
relevant state. Loading feedback keeps `role="status"` and `aria-live="polite"`.

### Charts and tables

Use the dark panel for Plotly and Matplotlib backgrounds. Keep readable axes,
visible grid lines, semantic series colors, tooltips, and the existing table
alternative where available. Data must be readable before any animation.

## Motion and interaction

Limit interaction feedback to 150ms color, border, and opacity changes. Do not
animate layout or add decorative movement. Keep focus visible. Under
`prefers-reduced-motion: reduce`, disable nonessential transitions and
animations.

## Surface treatment

Create depth with three adjacent dark surface values and clear borders. Do not
use shadows, gradients, glow, blur, or texture. Use 4px radii for controls, 8px
for cards, and 12px only for larger containers.

## Accessibility and responsive checks

- Normal text contrast is at least 4.5:1. Large text and chart graphics meet
  3:1 where the contrast criterion applies.
- Every interactive control has visible keyboard focus.
- Status never relies on color alone.
- Primary controls use a 44px target where Streamlit allows it.
- Check 375, 768, 1024, and 1440px widths.
- Honor `prefers-reduced-motion`.
- Keep chart grid lines distinct from the panel surface.

## Accepted debt

None. Do not add new design debt during this visual migration.
