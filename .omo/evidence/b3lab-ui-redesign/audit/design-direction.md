# B3Lab UI Redesign: Design Direction

## Superseding user direction

The user later requested a dark default mode and further UI refinement. This
replaces the light palette recommendation below for implementation. Keep the
B3Lab identity, Portuguese copy, accessibility rules, and data-dense workflow.
The active dark palette is defined in `DESIGN.md`.

## Previous decision, retained for history

Adopt **Direction A — the Layered Research Desk**. It makes the current light
editorial system feel more deliberate and less flat by using calibrated paper
layers, stronger grouping, and a clear reading sequence—not a new visual
identity. This is the safest way to borrow the *principle* behind
Stripe-inspired precision/elevation (purposeful hierarchy and explicit
boundaries) without copying Stripe's colours, assets, typography, layouts, or
copy.

The recommendation preserves B3Lab's defined identity: warm paper, deep ink,
serif titles, tabular figures, visible chart grids, useful information density,
and existing PT-BR content. [DESIGN.md:5-8] It also keeps the B3Lab flask and
candlestick identity intact. [brand/B3Lab-brand.md:5-10]

## Direction A — Layered Research Desk (recommended)

**Idea:** Make the screen read like an analyst's working sheet: a quiet page,
a bounded workbench for inputs and actions, and paper panels that collect
evidence and conclusions. “Elevation” comes from adjacent neutral tones and
rules, never from floating effects.

| System | Direction |
|---|---|
| Palette | Retain the existing canvas, panel, and muted-paper values (`#F4F1E9`, `#FFFDFA`, `#EAE6DD`) as three ordered layers. Keep ink and all five semantic accents in their existing roles; do not introduce a Stripe palette or a new decorative accent. [DESIGN.md:24-38] |
| Material | Canvas is the document field; panel is an evidence sheet; muted surface is the workbench, selected-row wash, or grouped-control tray. Separate only meaningful regions with the existing 1px borders, using the strong border only for form/control boundaries. This produces visible depth while remaining within the required borders-only, no-shadow surface strategy. [DESIGN.md:135-136] |
| Type | Keep Georgia only for 28–32px page identity; use system sans for section hierarchy and labels; reserve system mono with tabular figures for values, deltas, axes, and table columns. The contrast between editorial title and instrument-like data is the distinctive signature, not a larger type scale. [DESIGN.md:50-58] |
| Layout | At desktop widths, order each page as: page header → compact context/action strip → key metrics → chart or table evidence → interpretation/next step. Treat controls as a single muted “workbench” rather than scattering them among results. Let chart/table panels span the available analysis width; keep the document as the vertical scroll owner, the 1360px maximum, automatic sidebar, and single-column 375px fallback. [DESIGN.md:63-66] |
| Data density | Keep compact metric grids and visible chart grids. Use spacing to distinguish *question*, *input*, *evidence*, and *conclusion*; do not add vacant hero space, carousel cards, or a marketing-style narrative. The current contract explicitly frames this as a dashboard, not a marketing page. [DESIGN.md:5-6] |
| State | Active navigation and tabs use a muted-surface wash plus ink/semantic text and the existing `aria-current` or selected state—not a coloured card edge. Financial state continues to pair color with a sign, label, or icon. [DESIGN.md:39][DESIGN.md:74-75] |
| Motion | Limit feedback to the existing 150ms ease-out color, border, and opacity transitions; no lifting cards, no entrances for charts, and no decorative animation. Preserve visible focus and reduced-motion support. [DESIGN.md:127-130] |

### Why it answers “less flat”

The current visual contract already supplies the right materials but asks for a
borders-only surface. [DESIGN.md:135-136] Direction A makes that constraint
intentional: the difference between canvas, workbench, and evidence sheet is
structural, so a dense screen has depth even at rest. The result is closer to
a precise financial instrument than a stack of identical cards.

### Trade-offs

- **Gain:** more legible analysis flow and stronger hierarchy without changing
  the established visual language or consuming data area.
- **Cost:** sections need disciplined grouping; indiscriminate cards would
  erase the effect. One clear panel per analytical job is required.
- **Risk managed:** tonal depth is subtler than shadow-based “premium”
  elevation, but shadows, glow, gradients, and decorative noise are explicitly
prohibited. [DESIGN.md:9-15][brand/B3Lab-brand.md:41-42]

## Direction B — Editorial Evidence Ledger

**Idea:** Push the existing editorial side further: a strict baseline grid,
stronger typographic section breaks, and full-width table “ledgers” separated
by rules. It would feel like a compact market journal with calculations
embedded in the text flow.

| System | Direction |
|---|---|
| Palette/material | Same B3Lab warm neutrals and semantic accents, but fewer individual panels; tables and charts sit directly on the canvas, with rules doing most of the grouping. [brand/B3Lab-brand.md:22-42] |
| Type | Greater reliance on Georgia subheads and smaller mono labels to create an editorial rhythm, while retaining the defined system sans body text. [DESIGN.md:50-58] |
| Layout | A two-column editorial flow at wide widths: a narrow “thesis/controls” rail beside a wide evidence column, collapsing to the contract's single mobile column. [DESIGN.md:63-66] |
| Motion | The same constrained 150ms feedback; no story-like transitions or data entrance animation. [DESIGN.md:127-130] |

### Trade-offs

- **Gain:** most distinctive “research publication” character; tables can feel
  exceptionally intentional.
- **Cost:** it asks users to scan between a rail and a wide evidence column.
  That is less forgiving for Portfolio, Simulation, and Screener workflows
  where filters, KPI comparisons, and wide charts compete for attention.
- **Decision:** do not lead with this direction. It risks weakening the
  data-dense dashboard behavior that B3Lab identifies as essential.

## Implementation Guardrails

1. Keep all visible copy in PT-BR and preserve existing strings during a visual
   migration. [DESIGN.md:5-7][AGENTS.md:50]
2. Keep semantic B3Lab colors reserved for financial meaning and wayfinding;
   page accents remain on title, icon, and divider while surfaces stay neutral.
   [brand/B3Lab-brand.md:41-46]
3. Do not add external fonts, gradients, glow, shadows, blur, decorative
   texture/noise, all-caps eyebrows, or a forced-open narrow sidebar.
   [DESIGN.md:9-15][AGENTS.md:58]
4. Preserve 44px control targets where Streamlit allows, visible focus,
   text/icon state reinforcement, reduced motion, and the 375/768/1024/1440px
   responsive contract. [DESIGN.md:101-106][DESIGN.md:143-147]
5. Keep data behavior and page features unchanged: this is a styling and
   grouping direction, not a feature or navigation redesign. [AGENTS.md:56-57]

## Current-Source Conflicts to Resolve During Implementation

These are evidence findings, not changes made by this report:

1. **Glow and lift conflict with the design contract.** Primary button rules
   use green glow and a vertical hover transform; metric and general button
   rules also lift on hover. [style.css:370-379][style.css:403-406][style.css:495-499]
   This conflicts with no glow/shadow and with the specified non-lift,
   150ms surface/border interaction model. [DESIGN.md:127-130][DESIGN.md:135-136]
2. **Dark input surfaces break the warm-paper material system.** Text, number,
   date, select, and multiselect controls set `rgba(8, 12, 20, 0.8)` rather
   than the defined panel/control surface. [style.css:436-453] This undermines
   the required warm canvas and paper panels. [DESIGN.md:24-25]
3. **All-caps sidebar labels violate project guidance.** The shared
   `.sidebar-section-label` applies `text-transform: uppercase`. [style.css:134-144]
   Project rules prohibit all-caps eyebrows; new labels should be sentence
   case. [AGENTS.md:58]

## Evidence and Scope Check

- Local citations were checked against the approved design, brand, project, and
  shared UI/style sources.
- This task wrote only this report:
  `.omo/evidence/b3lab-ui-redesign/audit/design-direction.md`.
- The working tree contains pre-existing product, style, and test edits; none
  was changed by this task.
