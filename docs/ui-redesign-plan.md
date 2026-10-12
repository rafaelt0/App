# B3Lab UI redesign implementation plan

## Scope and decision

Modernize all five pages while retaining Next.js, React, FastAPI, SQLite, the local IBM Plex fonts, existing SVG assets, Portuguese copy, and every active feature. This is a presentation and reading-order change. Python remains responsible for financial policy, calculations, classifications, source normalization, and exports.

Two layouts were considered against the source and `DESIGN.md`:

| Approach | Structure | Assessment |
| --- | --- | --- |
| Staged research workspace | Full-width task area, compact source/context strip, prominent result summary, evidence below. Dividers and spacing group work. Desktop columns support selection and allocation. | Selected. It works for an empty selection, one company, twenty companies, simulations, and a long news feed. It preserves document scrolling and the existing inset frame. |
| Persistent analysis rail | A narrow control rail beside a large results canvas on every page. Setup remains visible during research. | Rejected for the common layout. Home search, manual portfolio shares, and simulation explanations do not fit a narrow rail. It would encourage nested scrolling and make prerequisites look like blank results. Screener can retain two columns because its criteria are naturally narrow, with both columns in document scrolling. |

The visible change is a consistent sequence of page identity, a clear task heading, task controls, compact context, a result heading with local actions, headline values, and evidence. Use the existing neutral palette and indigo brand. Keep task forms on the canvas. Reserve tiles for meaningful headline values, company summaries, presets, and enabled criteria. This is not a request to restyle every paragraph as a card.

## Findings from the current source

| Source | Current issue | Required change |
| --- | --- | --- |
| `frontend/app/page.tsx`, `home.css` | Empty Home renders `home-guide` as an open second column. Search and quick choices compete with a three-step guide. `DESIGN.md` calls for instructions after sectors and favorites in a closed disclosure. | Make the search and quick choices the full-width entry task. Move the complete guide into a native closed disclosure after the tools. Keep all instructions and links. |
| `home.css` | Quick choices are bordered, filled tiles, and each sector competes with its ticker. | Turn quick choices into compact ticker rows with a bottom rule, clear ticker, supporting sector, and indigo hover/focus. Use a responsive two-column grid at narrow widths. |
| `components/ui.tsx` | `TickerPicker` places all selected chips before the search. A large selection pushes the input down. It also creates 36px chip buttons. | Keep the label and search first, then selected chips and selection count. Make each chip action a 44px target. Preserve the combobox behavior and serialized selection callback. |
| `components/ui.tsx`, `globals.css` | `Section` is a heading followed by content with a fixed 40px heading margin. Section actions have no shared aligned heading pattern. Empty states center large recovery regions even when they follow a complete setup form. | Add an optional heading toolbar contract, support local actions, and make empty-state content concise and left-aligned within the existing dashed recovery region. Do not shrink its icon or recovery action. |
| `components/AppProvider.tsx`, `globals.css` | Active navigation uses a generic white fill. Sidebar search is 40px and favorite rows are 36px. | Use the existing indigo selected tint and retain icon, edge marker, label, and `aria-current`. Set the search and favorite rows to 44px. Improve grouping and spacing without adding new navigation behavior. |
| `portfolio/page.tsx`, `portfolio.css` | The setup has a section heading plus repeated task headings. Its result area has five core values, seven benchmark values, allocation, and evidence with similar section spacing. | Make setup headings match Home's task hierarchy. Lead the results with the existing run summary, core performance values, and allocation. Group benchmark/complementary values under their own quieter heading before historical evidence. Preserve every metric. |
| `simulacao/page.tsx`, `simulacao.css` | A long single methodology paragraph appears before the submit action. Presets, editable inputs, allocation, and technical explanation compete. | Keep the apply-on-submit and stale-result explanation visible. Move the remaining complete methodology text into a closed `Premissas e calibração` disclosure after the action. Make allocation a compact adjacent context column on wide screens. |
| `noticias/page.tsx`, `news.css` | The feed begins with two long methodology paragraphs before its filters. Five sample metrics share similar emphasis. Articles are separate filled cards with a dedicated score column. | Lead with portfolio/update context, the sample's score and coverage, then a clearly grouped filter toolbar and feed. Put full methodology into a closed disclosure near the filters. Present articles as separated editorial rows with a compact score rail, retaining all article details. |
| `screener/screener.css` | At a 960px workspace width, filters become sticky with `max-height` and `overflow-y: auto`. This creates a second vertical scroll despite the document-scroll requirement. Active chips, liquidity shortcuts, and other local actions are 36px. | Remove constrained filter height and vertical overflow. Keep columns in the normal document. Keep only the compact results context sticky if it does not obscure content. Make all interactive chip and quick-action targets 44px. |
| `screener/page.tsx` | Result count appears both in the active-criteria context and the metric group. Two separate column/help disclosures and selection actions precede the table without a shared toolbar. | Use a compact result toolbar, keep the live count in context, and retain all metrics and disclosures. Group table configuration and export visibly. Keep selected-row actions above the table and the selection replacement explanation. |

These are source findings. The parent agent's running preview supplies baseline screenshots and confirms their actual geometry before final acceptance.

## Common visual and component contracts

The shared worker owns these definitions. Page workers may consume them without editing their implementation.

### Existing components

- Keep `PageHeader`'s current signature and classes. Give its icon/title group a clearer baseline, a 32px desktop title, a 26px minimum mobile title, and a subtitle limited to 72 characters of visual line length. Preserve each current description in full. Keep the visible 44px desktop or 40px mobile icon tile and bottom rule.
- Extend `Section` with optional `description?: string`, `actions?: ReactNode`, and `className?: string`. Existing `title` and `children` stay required. Render a `.section-toolbar` containing a heading group and optional `.section-actions`, then children. The section remains semantic `<section>` with one `<h2>`. Description is a paragraph. Do not wrap the content in a filled panel. Existing calls remain valid.
- `Metrics`, `Metric`, `Table`, `ValueText`, `formatValue`, `Notice`, `Loading`, `ExportButton`, and `TickerPicker` retain their public types and behavioral contracts. Do not change numeric formatting or sorting while styling them.
- Remove the current first-heading margin dependency in `Section` by putting separation on the section wrapper. Preserve Home's custom `<h3>` tab sections. Give section toolbars a 24px gap before content and a 32px to 40px gap between major evidence sections.

### Shared CSS utilities

| Class | Contract |
| --- | --- |
| `.task-heading` | Flex heading group for a task, aligned at the top, 12px gap. No background, border, or enclosing padding. |
| `.task-step` | Noninteractive 28px square, indigo text/tint, small mono number, decorative `aria-hidden="true"`. The adjacent heading contains the actual task name. |
| `.task-heading-content` | `min-width: 0`; heading margin 0, 18px to 20px text; supporting paragraph margin 4px 0 0, muted 14px text. |
| `.section-toolbar` | Heading/description and local actions align at the bottom on desktop, wrap in DOM order on mobile. No painted background. |
| `.section-actions` | Flexible row, 8px gap, 44px action targets, wraps without horizontal overflow. |
| `.context-strip` | Flexible rows of source or submitted-run metadata, 12px to 24px gaps, 12px vertical padding and subtle top/bottom rules, transparent background. Labels muted, values secondary or primary. |
| `.result-lead` | A top rule, 32px to 40px top separation, local heading/action grouping; no card enclosing the result. |
| `.metrics-primary` | Wrapper for a headline `Metrics` group. It does not change numeric data. Values 30px to 32px desktop, 24px mobile. Page-specific first-tile emphasis remains in page CSS. |
| `.method-disclosure` | Native `<details>` on canvas, top rule, 44px `<summary>`, 14px body text at max 80ch. Closed by default. It preserves access to every original explanation. |

Do not add a generic card component, workflow state machine, extra client-side derived financial judgments, or another page shell.

### Dimensions and scrolling

- Retain 232px expanded and 64px collapsed desktop navigation, the 8px inset frame, 12px frame radius, and 1360px maximum content width.
- Use 32px desktop content padding, 24px at medium widths, and 16px at narrow widths. Keep the 48px workspace bar and its actual request activity line. Hide it below 769px as today.
- At widths below 1024px retain the explicit, initially closed Menu and full-width canvas. At wide widths the sidebar reserves a column.
- Use `container: workspace / inline-size` on the existing `.main-content`. Page columns use container widths, so collapsed navigation does not produce a mismatched layout.
- Use 4px spacing increments. Prefer 24px within a task, 32px between major task/result groups, and 12px between meaningful tiles.
- The document owns vertical scrolling. Wide tables own horizontal scrolling. All grid/flex children that can contain charts, controls, or long text use `min-width: 0`.
- Every interactive button, chip action, summary, search input, select, and numeric input is at least 44px high. Checkbox hit areas are at least 44px; the checkbox itself can remain 18px to 20px.

## Worker ownership

Run three Luna workers. They share the worktree but have disjoint edit sets. Workers must not commit, reset files, run formatters across the tree, or edit another worker's files. Ask the parent for a contract change through a message if required. The parent owns integration, final checks, the commit, PR, and merge.

| Worker | Files it may edit | Deliverable |
| --- | --- | --- |
| A, shared workspace | `frontend/app/globals.css`, `frontend/app/motion.css`, `frontend/components/ui.tsx`, only the `Shell` presentation JSX in `frontend/components/AppProvider.tsx` | Common hierarchy, utilities, `Section` toolbar, shell navigation treatment, picker order/targets, shared empty and loading treatment. |
| B, Home reference | `frontend/app/page.tsx`, `frontend/app/home.css`, `frontend/app/SelectionSummary.tsx`, `frontend/components/HomeResults.tsx` | Complete empty/selected/loaded Home design that establishes the shared pattern, without changing handlers, requests, result predicates, or tab logic. |
| C, four feature pages | `frontend/app/portfolio/page.tsx`, `frontend/app/portfolio/portfolio.css`, `frontend/components/PortfolioParts.tsx`, `frontend/app/simulacao/page.tsx`, `frontend/app/simulacao/simulacao.css`, `frontend/app/noticias/page.tsx`, `frontend/app/noticias/news.css`, `frontend/app/screener/page.tsx`, `frontend/app/screener/screener.css`, only presentation JSX/classes in `frontend/app/screener/ScreenerTable.tsx` | Propagate the task/context/result pattern across Portfólio, Simulação, Notícias, and Screener. Preserve all states and actions. |

Existing global CSS contains some Home/news-specific selectors. Worker A must leave those page-specific selectors alone, apart from shared target rules. Workers B and C make focused corrections in their owned page CSS with page-root scoping. Worker C can add `.portfolio-page`, `.simulation-page`, and `.screener-page` wrapper classes to scope its styles. It must scope all new rules under its corresponding page root. Avoid broad selectors such as `section button` or unscoped `[data-tone]` additions.

Worker A sends the parent and both page workers a brief contract-ready message when the shared utilities and `Section` props are in place. Worker B sends a Home-ready message so the parent can review an initial render, then requests parent checks of empty, selected, and loaded Home. Worker C completes Portfólio and Simulação first, then Notícias and Screener, using the same contracts. Page workers can start independently with the listed classes and existing component signatures. All browser automation and screenshots belong to the parent to avoid collisions in the shared browser session.

## Home implementation details

1. Keep PageHeader, restoration notices, undo notices, errors, and loading in their existing logical positions.
2. The empty entry task becomes one full-width search area with a `.task-heading`, a short instruction, `TickerPicker`, and quick choices. Remove the open guide from the task grid. The source/count row becomes `.context-strip` and remains truthful while suggestions load.
3. Keep selection actions directly after the picker. Keep the stale-selection explanation and Portfólio distinction visible. Preserve the 20-item ceiling and quick-start analysis sequence.
4. Keep `Setores e favoritos` closed by default. Follow it with a separate closed `Como funciona` disclosure containing the complete current guide. Preserve sector auto-selection, its replacement warning/undo, refresh, and all favorite actions.
5. The loaded selection summary becomes a compact context row with the tickers, count, and `Editar seleção`. Keep that edit button visible and focus restoration working.
6. Put the result heading and CSV action in a `.result-lead` toolbar. Company summaries retain quote, P/L, ROE, DY, company, sector/subsector, the textual synthesis count, source-labeled range, favorite toggle, and missing-data behavior. Use a 30px quote, compact facts, and a quieter synthesis footer. Cards stay visually stationary on hover.
7. Keep the typed tab list and ARIA keyboard behavior. Wrap tabs on narrow screens. Preserve all five tabs and comparison availability. Group each tab's sections with spacing rather than new enclosing cards. Health values must remain metric tiles or clearly aligned analytical rows, with their diagnostic text intact.
8. The portfolio next step becomes a compact rule-separated region with its existing action styled as a 44px `.button-link`. Keep the selection-specific text and route.

## Feature-page implementation details

### Portfólio

- Keep asset selection and parameters side by side at workspace widths of 880px or more. Use matching numbered task headings. Keep manual shares and Markowitz controls exactly where their corresponding strategy activates them.
- Group the primary load action ahead of quote refresh and clear. Retain disabled states, auto-recalculation explanation, source dates, validation notices, and the distinction between editing assets and editing parameters.
- Start the results with `ResultSummary`, a result heading, and the core performance/allocation overview. Keep the first total-return tile prominent and allocation text/swatch/weight encodings intact. Add the complementary metrics heading without changing visibility based on benchmark availability.
- Keep historical charts, monthly heatmap, stress controls, complete warnings, exports, and every analysis section. Make CSV actions local to the evidence they export. No financial ordering or calculations change.

### Simulação

- Preserve the prerequisite branches and restoration/reload actions. Make their primary recovery action and the future outputs easy to scan.
- Match the task headings and action row to Home. Retain preset labels, selected text/check, numeric validation, calibration selector, and all limits.
- Put the original apply-on-submit and outdated-results sentences beside the action. Put the remaining original technical paragraph in `Premissas e calibração`. No text about calendar assumptions or model limits disappears.
- Keep parameters before allocation in DOM order. At a 960px workspace width, use the current 280px to 320px allocation column with a simple left rule. On mobile it follows the form.
- Lead results with the submitted-run summary and three headline values. Keep lower-tail values, definitions, all fan/distribution/trajectory figures, display-count control, bootstrap comparison, synthesis, and export actions.

### Notícias

- Use a compact portfolio/update context row. Timestamp and actual Jev/lexicon engine evidence remain visible. Failure, retry, stale sample, and missing-classification notices remain above the sample results.
- Give score and weight coverage stronger hierarchy than article counts. Preserve their separate meanings, API assessments, absent score, coverage caveat, and complete chart/table.
- Align feed filters as a toolbar, then active chips and the live count. Keep a short visible statement that filters do not alter aggregate values. Move the complete source/method explanations into a closed disclosure near that toolbar.
- Change filled article cards into separated rows with a narrow desktop score rail and a compact top score row on mobile. Keep signed score, sentiment/intensity labels, engine, ticker chips, headline link, metadata, summary, importance triage, and model/lexicon details. Keep external destinations and link safety attributes.
- Retain the full per-asset tone table, its sorting, coverage contribution, all methodology explanations, and the existing reveal-more behavior.

### Screener

- At a 960px workspace width, use 304px to 320px filters plus the results column. Remove the filters' sticky height and vertical scroll. At narrower widths stack setup before results and retain the existing criterion grid where space permits.
- Make enabled criteria use a subdued indigo top rule, actual checkbox state, and textual `Ativo`. Inactive criteria still show only their toggle. Keep every range/input, exclusive-minimum explanation, validation, preset, and automatic refresh behavior.
- Group active filters and match count in compact context. Keep removal focus behavior. Remove gratuitous opacity changes only if the loading/stale status remains explicit; do not show a stale result as current.
- Add a results toolbar. Keep column count disclosure and indicator help disclosure, full CSV export, selected-row actions, replacement explanation, sorting, and progressive row reveal. The table owns horizontal overflow and retains a clear selected row.
- Keep source timestamp/age/missing timestamp and final Fundamentus/recommendation caveat visible. No unknown sector is inferred.

## Read-only fences

All workers must read but not modify `AGENTS.md`, `frontend/AGENTS.md`, `DESIGN.md`, `brand/`, fonts, public assets, route layouts, package manifests/lockfile, API proxy, `frontend/lib/`, `frontend/components/Chart.tsx`, `CommandPalette.tsx`, `RangeBar.tsx`, `backend/`, `utils/`, tests, persistence, run scripts, and dependencies. The plan file belongs to the planning agent; workers do not modify it.

In `AppProvider.tsx`, only `Shell` presentation markup is editable. Keep bootstrap, context types, enqueue/accept, session identity/revisions, selection/favorites mutations, request activity subscription, palette keyboard handling, navigation closing, collapsed-state persistence, and focus handling unchanged. In page files, request bodies, effects, callbacks, generation/revision checks, fingerprints, readiness predicates, API-result handling, and data transformations are read-only. Wrap or move existing JSX without rewriting those sections.

Do not replace active charts, tables, controls, or states with mock content. Do not invent prices, sentiment, market-open status, an account identity, source freshness, or browser-session recovery. Do not add dependencies or remote assets. No gradients, glow, blur, drop shadows, decorative textures, uppercase eyebrows, or color-only statuses. Existing inset hairline highlights are permitted by `DESIGN.md`.

## Verification and acceptance

Before editing TSX, workers read the TypeScript skill and its type-system-discipline prerequisite. Apply unslop to added copy. Read the local Next docs at `frontend/node_modules/next/dist/docs/01-app/01-getting-started/11-css.md`, `03-layouts-and-pages.md`, and relevant client-component/font sections. The installed docs note that imported global route CSS persists after navigation and that production CSS ordering can differ from development. Keep new page rules scoped and verify actual route navigation in the production build.

Each worker reports its changed files, the preserved feature checklist, static checks it ran, states the parent should inspect, and any remaining failure. Workers do not start a browser or another dev server. Do not claim success based only on compilation.

The parent has an existing preview at `http://localhost:3062`, API port 8062, and a Python 3.11 `.venv`. The baseline pytest run passed 247 tests. Baseline Home screenshots are `/tmp/b3lab-redesign-28e5f01e/home-before-desktop.png` and `home-before-mobile.png`. The desktop screenshot confirms the oversized open guide and unused lower space. The baseline Screener has five short interactive targets. Use the parent's existing Playwright session for further checks; T3 preview is unavailable in this environment.

The parent runs:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
PYTHONPATH=. pytest
```

Evaluate the existing `scripts/check_layout.js` and `scripts/check_ui_layout.js` on each of `/`, `/portfolio`, `/simulacao`, `/noticias`, and `/screener` at 375px, 768px, 1024px, and 1440px after content has loaded. The scripts are browser snippets, not standalone Node commands. Keep them unchanged and capture their reports. The parent may add and commit `scripts/verify_ui.cjs` as a rerunnable browser check that evaluates those snippets, checks keyboard/picker behavior, and captures screenshots through the available external Playwright installation. This script must not require a new project dependency. Also inspect screenshots because the snippets do not judge hierarchy, focus overlap, clipped chart labels, or the reading order of dense content.

Acceptance requires:

- Empty Home's first viewport contains its page title, search, and usable quick choices. Help is closed by default. Selected Home exposes analyze/clear. Loaded Home exposes edit, summaries, tabs, and full results.
- At 1440px the sidebar, inset frame, task sections, and result hierarchy visibly differ from baseline in a coherent way. The five pages share heading, action, context, and spacing conventions. Merely changing border colors is insufficient.
- At 375px there is no document horizontal scroll, clipped control, hidden page icon, forced-open navigation, or overlapping sticky area. Tables scroll locally and all touch targets pass the scripts.
- The document scrolls naturally through the complete Screener criteria and results. There is no nested vertical filter scroll.
- Desktop collapsed navigation remains useful. Mobile Menu starts closed, opens on activation, closes on route choice and Escape, and returns focus where the current behavior expects it.
- Keyboard checks cover skip link, visible focus, ticker search Arrow/Enter/Escape, selected-chip removal/favorite, Home tabs Arrow/Home/End, native disclosures, table sorting and row selection, palette shortcut/selection/Escape, and form submission.
- Check Home empty/selected/loaded and missing-source notices; portfolio empty/manual/Markowitz/loaded/stale; simulation prerequisite/ready/invalid/running/loaded/stale; news prerequisite/loaded/filter-empty/fallback/feed-failure; screener default/criteria/invalid/selected/no-match/loading. Existing deterministic tests and real UI fixtures can support these states. Keep mocked values out of production code.
- Run with reduced motion and verify final content remains visible. Charts keep their original data, accessible captions, units, and server-provided series colors.
- The final diff stays within the owned frontend presentation files, this plan, and the parent's optional `scripts/verify_ui.cjs` verification script. The parent checks for changed financial/session logic, dependency churn, missing exports, and removed explanations before opening the PR.

The user authorized opening a PR and merging to main after implementation. The parent prepares the concrete change, includes validation evidence in the PR, waits for required checks, and merges through the repository's normal mechanism. This planning agent does not commit or merge.
