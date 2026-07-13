# Alert CSS Fix & Empty-State Consistency

**Date:** 2026-07-13
**Status:** Approved

## Context

A follow-up UI/UX pass after the 2026-06-25 consistency+polish round
(commit `26d0440`). A live audit (screenshots of all six pages via headless
Chromium + CDP) surfaced one real regression and one consistency gap.
Two other candidate findings (Screener stat-row styling, "dead space" on
empty states) were investigated and ruled out — see Findings Ruled Out.

## Findings

### 1. Alert CSS is dead code (bug)

`style.css` styles `st.warning` / `st.info` / `st.success` / `st.error`
via:

```css
[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"] { ... }
```

DOM inspection of the running app (Streamlit 1.59.1) shows alerts now
render as:

```html
<div data-testid="stAlert">
  <div data-testid="stAlertContainer">
    <div data-testid="stAlertContentWarning">...</div>
  </div>
</div>
```

No `kind` attribute and no `data-baseweb="notification"` exist anywhere
in this structure, so all four kind-specific color rules never match.
`st.info()` happens to look acceptable because Streamlit's native default
blue is unobtrusive on the dark background; `st.warning()`'s native
default (solid dark olive/khaki with white text) visibly clashes and was
the finding that surfaced this. This affects every alert call site
app-wide: 17 `st.warning`, 19 `st.info`, 1 `st.success`, and all
`st.error` calls.

### 2. Unicode circled-digit glyphs (robustness)

`Main_Page.py`'s onboarding tips use `① Escolha`, `② Analise`,
`③ Valoração`. These Unicode characters (U+2460–U+2462) have no
guaranteed glyph in `Space Grotesk` or `JetBrains Mono` and may fall back
to an unstyled system glyph in some browsers/OSes.

### 3. Empty-state pattern is inconsistent (polish)

`pages/3_Notícias.py` has a custom empty-state card (icon + title +
message + CTA button) for "portfolio not configured yet." `pages/2_Simulação.py`
gates the identical underlying condition with a bare `st.warning()` +
`st.page_link()`. Same situation, two different visual treatments.

## Findings Ruled Out

- **Screener summary stat row "unstyled":** looked flat at normal
  screenshot resolution, but zooming into the raw pixels confirms the
  `.stMetric` panel background and border *are* applying — the contrast
  is just intentionally subtle. `data-testid="stMetric"` still matches
  the CSS selector correctly (unlike the alert case). No fix needed.
- **"Dead space" at the bottom of empty-state pages:** measured at a
  realistic 1440×900 viewport, the Simulação empty state is 813px tall —
  shorter than the viewport, no dead canyon. The original observation
  was an artifact of testing with an oversized 2400px-tall Chromium
  window. No fix needed.

## Design

### 1. Fix alert CSS selectors

In `style.css`, replace the four `[data-testid="stAlert"][kind="..."]` /
`div[data-baseweb="notification"][kind="..."]` selector pairs with:

```css
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { /* existing info rule body */ }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { /* existing warning rule body */ }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { /* existing success rule body */ }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { /* existing error rule body */ }
```

Colors/backgrounds stay exactly as already defined — only the selector
changes. `:has()` is supported in all current evergreen browsers.

### 2. Replace circled-digit glyphs

In `Main_Page.py`'s onboarding block, replace `① Escolha` / `② Analise` /
`③ Valoração` with a small inline circle badge per column, e.g.:

```html
<span style="display:inline-flex;align-items:center;justify-content:center;
  width:16px;height:16px;border-radius:50%;background:{color}22;
  border:1px solid {color};color:{color};font-size:0.62rem;font-weight:800;">1</span>
```

Same per-column colors as today (`#00ff87`, `#00d2ff`, `#a855f7`).

### 3. Shared `empty_state_card()` component

Add to `utils/ui.py`:

```python
def empty_state_card(icon_svg: str, title: str, message: str, cta_label: str, cta_page: str) -> None:
    ...
```

Renders the same markup currently hardcoded in `3_Notícias.py` (icon,
title, message with inline bold ticker/page references handled by the
caller passing pre-formatted `message` HTML, then a styled CTA link).
`3_Notícias.py` switches to calling it (no visual change). `2_Simulação.py`'s
`required_keys` gate switches from `st.warning()` + `st.page_link()` to
this same helper, using the same document icon Notícias uses (or a
portfolio-appropriate icon — implementer's call, kept consistent with
existing SVG icon style).

## Testing

Visual verification only (no test framework in this Streamlit app):
after implementation, re-screenshot all six pages via the existing CDP
screenshot workflow and confirm:
- `st.warning()` boxes show translucent gold background (not solid olive)
- `st.info()` / `st.success()` still look correct (regression check)
- Onboarding tip numbers render as clean circle badges
- Simulação's empty state visually matches Notícias' empty state
