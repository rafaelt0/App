# Alert CSS Fix & Empty-State Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the dead alert CSS (Streamlit's DOM changed, the `kind` attribute the CSS selects on no longer exists), replace two Unicode circled-digit glyphs with reliable styled badges, and unify the app's two "portfolio not configured yet" empty states behind one shared component.

**Architecture:** Pure CSS selector fix (no markup changes needed for alerts) plus a new small presentational helper in `utils/ui.py` (`empty_state_card`) that both `pages/3_Notícias.py` and `pages/2_Simulação.py` call instead of hand-rolling their own markup.

**Tech Stack:** Streamlit 1.59.1, plain CSS (`style.css`), Python f-string HTML via `st.markdown(..., unsafe_allow_html=True)`. No test framework in this repo — verification is static (grep/py_compile) plus a manual visual check.

## Global Constraints

- Do not change any color values already defined for alerts (info `#b8eeff`/`rgba(0,210,255,...)`, warning `#fff3b0`/`rgba(255,214,0,...)`, success `#b0ffe0`/`rgba(0,255,135,...)`, error `#ffc0c8`/`rgba(255,61,90,...)`) — only the CSS selectors change.
- Keep the per-column onboarding tip colors on Main Page exactly as today: `#00ff87` (col 1), `#00d2ff` (col 2), `#a855f7` (col 3).
- `empty_state_card`'s CTA must be a real, working `st.page_link` (not decorative text) — this fixes a latent issue where Notícias' current CTA is inert styled text pretending to be a link.
- Follow the existing import-ordering convention in touched files: alphabetical within the `from utils.ui import ...` line.

---

### Task 1: Fix stale alert CSS selectors

**Files:**
- Modify: `style.css:213-244`

**Interfaces:**
- Consumes: nothing (pure CSS).
- Produces: nothing consumed by other tasks — independent fix.

**Context:** DOM inspection of the running app (Streamlit 1.59.1) shows alerts render as `<div data-testid="stAlert"><div data-testid="stAlertContainer"><div data-testid="stAlertContentWarning">...`. There is no `kind` attribute and no `data-baseweb="notification"` element anywhere in that structure, so the current selectors never match. `st.info()` only looks acceptable today by coincidence (Streamlit's native default blue is inoffensive on the dark background); `st.warning()`'s native default (solid dark olive, white text) visibly clashes.

- [ ] **Step 1: Confirm the current dead selectors are still in place**

Run: `grep -n 'kind="warning"\|kind="info"\|kind="success"\|kind="error"' style.css`

Expected output (line numbers may differ slightly if the file changed, but content should match):
```
215:[data-testid="stAlert"][kind="info"],
216:div[data-baseweb="notification"][kind="info"] {
223:[data-testid="stAlert"][kind="warning"],
224:div[data-baseweb="notification"][kind="warning"] {
231:[data-testid="stAlert"][kind="success"],
232:div[data-baseweb="notification"][kind="success"] {
239:[data-testid="stAlert"][kind="error"],
240:div[data-baseweb="notification"][kind="error"] {
```

- [ ] **Step 2: Replace the four selector blocks**

In `style.css`, find this exact block (currently lines 213-244):

```css

/* Info */
[data-testid="stAlert"][kind="info"],
div[data-baseweb="notification"][kind="info"] {
  background: rgba(0, 210, 255, 0.06) !important;
  border-color: rgba(0, 210, 255, 0.35) !important;
  color: #b8eeff !important;
}

/* Warning */
[data-testid="stAlert"][kind="warning"],
div[data-baseweb="notification"][kind="warning"] {
  background: rgba(255, 214, 0, 0.06) !important;
  border-color: rgba(255, 214, 0, 0.35) !important;
  color: #fff3b0 !important;
}

/* Success */
[data-testid="stAlert"][kind="success"],
div[data-baseweb="notification"][kind="success"] {
  background: rgba(0, 255, 135, 0.06) !important;
  border-color: rgba(0, 255, 135, 0.35) !important;
  color: #b0ffe0 !important;
}

/* Error */
[data-testid="stAlert"][kind="error"],
div[data-baseweb="notification"][kind="error"] {
  background: rgba(255, 61, 90, 0.06) !important;
  border-color: rgba(255, 61, 90, 0.35) !important;
  color: #ffc0c8 !important;
}
```

Replace it with:

```css

/* Info */
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
  background: rgba(0, 210, 255, 0.06) !important;
  border-color: rgba(0, 210, 255, 0.35) !important;
  color: #b8eeff !important;
}

/* Warning */
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
  background: rgba(255, 214, 0, 0.06) !important;
  border-color: rgba(255, 214, 0, 0.35) !important;
  color: #fff3b0 !important;
}

/* Success */
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
  background: rgba(0, 255, 135, 0.06) !important;
  border-color: rgba(0, 255, 135, 0.35) !important;
  color: #b0ffe0 !important;
}

/* Error */
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
  background: rgba(255, 61, 90, 0.06) !important;
  border-color: rgba(255, 61, 90, 0.35) !important;
  color: #ffc0c8 !important;
}
```

(The `div[data-baseweb="notification"][kind="..."]` lines are dropped entirely — that attribute doesn't exist in the current DOM, so they were always dead weight.)

- [ ] **Step 3: Verify the old selectors are gone and the new ones are present**

Run: `grep -n 'kind="' style.css`
Expected: no output (zero matches).

Run: `grep -n 'stAlertContentWarning\|stAlertContentInfo\|stAlertContentSuccess\|stAlertContentError' style.css`
Expected: four lines, one per kind, each inside a `:has(...)` selector.

- [ ] **Step 4: Visual check**

If a local Streamlit instance is running (`streamlit run Main_Page.py`), open the **Simulação** page in a browser without a portfolio configured. Confirm the "Configure primeiro seu portfólio..." box now shows a translucent gold background (`rgba(255, 214, 0, 0.06)`) with light-gold text (`#fff3b0`), not a solid dark-olive box with white text. Also spot-check an `st.info()` box (e.g. Valuation's "Digite um ticker B3...") still shows translucent blue — this confirms no regression.

If no local instance is running, start one: `streamlit run Main_Page.py --server.headless true &` then visit `http://localhost:8501/Simula%C3%A7%C3%A3o` in a browser.

- [ ] **Step 5: Commit**

```bash
git add style.css
git commit -m "fix: repair dead alert CSS selectors after Streamlit DOM change

Streamlit's alert markup dropped the kind attribute this CSS relied on,
silently breaking themed colors for every st.warning/info/success/error
call in the app. Select on the stAlertContent{Kind} testid instead."
```

---

### Task 2: Replace Unicode circled-digit glyphs on Main Page

**Files:**
- Modify: `Main_Page.py:222-233`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by other tasks — independent fix.

**Context:** The onboarding tip labels use `① Escolha`, `② Analise`, `③ Valoração` (U+2460–U+2462). These glyphs have no guaranteed rendering in `Space Grotesk`/`JetBrains Mono` and may fall back to an unstyled system glyph. Replace each with a small inline circle badge built from a `<span>`, guaranteed to render identically everywhere.

- [ ] **Step 1: Confirm the current markup**

Run: `grep -n '① Escolha\|② Analise\|③ Valoração' Main_Page.py`

Expected:
```
223:      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00ff87;text-transform:uppercase;margin-bottom:0.5rem">① Escolha</div>
227:      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00d2ff;text-transform:uppercase;margin-bottom:0.5rem">② Analise</div>
231:      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a855f7;text-transform:uppercase;margin-bottom:0.5rem">③ Valoração</div>
```

- [ ] **Step 2: Replace the three tip blocks**

In `Main_Page.py`, find this exact block (currently lines 221-234):

```html
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00ff87;text-transform:uppercase;margin-bottom:0.5rem">① Escolha</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Filtre por setor na barra lateral ou digite o código da ação no campo de busca acima.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00d2ff;text-transform:uppercase;margin-bottom:0.5rem">② Analise</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Veja P/L, EV/EBITDA, ROE, ROIC, endividamento e posição no setor vs peers automaticamente.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a855f7;text-transform:uppercase;margin-bottom:0.5rem">③ Valoração</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Use a página <b style="color:#a855f7">Valuation</b> para o DCF completo McKinsey/Koller em 8 etapas.</div>
    </div>
  </div>
```

Replace it with:

```html
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00ff87;text-transform:uppercase;margin-bottom:0.5rem"><span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#00ff8722;border:1px solid #00ff87;color:#00ff87;font-size:0.62rem;font-weight:800;margin-right:4px;">1</span>Escolha</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Filtre por setor na barra lateral ou digite o código da ação no campo de busca acima.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00d2ff;text-transform:uppercase;margin-bottom:0.5rem"><span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#00d2ff22;border:1px solid #00d2ff;color:#00d2ff;font-size:0.62rem;font-weight:800;margin-right:4px;">2</span>Analise</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Veja P/L, EV/EBITDA, ROE, ROIC, endividamento e posição no setor vs peers automaticamente.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a855f7;text-transform:uppercase;margin-bottom:0.5rem"><span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#a855f722;border:1px solid #a855f7;color:#a855f7;font-size:0.62rem;font-weight:800;margin-right:4px;">3</span>Valoração</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Use a página <b style="color:#a855f7">Valuation</b> para o DCF completo McKinsey/Koller em 8 etapas.</div>
    </div>
  </div>
```

- [ ] **Step 3: Verify the glyphs are gone and syntax is valid**

Run: `grep -n '①\|②\|③' Main_Page.py`
Expected: no output (zero matches).

Run: `python3 -m py_compile Main_Page.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Visual check**

With a local Streamlit instance running, open the Main Page with no ticker selected. Confirm the three onboarding tip cards ("Escolha", "Analise", "Valoração") each show a small colored circle badge with "1"/"2"/"3" inside, in the same green/blue/purple as before.

- [ ] **Step 5: Commit**

```bash
git add Main_Page.py
git commit -m "fix: replace circled-digit unicode glyphs with styled badges

U+2460-U+2462 have no guaranteed glyph in the app's fonts. A small
inline-styled span renders identically everywhere."
```

---

### Task 3: Add `empty_state_card()` helper and migrate Notícias to it

**Files:**
- Modify: `utils/ui.py` (add function after `render_cards_grid`, i.e. after current line 60)
- Modify: `pages/3_Notícias.py:16` (import) and `pages/3_Notícias.py:532-550` (call site)

**Interfaces:**
- Produces: `empty_state_card(icon_svg: str, title: str, message: str, cta_label: str, cta_page: str) -> None` in `utils/ui.py`, importable as `from utils.ui import empty_state_card`. `icon_svg` is full standalone `<svg>...</svg>` markup (not wrapped via `svg_icon`, which is sized/positioned for inline text use). `message` may contain inline HTML (e.g. `<strong>`) and is inserted as-is via `unsafe_allow_html`. `cta_page` is a page path suitable for `st.page_link` (e.g. `"pages/1_Portfolio.py"`). Renders a centered card (icon, title, message) followed by a real `st.page_link` CTA below it.
- Consumed by: Task 4 (`pages/2_Simulação.py`).

- [ ] **Step 1: Add `empty_state_card` to `utils/ui.py`**

In `utils/ui.py`, insert this function immediately after `render_cards_grid` (after the current line 60, before the `def load_css` at line 63):

```python
def empty_state_card(icon_svg: str, title: str, message: str, cta_label: str, cta_page: str) -> None:
    """Render a centered empty-state card (icon, title, message) with a CTA link below it.

    `icon_svg` is full standalone SVG markup (not wrapped via `svg_icon`).
    `message` may contain inline HTML (e.g. `<strong>`) and is inserted as-is.
    `cta_page` is a page path suitable for `st.page_link` (e.g. "pages/1_Portfolio.py").
    """
    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:16px;padding:2.5rem;text-align:center;margin-top:2rem;">
        {icon_svg}
        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem">{title}</div>
        <div style="font-size:0.875rem;color:#94a3b8;max-width:400px;margin:0 auto 1.2rem;">{message}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.page_link(cta_page, label=cta_label, icon="➡️")
```

- [ ] **Step 2: Verify `utils/ui.py` still imports cleanly**

Run: `python3 -m py_compile utils/ui.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Update the import in `pages/3_Notícias.py`**

Current line 16:
```python
from utils.ui import load_css, loading_overlay, render_flow_sidebar, svg_icon
```

Replace with:
```python
from utils.ui import empty_state_card, load_css, loading_overlay, render_flow_sidebar, svg_icon
```

- [ ] **Step 4: Replace the empty-state call site in `pages/3_Notícias.py`**

Find this exact block (currently lines 532-550):

```python
if "peso_manual_df" not in st.session_state or st.session_state["peso_manual_df"] is None:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:16px;padding:2.5rem;text-align:center;margin-top:2rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
            <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem">Portfólio não configurado</div>
        <div style="font-size:0.875rem;color:#94a3b8;max-width:400px;margin:0 auto 1.2rem;">
            Configure e carregue seu portfólio na página <strong style="color:#00ff87">Portfolio</strong> para que as notícias sejam filtradas para os ativos da sua carteira.
        </div>
        <div style="display:inline-block;background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.25);border-radius:8px;padding:0.5rem 1.2rem;font-size:0.85rem;color:#00ff87;font-weight:600;">
            → Acesse Portfolio na barra lateral
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
```

Replace with:

```python
if "peso_manual_df" not in st.session_state or st.session_state["peso_manual_df"] is None:
    empty_state_card(
        icon_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
            <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
        </svg>""",
        title="Portfólio não configurado",
        message='Configure e carregue seu portfólio na página <strong style="color:#00ff87">Portfolio</strong> para que as notícias sejam filtradas para os ativos da sua carteira.',
        cta_label="Ir para Portfolio",
        cta_page="pages/1_Portfolio.py",
    )
    st.stop()
```

Note: this changes Notícias' CTA from inert styled text ("→ Acesse Portfolio na barra lateral") into a real, clickable `st.page_link` — a small functional improvement, not just a refactor.

- [ ] **Step 5: Verify**

Run: `python3 -m py_compile "pages/3_Notícias.py"`
Expected: no output, exit code 0.

With a local Streamlit instance running, open Notícias with no portfolio configured. Confirm the card looks the same as before (icon, "Portfólio não configurado", message) and that "Ir para Portfolio" is now a real clickable link that navigates to the Portfolio page.

- [ ] **Step 6: Commit**

```bash
git add utils/ui.py "pages/3_Notícias.py"
git commit -m "refactor: extract empty_state_card() helper, use it in Notícias

Pulls Notícias' hardcoded empty-state markup into a reusable component
so Simulação's equivalent gate (next commit) can match it visually.
Also upgrades the CTA from static text to a real st.page_link."
```

---

### Task 4: Migrate Simulação's portfolio gate to `empty_state_card()`

**Files:**
- Modify: `pages/2_Simulação.py:9` (import) and `pages/2_Simulação.py:125-133` (call site)

**Interfaces:**
- Consumes: `empty_state_card(icon_svg, title, message, cta_label, cta_page)` from `utils/ui.py` (produced in Task 3).

- [ ] **Step 1: Update the import in `pages/2_Simulação.py`**

Current line 9:
```python
from utils.ui import load_css, render_flow_sidebar, svg_icon
```

Replace with:
```python
from utils.ui import empty_state_card, load_css, render_flow_sidebar, svg_icon
```

- [ ] **Step 2: Replace the gate at lines 125-133**

Find this exact block:

```python
# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        st.warning(
            "Configure primeiro seu portfólio na aba **Portfolio** para liberar a Simulação Monte Carlo."
        )
        st.page_link("pages/1_Portfolio.py", label="Ir para Portfolio", icon="➡️")
        st.stop()
```

Replace with:

```python
# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        empty_state_card(
            icon_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
                <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
                <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
                <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            </svg>""",
            title="Portfólio não configurado",
            message='Configure primeiro seu portfólio na página <strong style="color:#00ff87">Portfolio</strong> para liberar a Simulação Monte Carlo.',
            cta_label="Ir para Portfolio",
            cta_page="pages/1_Portfolio.py",
        )
        st.stop()
```

- [ ] **Step 3: Verify**

Run: `python3 -m py_compile "pages/2_Simulação.py"`
Expected: no output, exit code 0.

With a local Streamlit instance running, open Simulação with no portfolio configured. Confirm it now shows the same card layout as Notícias (icon, "Portfólio não configurado" heading, message, real page-link CTA) instead of the old plain warning box + separate link.

- [ ] **Step 4: Commit**

```bash
git add "pages/2_Simulação.py"
git commit -m "refactor: use empty_state_card() for Simulação's portfolio gate

Matches Notícias' empty-state visual treatment instead of a bare
st.warning + separate page_link."
```

---

## Self-Review Notes

- **Spec coverage:** Design section 1 (alert CSS) → Task 1. Design section 2 (circled digits) → Task 2. Design section 3 (shared component) → Task 3 (helper + Notícias) and Task 4 (Simulação). All three spec sections have a task.
- **Placeholder scan:** no TBD/TODO; every step has literal file content, not descriptions.
- **Type/interface consistency:** `empty_state_card`'s signature (`icon_svg, title, message, cta_label, cta_page`) is defined once in Task 3 Step 1 and used identically (same parameter names, same call shape) in Task 3 Step 4 and Task 4 Step 2.
