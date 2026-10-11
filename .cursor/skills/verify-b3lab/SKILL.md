---
name: verify-b3lab
description: Launch an isolated B3Lab instance (Next.js UI + private FastAPI, scratch SQLite) and drive its five Portuguese pages (Fundamentos, Portfólio, Simulação, Notícias, Screener) in headless Chromium over CDP, capturing ARIA snapshots, screenshots, CSV downloads, and SQLite rows as proof. Use when you must prove a B3Lab UI/API change works end to end the way a user would exercise it.
---

# Verify B3Lab

B3Lab's primary surface is the **web UI** at five routes: `/` (Fundamentos), `/portfolio`, `/simulacao`, `/noticias`, `/screener`. The browser only talks to the Next proxy at `/api/*`, which forwards to the private FastAPI (`backend.app:app`). Secondary surfaces: the API itself (`scripts/smoke_api.py` drives it over HTTP) and the pytest suite (mocked; not proof of user behavior).

Everything goes through one helper. Set it once per shell:

```bash
V=.cursor/skills/verify-b3lab/scripts/b3lab-verify   # run from the repo root
```

`$V` is bash; `$V browser …` execs `scripts/cdp.mjs` (Node ≥ 22, no npm deps) against the run's Chromium. `node .cursor/skills/verify-b3lab/scripts/cdp.mjs --help` lists every browser command.

## Launch

Prerequisites (README quickstart): `.venv` with `pip install -r requirements-dev.txt`, `npm --prefix frontend ci`, `/usr/bin/chromium`, network access to Yahoo Finance, Fundamentus, BCB, and Google News.

```bash
$V up                    # dev mode: python run.py with isolated env; ~10 s
export B3LAB_VERIFY_RUN=<id printed by up>
```

`up` does, in order:

- Picks the first free port triple: web `3100+n`, API `8100+n`, CDP `9300+n`.
- Starts `python run.py` (the documented launcher) with `B3LAB_PORT`, `B3LAB_API_PORT`, `B3LAB_ORIGIN=http://127.0.0.1:<web>`, and `B3LAB_DB=/tmp/b3lab-verify/runs/<id>/b3lab.db`. Your real `b3_data.db` is never touched.
- Starts headless Chromium with a throwaway profile, so you get a fresh anonymous session cookie, at 1440×1000.
- Reports ready only when API `/health`, the web root, and CDP all answer. It prints `ready.` and the `export` line.

Options: `--run <id>` names the run; `--production` runs `npm --prefix frontend run build` then `run.py --production`.

**Isolation limits.** Next allows only **one `next dev` per checkout** (`frontend/.next/dev/lock`). If the user already runs `python run.py` / `npm run dev`, `up` refuses. Do **not** stop their server: use `$V up --production` (separate `.next` output, coexists with their dev server) or ask. `--production` rebuilds `frontend/.next`, so never use it while the user or another verification run serves `run.py --production` from this checkout. Run one verification instance per checkout at a time. `http_cache.sqlite` (repo root, requests-cache) and the Hugging Face FinBERT cache are shared with the user's instance; both are caches, not user state.

## Doctor

Run first, and whenever anything looks off:

```bash
$V doctor
```

Read-only. Checks: launcher alive; web and API ports owned by **this run's** process tree; API `/health`; web root serves `<title>B3Lab`; the API process env has the scratch `B3LAB_DB` and the matching `B3LAB_ORIGIN`; Chromium CDP answers; evidence dir exists. Also prints the git revision and changed-path count at launch vs now. If the count changed, the dev server hot-reloaded edits made after launch. Ends with `doctor: healthy` or exits non-zero. `$V logs [N]` tails the combined uvicorn and Next log. `$V list` shows runs whose scratch state still exists.

## Drive

Browser commands take the user's visible names: button text or `aria-label`, link text, and `<label>` text for inputs and selects. They use real CDP mouse and keyboard events, and they fail loudly with the candidate names when nothing matches, the target is disabled, or the target is covered.

```bash
$V browser open /portfolio                         # path relative to the run URL
$V browser wait-text "Montagem da carteira"         # poll visible text; --gone, --timeout S (default 120)
$V browser fill "Ações da carteira" PETR4           # type into a labelled input (select-all + insertText)
$V browser press Enter                              # Enter|Escape|Tab|ArrowDown|ArrowUp
$V browser click VALE3 --role option --contains     # roles: button (default) link option checkbox summary any; --nth N
$V browser click "Carregar portfólio"
$V browser select "Estratégia de alocação" "Otimização Hierarchical Risk Parity (HRP)"   # by option text or value; several for multi-selects
$V browser set "Liquidez média 2 meses mínima (R$)" 5000000   # range sliders only
$V browser click "Exportar alocação (CSV)" --download          # waits for the file in <evidence>/downloads/
$V browser text main                                # visible text of <main>
$V browser snapshot <evidence>/<feature>/NN-step.aria.txt
$V browser screenshot <evidence>/<feature>/NN-step.png [--full]
$V browser viewport 375 812                         # responsive checks (<1024 px hides the sidebar behind "Menu")
$V db "select uid, tickers, weights from portfolio" # read-only SQL against the run's scratch DB
```

`browser eval <js>` exists for read-only inspection. Never use it, or `curl` to `/api`, to change app state: that skips the user path.

Shared UI handles in every page:

- Nav links in `nav[aria-label="Navegação principal"]`: `Fundamentos`, `Portfólio`, `Simulação`, `Notícias`, `Screener`.
- Ticker picker: a combobox labelled `Escolha ações para analisar` (Fundamentos) or `Ações da carteira` (Portfólio). Options are `[role=option]` buttons whose text starts with the ticker. Chips have `Remover <TICKER>` buttons. The caption reads `N/20 ativos selecionados.`
- Loading: `role=status` with a page-specific message. Errors: `role=alert` notices; `wait-text` timeouts print them.
- Table row pickers are buttons named `Selecionar <TICKER>`.

Feature recipes live in [`features/README.md`](features/README.md). Read the index, then the feature file, before driving.

## Evidence

Everything goes under `/tmp/b3lab-verify/evidence/<run-id>/`. It survives `down`. `down` also copies `app.log` there.

- Per feature: `<feature>/NN-<step>.aria.txt` and `.png` for each **action and resulting state**, not just the last screen.
- Downloads: `downloads/*.csv`, saved by the real `Export…` buttons.
- Side effects: save `$V db "…"` output as a `.tsv` next to the screenshots. The `portfolio`, `watchlist`, and `browser_sessions` tables hold the per-owner state.

Proof standards:

- Drive the real user path: nav links, buttons, form fields. No internal setters, test-only endpoints, or direct API calls standing in for clicks.
- Verify side effects alongside the UI. Examples: a `portfolio` row after `Carregar portfólio`, CSV values that match the on-screen weights, a `watchlist` row after `☆ Favoritar`.
- Use only real providers. pytest mocks networks, so a green pytest proves nothing about the UI. If a provider fails, report the failure as observed (the UI surfaces it as a warning or alert). Never substitute fixtures.
- Name the feature ID and entry point with every artifact. An entry point you did not drive is unverified, even if another path reaches the same state.

## Cleanup

```bash
$V down
```

`down` stops only what this run started: its Chromium pid, its `run.py` pid (which terminates its own uvicorn and Next groups), then the API and Next process groups recorded at `up`. It warns, but does not kill, if a port is still held by something else. It deletes `/tmp/b3lab-verify/runs/<id>/` (scratch DB, Chromium profile, logs) and lists the evidence it kept. Never `pkill node|python|chromium`: the user may run B3Lab or other Next apps. Run `down` after every failed attempt too. `$V list` finds stranded runs; `B3LAB_VERIFY_RUN=<id> $V down` removes them.

## Helpers

- `scripts/b3lab-verify`: `up [--production] [--run ID]`, `doctor`, `browser <cmd>…`, `db "<SELECT>"`, `logs [N]`, `down`, `list`. State lives in `/tmp/b3lab-verify/runs/<id>/state.env`; override the root with `B3LAB_VERIFY_HOME`.
- `scripts/cdp.mjs`: browser driver; see `--help`. Uses env `CDP_PORT`, `B3LAB_URL`, `DOWNLOAD_DIR`, which `b3lab-verify browser` sets.
- `scripts/tour <label>`: on a **fresh** run, captures full-page PNGs of all five routes at 1440/1024/768/375 px, first empty and then with a loaded PETR4/VALE3/ITUB4 portfolio. Writes `audit.jsonl` and `summary.tsv` (horizontal page overflow, controls under 44 px, document height) to `<evidence>/tour-<label>/`. Use it for visual or layout changes. Run it once before the edit and once after, then compare `summary.tsv` rows and the same-named PNGs. Takes about 4 minutes. Run it as `B3LAB_VERIFY_RUN=<id> .cursor/skills/verify-b3lab/scripts/tour before`.
- `browser wait-idle` waits until no loading spinner has been visible for 750 ms. `browser audit` prints those layout metrics for the current viewport.
