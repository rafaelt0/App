# B3Lab verification map

This directory is the maintained source for verifying B3Lab's user-facing behavior. Read this index before driving the app, then use the matching feature file as the recipe. All commands assume `V=.cursor/skills/verify-b3lab/scripts/b3lab-verify` from the repo root and `B3LAB_VERIFY_RUN` exported (see `../SKILL.md`).

## Baseline preconditions

- `$V up` started this run, and `$V doctor` reports `doctor: healthy`.
- The run has a fresh Chromium profile, so it starts with a new anonymous session, an empty selection, no favorites, and no saved portfolio, in a scratch SQLite DB.
- Live network access to Yahoo Finance (prices, consensus), Fundamentus (fundamentals, screener), BCB (Selic), and Google News (news). Nothing is mocked.
- `EV=/tmp/b3lab-verify/evidence/$B3LAB_VERIFY_RUN` is the evidence root.
- Never drive an instance that this run did not start. A user-started `python run.py` on 3000/8000 is off limits.

## Driving conventions

- The session is shared across pages. The selection made on one page is the selection on every page, and the server bumps `revision` on each change. Plan recipes in order: Portfólio must be loaded before Simulação and Notícias have anything to show.
- Navigate the way a user does: `$V browser click "<Nav label>" --role link`. Use `browser open /route` only for the first page of a recipe or a deliberate deep link.
- Address controls by visible label or button text, exactly as written here. They are Brazilian Portuguese, with accents.
- After every action, `wait-text` for the resulting state instead of sleeping. Analyses take 1–10 s with warm caches. The first FinBERT load or a cold Yahoo fetch can take minutes, so pass `--timeout`.
- Reset between independent recipes with `Limpar carteira` on Portfólio, which deletes the saved portfolio and the selection. Or use a fresh run.

## Proof and skip reporting

- Capture `NN-<step>.aria.txt` + `.png` under `$EV/<feature>/` for each user action and its resulting state.
- Save CSVs through the real export buttons (`click … --download`); they land in `$EV/downloads/`.
- Persisted mutations (portfolio, favorites) need a read-only second view: `$V db "<SELECT>"` saved as `.tsv`, or a reload that shows restored state.
- Record the feature ID and entry point with every artifact.
- If a provider is down, report the exact alert or warning text and the command that hit it. Do not count the step as verified.
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with b3lab-verify` starts with `Preconditions:` and uses labeled bullets. Each bullet pairs a user action with an exact command and an observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Portfólio](./portfolio.md) covers selection, optimization strategies, results, CSV export, the persisted portfolio, and stress history.
- [Fundamentos](./fundamentos.md) covers ticker and sector selection, Analisar, fundamentals, consensus, favorites, and CSV export.
- [Screener](./screener.md) covers presets, criteria, sorting, columns, row selection, the handoff to Fundamentos, favorites, and CSV export.
- [Simulação](./simulacao.md) covers Monte Carlo runs on the loaded portfolio, presets, display-only trajectories, and two CSV exports.
- [Notícias](./noticias.md) covers news for the loaded portfolio, the FinBERT or lexical engine, filters, pagination, and refresh.

## Last full drive

On 2026-10-10, after the page rebrand, every recipe was driven against `2dc50f5` in run `skillproof` with live providers. The pass covered the main steps of all five features: Portfólio load/export/DB/HRP/manual/stale/clear; Simulação run/display/both CSVs; Notícias analyze (FinBERT)/filter; Screener preset/criteria/select/favorite/export/columns/handoff; Fundamentos analyze/export/favorite/sector/unfavorite. These steps were not driven: `pf-window`, `pf-stress`, `pf-refresh`, `news-refresh`, `news-paging`, `scr-more`, `home-consensus`, `home-source`, `sim-restore`, and the `/?unavailable=<T>` deep link. After any UI change, re-drive a recipe before trusting its labels.
