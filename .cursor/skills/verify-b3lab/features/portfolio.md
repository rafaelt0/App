# Portfólio

Portfólio lets a user pick 2–20 B3 tickers, choose a history window, capital, and allocation strategy, and press `Carregar portfólio`. The page shows optimized weights, a share-purchase table, charts, risk metrics, and crisis stress tests. It exports the allocation as CSV and saves the weighted portfolio for the anonymous session, which unlocks Simulação and Notícias.

## Sub-features

- `pf-select` adds and removes tickers with the `Ações da carteira` combobox. The caption reads `N/20 ativos selecionados.`
- `pf-load` runs `Carregar portfólio`, which shows `Pesos do Portfólio (%)`, `Sugestão de Compra de Cotas`, and `Resumo de Desempenho`.
- `pf-strategy` switches between `Otimização de Mínima Volatilidade`, `Otimização de Markowitz (Média-Variância)`, `Otimização Hierarchical Risk Parity (HRP)`, and `Alocação Manual` (which shows `Cotas de <TICKER>` inputs). After a load, changing a parameter recalculates without another click.
- `pf-window` changes the history window with `Período de análise` (`2 Anos (Padrão)`, `1 Ano`, `3 Anos`, `5 Anos`, `6 Meses`, `Personalizado (Dias)` with `Dias de lookback`).
- `pf-export` downloads `portfolio_alocacao_<date>.csv` via `Exportar alocação (CSV)`.
- `pf-persist` shows `Alocação salva` and creates a `portfolio` row for the session uid. `Limpar carteira` removes it.
- `pf-stress` loads crisis history with the `Carregar histórico longo para crises (desde 2008)` checkbox.
- `pf-refresh` runs `Atualizar cotações`, which clears cached quotes and requires a new load.
- `pf-handoff` sends the user to Simulação with `Abrir Simulação`.

## How to get to it (user POV)

- Click `Portfólio` in the sidebar nav (or under `Menu` below 1024 px).
- Click `Abrir Portfolio` in the next-step card after Fundamentos results.
- Click `Configurar portfólio` / `Ir para Portfolio` in the empty states on Fundamentos, Simulação, and Notícias.
- Deep link: `/portfolio?portfolio_tickers=PETR4,VALE3` preloads a selection and shows `Seleção carregada da análise de ativos…`.

## Driving it with b3lab-verify

Preconditions:

- Baseline from `README.md`, a healthy `doctor`, and an empty selection (fresh run, or `Limpar carteira` first).
- Yahoo Finance reachable.

- **Open page.** Run `$V browser open /` then `$V browser click "Portfólio" --role link` and `$V browser wait-text "Montagem da carteira"`. The empty state `Monte seu portfólio` is visible.
- **Add tickers.** Run `$V browser fill "Ações da carteira" PETR4`, `$V browser press Enter`, `$V browser wait-text "1/20 ativos selecionados"`, then `$V browser fill "Ações da carteira" VALE3`, `$V browser click VALE3 --role option --contains`, and `$V browser wait-text "2/20 ativos selecionados"`. Chips `PETR4` and `VALE3` show, with the notice `Confira a seleção e clique em Carregar portfólio.` Capture `snapshot $EV/portfolio/01-selection.aria.txt` + `screenshot …/01-selection.png`.
- **Load.** Run `$V browser click "Carregar portfólio"` and `$V browser wait-text "Pesos do Portfólio (%)" --timeout 300`. The text shows `2 ativos selecionados · N retornos diários completos (…)` and `Otimização de Mínima Volatilidade · Mínima Volatilidade · Alocação salva`. Weights for PETR4 and VALE3 sum to 100%. Capture `02-result.aria.txt`, `screenshot --full …/02-result.png`, and `$V browser text main > $EV/portfolio/02-result.txt`.
- **Export.** Run `$V browser click "Exportar alocação (CSV)" --download`. It prints `downloaded …/downloads/portfolio_alocacao_<date>.csv`. The file starts with a BOM, has header `,Peso (%)`, and has one row per ticker matching the on-screen weights (e.g. `PETR4,51.44`).
- **Persisted row.** Run `$V db "select uid, tickers, weights, datetime(updated_at,'unixepoch') from portfolio" > $EV/portfolio/db-portfolio.tsv`. Exactly one row has `["PETR4", "VALE3"]` and weights keyed `PETR4.SA`/`VALE3.SA` matching the CSV.
- **Strategy change.** Run `$V browser select "Estratégia de alocação" "Otimização Hierarchical Risk Parity (HRP)"` and `$V browser wait-text "(HRP) · Alocação salva" --timeout 300`. The mode line reads `Otimização Hierarchical Risk Parity (HRP) · Otimização Hierarchical Risk Parity (HRP) · Alocação salva`, and results recompute without clicking `Carregar portfólio`. Concentration warnings appear as `role=alert`; they are content, not failures.
- **Manual allocation.** Run `$V browser select "Estratégia de alocação" "Alocação Manual"`, `$V browser fill "Cotas de PETR4" 10`, `$V browser fill "Cotas de VALE3" 5`, and wait for `Pesos do Portfólio (%)`. With all shares at 0 the warning `A alocação manual precisa ter ao menos uma cota positiva.` shows and the button is disabled.
- **Stale selection.** Remove a ticker with `$V browser click "Remover VALE3"`. Results disappear and `A seleção mudou. Clique em Carregar portfólio para atualizar a análise.` or `Confira a seleção…` shows. `Carregar portfólio` is disabled below 2 tickers.
- **Clear.** Run `$V browser click "Limpar carteira"` and `$V browser wait-text "Carteira e seleção removidas."`. The `$V db "select count(*) from portfolio"` row for that uid is gone.

## Gotchas

- `Carregar portfólio` is required after every **ticker** change. Parameter changes (strategy, capital, window, gamma) auto-recompute only after a load.
- The first `browser open` in dev mode compiles the route and can take 20–30 s. `open` waits for load, but the page may still show `Carregando a sessão e a lista de ativos.`, so always `wait-text` a page heading.
- Results render below the fold. Use `screenshot --full`, or the viewport shot shows only the form.
- The dev-mode Next "N" badge sits bottom-left over the sidebar footer. If a click reports `covered by`, scroll or target a different handle.
- Two heavy analyses at once return `Duas análises já estão em execução…` (HTTP 503). Do not fire parallel loads from several runs at the same API.
- Weights CSV values are percentages with a dot decimal. The UI shows a pt-BR comma (`51,44%`).
