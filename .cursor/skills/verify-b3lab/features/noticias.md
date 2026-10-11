# Notícias

Notícias fetches Google News RSS for each ticker in the loaded portfolio (last 7 days, up to 10 unique items per ticker). It classifies each article with FinBERT-PT-BR, or with a clearly labelled lexical fallback, and shows a weight-based tone score (0–100), coverage, per-ticker tone, and a filterable, paginated article feed.

## Sub-features

- `news-gate` shows the empty state `Portfólio não configurado` (no selection) or `Atualize o portfólio` (selection without a current load).
- `news-analyze` runs automatically when the page opens with a loaded portfolio and shows `Tom da amostra disponível`.
- `news-engine` shows the badge `FinBERT-PT-BR disponível` or `PLN Léxico (fallback)`. Each article's `Detalhes do algoritmo PLN (…)` names its engine.
- `news-filters` covers `Filtrar por ativo`, `Ordenar por`, `Filtrar por sentimento`, and `Filtrar por intensidade`. They do not change the score.
- `news-paging` uses `Ver mais N notícias (M restantes)`.
- `news-refresh` uses `Atualizar notícias`, which clears the shared RSS caches and reclassifies.

## How to get to it (user POV)

- Click `Notícias` in the sidebar nav.
- Click `Abrir Notícias` in the Simulação analyst synthesis.

## Driving it with b3lab-verify

Preconditions:

- Baseline from `README.md`, plus a **loaded** portfolio from the Portfólio recipe in this run.
- Google News reachable. The FinBERT model is downloaded from Hugging Face on first use; later runs use the HF cache.

- **Gate.** With a selection but no load, run `$V browser click "Notícias" --role link` and `$V browser wait-text "Atualize o portfólio"`.
- **Analyze.** After a Portfólio load, run `$V browser click "Notícias" --role link` and `$V browser wait-text "Tom da amostra disponível" --timeout 600`. The `Carteira e pesos` list shows `PETR4 49,8%` / `VALE3 50,2%` (Portfólio weights at 1 decimal), with `Amostra em <timestamp>`, an engine badge, `Nota do tom da carteira (/100)`, and `Feeds indisponíveis: nenhum.` (or the failed tickers). Capture to `$EV/noticias/`.
- **Filter.** Run `$V browser select "Filtrar por sentimento" Otimistas` and `$V browser wait-idle`. The line `Mostrando X de Y notícias após filtros` changes (e.g. `15 de 20` → `1 de 1`), while `Nota do tom da carteira: N/100` stays identical.
- **Paging.** If more than 15 articles match, run `$V browser click "Ver mais" --contains`. The visible count grows by up to 15, and once all are shown, `Todas as N notícias carregadas.` appears.
- **Refresh.** Run `$V browser click "Atualizar notícias"`, then `$V browser wait-text "Buscando feeds" --timeout 30` and `$V browser wait-text "Buscando feeds" --gone --timeout 600`. A new `Amostra em <timestamp>` shows.

## Gotchas

- The first load can spend minutes loading FinBERT. Use `--timeout 600`, and do not mistake the loading status for a hang.
- A per-ticker `Feed de notícias indisponível para <T>` warning is a real source failure. Report it; do not count it as a pass.
- `PLN Léxico (fallback)` is a legitimate state when the model is unavailable. Verify it is labelled, never presented as FinBERT.
- News content changes hourly. Assert structure, labels, and invariants (score unaffected by filters), not specific headlines.
- `Atualizar notícias` clears caches shared by all sessions in this process. That is harmless in an isolated run.
