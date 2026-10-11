# Screener

The screener filters the full Fundamentus market with seven starting presets and optional criteria (P/L range, ROE, 5-year revenue growth, dividend yield, 2-month liquidity). It sorts the results, lets the user choose visible columns, select a row to favorite it or send it to Fundamentos, and export all matches as CSV.

## Sub-features

- `scr-preset` uses the `Ponto de partida` select: `Explorar B3` (default), `Lucro a preço moderado`, `Renda atual`, `Qualidade rentável`, `Alta liquidez`, `Crescimento com lucro`, `Personalizado`.
- `scr-criteria` toggles checkboxes `Aplicar faixa de P/L`, `Aplicar ROE mínimo`, `Aplicar crescimento da receita (5 anos)`, `Aplicar dividend yield mínimo`, `Aplicar liquidez mínima`, with their numeric inputs and the liquidity slider.
- `scr-sort` uses the `Ordenar resultados` select.
- `scr-columns` uses the `Colunas exibidas` checkbox chips (one checkbox per column, named by its label). It changes only the visible table, not the selection or the CSV.
- `scr-more` shows 25 rows at a time. `Ver mais N ações (M restantes)` reveals 25 more; `Todas as N ações exibidas.` shows once all rows are visible.
- `scr-select` uses the row buttons `Selecionar <T>`, which show the `Ação selecionada: <T>` panel directly above the table.
- `scr-handoff` uses `Ver preço-alvo`, which replaces the selection and opens Fundamentos.
- `scr-favorite` uses `☆ Favoritar` / `★ Favoritado` on the selected row.
- `scr-export` uses `Exportar todos os N resultados como CSV`.
- `scr-reset` uses `Redefinir para Explorar B3`. `scr-source` uses `Atualizar dados Fundamentus`.

## How to get to it (user POV)

- Click `Screener` in the sidebar nav.

## Driving it with b3lab-verify

Preconditions:

- Baseline from `README.md`.
- Fundamentus reachable.

- **Open page.** Run `$V browser click "Screener" --role link` (or `browser open /screener` for the first page) and `$V browser wait-text "ativos correspondem aos filtros" --timeout 180`. The default line reads like `N de M ativos correspondem aos filtros · Liquidez 2m ≥ R$ 1.000.000`.
- **Preset.** Run `$V browser select "Ponto de partida" "Renda atual"` and `$V browser wait-text "DY ≥ 5%"`. The summary lists `Liquidez 2m ≥ R$ 1.000.000 · ROE ≥ 10% · DY ≥ 5%` and the match count drops. Capture to `$EV/screener/`.
- **Criterion.** Run `$V browser click "Aplicar faixa de P/L" --role checkbox`, `$V browser fill "P/L máximo" 10`, and `$V browser wait-text "P/L"`. The active-filter line gains a P/L entry. If the minimum is above the maximum, the alert `O P/L mínimo não pode ser maior que o P/L máximo.` shows.
- **Select row.** Run `$V browser click "Selecionar PETR4"` (or `--contains --nth 0` on `Selecionar`) and `$V browser wait-text "Ação selecionada:"`. The panel shows `Ação selecionada: <T>`.
- **Favorite.** Run `$V browser click "☆ Favoritar"` then `$V browser wait-text "★ Favoritado"`. Second view: `$V db "select ticker from watchlist"`.
- **Export.** Run `$V browser click "Exportar todos os" --contains --download`. The CSV row count equals N from `N de M ativos…`, and the CSV has column `Papel`.
- **Columns.** Run `$V browser click "P/VP" --role checkbox`. The `P/VP` column header disappears from the table; the CSV export is unchanged.
- **More rows.** Run `$V browser click "Ver mais" --contains` and confirm the table grows from 25 to 50 `Selecionar` rows (`$V browser eval 'document.querySelectorAll("tbody tr").length'`).
- **Handoff.** Run `$V browser click "Ver preço-alvo"` and `$V browser wait-text "Selecione ativos para analisar"`. The URL is `/` and the only chip is the selected ticker.

## Gotchas

- Filters apply after a 200 ms debounce, and every change re-queries. Always `wait-text` for the new active-filter text, not just the old count line.
- `Ver preço-alvo` **replaces** the shared selection. Any loaded portfolio becomes stale, and Simulação and Notícias then show `Atualize o portfólio`.
- Missing data never passes a criterion. A low count is not a bug by itself.
- The liquidity control is a range slider. Use `browser set "Liquidez média 2 meses mínima (R$)" <value>`, not `fill`.
- The row button's accessible name is its `aria-label` `Selecionar <T>`. Its visible text is only `Selecionar`.
- Rows past the first 25 are not in the DOM until `Ver mais` is clicked. `Selecionar <T>` for a low-liquidity ticker fails with "no visible button" until you reveal it.
