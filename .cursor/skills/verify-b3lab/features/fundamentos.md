# Fundamentos

Fundamentos is the home page. The user selects up to 20 tickers (by search, quick picks, sectors, or favorites) and clicks `Analisar`. The page then shows sector identity, market data, financial indicators, debt health, a multiples comparison chart, Yahoo price-target consensus, and an analyst synthesis per company. The user can favorite tickers and export the raw fundamentals CSV.

## Sub-features

- `home-select` builds the selection via the `Escolha ações para analisar` combobox or the quick-pick buttons (`WEGE3`, `PETR4`, `VALE3`, `ITUB4`, `RENT3`, `ABEV3`, `EGIE3`, `RADL3`) shown while the selection is empty.
- `home-sectors` opens `Setores e favoritos` and uses the multi-select `Escolha um ou mais setores` or the sector buttons (`Bancos`, `Petróleo e Gás`, …). These auto-pick the most liquid tickers and show `Selecionamos os N ativos mais líquidos…`.
- `home-analyze` uses `Analisar` to show sections `Setor`, `Informações de Mercado`, `Indicadores Financeiros`, `Saúde Financeira`, `Preço-alvo e consenso`, and `Síntese do Analista`. With 2 or more tickers it also shows `Comparação Gráfica de Múltiplos`.
- `home-favorite` toggles `☆ Favoritar · <T>` / `★ Favoritado · <T>`. The ticker then appears under `Favoritos` with a `Remover <T> dos favoritos` button.
- `home-export` downloads the raw fundamentals CSV via `Exportar dados fundamentalistas (CSV bruto)`.
- `home-consensus` refreshes Yahoo targets only via `Atualizar consensos`.
- `home-source` runs `Atualizar dados Fundamentus`, which clears the Fundamentus cache and shows `Cache Fundamentus atualizado (…)`.
- `home-clear` empties the selection with `Limpar`.

## How to get to it (user POV)

- Click `Fundamentos` or the `B3Lab, página inicial` logo link in the sidebar.
- Click `Ver preço-alvo` on a selected Screener row. This replaces the selection with that ticker.
- Deep link `/?unavailable=<T>` shows the "fora da lista local B3" notice. It is reached from the Screener for unlisted tickers.

## Driving it with b3lab-verify

Preconditions:

- Baseline from `README.md` with an empty selection.
- Fundamentus and Yahoo Finance reachable.

- **Open page.** Run `$V browser open /` and `$V browser wait-text "Selecione ativos para analisar"`. The quick-pick grid `Seleção rápida · ações líquidas` is visible.
- **Quick pick.** Run `$V browser click "WEGE3" --role button --contains`. The chip `WEGE3` and `1/20 ativos selecionados.` show, and so does the notice `Clique em Analisar para buscar os indicadores…`.
- **Search add.** Run `$V browser fill "Escolha ações para analisar" ITUB4` and `$V browser press Enter`. The caption reads `2/20 ativos selecionados.`
- **Analyze.** Run `$V browser click "Analisar"` and `$V browser wait-text "Preço-alvo e consenso" --timeout 300`. Sections `Setor`, `Informações de Mercado`, `Comparação Gráfica de Múltiplos`, and `Síntese do Analista` show cards `WEGE3 · …` and `ITUB4 · …`. Capture `snapshot` + `screenshot --full` to `$EV/fundamentos/`.
- **Export.** Run `$V browser click "Exportar dados fundamentalistas (CSV bruto)" --download`. A CSV lands in `$EV/downloads/` with one row per analyzed ticker.
- **Favorite.** Run `$V browser click "☆ Favoritar · WEGE3"` then `$V browser wait-text "★ Favoritado · WEGE3"`. Second view: `$V db "select uid, ticker from watchlist"` lists `WEGE3`. `$V browser click "Setores e favoritos" --role summary` then shows `WEGE3` with `Remover WEGE3 dos favoritos`.
- **Sector auto-pick.** Run `$V browser click "Limpar"`, `$V browser click "Setores e favoritos" --role summary`, and `$V browser click "Bancos"`. The notice `Selecionamos os N ativos mais líquidos de M disponíveis…` shows and the chips are filled.

## Gotchas

- Changing the selection discards results immediately. `Analisar` must be clicked again; results never auto-refresh.
- Quick-pick and sector buttons render only while the selection is empty.
- Fundamentos favorites and Screener favorites share the same `watchlist` table for the session.
- A ticker missing from Fundamentus shows a `Dados Fundamentus indisponíveis para …` warning. That is a source failure to report, not a pass.
- The selection is shared with Portfólio. Analyzing here does **not** load a portfolio.
