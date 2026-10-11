# Simulação

Simulação projects the loaded portfolio with Monte Carlo: 10–3,000 paths over 1–10 years. It compares a multivariate normal model with joint historical-day resampling and shows statistics, model comparison, individual paths, percentile fans, the terminal distribution, and an analyst synthesis. It exports a summary CSV and a percentiles-by-date CSV.

## Sub-features

- `sim-gate` shows the empty state (`Portfólio não configurado` / `Atualize o portfólio`) with `Ir para Portfólio` when no current portfolio is loaded.
- `sim-presets` uses the `Pontos de partida` group buttons `Rápida` (500 paths, 1 year), `Padrão` (2,000, 1), and `Longo prazo` (2,000, 5). Each button's name includes its description, and the active one reads `✓ <name> (selecionado)`.
- `sim-inputs` covers `Número de simulações (trajetórias)`, `Capital inicial (R$)`, `Horizonte (anos)`, and `Janela de calibração` (`Histórico do Portfólio`, `2 Anos`, `3 Anos`, `5 Anos`).
- `sim-run` uses `Rodar Simulação`, which shows `Estatísticas da Simulação Monte Carlo`, `Comparação de modelos`, `Faixas de percentis`, and `Síntese do Analista`.
- `sim-display` uses `Número de trajetórias exibidas` (5–50). It changes only the chart and does not rerun the simulation.
- `sim-export` uses `Resumo (CSV)` and `Percentis por Data (CSV)`.
- `sim-restore` restores a saved portfolio after a session reload with `Carteira salva restaurada…`.

## How to get to it (user POV)

- Click `Simulação` in the sidebar nav.
- Click `Abrir Simulação` in the Portfólio next-step card.

## Driving it with b3lab-verify

Preconditions:

- Baseline from `README.md`, plus a **loaded** portfolio: run the Portfólio recipe through **Load** in this same run.

- **Gate.** Before loading a portfolio, run `$V browser click "Simulação" --role link` and `$V browser wait-text "Portfólio não configurado"`. The link `Ir para Portfólio` is visible.
- **Open with portfolio.** After a Portfólio load, run `$V browser click "Simulação" --role link` and `$V browser wait-text "Alocação usada na simulação" --timeout 120`. The weights match the Portfólio weights (1 decimal).
- **Preset + input.** Run `$V browser click "Rápida" --contains` and `$V browser fill "Horizonte (anos)" 2`. `fill` prints `-> "2"`, not `"12"`.
- **Run.** Run `$V browser click "Rodar Simulação"` and `$V browser wait-text "Estatísticas da Simulação Monte Carlo" --timeout 300`. The text shows `Projeção 2 ano(s) — 500 simulações` and `…% dos cenários do modelo normal terminaram acima do capital inicial.` Capture to `$EV/simulacao/`.
- **Display only.** Save `$V browser text main > $EV/simulacao/01-run.txt`, then run `$V browser fill "Número de trajetórias exibidas" 20`, `$V browser press Tab`, and `$V browser wait-idle`, and save `02-display.txt`. In `diff`, only the `Trajetórias Individuais` chart's axis ticks may change (e.g. `R$ 12,000` → `R$ 15,000`). Every statistic, percentile, and synthesis line stays identical.
- **Export.** Run `$V browser click "Resumo (CSV)" --download` and `$V browser click "Percentis por Data (CSV)" --download`. They land as `monte_carlo_resumo_<date>.csv` and `monte_carlo_percentis_<date>.csv`. The percentiles CSV has a header `,P5,P25,P50,P75,P95` plus `years×252+1` data rows (506 lines for 2 years). Its first data row is all `10000.0`.
- **Stale after input edit.** Change `Capital inicial (R$)` without running. Results hide and `Configure os parâmetros acima e clique em Rodar Simulação…` shows.

## Gotchas

- Simulation needs a **current** portfolio. Any selection change elsewhere (including Screener `Ver preço-alvo`) turns this page into `Atualize o portfólio`.
- Results are seeded (42/43), so identical inputs reproduce identical numbers. Compare numbers across runs only with identical inputs and identical data.
- P5 is not VaR. Assert the copy says so; do not "fix" it.
- Results are process-local and expire after 2 h idle. After `down`/`up`, export links from old runs 404.
