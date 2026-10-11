# Simulação

Simulação projects the loaded portfolio with Monte Carlo: 10–3,000 paths over 1–10 years. It compares a multivariate normal model with joint historical-day resampling and shows statistics, model comparison, individual paths, percentile fans, the terminal distribution, and an analyst synthesis. It exports a summary CSV and a percentiles-by-date CSV.

## Sub-features

- `sim-gate` shows the empty state (`Portfólio não configurado` / `Atualize o portfólio`) with `Ir para Portfolio` when no current portfolio is loaded.
- `sim-presets` uses buttons `Rápida` (500 paths, 1 year), `Padrão` (2,000, 1), and `Longo prazo` (2,000, 5).
- `sim-inputs` covers `Janela de calibração` (`Histórico do Portfolio`, `2 Anos`, `3 Anos`, `5 Anos`), `Número de Simulações`, `Capital Inicial (R$)`, and `Anos`.
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

- **Gate.** Before loading a portfolio, run `$V browser click "Simulação" --role link` and `$V browser wait-text "Portfólio não configurado"`. The button `Ir para Portfolio` is visible.
- **Open with portfolio.** After a Portfólio load, run `$V browser click "Simulação" --role link` and `$V browser wait-text "Alocação usada na simulação" --timeout 120`. The weights match the Portfólio weights (1 decimal).
- **Preset + input.** Run `$V browser click "Rápida"` and `$V browser fill "Anos" 2`.
- **Run.** Run `$V browser click "Rodar Simulação"` and `$V browser wait-text "Estatísticas da Simulação Monte Carlo" --timeout 300`. The text shows `Projeção 2 ano(s) — 500 simulações` and `…% dos cenários do modelo normal terminaram acima do capital inicial.` Capture to `$EV/simulacao/`.
- **Display only.** Run `$V browser fill "Número de trajetórias exibidas" 20` and `$V browser press Tab`. The statistics text stays identical, so diff `browser text main` before and after.
- **Export.** Run `$V browser click "Resumo (CSV)" --download` and `$V browser click "Percentis por Data (CSV)" --download`. The percentiles CSV has `years×252+1` rows (505 for 2 years) and its first row is all `10000`.
- **Stale after input edit.** Change `Capital Inicial (R$)` without running. Results hide and `Configure os parâmetros acima e clique em Rodar Simulação…` shows.

## Gotchas

- Simulation needs a **current** portfolio. Any selection change elsewhere (including Screener `Ver preço-alvo`) turns this page into `Atualize o portfólio`.
- Results are seeded (42/43), so identical inputs reproduce identical numbers. Compare numbers across runs only with identical inputs and identical data.
- P5 is not VaR. Assert the copy says so; do not "fix" it.
- Results are process-local and expire after 2 h idle. After `down`/`up`, export links from old runs 404.
