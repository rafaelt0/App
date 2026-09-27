# Utility Modules

## OVERVIEW
Shared Python modules provide data access, calculations, persistence, and reusable UI for the Streamlit pages.

## WHERE TO LOOK
| Area | Location | Notes |
|------|----------|-------|
| Market data | `market_data.py` | Listed tickers, Fundamentus data, and target prices |
| Home analysis | `home_data.py`, `home_render.py` | Fundamentals and home-page panels |
| Portfolio | `portfolio_data.py`, `portfolio_charts.py` | Prices, risk calculations, and portfolio charts |
| Simulation | `simulation.py` | Monte Carlo and bootstrap calculations |
| News | `news.py` | RSS parsing, ranking, and sentiment helpers |
| Screener | `screener.py` | Presets, pure filtering, and CSV preparation |
| Persistence | `db.py`, `identity.py` | SQLite cache, watchlist, portfolio, and session identity |
| Shared presentation | `ui.py`, `charts.py`, `icons.py`, `formatting.py` | Reused page UI and chart formatting |

## CONVENTIONS
- Keep reusable data and calculation logic in these modules; page scripts own Streamlit page orchestration.
- Screener percentage thresholds are decimal fractions internally; convert only when displaying or exporting percentages.
- `_conn()` creates SQLite tables and runs `_migrate_legacy_watchlist()`; preserve that path when changing the schema.
- `cache_get()` defaults to a one-hour TTL; callers can choose shorter freshness windows.
- The legacy watchlist migration preserves old shared rows under the `legacy` uid.
- `portfolio_get()` returns `([], {})` when no saved row is available.
- Screener changes should stay aligned with `../docs/screener-implementation-brief.md` and its acceptance checks.

## ANTI-PATTERNS
- Do not let disabled screener criteria affect results. An enabled criterion with a missing required value must fail that row.
- Do not stack a newly selected screener preset on top of the previous criteria; preset selection replaces the filter state.
- Do not weaken screener results when a required input column is missing; `filter_stocks()` raises instead.
