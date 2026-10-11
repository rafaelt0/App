# Project knowledge base

## Overview

B3Lab uses Next.js and React for its five-page interface and a Python 3.11 FastAPI backend for Brazilian-equity calculations, data providers, persistence, and sentiment classification.

## Structure

```text
./
├── run.py                    # Starts Next.js and the private Python API
├── backend/                  # Domain API handlers and owned analysis lifecycle
├── frontend/app/             # Five pages, globals.css, and API proxy
├── frontend/components/      # Shared state, shell, controls, tables, and charts
├── frontend/public/          # Brand assets and page icons
├── utils/                    # Financial sources, calculations, cache, and SQLite
├── tests/                    # Deterministic API and calculation regressions
├── scripts/smoke_api.py       # Real-provider HTTP verification
├── docs/                     # Product briefs and historical implementation notes
├── brand/                    # B3Lab visual and accessibility rules
└── requirements*.txt         # Pinned Python dependencies
```

## Where to look

| Task | Location | Notes |
|---|---|---|
| Start the application | `run.py`, `README.md` | Default frontend port 3000, private API port 8000 |
| Change a feature page | `frontend/app/`, `frontend/lib/` | Home, portfolio, simulacao, noticias, screener |
| Change financial orchestration | `backend/home.py`, `portfolio.py`, `simulation.py`, `news.py`, `screener.py` | Python owns financial policy and source units |
| Change shared calculations or data | `utils/` | Preserve numerical-library behavior and provider distinctions |
| Change sessions or readiness | `backend/state.py`, `backend/app.py` | Server-issued identity, revisions, generations, in-flight leases |
| Change visual design | `DESIGN.md`, `brand/B3Lab-brand.md`, `frontend/app/globals.css` | Neutral dark system, one indigo brand, separate financial colors |
| Add or update tests | `tests/` | Isolated SQLite, deterministic fixtures, network mocks |
| Change CI or dependencies | `.github/workflows/tests.yml`, `requirements*.txt`, `frontend/package*.json` | Python tests and Next production build |

## Code map

| Symbol | Location | Role |
|---|---|---|
| `resolve_session` | `backend/state.py` | Resolves opaque cookies to anonymous owner state |
| `begin`, `publish` | `backend/state.py` | Prevent stale analysis publication |
| `require_portfolio` | `backend/state.py` | Separates weight-ready from return-ready portfolio context |
| `get_portfolio_prices` | `utils/portfolio_data.py` | Shared adjusted Yahoo price retrieval |
| `filter_stocks` | `utils/screener.py` | Applies only enabled fundamental criteria |
| `simulate_portfolio` | `utils/simulation.py` | Deterministic daily-rebalanced paths |
| `parse_rss_items` | `utils/news.py` | Recent-news parsing and deduplication |
| `useApp` | `frontend/components/AppProvider.tsx` | Shared selection, watchlist, and serialized context transitions |
| `PageHeader`, `Metrics`, `Table` | `frontend/components/ui.tsx` | Shared presentation with explicit units and nulls |

## Conventions

- Keep user-facing copy in Brazilian Portuguese and follow `DESIGN.md`.
- Keep financial calculations, source normalization, thresholds, classification, and exports in Python. React formats and presents calculated values.
- Distinguish fractions, percentage points, currency, multiples, and unitless ratios. Home detail scraper corrections do not apply to full-market data.
- Adjusted prices feed returns and optimizers. Unadjusted prices feed cotas and purchase estimates.
- Preserve PyPortfolioOpt cleaned weights, separate Markowitz L2 solve, simulation seeds 42 and 43, and chunk size 64.
- Validate requests at API boundaries. Never accept an arbitrary owner UID from a body or URL.
- Publish owner results only after revision and operation-generation checks. Protect in-flight owner contexts from eviction.
- Keep tests deterministic and isolated. Mock network-dependent behavior in pytest; use `scripts/smoke_api.py` for live-provider checks.
- The Next proxy forwards transport only. It does not calculate, classify, or cache private responses.
- Shared page components and icons live in `frontend/`; do not reintroduce Python UI rendering.

## Anti-patterns

- Do not rewrite whole files for focused changes, delete active features, or add dependencies without a reason.
- Do not change financial behavior unless the task explicitly calls for it or fixes a verified defect.
- Do not use gradients, glow, shadows, decorative card noise, uppercase eyebrows, color-only status, or forced-open mobile navigation.
- Do not classify unknown stock sectors.
- Do not present an anonymous session as a durable user account or invent identity recovery.
- Do not fabricate market data or neutral sentiment when a source fails. Identify real fallbacks and missing data.
- Do not run multiple Python workers with process-local analysis artifacts and FinBERT state.

## Commands

```bash
pip install -r requirements-dev.txt
npm --prefix frontend ci
python run.py
PYTHONPATH=. pytest
npm --prefix frontend run typecheck
npm --prefix frontend run build
python scripts/smoke_api.py
```

## Notes

Portfolio and watchlist records remain in SQLite under random owner identities. Cookies are browser-session cookies; analysis results are temporary. Existing data is not erased or assigned to a new visitor. Historical documents may describe the removed Streamlit implementation and are not current launch instructions.
