# PROJECT KNOWLEDGE BASE

**Generated:** 2026-09-27
**Commit:** a5caef5
**Branch:** feature/omo

## OVERVIEW
B3 Explorer is a Python 3.11 Streamlit app for quantitative analysis of Brazilian stocks, with fundamentals, portfolios, Monte Carlo simulation, news, and a market screener.

## STRUCTURE
```text
./
├── Main_Page.py              # Streamlit home page and fundamentals
├── pages/                     # Portfolio, simulation, news, and screener pages
├── utils/                     # Shared data, calculations, persistence, and UI
├── tests/                     # Pytest coverage for pages and utility modules
├── docs/                      # Product briefs, designs, and implementation notes
├── brand/                     # B3Lab visual and accessibility rules
├── .streamlit/config.toml     # Theme and Streamlit server settings
├── requirements*.txt          # Pinned application and test dependencies
└── style.css                  # Shared application styling
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Start the app or change the home page | `Main_Page.py` | Streamlit entry point |
| Change a feature page | `pages/` | Streamlit discovers these scripts as multipage entries |
| Change shared data or calculations | `utils/` | Module-specific guidance is in `utils/AGENTS.md` |
| Change screener requirements | `docs/screener-implementation-brief.md` | Product direction and acceptance checks |
| Change visual design | `DESIGN.md`, `brand/B3Lab-brand.md`, `style.css`, `.streamlit/config.toml` | Design contract, brand identity, shared CSS, and Streamlit theme |
| Add or update tests | `tests/` | Page behavior and utility tests use pytest |
| Change CI or local environment | `.github/workflows/tests.yml`, `requirements*.txt` | CI uses Python 3.11 and `PYTHONPATH=. pytest` |

## CODE MAP
`Refs` are literal source-text match counts across the repository, not semantic LSP counts.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|-----:|------|
| `get_browser_uid` | Function | `utils/identity.py` | 21 | Random identity scoped to the current Streamlit session |
| `get_portfolio_prices` | Function | `utils/portfolio_data.py` | 16 | Shared portfolio and simulation price retrieval |
| `filter_stocks` | Function | `utils/screener.py` | 16 | Applies enabled screener criteria |
| `simulate_portfolio` | Function | `utils/simulation.py` | 14 | Monte Carlo portfolio simulation |
| `render_page_header` | Function | `utils/ui.py` | 12 | Shared page title, description, and icon |
| `get_listed_stocks` | Function | `utils/market_data.py` | 12 | Loads and normalizes the listed-stock universe |
| `portfolio_get` | Function | `utils/db.py` | 12 | Loads a session's saved portfolio |
| `parse_rss_items` | Function | `utils/news.py` | 10 | Parses and filters RSS items |

## CONVENTIONS
- Keep user-facing copy in Brazilian Portuguese; the app uses the dark editorial research system in `DESIGN.md`.
- Tests use pytest, deterministic input data, injected timestamps where needed, and `tmp_path`/`monkeypatch` to isolate side effects.
- Mock network-dependent behavior in tests; CI runs `PYTHONPATH=. pytest` on Python 3.11.
- Shared page headers and presentation helpers live in `utils/`; follow the B3Lab page accents and semantic color tokens.

## ANTI-PATTERNS (THIS PROJECT)
- Do not rewrite whole files for focused changes, delete existing features, or add dependencies without a clear reason.
- Do not change existing behavior unless the task is fixing a bug.
- Do not use gradients, glow, all-caps eyebrows, decorative card noise, color-only state indicators, or a forced-open sidebar on narrow screens.
- Do not present unknown stock sectors as classified; sector coverage is incomplete.
- Do not treat the random session identity as a durable user account; it is intentionally not derived from URL parameters.

## UNIQUE STYLES
- The Streamlit UI is in Brazilian Portuguese and uses B3Lab's layered charcoal surfaces, serif page titles, and semantic financial accents.
- The home page and four numbered scripts in `pages/` form the app's feature flow; shared helpers are imported from `utils/`.
- Portfolio and watchlist records are keyed by a random session identity in SQLite and may not survive the Streamlit session.

## COMMANDS
```bash
pip install -r requirements-dev.txt
streamlit run Main_Page.py
PYTHONPATH=. pytest
python -m py_compile <file>
```

## NOTES
- `Main_Page.py`, `pages/1_Portfolio.py`, `pages/2_Simulação.py`, and `pages/3_Notícias.py` exceed 500 lines; inspect their shared helpers before changing a large page flow.
- The Python LSP (`basedpyright`) and ast-grep binary were unavailable when this map was generated; symbol references above came from source-text searches.
