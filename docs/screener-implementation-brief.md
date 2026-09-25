# Screener: implementation brief

This brief records the accepted product direction and acceptance criteria for a trustworthy, simpler B3 screener—not a claim to reproduce famous investment strategies. The implementation lives in `pages/5_Screener.py` and `utils/screener.py`; keep this document aligned with their behavior. The market-target panel is integrated into `Main_Page.py`.

## Agreed product direction

Replace the four branded presets (Bazin/Barsi, Graham, Lynch, Magic Formula) with **three transparent, editable starting points**:

| Preset | Only active criteria at selection | Meaning |
| --- | --- | --- |
| **Explorar B3** (initial/reset state) | `liq2m >= R$ 1,000,000` | Liquid stocks; no hidden valuation, profitability, or dividend limits. |
| **Lucro a preço moderado** | `0 < pl <= 15`, `roe >= 12%`, `liq2m >= R$ 1,000,000` | Profitable companies at a moderate P/L; not a buy recommendation. |
| **Renda atual** | `dy >= 5%`, `roe >= 10%`, `liq2m >= R$ 1,000,000` | Current/historical yield, **not** evidence of dividend consistency or future payouts. |

Use decimal units internally (`roe >= .12`, `dy >= .05`); the form displays percentages. Thresholds are adjustable heuristics, not investment advice. A manually changed preset becomes **Personalizado** (or visibly “modified”) so its label cannot imply unchanged criteria. Selecting a preset replaces the previous filter state; do not stack today’s P/L, P/VP, EV/EBITDA, etc. defaults behind it. Only an **enabled** criterion filters rows, and rows missing its required value must fail it. Disabled criteria must not filter anything, even missing values. Sorting must **never** change membership. Prefer liquidity-descending as a neutral initial sort over the composite Score.

A fourth **Crescimento com lucro** preset was discussed but is **deferred**, not required for this implementation: 5-year revenue growth >= 10%, ROE >= 12%, liquidity >= R$ 1m. Do not bring back “Lynch PEG”: current `peg` divides P/L by *revenue* growth, not earnings growth. Likewise, do not call `P/L × P/VP` a “Graham Number” (the page currently does). Avoid Magic Formula until sector coverage and eligibility are reliable.

## UX changes

- Show **matched / total**, real data age, and all active limits **above** the results (including the liquidity threshold in Explorar B3). Provide one clear reset to Explorar B3. Don't label active default filters simply “configuração padrão.”
- Use **one visible preset selector**; remove the duplicate quick-preset buttons. Keep custom controls grouped (e.g., Valuation, Quality, Dividends, Liquidity), with optional/disabled state clear. Validate `min <= max` for ranges and make the zero-results state actionable.
- Make the table the focus: keep numeric columns numeric so header sorting is numeric; use Streamlit's native numeric formatting where practical. Show a manageable set of columns initially, with a way to see the others. Remove the three “Destaques” cards if they cannot represent the selected sort; allow **any** displayed ticker, not only the top three, to be opened in `Main_Page.py` with its ticker loaded in the market-target panel, or favorited.
- Export **all** matched rows to CSV and compute summary averages over the same full result set. `df_filtrado.head(200)` currently runs before both export and averages. With ~1,000 source rows, removing the cap entirely is simpler than adding pagination. If a cap remains, it may apply only to the on-screen display and must be explicit.
- Treat the composite Score as a heuristic, not an investment recommendation; avoid highlighting its top three as “picks.” Avoid extra dependencies, new strategy engines, or complex ranking logic.

## Data correctness to fix alongside the UI

1. The debt metric is mapped from Fundamentus' `Dív.Líq/ Patrim.` header. No debt filter is currently exposed; if one is added, label it *net debt/equity* and make missing data fail while active.
2. Freshness is implemented by evicting Fundamentus' non-expiring `/resultado.php` HTTP response whenever the one-hour Streamlit cache misses. The page shows the successful fetch timestamp; keep refresh/cache invalidation covered by offline tests.
3. **Sector coverage is incomplete.** `get_listed_stocks()` uses `acoes-listadas-b3.csv`; in the local cached snapshot, 333 of 994 Fundamentus tickers match. The old Magic exclusion lets unknown sectors through via `~df['setor'].isin(...)`. The three new presets are not sector-dependent, so **defer** rebuilding sector coverage, but do not silently reintroduce sector-based exclusions or present unknown sectors as classified.
4. Keep data-source failure and missing required columns distinct from “zero matches”; no preset should silently turn into a weaker one when Fundamentus changes a header.

## Relevant code and acceptance checks

- Page and filter state: `pages/5_Screener.py`; shared bulk data: `utils/market_data.py`; cache clearing: `utils/home_data.py:clear_fundamentus_cache`; ticker/sector CSV: `acoes-listadas-b3.csv`; existing test conventions: `tests/test_market_data.py`.
- Check with small fixture rows that each preset has **exactly** its listed predicates, a missing active field fails, disabled filters have no effect, manual edits show modified/custom state, and changing sort doesn't change the matched ticker set.
- Check that the CSV row count equals the total shown, numeric columns still sort numerically, reset restores only Explorar B3, and refresh can actually obtain newer data. Keep any network-dependent test mocked; do not rely on the live Fundamentus site.
- The market-target panel lives in `Main_Page.py`; preserve its ticker handoff/query parameter when updating screener navigation.
