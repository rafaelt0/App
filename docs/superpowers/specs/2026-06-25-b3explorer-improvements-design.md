# B3 Explorer — Improvements Design

**Date:** 2026-06-25  
**Approach:** A — Bugs first, then performance, then UX, then Screener ranking  
**Files touched:** `pages/1_Portfolio.py`, `pages/2_Simulação.py`, `pages/4_Valuation.py`, `pages/5_Screener.py`

---

## 1. Bug Fixes

### 1.1 Correlation heatmap on prices instead of returns (`Portfolio.py:747`)

**Problem:** `data_yf.corr()` computes correlation of price levels. Price correlation is spurious — stocks trend upward together, so nearly all pairs show high positive correlation regardless of actual co-movement.

**Fix:** Replace with `returns.corr()`. Strip `.SA` suffix from column names after.

**Impact:** Heatmap now shows meaningful return correlation.

---

### 1.2 Deprecated monthly returns groupby (`Portfolio.py:793`)

**Problem:** `groupby([index.year, index.month]).apply(lambda x: (1+x).prod()-1)` is deprecated in pandas ≥ 2.2 and will raise a warning or break.

**Fix:** Replace with:
```python
monthly_ret = portfolio_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
monthly_ret_df = monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month]).first().unstack() * 100
```
Then rename columns with month abbreviations as before.

**Impact:** Forward-compatible monthly returns table.

---

### 1.3 GBM uses arithmetic returns instead of log-returns (`Simulação.py:196`)

**Problem:** The simulation computes `mu = aligned_returns.mean()` (arithmetic daily returns) and then does `exp(cumsum(r))`. This formula is only correct when `r` are log-returns. Using arithmetic returns overestimates expected terminal wealth due to Jensen's inequality.

**Fix:** Change two lines:
```python
# after
log_returns = np.log(1 + aligned_returns)
mu = log_returns.mean().values
cov = log_returns.cov().values
```

**Impact:** Monte Carlo projections are no longer systematically optimistic.

---

### 1.4 BCB package inconsistency (`Valuation.py:59` vs `Portfolio.py:17`)

**Problem:** `Portfolio.py` uses `from bcb import sgs`. `Valuation.py` uses `from python_bcb import sgs`. Different PyPI packages — if only one is installed, one page fails.

**Fix:** Change `Valuation.py` to use `from bcb import sgs`, matching `Portfolio.py`.

**Impact:** Single BCB dependency, consistent Selic rate source across pages.

---

## 2. Performance

### 2.1 Cache IBOVESPA download (`Portfolio.py:764`)

**Problem:** `yf.download("^BVSP", ...)` is called uncached inside the button block — re-downloads on every button press.

**Fix:** Extract into `@st.cache_data(ttl=3600)` function `get_benchmark_prices(start_date)`.

**Impact:** IBOVESPA fetched once per session per date range.

---

### 2.2 Cache efficient frontier random simulation (`Portfolio.py:206`)

**Problem:** 5,000 random portfolio simulations + 25 quadratic programs run on every button click, even with identical inputs.

**Fix:** Extract numerical computation into a cached helper taking hashable inputs (tuples). Plotly figure construction stays outside the cache.

**Impact:** Repeated button clicks with same inputs skip the simulation.

---

## 3. UX — Matplotlib → Plotly

Three charts in `Portfolio.py` use `matplotlib` + `st.pyplot`: drawdown (line 1326), rolling beta (line 1394), rolling Sharpe (line 1413). They are static and theme-inconsistent.

- **Drawdown:** `go.Scatter(fill='tozeroy', fillcolor='rgba(255,23,68,0.25)', line=dict(color='#ff1744'))`
- **Rolling beta:** `go.Scatter` + `add_hline(y=1, line_dash='dash', line_color='#ffd600')`
- **Rolling Sharpe:** `go.Scatter` + `add_hline(y=0, line_dash='dash', line_color='#94a3b8')`

All use `apply_plotly_theme`. Same data, better rendering.

---

## 4. Screener — Magic Formula Composite Rank (`Screener.py`)

**Problem:** "Fórmula Mágica" preset uses hard thresholds. Greenblatt's actual method is a composite rank: rank by ROIC descending + rank by EV/EBIT ascending, sum ranks, sort ascending.

**Fix:** When active preset is "Fórmula Mágica", compute and display `rank_magic` column:
```python
df_filtrado["rank_roic"]   = df_filtrado["roic"].rank(ascending=False)
df_filtrado["rank_evebit"] = df_filtrado["evebit"].rank(ascending=True)
df_filtrado["rank_magic"]  = df_filtrado["rank_roic"] + df_filtrado["rank_evebit"]
df_filtrado = df_filtrado.sort_values("rank_magic")
```
Existing filters remain as pre-screen to exclude negative-ROIC / negative-EV/EBIT companies.

**Scope:** ~25 lines in `Screener.py`. Other presets unaffected.

---

## Implementation Order

1. Bug fixes (1.1 → 1.4)
2. Performance (2.1 → 2.2)
3. UX charts (3.1 → 3.3)
4. Screener rank (4)

## Files Changed

| File | Changes |
|------|---------|
| `pages/1_Portfolio.py` | Fixes 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 3.3 |
| `pages/2_Simulação.py` | Fix 1.3 |
| `pages/4_Valuation.py` | Fix 1.4 |
| `pages/5_Screener.py` | Feature 4 |
