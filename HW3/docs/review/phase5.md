# Phase 5 Performance Recommendations — Experiment Execution

`backtest/single_feature_ic.py`, `backtest/quintile.py`, `backtest/model.py`, `backtest/portfolio.py`, `backtest/portfolio_batch.py`, `backtest/robustness.py`, `backtest/universe.py`, `backtest/_stats.py`

## Recommendations

### 1. Add shared in-memory cache for features parquet

IC, quintile, and robustness modules each independently read the full features parquet and call `ensure_forward_returns`. In a `run_all.py` execution this is ~1 GB of redundant I/O.

**Fix:** Add a module-level LRU cache in a shared location (e.g., `backtest/splits.py`):

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _cached_read_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
```

Each module calls `_cached_read_features(features_path)` instead of `pd.read_parquet(features_path)`. The cache is automatically evicted if the path changes (which it does not within a single run).

**Risk:** None.

---

### 2. Derive yearly and sector IC from monthly IC instead of recomputing

After computing monthly IC for all (feature, horizon, SignalType) combos, the code recomputes from scratch for yearly and sector groupings — adding ~700 redundant ProcessPoolExecutor submissions.

**Fix:**
- **Yearly IC:** group monthly IC by year and take the mean. This is the standard quant finance definition of "annual IC."
- **Sector IC:** compute per-sector-per-month IC once; derive the per-sector summary and monthly sector-split table from the same intermediate.

Eliminates ~67% of Executor submissions.

**Risk:** None. "Mean of monthly ICs within a year" is standard practice and mathematically documented.

---

### 3. Pre-index signals DataFrame by date in portfolio simulator

Each of ~200 rebalances does a two-sided boolean mask scan of the full signals DataFrame:

```python
eligible = signals[(signals["date"] >= lookback_start) & (signals["date"] <= date)]
```

**Fix:** At construction time, set a sorted DatetimeIndex:

```python
self._signals = signals.set_index("date").sort_index()
```

Then at each rebalance:

```python
eligible = self._signals.loc[lookback_start:date]
```

This switches from O(N) boolean scan to O(log N) index lookup per rebalance.

**Risk:** Low. `.loc` on a sorted DatetimeIndex produces identical row subsets. Ensure `lookback_start` and `date` are within bounds.

---

### 4. Replace `dedup_latest_per_ticker` sort with `groupby.last()` in IC path

Each month group calls `dedup_latest_per_ticker(gdf)` which sorts by `call_entry_date` then drops duplicates. The data is already chronologically ordered from the feature engineering pipeline, making the O(n log n) sort redundant. Called 350–1050 times during IC analysis.

**Fix:** Replace with `gdf.groupby("BESTTICKER").last()`. Since data is sorted before grouping into months and per-month order preserves relative chronology, this is equivalent.

**Risk:** Low. Document that the function assumes chronologically-ordered input. If this invariant changes, `groupby.last()` would pick the wrong row — add an assertion (`assert gdf["call_entry_date"].is_monotonic_increasing`) as a safety net.

---

### 5. Use `scipy.stats.spearmanr` in `_stats.py:spearman`

Current implementation wraps arrays in `pd.Series` and calls `.corr(method="spearman")`, creating temporary objects and going through pandas dispatch. Called hundreds of times across model evaluation, IC computation, and quintile results.

**Fix:**

```python
from scipy.stats import spearmanr
return float(spearmanr(a, b).statistic)
```

`scipy` is already imported in `single_feature_ic.py`.

**Risk:** None. `spearmanr` produces identical results.

---

### 6. Pass `price_table` to `compute_capacity_metrics` instead of reloading

`compute_capacity_metrics` reloads 500–1500 individual ticker parquet files, but the portfolio simulator already has the full `_price_table` in memory.

**Fix:** Add an optional `price_table` parameter. When provided, use it instead of loading from disk. Caller passes `sim._price_table`.

**Risk:** None. The function already handles missing tickers/dates gracefully.

---

### 7. Vectorize daily PnL loop with numba (highest reward, needs caution)

The daily PnL loop (`_compute_daily_pnl`) iterates ticker-by-ticker with mutable state — ~320K ticker-day iterations, the hottest path in the entire project.

**Why numba helps here.** The per-ticker Python loop has three costs: (a) Python interpreter dispatch for each iteration, (b) per-ticker dict lookups for `last_prices`, `missing_streaks`, and `censored_tickers`, and (c) float arithmetic on individual price ratios. Numba eliminates (a) and (c) by compiling the loop body to machine code, and restructured numpy arrays eliminate (b) by replacing dicts with flat integer-indexed arrays. The arithmetic itself (price ratios, weighted sums) is trivial for numba's LLVM backend.

#### Step 1: Restructure mutable state into flat numpy arrays

The current loop tracks per-ticker state in Python dicts. Replace with parallel numpy arrays indexed by a ticker-to-column-id mapping:

```python
# Build once at construction time
ticker_to_idx: dict[str, int]    # ticker -> column index (0..n_tickers-1)
n_tickers = len(ticker_to_idx)

# Per-ticker state arrays (updated in-place each day)
last_prices      = np.full(n_tickers, np.nan, dtype=np.float64)  # last valid close
missing_streaks  = np.zeros(n_tickers, dtype=np.int32)           # consecutive missing days
is_censored      = np.zeros(n_tickers, dtype=np.bool_)           # True = excluded from PnL
entry_dates      = np.full(n_tickers, -1, dtype=np.int64)        # actual entry date as int (ordinal or unix)
```

A 2D price matrix `prices[i_ticker, i_day]` with `np.nan` for missing quotes is the input. A 2D weight matrix `weights[i_ticker, i_day]` (sparse — mostly zeros) drives the PnL.

#### Step 2: Numba-compiled daily loop

```python
from numba import njit

@njit(cache=True)
def _compute_daily_pnl_numba(
    prices: np.ndarray,          # (n_tickers, n_days) float64, NaN where quote missing
    weights: np.ndarray,         # (n_tickers, n_days) float64
    first_trading_day: np.int64, # int index into day dimension
    max_gap: np.int32 = 2,
) -> tuple[np.ndarray, ...]:
    """
    Returns (daily_gross, daily_net, gap_columns, ...) matching existing output schema.
    """
    n_tickers, n_days = prices.shape
    daily_gross = np.zeros(n_days, dtype=np.float64)
    daily_net   = np.zeros(n_days, dtype=np.float64)

    last_prices     = np.full(n_tickers, np.nan, dtype=np.float64)
    missing_streaks = np.zeros(n_tickers, dtype=np.int32)
    is_censored     = np.zeros(n_tickers, dtype=np.bool_)
    # gap accounting accumulators...

    for d in range(n_days):
        # --- Pass 1: classify tickers for day d ---
        # valid_next_price = ~is_censored & weights[:, d] != 0 & ~np.isnan(prices[:, d])
        # missing_today    = ~is_censored & weights[:, d] != 0 &  np.isnan(prices[:, d])

        # --- Pass 2: vectorized PnL for tickers with valid next price ---
        # mask = valid_next_price & ~np.isnan(last_prices)
        # ret  = prices[mask, d] / last_prices[mask] - 1.0
        # daily_gross[d] += np.sum(weights[mask, d] * ret)
        # last_prices[valid_next_price] = prices[valid_next_price, d]
        # missing_streaks[valid_next_price] = 0

        # --- Pass 3: streak updates for tickers with missing price ---
        # missing_streaks[missing_today] += 1
        # streak_1d = missing_today & (missing_streaks == 1)
        # streak_2d = missing_today & (missing_streaks == 2)
        # long_gap   = missing_today & (missing_streaks > 2)
        # ... accumulate gap accounting columns ...
        # is_censored[long_gap] = True

        # --- Pass 4: handle newly entered positions ---
        # newly_entered = (entry_dates == d)  # vectorized integer compare
        # last_prices[newly_entered] = prices[newly_entered, d]
        # missing_streaks[newly_entered] = 0

    return (daily_gross, daily_net, ...)

```

#### Step 3: Why this is safe

| Concern | Resolution |
|---|---|
| Iteration order | All tickers on day `d` are independent — vectorized boolean masks produce identical results to sequential per-ticker processing. Day order is preserved by the outer `for d` loop. |
| `last_prices` carry-over | `last_prices` is only updated at end-of-day after all tickers have used the previous day's value. This matches the sequential semantic. |
| Gap streak state machine | Streaks increment for missing, reset to 0 for present, censor at threshold — all done via vectorized boolean indexing on `missing_streaks[bool_mask]`. Same logic, same outcome. |
| Entry date delayed activation | `entry_dates` is set at rebalance time (outside the numba loop). Inside, a simple `entry_dates == d` vector compare activates positions on their actual entry day. |

#### Step 4: Integration

Replace the body of `_compute_daily_pnl` with:

```python
def _compute_daily_pnl(self, ...):
    # Build flat arrays from SimulationState (ticker_to_idx, prices, weights, entry_dates)
    # Call _compute_daily_pnl_numba(prices, weights, entry_dates, ...)
    # Unpack result tuple into existing gap accounting columns
```

#### Expected speedup

~10-30x on the daily loop body. For a full SP500 enhanced run (~320K ticker-days), this takes the PnL computation from ~5-15 seconds to well under 1 second.

**Risk:** Medium. The mutable state (`censored_tickers`, `missing_streaks`, `last_prices`) must be reproduced identically from the sequential Python version. **Mitigation:** (1) Write the pure-Python vectorized version first using numpy boolean masks (no numba), verify equivalence against the existing per-ticker loop with an assertion-based regression test on synthetic fixtures; (2) once the vectorized logic is validated, add `@njit` which is a one-line decorator change. The numba fallback path is the validated vectorized numpy path.

#### Recommendation

Implement after all lower-risk Phase 5 items (1-6, 8) are done and tests pass. Ship the pure-numpy vectorized version first; add `@njit(cache=True)` once regression equivalence is confirmed.

---

### 8. Minor: pre-allocate arrays in block bootstrap

Each of 2000 bootstrap iterations builds a Python list, extends it with block samples, and converts to numpy. This creates 2000 temporary lists.

**Fix:** Pre-allocate `samples = np.empty((n_boot, n))` and fill by row.

**Risk:** None.

---

### Not worth changing

- **`_next_valid_quote` linear scan** — bounded at 5 trading days, O(1) in practice.
- **`_coverage_by_rebalance_date` per-date loop** — informational only, not on the critical path.
- **`_run_beta_window_section` calling momentum 3x** — one-time robustness check.
- **`filter_to_universe` DataFrame double-copy** — small gain relative to refactoring complexity.
- **`_market_cache` key omitting `universe_name`** — latent cache-collision but never triggered in practice (each simulator instance is scoped to one universe).
