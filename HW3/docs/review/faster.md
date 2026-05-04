# Implementation plan: vectorize `_compute_daily_pnl`

`backtest/portfolio.py:_compute_daily_pnl` (line 1029) is the hottest path in the
project. Called once per trading day (~2500 times) with a Python `for` loop iterating
every active ticker (~320K ticker-day iterations for SP500). This plan eliminates the
inner ticker loop via numpy vectorization, with an optional final step to add `@njit`.

## Step 0: Write 10 unit tests for the PnL state machine

**File:** `tests/test_portfolio_pnl.py`

Extract the core daily-PnL logic into a free function so it can be tested independently
of the simulator. The function takes flat data (not DataFrames) and mutates state arrays.

### Function to test

```python
# backtest/portfolio.py (new free function)

def _advance_daily_state(
    tickers: list[str],
    weights: dict[str, float],
    last_prices: dict[str, float],
    missing_streaks: dict[str, int],
    censored_tickers: set[str],
    p_today: dict[str, float],    # ticker -> today's close (NaN if missing)
    p_next: dict[str, float],     # ticker -> next day's close (NaN if missing)
) -> dict:
    """Advance one day of the gap-state machine. Returns a daily record dict.
    
    Pure function over plain dicts — no pandas, no SimulationState, no
    close_matrix slicing. This is the logic kernel we vectorize in Step 3.
    """
```

This is a pure extraction of lines 1040-1124 from the current `_compute_daily_pnl`.
The existing method becomes a thin wrapper that slices `close_matrix` and delegates.

### Test 1: empty weights produce zero record

No active weights → pnl=0, gross=0, net=0, n_positions=0. All gap fields zero.

### Test 2: single ticker, valid both days, correct PnL

One ticker with weight 0.5, today_px=100, next_px=110. Expected: pnl = 0.5 * (110/100 - 1) = 0.05, n_positions=1, no gaps.

### Test 3: single ticker, missing next price (1-day gap)

One ticker, next_px=NaN. Expected: pnl=0, ffill_1d_weight=|weight|, n_ffill_1d=1, streak=1.

### Test 4: ticker with two consecutive missing days (2-day gap)

Pre-set `missing_streaks[t]=1`. Next_px=NaN again. Expected: ffill_2d_weight=|weight|, n_ffill_2d=1, streak=2. Ticker NOT yet censored.

### Test 5: ticker with three consecutive missing days (censored)

Pre-set `missing_streaks[t]=2`. Next_px=NaN again. Expected: streak hits 3 → ticker added to `censored_tickers`, `long_gap_records` populated. No PnL.

### Test 6: already-censored ticker is skipped

Ticker in `censored_tickers`. Has valid next_px. Expected: no PnL contribution, not counted in positions. If next_px is NaN → `long_gap` flagged (line 1071).

### Test 7: setdefault initializes last_prices on first valid today price

Ticker not in `last_prices`. today_px=100, next_px=110. Expected: `last_prices[t]` set to 100 (via setdefault), then PnL computed as 0.5 * (110/100 - 1) = 0.05.

### Test 8: ticker without last_prices skips streak on missing next

Ticker not in `last_prices`. today_px=NaN or zero, next_px=NaN. Expected: streak NOT incremented (line 1086: `if tkr not in state.last_prices: continue`).

### Test 9: multiple tickers, mixed valid/gap

Three tickers: A (valid, weight=0.4), B (1d gap, weight=0.3), C (2d gap pre-streak=1, weight=0.3). Expected: pnl only from A, correct gap weight sums, correct streak counts.

### Test 10: gap recovery captures cumulative return

Day 1: ticker has valid base price 100, next_px=NaN → streak=1, no PnL.  
Day 2: ticker still has `last_prices=100` (carried from day 1), next_px=105 (quote reappears).  
Expected: PnL computed as 0.5 * (105/100 - 1) = 0.025 (cumulative return captured in one step). Streak resets to 0.

### Why test against the dict-based function

These tests verify the state machine *semantics*, not the implementation. After
vectorization (Step 3), the same 10 tests run against the numpy version — same
inputs, same expected outputs, different internal representation. This is the
regression suite that gates Step 3.

---

## Step 1: Build flat price matrix once (upstream of the loop)

**File:** `backtest/portfolio.py` — `PortfolioSimulator.run_simulation`

After `_load_calendar_prices_and_coverage` returns `close_matrix` (line 1288),
convert it once:

```python
ticker_to_idx = {t: i for i, t in enumerate(close_matrix.columns)}
idx_to_ticker = list(close_matrix.columns)
day_index = {d: i for i, d in enumerate(close_matrix.index)}
prices_2d = close_matrix.values.T.astype(np.float64)  # (n_tickers, n_days)
```

`prices_2d[ticker_idx, day_idx]` replaces `close_matrix.loc[date, ticker]` —
two integer-indexed lookups instead of DataFrame `.loc` + `.reindex`.

**Risk:** None. Same underlying float data, different container.

---

## Step 2: Replace SimulationState dicts with numpy arrays

**File:** `backtest/portfolio.py` — `SimulationState` dataclass (line 468) and
`_process_rebalance` (line 951).

Add parallel array fields to `SimulationState`:

```python
# New fields (populated after ticker_to_idx is built)
last_prices_arr:     np.ndarray | None = None   # (n_tickers,) float64, NaN = no price
missing_streaks_arr: np.ndarray | None = None   # (n_tickers,) int32
is_censored_arr:     np.ndarray | None = None   # (n_tickers,) bool
weights_arr:         np.ndarray | None = None   # (n_tickers,) float64
```

`_process_rebalance` (line 984-1004) populates these arrays instead of (or in
addition to) the existing dicts:

```python
# Reset all tickers to neutral
state.weights_arr[:] = 0.0
state.last_prices_arr[:] = np.nan
state.missing_streaks_arr[:] = 0
state.is_censored_arr[:] = False

# Set active positions
for tkr, w in new_weights.items():
    idx = ticker_to_idx[tkr]
    state.weights_arr[idx] = w
    state.last_prices_arr[idx] = float(entry_prices[tkr])
```

Keep the dict fields initially — the 10 unit tests from Step 0 validate them.
Remove the dicts as a final cleanup once the numpy path passes all tests.

**Risk:** Low. Data layout change only; values are identical.

---

## Step 3: Rewrite `_compute_daily_pnl` in pure numpy

**File:** `backtest/portfolio.py:_compute_daily_pnl` (line 1029)

The per-ticker Python `for` loop (lines 1062-1096) is replaced with boolean-mask
passes over numpy arrays. The outer day loop stays in Python.

```python
def _compute_daily_pnl_vectorized(
    state: SimulationState,
    prices_2d: np.ndarray,        # (n_tickers, n_days)
    ticker_to_idx: dict[str, int],
    day_index: dict[pd.Timestamp, int],
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    did_rebalance: bool,
) -> None:
    d = day_index[date]
    d_next = day_index[next_date]

    # Active tickers: non-zero weight, entered, not censored
    active = (
        (state.weights_arr != 0.0)
        & ~state.is_censored_arr
        & (state.entry_dates_arr >= 0)
        & (state.entry_dates_arr <= d)
    )
    if not active.any():
        # append zero record, return
        ...

    active_idx = np.flatnonzero(active)
    weights_a  = state.weights_arr[active_idx]
    p_today    = prices_2d[active_idx, d]
    p_next     = prices_2d[active_idx, d_next]

    # --- Pass 1: setdefault for newly-seen valid prices ---
    need_init = (
        np.isnan(state.last_prices_arr[active_idx])
        & np.isfinite(p_today) & (p_today > 0)
    )
    state.last_prices_arr[active_idx[need_init]] = p_today[need_init]

    # --- Pass 2: compute PnL for tickers with valid next price ---
    can_compute = (
        np.isfinite(p_next) & (p_next > 0)
        & np.isfinite(state.last_prices_arr[active_idx])
        & (state.last_prices_arr[active_idx] > 0)
    )
    pnl = 0.0
    if can_compute.any():
        ret = p_next[can_compute] / state.last_prices_arr[active_idx[can_compute]] - 1.0
        pnl = float(np.dot(weights_a[can_compute], ret))
        state.last_prices_arr[active_idx[can_compute]] = p_next[can_compute]
        state.missing_streaks_arr[active_idx[can_compute]] = 0

    # --- Pass 3: missing quotes → increment streaks ---
    has_last = np.isfinite(state.last_prices_arr[active_idx])
    missing = ~np.isfinite(p_next) & has_last
    if missing.any():
        state.missing_streaks_arr[active_idx[missing]] += 1
        streaks = state.missing_streaks_arr[active_idx]
        is_1d = missing & (streaks == 1)
        is_2d = missing & (streaks == 2)
        long_gap = missing & (streaks > 2)
        state.is_censored_arr[active_idx[long_gap]] = True

    # --- Pass 4: gap accounting ---
    ffill_1d_weight = float(np.abs(weights_a[is_1d]).sum()) if is_1d.any() else 0.0
    ffill_2d_weight = float(np.abs(weights_a[is_2d]).sum()) if is_2d.any() else 0.0
    ...

    # Append daily record (identical to current lines 1126-1142)
    state.daily_records.append({...})
```

**Equivalence proof.** The 4 passes reproduce the sequential logic exactly:

| Sequential (current) | Vectorized (new) | Why they match |
|---|---|---|
| `setdefault(t, today)` | `np.where(isnan & valid, today, last)` | `setdefault` only fills missing keys |
| `if t in censored: continue` | `is_censored` excluded from `active` mask | Same skip semantic |
| `base = last_prices.get(t); pnl += w*(next/base-1)` | `np.dot(weights[mask], ret_vec)` | Dot product equals sum of element-wise products |
| `last_prices[t] = next_px` | `last_prices[idx[mask]] = p_next[mask]` | Same update, batched |
| `streak += 1; if 1/2/else` | `streaks += 1; is_1d = streaks==1; ...` | Same state machine |

Within a day, every ticker is independent — no cross-ticker data dependencies. So
batch boolean-mask operations produce identical results to sequential iteration.

---

## Step 4: Run the 10 unit tests against both implementations

Same test file from Step 0. Add a parametrized fixture that runs each test case
through both the dict-based implementation and the numpy implementation. Assert
identical `pnl`, `gross_exposure`, `net_exposure`, `ffill_1d_weight`,
`ffill_2d_weight`, `n_positions`, `n_ffill_1d`, `n_ffill_2d`, `long_gap_records`,
`censored_tickers` membership.

Also run a full SP500 simulation with both code paths and assert:
- `daily_df["pnl"].sum()` matches to 1e-10 relative tolerance
- Gap weight sums match exactly
- Trade log is byte-identical

---

## Step 5 (optional): Add `@njit`

Once Step 4 passes, decorate the numpy function with `@njit(cache=True)`. This
fuses all passes into one register-level loop — zero temporary arrays, zero Python
call overhead. Expected additional 2-3x over pure numpy.

Keep the pure-numpy version as the fallback path. If numba ever produces a
divergent result, swap the call site back.

---

## Expected speedup

| Stage | PnL loop time (SP500) |
|---|---|
| Current (Python for-loop + dicts + `.loc`) | ~5–15 s |
| After Steps 1+2+3 (pure numpy) | ~0.5–2 s |
| After Steps 1+2+3+5 (numba) | < 0.3 s |

---

## Risk summary

| Step | Risk | Mitigation |
|---|---|---|
| 0 | N/A (tests only) | — |
| 1 | None | Same float data, different container |
| 2 | Low — data layout change | Values identical to dict fields |
| 3 | Medium — state machine ordering | Step 4 side-by-side test on fixtures + real run |
| 4 | N/A (validation only) | — |
| 5 | Low after Step 4 | Fallback to validated pure-numpy path |
