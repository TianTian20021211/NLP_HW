# Phase 2 Performance Recommendations — Feature Engineering

`features/engineer.py`

## Recommendations

### 1. Eliminate duplicate `_group_aspect_theme_cols` call

`_aspect_features` and `_theme_features` each call `_group_aspect_theme_cols(df)` independently, producing identical dicts. For stretch tier this iterates 400+ column names twice.

**Fix:** Compute once in `compute_row_features`, pass to both functions:

```python
at_groups = _group_aspect_theme_cols(df)
aspect_feat = _aspect_features(df, at_groups)
theme_feat  = _theme_features(df, at_groups)
```

**Risk:** None. Dict is read-only in both functions.

---

### 2. Share sorted intermediate between `compute_qoq_deltas` and `compute_4q_trend`

Both functions independently sort by the same 5 keys, groupby-collapse, and hash-merge. The O(n log n) sort runs twice on the same data.

**Fix:** Move the shared sort + groupby-collapse + merge into a helper that returns a reusable structure. The trickiest part is `compute_4q_trend` needs `atc_by_date` multi-indexed by `(BESTTICKER, SignalType, availability_date)` while `compute_qoq_deltas` needs a flat merge — but both start from the same sorted base.

**Risk:** Medium. The `last()` collapse within same-day groups depends on `_row_id` ordering — must preserve existing tiebreaker semantics.

---

### 3. Stream groupby instead of materializing to dict in `_merge_momentum_metrics`

```python
events_by_tkr = dict(list(events.groupby(ticker_col)))
```

This doubles the memory footprint of the events DataFrame by holding all groups simultaneously.

**Fix:**

```python
for tkr, ev in events.groupby(ticker_col):
    if tkr not in metrics_by_tkr:
        continue
    ...
```

**Risk:** None. Groups are concatenated and sorted downstream anyway.

---

### 4. Restructure Fenwick tree to avoid redundant setup per column

`compute_pit_percentiles` calls `_strict_historical_percentile` once per percentile column (up to 6) per group. Each call independently converts history/cutoff dates to numpy arrays, finds valid indices, sorts by cutoff date, and builds Fenwick tree state. Since all columns share the same dates, this is 6x redundant setup.

**Fix:** Split `_strict_historical_percentile` into:
- Pre-process step: sorted history indices, sorted query cutoff indices → return once per group
- Per-column step: apply Fenwick tree walk to a single value column using pre-sorted indices

For the Numba path, add a variant that accepts pre-sorted arguments.

**Risk:** Medium. Fenwick tree state machine must be reproduced identically; off-by-one in the refactored walk would shift percentiles.

---

### 5. Minor: `pd.get_dummies` for sector one-hot

11 independent string comparisons (`df["SECTOR"] == sector` × 11) can be replaced with a single `pd.get_dummies` + `reindex` to guarantee all 11 columns exist.

**Risk:** None.

---

### Not worth changing

- **Per-ticker momentum DataFrame construction** — rolling beta is inherently per-ticker and the I/O (loading price parquet per ticker) dominates over DataFrame allocation cost. Parallelizing across tickers would help more than eliminating allocations.
- **Three groupby-shift calls in `compute_4q_trend`** — GroupBy overhead is negligible relative to the shift operation itself.
- **Multiple `pd.concat` calls in `build_features`** — standard pandas pipeline construction, each concat copies column references not data.
