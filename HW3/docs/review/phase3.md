# Phase 3 Performance Recommendations — Automated Look-Ahead Tests

`features/audit.py`

## Recommendations

### 1. Remove `gc.collect()` from per-date streaming loop

Line ~249, inside `_compare_one_streaming_date`: after deleting local DataFrames, the code calls `gc.collect()`. This runs 15–30 times per audit and each call scans the entire Python heap (including DataFrames with millions of rows). Adds 0.3–3s of wall-clock time per run.

**Fix:** Delete the `gc.collect()` call. The deleted variables are pandas DataFrames (no reference cycles) — Python's reference counting frees them immediately. A single `gc.collect()` after the streaming loop (outside the per-date body) is sufficient if cycle cleanup is desired.

**Risk:** None. No objects in this path create reference cycles.

---

### 2. Swap loop nesting in `_compute_target_pit_percentiles`

Outer loop: iterate over PIT feature columns (~6). Inner loop: iterate over groups. Each group's history is fetched from `base_groups` and passed to Fenwick tree construction once per column — 6x redundant.

**Fix:** Put groups in the outer loop, columns in the inner loop. Each group's history is fetched once and the Fenwick tree is built once per group (not per column).

```python
for group_key, hist_df in target_groups.items():
    base_hist = base_groups.get(group_key)
    for col in pit_cols:
        ...
```

**Risk:** None. Same data, same `_strict_historical_percentile_queries` calls, just reordered.

---

### 3. Pre-group batch DataFrame in `assert_feature_parity`

For each sampled date, `batch.loc[batch["availability_date"] == d]` does a linear scan of the full batch DataFrame (potentially millions of rows) — 15–30 times.

**Fix:** Pre-group before the loop:

```python
batch_by_date = dict(list(batch.groupby("availability_date")))
# then: batch_rows = batch_by_date.get(d)
```

**Risk:** None. Produces identical row subsets.

---

### 4. Minor: drop redundant `sorted()` in `_sample_availability_dates`

`pd.DatetimeIndex.unique()` already returns sorted values. The explicit `sorted()` call is redundant.

**Fix:** `dates.dropna().unique().tolist()`.

**Risk:** None.

---

### Not worth changing

- **Double `np.isclose` in `_compare_parity`** — mismatch branch is almost never taken (all checks pass); the short-circuit `np.allclose` call is the right optimization.
- **Column-by-column Python loop in `_compare_parity`** — only runs 15–30 times, each with 85–405 columns. Python overhead is negligible compared to feature-building cost.
- **Fenwick tree memory in numba fallback** — ~4 MB per column for large datasets, acceptable on any modern machine.
- **`gc.collect()` at line 354** — single call before the streaming loop begins, keeps the heap clean for the loop. Fine as-is.
