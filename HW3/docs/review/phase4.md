# Phase 4 Performance Recommendations — Splits, Forward Returns, Hyperparameters

`backtest/splits.py`, `backtest/model.py`

## Recommendations

### 1. Extract `pd.to_datetime` from repeated calls in `_prepare_fold_sample`

The same column `df[availability_col].iloc[train_idx]` is converted with `pd.to_datetime` four separate times: sorting, DatetimeIndex construction, `train_start`, and `train_end`. Called ~130 times total (26 folds × 5 horizons), resulting in 520 conversions.

**Fix:** Convert once at the top:

```python
avail_dates = pd.to_datetime(df[availability_col].iloc[train_idx], errors="coerce")
avail_dates_np = avail_dates.to_numpy(dtype="datetime64[ns]")
```

Reuse `avail_dates_np` for `np.argsort` and `avail_dates` for the index and min/max.

**Risk:** None.

---

### 2. Avoid full DataFrame copy in `generate_folds`

`df = df.reset_index(drop=True)` copies the entire features DataFrame (millions of rows × hundreds of columns), but only two date columns are used downstream.

**Fix:** Extract only what's needed:

```python
call_entry = pd.to_datetime(df["call_entry_date"].to_numpy())
avail_date = pd.to_datetime(df[availability_col].to_numpy())
```

Remove the `reset_index(drop=True)`. The function already operates on numpy arrays after extraction; call sites use `df.iloc[train_idx]` (positional, not label-based).

**Risk:** None.

---

### 3. Parallelize `tune_ridge` over alphas

`tune_lightgbm` and `tune_xgboost` use `joblib.Parallel(n_jobs=-1)` for their trial loops. `tune_ridge` has 6 alphas × 5 CV folds = 30 fits, all serial.

**Fix:** Extract a `_fit_ridge_alpha` helper and parallelize per fold:

```python
fold_scores = Parallel(n_jobs=-1)(
    delayed(_fit_ridge_alpha)(a, X_tr, y[tr], X_vl, y[vl])
    for a in alphas
)
```

**Risk:** None. Alphas within a fold are independent. Use `n_jobs=1` on each Ridge to avoid nested parallelism.

---

### 4. Remove redundant `.copy()` in `filter_model_sample`

`out = df.copy()` creates a full deep copy, then `out = out[out["SignalType"] == signal_type].copy()` creates a second copy. The first copy is discarded if filtering is applied, and wasteful if not.

**Fix:** Remove the initial `df.copy()`. The `_orig_df_index` column assignment is safe on the original df (adding a new column, not modifying existing ones).

**Risk:** Low. Callers do not depend on the original DataFrame remaining unmodified.

---

### 5. Remove redundant `.copy()` after `.sort_values()`

In `_compute_ticker_forward_return_matches`, `events.sort_values("_entry_date").copy()` creates a redundant allocation — `sort_values` already returns a new DataFrame. Runs ~2600 times during forward-returns computation.

**Fix:** `ev = events.sort_values("_entry_date")`

**Risk:** None.

---

### 6. Add dtype guard in `purge_train_for_horizon`

`pd.to_datetime(df[date_col].values)` is called per fold per horizon, even though the column is already `datetime64[ns]`. Each call creates a new numpy array.

**Fix:**

```python
if df[date_col].dtype == "datetime64[ns]":
    target_dates = df[date_col].to_numpy()
else:
    target_dates = pd.to_datetime(df[date_col].values)
```

**Risk:** None.

---

### 7. Consider parallelizing `compute_forward_returns` over tickers (higher effort)

Tickers are processed sequentially in a for-loop, each doing independent parquet I/O and `merge_asof`. This is the bottleneck of forward-returns computation.

**Fix:** Use `joblib.Parallel` — each worker returns `(orig_pos, ret_by_h, date_by_h)` tuples instead of mutating shared dicts. Cap at `n_jobs=4` to avoid I/O contention on the price cache.

**Risk:** Low. Tickers are independent. Requires refactoring from shared-dict mutation to return-then-merge pattern.

---

### 8. Consider horizon-level parallelism in `run_walk_forward` (higher effort)

26 folds run sequentially per horizon. Horizons are independent and could run in parallel.

**Fix:** Parallelize at the horizon level (each horizon's folds run serially within one worker). This avoids the `monitor_fit_calls` thread-safety concern since different horizons use different target columns. Fold-level parallelism within a horizon is riskier due to shared audit tooling state.

**Risk:** Low for horizon-level; Medium for fold-level (audit tooling thread safety).

---

### Not worth changing

- **Source hash recomputation** — 5–15ms per run, negligible.
- **Price manifest glob** — ~50–200ms, called once per pipeline execution.
- **Duplicate fold manifest writes** — `<100ms total`.
- **Import statements inside trial functions** — `sys.modules` cache makes them ~1µs each.
- **`_write_forward_returns_manifest` on every cache hit** — 5ms per run.
