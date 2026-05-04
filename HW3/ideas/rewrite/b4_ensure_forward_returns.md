# B4. Add `ensure_forward_returns`

Part of [Part B — Behavior-Preserving Consolidation](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Add to `backtest/splits.py`:

```python
def ensure_forward_returns(
    df: pd.DataFrame,
    features_path: Path,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    horizons: Sequence[int] | None = None,
    entry_date_col: str = "availability_date",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return df with requested forward returns and target-available dates."""
```

Parameters:
- `df`: feature DataFrame.
- `features_path`: feature parquet path used by the cache signature.
- `price_cache_dir`: per-ticker price parquet directory.
- `horizons`: requested horizons. Default is global `HORIZONS`.
- `entry_date_col`: entry date column passed to forward-return computation.
- `overwrite`: when true, recompute requested forward-return and target-date
  columns even if present.

Return: DataFrame with requested columns present. Preserve row order and index.

Required columns per horizon:
- `forward_return_{h}d`
- `target_available_date_{h}d`

Implementation rules:
- **CRITICAL**: ``ensure_forward_returns`` must ONLY be called on the full,
  unfiltered DataFrame loaded directly from the features parquet file.  Never
  call it on a subset (e.g. after universe filtering or row slicing).  The
  underlying ``get_forward_returns_cached`` uses ``features_path`` as the
  primary cache key and excludes ``df`` from the joblib key, so calling with a
  filtered/subset DataFrame with the same ``features_path`` will return cached
  forward returns for the full row set — causing silent row misalignment and
  forward-return contamination.  To reinforce this, the cache signature
  includes ``len(df)`` and a deterministic row-position fingerprint; a
  size/order mismatch between the cached and caller ``df`` raises ``ValueError``.
- If all required columns exist and `overwrite=False`, return `df`.
- If any requested columns are missing, compute the full requested set and drop
  stale requested return/target columns before assigning new values.
- Assign new columns by values aligned to row order:

```python
out = df.drop(columns=drop_cols, errors="ignore").copy()
for col in fwd.columns:
    out[col] = fwd[col].to_numpy()
return out
```

Caller changes:
- `backtest/single_feature_ic.py::run_single_feature_ic(features_path: Path, universe_name: str, price_cache_dir: Path = PRICE_CACHE_DIR, output_dir: Path | None = None) -> dict[str, pd.DataFrame]`
- `backtest/quintile.py::run_quintile_analysis(features_path: Path, universe_name: str, price_cache_dir: Path = PRICE_CACHE_DIR, output_dir: Path | None = None) -> dict[str, Any]`
- `backtest/robustness.py::run_all_robustness(features_path: Path, universe_name: str = "sp500", price_cache_dir: Path = PRICE_CACHE_DIR, shares_cache_dir: Path = SHARES_CACHE_DIR, output_dir: Path | None = None, features: list[str] | None = None, skip_mcap: bool = False) -> RobustnessResult`
- `backtest/model.py::main()`
