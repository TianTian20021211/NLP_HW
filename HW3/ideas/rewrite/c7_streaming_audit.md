# C7. Split `features/audit.py::run_streaming_vs_batch_test`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

`run_all_audits(signals_path: Path = SIGNALS_PARQUET, price_cache_dir: Path = PRICE_CACHE_DIR, tier: str = "enhanced", small_only: bool = False, full_dates: int = 15) -> dict[str, Any]` already delegates to assertion helpers. Do not split it
in this refactor.

Add these helpers for `run_streaming_vs_batch_test(df: pd.DataFrame, price_cache_dir: Path = PRICE_CACHE_DIR, tier: str = "enhanced", n_sample_dates: int | None = None, include_momentum: bool = True, rtol: float = 1e-9, atol: float = 1e-12) -> tuple[bool, pd.DataFrame, dict[str, Any]]`:

```python
def _build_batch_features_for_audit(
    df: pd.DataFrame,
    price_cache_dir: Path,
    tier: str,
    include_momentum: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build full batch features and partition columns into strict/xsectional."""
```

Return: `(batch, strict_cols, xsectional_cols)`.

```python
def _prepare_streaming_targets_for_audit(
    batch: pd.DataFrame,
    sampled_dates: list[pd.Timestamp],
    strict_cols: list[str],
    xsectional_cols: list[str],
    include_momentum: bool,
) -> pd.DataFrame | None:
    """Use target-only streaming path when momentum is disabled."""
```

Return: streaming target DataFrame or `None`.

```python
def _compare_one_streaming_date(
    original_df: pd.DataFrame,
    batch: pd.DataFrame,
    streaming_targets: pd.DataFrame | None,
    date: pd.Timestamp,
    price_cache_dir: Path,
    tier: str,
    include_momentum: bool,
    strict_cols: list[str],
    xsectional_cols: list[str],
    rtol: float,
    atol: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int, bool]:
    """Compare one sampled date between batch and streaming results."""
```

Return:
- strict mismatch DataFrame or `None`
- cross-sectional mismatch DataFrame or `None`
- rows compared
- row-count mismatch flag

```python
def _summarize_streaming_mismatches(
    strict_parts: list[pd.DataFrame],
    xsectional_parts: list[pd.DataFrame],
    n_dates_tested: int,
    n_rows_compared: int,
    n_row_count_mismatch: int,
    elapsed_s: float,
) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
    """Combine mismatch tables and return the public audit tuple."""
```

Return: `(passed, mismatch_df, summary_dict)`.
