# C3. Split `backtest/robustness.py::run_all_robustness`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Keep `RobustnessResult` unchanged.

## C3.1 `_load_robustness_features`

```python
def _load_robustness_features(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load features, ensure forward returns, filter PIT universe, and add year_month."""
```

Return:
- filtered DataFrame
- available feature list

## C3.2 `_run_subperiod_ic_section`

```python
def _run_subperiod_ic_section(
    df: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run and persist subperiod IC robustness."""
```

Return: DataFrame from `run_subperiod_ic`.

## C3.3 `_run_subperiod_quintile_section`

```python
def _run_subperiod_quintile_section(
    df: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run and persist subperiod quintile robustness."""
```

Return: DataFrame from `run_subperiod_quintile`.

## C3.4 `_run_sector_neutral_section`

```python
def _run_sector_neutral_section(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare raw vs sector-neutral ATC quintiles for Total across horizons."""
```

Return: DataFrame persisted to `robustness_sector_neutral.parquet`.

## C3.5 `_run_mcap_bucket_section`

```python
def _run_mcap_bucket_section(
    df: pd.DataFrame,
    universe_name: str,
    price_cache_dir: Path,
    shares_cache_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run market-cap bucket robustness for horizons 1, 5, and 20."""
```

Return: DataFrame persisted to `robustness_mcap_buckets.parquet`.

## C3.6 `_run_weighting_section`

```python
def _run_weighting_section(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare equal-weight and score-weighted ATC quintiles."""
```

Return: DataFrame persisted to `robustness_weighting.parquet`.

## C3.7 `_run_ofat_quantile_section`

```python
def _run_ofat_quantile_section(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run top-N cutoff sensitivity for ATCClassifierScore / Total."""
```

Return: DataFrame persisted to `robustness_ofat_quantile.parquet`.

## C3.8 `_run_ofat_cost_section`

```python
def _run_ofat_cost_section(
    universe_name: str,
    output_dir: Path,
    portfolio_dir: Path = RESULTS_DIR / "portfolio",
) -> pd.DataFrame:
    """Load weekly 5d portfolio returns if available and run cost sensitivity."""
```

Return: DataFrame. Empty when no candidate portfolio daily-return file exists.

## C3.9 `_run_bootstrap_section`

```python
def _run_bootstrap_section(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Run quarterly-block bootstrap on monthly ATC decile L/S returns."""
```

Rules:
- ``block_bootstrap`` defaults to ``periods_per_year=252`` (daily).
  This helper MUST pass ``periods_per_year=12`` because it operates on
  monthly decile L/S returns from ``_build_equity_curves``.
  Omitting it would annualize the Sharpe ratio with √252 instead of √12,
  inflating reported Sharpe CIs by ~4.6×.

Return: JSON-serializable dictionary persisted to
`robustness_bootstrap_ci.json`.

## C3.10 `_run_beta_window_section`

```python
def _run_beta_window_section(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run R8 idiosyncratic residual IC sign check."""
```

Return: DataFrame persisted to `robustness_r8_beta_window.parquet`.

## C3.11 New `run_all_robustness()` Shape

`run_all_robustness(features_path: Path, universe_name: str = "sp500", price_cache_dir: Path = PRICE_CACHE_DIR, shares_cache_dir: Path = SHARES_CACHE_DIR, output_dir: Path | None = None, features: list[str] | None = None, skip_mcap: bool = False) -> RobustnessResult` should become an orchestration
function only:

1. Resolve output directory.
2. Load/filter features with `_load_robustness_features`.
3. Call ``_run_subperiod_ic_section``, ``_run_subperiod_quintile_section``,
   ``_run_sector_neutral_section`` unconditionally.
4. Call ``_run_mcap_bucket_section`` only when ``skip_mcap`` is ``False``.
5. Call ``_run_weighting_section``, ``_run_ofat_quantile_section``,
   ``_run_ofat_cost_section``, ``_run_bootstrap_section``,
   ``_run_beta_window_section`` unconditionally.
6. Set `result.config`.
7. Return `result`.
