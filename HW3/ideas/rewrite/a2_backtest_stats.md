# A2. `backtest/_stats.py`

Part of [Part A — New Shared Modules](main.md#module-index). The cross-cutting
rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is, Fixes To The
Previous Draft, Testing Strategy, Specific Regression Risks To Watch) apply to
this plan.

Create `backtest/_stats.py` as a leaf utility module.

## A2.1 Imports

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
```

## A2.2 `spearman`

```python
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-d arrays."""
```

Parameters:
- `a`: first numeric vector.
- `b`: second numeric vector.

Return: `float`.

Caller changes:
- Replace `backtest/model.py::_spearman`.
- Replace `backtest/splits.py::_spearman`.

## A2.3 `make_median_imputer`

```python
def make_median_imputer() -> Any:
    """SimpleImputer(strategy='median') with keep_empty_features fallback."""
```

Parameters: none.

Return: scikit-learn `SimpleImputer`.

Caller changes:
- Replace `backtest/model.py::_make_median_imputer`.
- Replace `backtest/splits.py::_make_median_imputer`.

## A2.4 `dedup_latest_per_ticker`

```python
def dedup_latest_per_ticker(
    df: pd.DataFrame,
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Sort by date_col ascending and keep the last row per ticker_col."""
```

Parameters:
- `df`: current grouped DataFrame.
- `date_col`: date column used for "latest" ordering.
- `ticker_col`: ticker identifier column.

Return: deduplicated DataFrame.

Replace only the exact current pattern:

```python
gdf.sort_values("call_entry_date").drop_duplicates(
    subset=["BESTTICKER"], keep="last"
)
```

Do not apply before broader groupby operations unless the old code did.

## A2.5 `bucket_returns`

```python
def bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int,
    min_samples_per_bucket: int = 5,
    group_col: str = "year_month",
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Equal-size buckets by feature_col; mean return_col per bucket and group."""
```

Parameters:
- `df`: input event DataFrame.
- `feature_col`: signal column to sort.
- `return_col`: forward-return target column.
- `n_buckets`: qcut bucket count.
- `min_samples_per_bucket`: minimum samples required per requested bucket.
- `group_col`: time grouping column, currently `year_month`.
- `date_col`: latest-event date for per-ticker dedupe.
- `ticker_col`: ticker column for per-ticker dedupe.

Return: DataFrame with columns `[group_col, "bucket", "ret", "n"]`.

Implementation details:
- Return an empty standard-schema DataFrame if no valid rows.
- Copy `gdf` after deduplication before assigning `_bucket`.
- Preserve current `pd.qcut(..., duplicates="drop")` behavior.

Caller changes:
- Replace `backtest/quintile.py::_bucket_returns`.
- Replace `backtest/robustness.py::_monthly_bucket_returns`.

## A2.6 `build_equity_curves`

```python
def build_equity_curves(
    bucket_returns_df: pd.DataFrame,
    n_buckets: int,
    group_col: str = "year_month",
) -> pd.DataFrame:
    """Build monthly bucket, cumulative bucket, and long/short equity curves."""
```

Parameters:
- `bucket_returns_df`: output from `bucket_returns`.
- `n_buckets`: expected number of buckets.
- `group_col`: index column.

Return: DataFrame indexed by `group_col` with **observed bucket columns only**
(when ``pd.qcut(duplicates="drop")`` removes buckets, only the surviving
bucket columns appear):
- ``bucket_{i}`` and ``cum_bucket_{i}`` for each observed bucket *i*
- ``long_only``, ``short_only``, ``long_short`` (when both top and bottom
  bucket columns are present)
- ``cum_long_only``, ``cum_short_only``, ``cum_long_short``

Caller changes:
- Replace `backtest/quintile.py::_build_equity_curves`.
- Replace `backtest/robustness.py::_build_equity_curves`.

## A2.7 `max_drawdown_from_equity`

```python
def max_drawdown_from_equity(cum: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a cumulative equity series."""
```

Parameters:
- `cum`: cumulative equity curve, not raw period returns.

Return: `float`, or `np.nan` for empty input.

Caller changes:
- Replace nested max-drawdown logic in `backtest/quintile.py`.
- Replace `backtest/robustness.py::_max_drawdown`.

## A2.8 `portfolio_stats`

```python
def portfolio_stats(
    equity_curves: pd.DataFrame,
    leg: str = "long_short",
    ann_factor: float = 12.0,
    count_key: str = "n_months",
) -> dict[str, float]:
    """Compute annualized return, vol, Sharpe, rolling Sharpe, drawdown, and count."""
```

Parameters:
- `equity_curves`: output of `build_equity_curves` or another DataFrame with
  `leg` and `cum_{leg}` columns.
- `leg`: `long_only`, `short_only`, or `long_short`.
- `ann_factor`: annualization factor. Keep `12.0` for monthly bucket outputs.
- `count_key`: output count key. Default must stay `n_months`.

Return: dictionary with keys:
- `ann_return`
- `ann_vol`
- `sharpe`
- `rolling_sharpe_12m`
- `max_drawdown`
- `count_key` (default `n_months`)

Return `{}` when the requested leg is unavailable or has no observations.

Caller changes:

- `backtest/quintile.py::_portfolio_stats(equity_curves, leg)` — drop and replace
  every call with `portfolio_stats(equity_curves, leg=leg)` directly. No
  identifying key to preserve.

- `backtest/robustness.py::run_subperiod_quintile` (~line 185-199):
  ```python
  s = portfolio_stats(eq, leg="long_short")
  rows.append({"subperiod": sub, "feature": feat, "horizon": horizon,
               **s} if s else {"subperiod": sub, "feature": feat, "horizon": horizon})
  ```
  Identifying keys must come BEFORE the `**s` spread so the stats keys cannot
  silently overwrite them.

- `backtest/robustness.py::mcap_bucket_quintile` (~line 524-534):
  ```python
  s = portfolio_stats(eq, leg="long_short")
  rows.append({"bucket": bucket, "horizon": horizon, **s})
  ```
  Caller currently keeps only `ann_return` and `sharpe`; the extra keys returned
  by `portfolio_stats` are harmless but show up in the persisted parquet —
  document this column-set widening in the verification log.

- `backtest/robustness.py::ofat_quantile_cutoff` (~line 706-718):
  ```python
  s = portfolio_stats(eq, leg="long_short")
  rows.append({"cutoff": cutoff, **s})
  ```
  This call site uses `_build_equity_curves_2leg` output, which still contains
  `long_short` and `cum_long_short` columns, so `portfolio_stats` works without
  modification. **Correctness fix bundled in:** the current code computes
  drawdown via `_max_drawdown(eq["long_short"])` — i.e. on raw period returns,
  not cumulative equity. `portfolio_stats` correctly uses `cum_long_short`.
  Document the resulting `max_drawdown` numerical change in the verification
  log; expect drawdowns to look more negative (peak-to-trough on cumulative is
  always at least as deep as on raw periods).

- `backtest/robustness.py::compare_weighting_schemes` (~line 819-832):
  ```python
  s = portfolio_stats(eq, leg="long_short")
  rows.append({"weighting": scheme, **s} if s else {"weighting": scheme})
  ```
  Required because `portfolio_stats({}) -> {}` would otherwise drop the row
  entirely; the empty-dict guard preserves the pre-refactor row count.
