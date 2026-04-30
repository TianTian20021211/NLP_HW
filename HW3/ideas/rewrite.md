# Code Restructuring Plan — NLP_HW Backtesting Codebase

## Context

The codebase (~10,000 lines across 15+ files) has grown organically through 5 phases of implementation. Functions are long (13 over 100 lines, 4 over 200 lines), and copy-paste duplication exists across 10+ patterns. The goal is to make the code more readable and maintainable by: (a) extracting shared utilities to eliminate duplication, (b) splitting monster functions into single-responsibility pieces under 20 lines where practical, and (c) keeping all existing behavior and CLI contracts intact.

## What NOT to Change

These functions are long but well-scoped, performance-critical, or algorithmically complex — splitting would hurt readability:

- `build_sp500_pit()` (88 lines) — reverse-replay algorithm, single conceptual unit
- `stream_chunks()` (100 lines) — performance-critical streaming CSV→Parquet pipeline
- `_download_batch()` (73 lines) — well-scoped yfinance batch + retry integration
- Fenwick tree / numba percentiles (3 variants across engineer.py + audit.py) — the audit versions MUST remain separate (they cross-check each other for look-ahead)
- `compute_qoq_deltas()` / `compute_4q_trend()` (64/49 lines) — tightly coupled sort/shift/merge, single unit each
- `assign_market_cap_buckets()` (167 lines) — already vectorized, splitting would require passing many intermediates
- Block bootstrap (69 lines) — single conceptual loop
- `_build_cohort_weights()` (114 lines) — single transformation pipeline

---

## Part A: New Shared Modules

### A1. `data/_utils.py` — Data-Layer Shared Utilities

Canonical home for patterns duplicated across `load_prices.py` and `load_shares.py`.

#### A1.1 `FetchResult` dataclass

Identical in `load_prices.py:78` and `load_shares.py:54`.

```python
from dataclasses import dataclass

@dataclass
class FetchResult:
    ticker: str
    status: str          # "success" | "empty" | "error"
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None = None
```

Delete from both callers, import from `data._utils`.

#### A1.2 Manifest helpers

Currently both files hardcode their own manifest path constants (`PRICE_MANIFEST`, `SHARES_MANIFEST`) inside the function body. Shared versions take the path explicitly.

```python
from pathlib import Path
import json
import threading

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest, returning a default if missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {"updated_at": None, "tickers": {}}


def save_manifest(manifest: dict[str, Any], path: Path, lock: threading.Lock | None = None) -> None:
    """Stamp updated_at, create parent dirs, write manifest JSON.

    *lock* is optional — ``load_prices.py`` passes ``_MANIFEST_LOCK`` because it
    uses threads; ``load_shares.py`` passes ``None``.
    """
    from data.config import utc_now_iso

    manifest["updated_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if lock is not None:
        with lock:
            path.write_text(payload)
    else:
        path.write_text(payload)
```

**Caller changes:**

- `load_prices.py`: `load_manifest()` → `load_manifest(PRICE_MANIFEST)`
  `save_manifest(m)` → `save_manifest(m, PRICE_MANIFEST, lock=_MANIFEST_LOCK)`
- `load_shares.py`: `load_manifest()` → `load_manifest(SHARES_MANIFEST)`
  `save_manifest(m)` → `save_manifest(m, SHARES_MANIFEST)`

#### A1.3 Failed-log helpers

`load_prices.py` operates on batches with a threading lock; `load_shares.py` operates on single results in memory. The shared module exposes four helpers so each caller can compose the behavior it needs.

```python
import pandas as pd
from pathlib import Path

def load_failed_log(path: Path) -> pd.DataFrame:
    """Read the failed-ticker CSV at *path*. Returns empty DataFrame on miss."""
    columns = ["ticker", "status", "rows", "first_date", "last_date", "error"]
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=columns)


def apply_failed_log_result(existing: pd.DataFrame, result: FetchResult) -> pd.DataFrame:
    """Apply ONE FetchResult to an in-memory failed-ticker DataFrame.

    Removes any previous row for the same ticker. If *result* is ``"success"``
    the ticker is cleared. Non-success rows are appended.
    Returns a NEW DataFrame (does not mutate *existing* in place).
    """
    existing = existing[existing["ticker"] != result.ticker].copy()
    if result.status != "success":
        row = pd.DataFrame([{
            "ticker": result.ticker,
            "status": result.status,
            "rows": str(result.rows),
            "first_date": result.first_date or "",
            "last_date": result.last_date or "",
            "error": result.error or "",
        }])
        existing = pd.concat([existing, row], ignore_index=True)
    return existing


def apply_failed_log_results(existing: pd.DataFrame, results: list[FetchResult]) -> pd.DataFrame:
    """Apply a LIST of FetchResult to an in-memory failed-ticker DataFrame.

    Removes previously failed tickers that now succeeded, then appends new
    non-success rows. Returns a NEW DataFrame.
    """
    success_tickers = {r.ticker for r in results if r.status == "success"}
    existing = existing[~existing["ticker"].isin(success_tickers)].copy()
    new_rows = []
    for r in results:
        if r.status != "success":
            new_rows.append({
                "ticker": r.ticker,
                "status": r.status,
                "rows": str(r.rows),
                "first_date": r.first_date or "",
                "last_date": r.last_date or "",
                "error": r.error or "",
            })
    if new_rows:
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    return existing


def save_failed_log(df: pd.DataFrame, path: Path, lock: threading.Lock | None = None) -> None:
    """Write the failed-ticker DataFrame to CSV at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is not None:
        with lock:
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
```

**Caller changes:**

- `load_shares.py`: Replace `load_failed_log()`, `save_failed_log(existing)`, `apply_failed_log_result(existing, result)`, and `update_failed_log(result)` — the last one becomes:
  ```python
  def update_failed_log(result: FetchResult) -> None:
      existing = load_failed_log(SHARES_FAILED_TICKERS)
      existing = apply_failed_log_result(existing, result)
      save_failed_log(existing, SHARES_FAILED_TICKERS)
  ```
  Or eliminate the wrapper entirely and inline the three calls.

- `load_prices.py`: Replace the inline batch failed-log code (lines 155-189) with:
  ```python
  existing = load_failed_log(PRICE_FAILED_TICKERS)
  existing = apply_failed_log_results(existing, results)
  save_failed_log(existing, PRICE_FAILED_TICKERS, lock=_FAILED_LOG_LOCK)
  ```

**Important:** Do NOT silently change failed-log semantics. Prices currently appends batch failures and clears successes; shares keeps one latest row per ticker. The implementations above preserve each behavior. Compare failed-ticker CSV output before/after.

#### A1.4 Exponential backoff

Both files use the identical formula `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` with different constants.

```python
def exponential_backoff(attempt: int, base: float, max_val: float) -> float:
    """min(max_val, base * 2**attempt)"""
    return min(max_val, base * (2 ** attempt))
```

**Caller changes:**

- `load_prices.py:270`: `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` → `exponential_backoff(attempt, base_sleep, MAX_BACKOFF)`
- `load_shares.py:180`: `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` → `exponential_backoff(attempt, base_sleep, MAX_BACKOFF)`

#### A1.5 Logger setup

11 files repeat the same `logging.basicConfig` + `getLogger` pattern. The shared helper lives in `data/_utils.py` for use by `data.*` modules. Backtest modules keep their own logging setup — some (`splits.py`, `universe.py`) intentionally skip `basicConfig` and rely on the importing module to configure logging.

```python
import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return a named logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)
```

**Caller changes (data layer only):**

- `load_prices.py:51-55` → `log = setup_logger("load_prices")`
- `load_shares.py:35-39` → `log = setup_logger("load_shares")`
- `load_universes.py:43-47` → `log = setup_logger("load_universes")`
- `load_signals.py:92-96` → `log = setup_logger("load_signals")`

#### A1.6 `suppress_yfinance_logging`

Currently inline in `load_prices.py:251-252`. Extract so `load_shares.py` can also call it. Must be called **immediately after** `import yfinance as yf` because yfinance may reconfigure loggers during import.

```python
def suppress_yfinance_logging() -> None:
    """Silence yfinance/peewee log spam after lazy import."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)
```

**Caller changes:**

- `load_prices.py:251-252`: replace two lines with `suppress_yfinance_logging()`
- `load_shares.py:127`: after `import yfinance as yf`, add `suppress_yfinance_logging()` (harmless for `yf.Ticker()` but consistent)

#### A1.7 `US_TICKER_RE` regex

Currently only in `load_prices.py:71`. Move to `data/_utils.py` for potential reuse.

```python
import re

US_TICKER_RE = re.compile(r"^[A-Z]{1,5}([\.\-][A-Z])?$")
```

**Caller changes:**

- `load_prices.py:71`: delete, import `from data._utils import US_TICKER_RE`
- `load_prices.py:115`: unchanged (uses the same imported name)

---

### A2. `backtest/_stats.py` — Backtest Shared Statistics

Canonical home for statistics/portfolio helpers duplicated across backtest modules.

#### A2.1 `spearman`

Identical in `model.py:625` and `splits.py:794`.

```python
import numpy as np
import pandas as pd

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-d arrays."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
```

**Caller changes:**

- `model.py:625`: delete `_spearman`, import `spearman` from `backtest._stats`
- `splits.py:794`: delete `_spearman`, import `spearman` from `backtest._stats`

#### A2.2 `make_median_imputer`

Identical in `model.py:46` and `splits.py:611`.

```python
from typing import Any

def make_median_imputer() -> Any:
    """SimpleImputer(strategy='median') with scikit-learn version fallback."""
    from sklearn.impute import SimpleImputer

    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older scikit-learn fallback
        return SimpleImputer(strategy="median")
```

**Caller changes:**

- `model.py:46`: delete `_make_median_imputer`, import `make_median_imputer` from `backtest._stats`
- `splits.py:611`: delete `_make_median_imputer`, import `make_median_imputer` from `backtest._stats`

#### A2.3 `dedup_latest_per_ticker`

7 identical occurrences across `quintile.py:103`, `single_feature_ic.py:118`, `robustness.py:79,239,639,734,855`.

```python
import pandas as pd

def dedup_latest_per_ticker(
    df: pd.DataFrame,
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Sort by *date_col* ascending, keep last row per *ticker_col*.

    Preserves row content and only reorders within the current DataFrame.
    Do NOT call before a broader groupby unless the old code did so.
    """
    return df.sort_values(date_col).drop_duplicates(subset=[ticker_col], keep="last")
```

**Caller changes:** Replace each of the 7 inline occurrences:

```python
# Before:
gdf = gdf.sort_values("call_entry_date").drop_duplicates(subset=["BESTTICKER"], keep="last")

# After:
gdf = dedup_latest_per_ticker(gdf)
```

#### A2.4 `bucket_returns`

Near-duplicate in `quintile.py:82` (`_bucket_returns`) and `robustness.py:842` (`_monthly_bucket_returns`). The only material differences are the function name, the parameter name (`min_samples_per_bucket` vs `min_per_bucket`), and an early-return guard (quintile has it, robustness doesn't need it because an empty groupby produces zero records anyway). Unify both.

```python
import pandas as pd
import numpy as np
from typing import Any

def bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int,
    min_samples_per_bucket: int = 5,
) -> pd.DataFrame:
    """Form equal-size buckets by *feature_col*, compute mean *return_col* per bucket.

    Groups by ``year_month`` to create a time series of bucket returns.
    Returns DataFrame with columns ``[year_month, bucket, ret, n]``.
    """
    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=["year_month", "bucket", "ret", "n"])

    records: list[dict[str, Any]] = []
    for month, gdf in sub.groupby("year_month", observed=True):
        gdf = dedup_latest_per_ticker(gdf)
        if len(gdf) < n_buckets * min_samples_per_bucket:
            continue
        try:
            gdf["_bucket"] = pd.qcut(
                gdf[feature_col], q=n_buckets, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        buckets = gdf.groupby("_bucket")[return_col]
        for b_idx, b_ret in buckets.mean().items():
            records.append({
                "year_month": month,
                "bucket": int(b_idx),
                "ret": float(b_ret),
                "n": int(buckets.size()[b_idx]),
            })
    return pd.DataFrame(records)
```

**Caller changes:**

- `quintile.py:82`: delete `_bucket_returns`, import `bucket_returns` from `backtest._stats`. Update call site to use new name.
- `robustness.py:842`: delete `_monthly_bucket_returns`, import `bucket_returns` from `backtest._stats`. Update call site.

#### A2.5 `build_equity_curves`

Near-duplicate in `quintile.py:128` and `robustness.py:879`. Quintile version pre-computes cumulative returns, robustness version inlines per-column. Output is byte-for-byte the same. Use the quintile version (slightly cleaner).

```python
def build_equity_curves(
    bucket_returns: pd.DataFrame,
    n_buckets: int,
) -> pd.DataFrame:
    """Build cumulative equity curves from monthly bucket returns.

    Returns DataFrame indexed by ``year_month`` with columns:
    ``bucket_0`` … ``bucket_{n-1}`` (monthly returns),
    ``cum_bucket_0`` … ``cum_bucket_{n-1}`` (cumulative),
    ``long_only`` / ``short_only`` / ``long_short`` (leg returns),
    ``cum_long_only`` / ``cum_short_only`` / ``cum_long_short`` (leg cumulative).
    Long = top bucket, Short = bottom bucket.
    """
    if bucket_returns.empty:
        return pd.DataFrame()

    piv = bucket_returns.pivot_table(
        index="year_month", columns="bucket", values="ret", aggfunc="mean"
    )
    piv.columns = [f"bucket_{int(c)}" for c in piv.columns]
    piv = piv.sort_index()

    cum = (1 + piv).cumprod()
    for col in piv.columns:
        piv[f"cum_{col}"] = cum[col]

    top = f"bucket_{n_buckets - 1}"
    bot = "bucket_0"
    if top in piv.columns and bot in piv.columns:
        piv["long_only"] = piv[top]
        piv["short_only"] = -piv[bot]
        piv["long_short"] = piv[top] - piv[bot]
        piv["cum_long_only"] = (1 + piv["long_only"]).cumprod()
        piv["cum_short_only"] = (1 + piv["short_only"]).cumprod()
        piv["cum_long_short"] = (1 + piv["long_short"]).cumprod()

    return piv
```

**Caller changes:**

- `quintile.py:128`: delete `_build_equity_curves`, import from `backtest._stats`
- `robustness.py:879`: delete `_build_equity_curves`, import from `backtest._stats`

#### A2.6 `max_drawdown_from_equity`

Two variants: quintile.py nested inside `_portfolio_stats` (no empty guard) and robustness.py module-level (has empty guard + docstring). Consolidate to the safer robustness version.

```python
def max_drawdown_from_equity(cum: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a cumulative return series.

    Accepts cumulative equity (e.g. ``cum_long_short`` column from
    ``build_equity_curves``). For raw period returns, compute the cumulative
    product first.
    """
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()) if len(dd) > 0 else np.nan
```

#### A2.7 `portfolio_stats`

Currently `_portfolio_stats` in `quintile.py:166` takes `(equity_curves, leg)` and returns `dict[str, float]`. The same logic is repeated inline in 4 places in `robustness.py`:
- `run_subperiod_quintile` (~lines 185-199): `ann_return`, `ann_vol`, `sharpe`, `max_drawdown`
- `mcap_bucket_quintile` (~lines 524-534): `ann_return`, `sharpe` only
- `ofat_quantile_cutoff` (~lines 706-718): `ann_return`, `sharpe`, `max_drawdown`
- `compare_weighting_schemes` (~lines 819-832): `ann_return`, `sharpe`, `max_drawdown`

Unify to a single function that all 5 call sites use:

```python
def portfolio_stats(
    equity_curves: pd.DataFrame,
    leg: str = "long_short",
    ann_factor: float = 12.0,
) -> dict[str, float]:
    """Compute summary statistics for a portfolio leg.

    Parameters
    ----------
    equity_curves:
        Output of ``build_equity_curves``. Must contain *leg* and ``cum_{leg}`` columns.
    leg:
        Leg name, e.g. ``"long_short"``.
    ann_factor:
        Annualisation factor (12 for monthly, 252 for daily).

    Returns
    -------
    dict with keys: ann_return, ann_vol, sharpe, max_drawdown, rolling_sharpe_12m, n_periods.
    Returns empty dict if the leg is missing or has < 2 observations.
    """
    cum_col = f"cum_{leg}"
    if cum_col not in equity_curves.columns or len(equity_curves) < 2:
        return {}
    rets = equity_curves[leg].dropna()
    if len(rets) == 0:
        return {}

    def _rolling_sharpe(r: pd.Series, window: int = 12) -> float:
        roll = r.rolling(window).mean() / r.rolling(window).std()
        return float(roll.mean())

    return {
        "ann_return": float(rets.mean() * ann_factor),
        "ann_vol": float(rets.std() * np.sqrt(ann_factor)),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(ann_factor))
        if rets.std() > 0
        else np.nan,
        "rolling_sharpe_12m": _rolling_sharpe(rets, 12),
        "max_drawdown": max_drawdown_from_equity(equity_curves[cum_col]),
        "n_periods": len(rets),
    }
```

**Caller changes:**

- `quintile.py:166`: delete `_portfolio_stats` (and its nested `_max_drawdown` + `_rolling_sharpe`), import `portfolio_stats` from `backtest._stats`
- `robustness.py`: replace each of the 4 inline stats blocks:
  ```python
  # Before (e.g. run_subperiod_quintile:185-199):
  rets = eq[lsp_col].dropna() if lsp_col in eq.columns else pd.Series(dtype=float)
  n_m = len(rets)
  rows.append({
      "ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
      "ann_vol": float(rets.std() * np.sqrt(12)) if n_m > 1 else np.nan,
      "sharpe": ...,
      "max_drawdown": float(_max_drawdown(eq[lsp_col])) if lsp_col in eq.columns else np.nan,
  })
  # After:
  stats = portfolio_stats(eq, leg="long_short")
  if stats:
      rows.append(stats)
  ```
  Note: callers that only need a subset of keys (e.g. mcap_bucket_quintile only uses `ann_return` + `sharpe`) can still subscript the returned dict. The overhead of computing the extra keys is trivial.

- `robustness.py:901`: delete `_max_drawdown` (module-level), import from `backtest._stats` wherever it's used directly.

#### A2.8 `_stats.py` imports summary

```python
# backtest/_stats.py
from typing import Any
import numpy as np
import pandas as pd
```

No imports from any `backtest.*` module — this is a pure leaf utility.

---

## Part B: Consolidate Duplication

### B1. Refactor data layer to use `data/_utils.py`

**Files:** `load_prices.py`, `load_shares.py`, `load_universes.py`, `load_signals.py`

Specific changes per file:

**`load_prices.py`:**
- Delete `FetchResult` dataclass (line 78). Import from `data._utils`.
- Replace `load_manifest()` with `load_manifest(PRICE_MANIFEST)`.
- Replace `save_manifest(m)` with `save_manifest(m, PRICE_MANIFEST, lock=_MANIFEST_LOCK)`.
- Replace inline `update_failed_log` (lines 155-189) with `load_failed_log` + `apply_failed_log_results` + `save_failed_log` pattern.
- Replace `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` (line 270) with `exponential_backoff(attempt, base_sleep, MAX_BACKOFF)`.
- Replace logging boilerplate (lines 51-55) with `log = setup_logger("load_prices")`.
- Replace lines 251-252 with `suppress_yfinance_logging()`.
- Delete `_US_TICKER_RE` (line 71). Import from `data._utils`.

**`load_shares.py`:**
- Delete `FetchResult` dataclass (line 54). Import from `data._utils`.
- Replace `load_manifest()` with `load_manifest(SHARES_MANIFEST)`.
- Replace `save_manifest(m)` with `save_manifest(m, SHARES_MANIFEST)`.
- Replace `load_failed_log()`, `save_failed_log(existing)`, `apply_failed_log_result(existing, result)`, `update_failed_log(result)` with imports from `data._utils`. Either keep a thin `update_failed_log` wrapper or inline the three calls at the single call site.
- Replace `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` (line 180) with `exponential_backoff(attempt, base_sleep, MAX_BACKOFF)`.
- Replace logging boilerplate (lines 35-39) with `log = setup_logger("load_shares")`.
- Add `suppress_yfinance_logging()` after `import yfinance as yf` (line 127).

**`load_universes.py`:**
- Replace logging boilerplate (lines 43-47) with `log = setup_logger("load_universes")`.

**`load_signals.py`:**
- Replace logging boilerplate (lines 92-96) with `log = setup_logger("load_signals")`.
- Note: `load_signals.py` does NOT import `FetchResult` from `load_prices.py` (verified — it only imports from `data.config` and `data.progress`). No import fix needed.

### B2. Eliminate `_filter_to_universe` wrappers

**Files:** `quintile.py`, `single_feature_ic.py`, `robustness.py`

The wrappers in `quintile.py:60` and `single_feature_ic.py:139` are identical pass-throughs:

```python
def _filter_to_universe(df, universe_name, tolerance_days=None):
    from backtest.universe import filter_to_universe
    return filter_to_universe(df, universe_name, date_col="call_entry_date",
                              ticker_col="BESTTICKER", tolerance_days=tolerance_days)
```

`date_col="call_entry_date"` and `ticker_col="BESTTICKER"` are already the **defaults** of `filter_to_universe`. The wrappers are completely redundant.

**Changes:**

- `quintile.py:60-74`: delete `_filter_to_universe`. Change call site (line 225):
  ```python
  # Before:
  df = _filter_to_universe(df, universe_name)
  # After:
  df = filter_to_universe(df, universe_name)
  ```
  Add `from backtest.universe import filter_to_universe` to the imports of `run_quintile_analysis` (currently it's imported lazily inside the wrapper).

- `single_feature_ic.py:139-153`: delete `_filter_to_universe`. Change call site (line 214):
  ```python
  # Before:
  df = _filter_to_universe(df, universe_name)
  # After:
  df = filter_to_universe(df, universe_name)
  ```
  Add `from backtest.universe import filter_to_universe` to the imports of `run_single_feature_ic`.

- `robustness.py:948`: change import from `from backtest.quintile import _filter_to_universe` to `from backtest.universe import filter_to_universe`. Change call site (line 968):
  ```python
  # Before:
  df = _filter_to_universe(df, universe_name)
  # After:
  df = filter_to_universe(df, universe_name)
  ```

### B3. Consolidate backtest statistics to `backtest/_stats.py`

**Files:** `model.py`, `splits.py`, `quintile.py`, `robustness.py`, `single_feature_ic.py`

Summary of deletions and replacements:

| Delete from | Symbol | Replace with |
|-------------|--------|--------------|
| `model.py:46` | `_make_median_imputer` | `from backtest._stats import make_median_imputer` |
| `model.py:625` | `_spearman` | `from backtest._stats import spearman` |
| `splits.py:611` | `_make_median_imputer` | `from backtest._stats import make_median_imputer` |
| `splits.py:794` | `_spearman` | `from backtest._stats import spearman` |
| `quintile.py:82` | `_bucket_returns` | `from backtest._stats import bucket_returns` |
| `quintile.py:128` | `_build_equity_curves` | `from backtest._stats import build_equity_curves` |
| `quintile.py:166` | `_portfolio_stats` (and nested `_max_drawdown`, `_rolling_sharpe`) | `from backtest._stats import portfolio_stats` |
| `robustness.py:842` | `_monthly_bucket_returns` | `from backtest._stats import bucket_returns` |
| `robustness.py:879` | `_build_equity_curves` | `from backtest._stats import build_equity_curves` |
| `robustness.py:901` | `_max_drawdown` (module-level) | `from backtest._stats import max_drawdown_from_equity` |
| `robustness.py:185-199` | inline stats (subperiod_quintile) | `portfolio_stats(eq, leg="long_short")` |
| `robustness.py:524-534` | inline stats (mcap_bucket_quintile) | `portfolio_stats(eq, leg="long_short")` |
| `robustness.py:706-718` | inline stats (ofat_quantile_cutoff) | `portfolio_stats(eq, leg="long_short")` |
| `robustness.py:819-832` | inline stats (compare_weighting_schemes) | `portfolio_stats(eq, leg="long_short")` |

Also replace all 7 inline `sort_values + drop_duplicates` occurrences with `dedup_latest_per_ticker(gdf)`:
- `single_feature_ic.py:118-119`
- `quintile.py:103-104` (absorbed inside `bucket_returns` → `backtest/_stats.py`)
- `robustness.py:79-80, 239-240, 639-640, 734-735, 855-856` (the last one absorbed inside `bucket_returns` → `backtest/_stats.py`)

Note: `score_weighted_bucket_returns()` and `_top_n_bucket_returns()` in `robustness.py` are NOT equivalent to equal-size bucket returns and stay local.

### B4. Extract forward-return join helper

**Files:** `quintile.py`, `single_feature_ic.py`, `robustness.py`, `model.py`

Add `ensure_forward_returns` to `backtest/splits.py`:

```python
from pathlib import Path
import pandas as pd

def ensure_forward_returns(
    df: pd.DataFrame,
    features_path: Path,
    price_cache_dir: Path,
    horizons: list[int] | None = None,
    entry_date_col: str = "availability_date",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Ensure forward-return columns are attached to *df*.

    Parameters
    ----------
    df:
        Feature DataFrame. Columns are added in-place AND the DataFrame is returned.
    features_path:
        Path to the features parquet (used as cache key).
    price_cache_dir:
        Directory of per-ticker price parquets.
    horizons:
        Forward-return horizons. Defaults to ``HORIZONS``.
    entry_date_col:
        Column used as entry date for forward returns.
    overwrite:
        If False (default), compute only when ``forward_return_{h}d`` columns
        are missing. If True, recompute and replace.

    Returns
    -------
    The same DataFrame with ``forward_return_{h}d`` and
    ``target_available_date_{h}d`` columns added (or replaced).
    Row count, row order, and original index are preserved.
    """
    if horizons is None:
        from backtest.splits import HORIZONS
        horizons = list(HORIZONS)

    existing_cols = {f"forward_return_{h}d" for h in horizons}
    has_all = existing_cols.issubset(set(df.columns))

    if has_all and not overwrite:
        log.info("forward returns already present; skipping (use overwrite=True to force)")
        return df

    # Drop any stale columns before recomputing when overwrite=True
    if overwrite:
        drop_cols = [c for c in df.columns
                     if c.startswith("forward_return_") or c.startswith("target_available_date_")]
        df = df.drop(columns=drop_cols)

    from backtest.splits import get_forward_returns_cached

    fwd = get_forward_returns_cached(
        features_path,
        df,
        price_cache_dir,
        horizons=horizons,
        entry_date_col=entry_date_col,
    )
    # Use pd.concat with index reset to match model.py pattern (safe for all callers).
    # The original 3-caller column-loop pattern mutates df in place; concat is equivalent
    # because fwd is always row-aligned with df.
    df = pd.concat([df.reset_index(drop=True), fwd.reset_index(drop=True)], axis=1)

    if overwrite:
        log.info("forward returns recomputed: %d cols", len(fwd.columns))
    else:
        log.info("forward returns added: %d cols", len(fwd.columns))
    return df
```

**Caller changes:**

All four callers replace their inline block:

```python
# Before (quintile.py, single_feature_ic.py):
from backtest.splits import get_forward_returns_cached
fwd = get_forward_returns_cached(features_path, df, price_cache_dir, entry_date_col="availability_date")
for col in fwd.columns:
    df[col] = fwd[col]

# Before (robustness.py):
has_returns = any(c.startswith("forward_return_") for c in df.columns)
if not has_returns:
    from backtest.splits import get_forward_returns_cached
    fwd = get_forward_returns_cached(features_path, df, price_cache_dir, entry_date_col="availability_date")
    for col in fwd.columns:
        df[col] = fwd[col]

# Before (model.py):
from backtest.splits import get_forward_returns_cached
fwd = get_forward_returns_cached(args.features, feat_df, horizons=args.horizons, entry_date_col=args.availability_col)
feat_df = pd.concat([feat_df.reset_index(drop=True), fwd.reset_index(drop=True)], axis=1)

# After (all four):
from backtest.splits import ensure_forward_returns

df = ensure_forward_returns(df, features_path, price_cache_dir)
# model.py passes explicit horizons + entry_date_col:
feat_df = ensure_forward_returns(feat_df, args.features, PRICE_CACHE_DIR,
                                 horizons=args.horizons, entry_date_col=args.availability_col)
# robustness.py drops the has_returns guard — ensure_forward_returns handles it internally.
```

---

## Part C: Split Large Functions

### C1. `PortfolioSimulator.run` (470 lines) — HIGHEST PRIORITY

**File:** `backtest/portfolio.py`

Extract 6 methods plus a `SimulationState` dataclass.

#### C1.1 `SimulationState` dataclass

```python
from dataclasses import dataclass, field

@dataclass
class SimulationState:
    """Mutable state shared across the PortfolioSimulator daily loop."""
    current_weights: dict[str, float] = field(default_factory=dict)
    prev_weights: dict[str, float] = field(default_factory=dict)
    current_entry_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    daily_records: list[dict[str, Any]] = field(default_factory=list)
    trade_records: list[dict[str, Any]] = field(default_factory=list)
    long_gap_records: list[dict[str, Any]] = field(default_factory=list)
    weights_history: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    cohort_weights_history: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    turnover: float = 0.0
```

#### C1.2 Extracted methods

| Method | Signature | Return | Purpose |
|--------|-----------|--------|---------|
| `_validate_and_filter` | `(self, signals: pd.DataFrame, start: pd.Timestamp \| None, end: pd.Timestamp \| None) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int, set[str], set[str], set[str]]` | `(signals, start, end, n_before, tickers, coverage_tickers, price_tickers)` | Copy signals, coerce dates, filter date range, extract ticker sets |
| `_load_calendar_and_prices` | `(self, tickers: set[str], coverage_tickers: set[str], start: pd.Timestamp, end: pd.Timestamp, lookback: int) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]` | `(close_matrix, calendar, rebal_dates)` | Load calendar, price table, close matrix, coverage series |
| `_process_rebalance` | `(self, state: SimulationState, signals: pd.DataFrame, close_matrix: pd.DataFrame, cal_list: list[pd.Timestamp], cal_pos: int, rebal_set: set[pd.Timestamp], lookback: int, long_frac: float) -> None` | None (mutates `state`) | Build cohort weights, look up entry prices, compute turnover, append trade records |
| `_compute_daily_pnl` | `(self, state: SimulationState, close_matrix: pd.DataFrame, cal_list: list[pd.Timestamp], cal_pos: int, date: pd.Timestamp) -> None` | None (mutates `state`) | Compute daily P&L with forward-fill gap logic |
| `_resolve_price_gaps` | `(daily_df: pd.DataFrame, long_gap_records: list, close_matrix: pd.DataFrame, cal_list: list[pd.Timestamp]) -> pd.DataFrame` | `daily_df` (mutated copy) | Pure function: classify positions as normal/ffill1d/ffill2d/long-gap; apply gap recovery |
| `_post_process` | `(self, state: SimulationState, close_matrix: pd.DataFrame, cal_list: list[pd.Timestamp], rebal_dates: pd.DataFrame, universe_coverage: pd.Series, signals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cadence: str, lookback: int, long_frac: float, transaction_cost_bps: float) -> PortfolioResult` | `PortfolioResult` | Long-gap audit, trade log validation, build and return result |

`run()` becomes ~50 lines:

```python
def run(self, signals, cadence="weekly", lookback=5, start=None, end=None,
        long_frac=0.2, transaction_cost_bps=5.0) -> PortfolioResult:
    # 1. Validate & filter
    signals, start, end, n_before, tickers, coverage_tickers, price_tickers = \
        self._validate_and_filter(signals, start, end)
    # 2. Load calendar + prices
    close_matrix, calendar, rebal_dates = \
        self._load_calendar_and_prices(tickers, coverage_tickers, start, end, lookback)
    # 3. Daily loop
    state = SimulationState()
    cal_list = list(calendar)
    rebal_set = set(rebal_dates)
    for pos, date in enumerate(cal_list):
        if date in rebal_set:
            self._process_rebalance(state, signals, close_matrix, cal_list, pos,
                                    rebal_set, lookback, long_frac)
        self._compute_daily_pnl(state, close_matrix, cal_list, pos, date)
    # 4. Post-process
    return self._post_process(state, close_matrix, cal_list, rebal_dates,
                              universe_coverage, signals, start, end,
                              cadence, lookback, long_frac, transaction_cost_bps)
```

### C2. `compute_momentum_features` (206 lines)

**File:** `features/engineer.py:659`

Signature:
```python
def compute_momentum_features(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
```
Returns a 3-column DataFrame: `pre_event_ret_21d`, `pre_event_ret_21d_sector_rel`, `pre_event_idio_resid_5d`.

Extract each numbered section to a private module-level function:

| Function | Signature | Returns | Lines |
|----------|-----------|---------|-------|
| `_load_ticker_prices` | `(unique_tickers: list[str], price_cache_dir: Path, calendar: pd.DatetimeIndex) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]` | `(price_cache, daily_ret_cache)` | ~18 |
| `_build_sector_daily_returns` | `(daily_ret_cache: dict[str, pd.Series], ticker_to_sector: dict[str, str]) -> dict[str, pd.DataFrame]` | `sector_metrics` dict keyed by sector | ~20 |
| `_precompute_ticker_metrics` | `(price_cache: dict[str, pd.DataFrame], daily_ret_cache: dict[str, pd.Series], sector_metrics: dict[str, pd.DataFrame], ticker_to_sector: dict[str, str], calendar: pd.DatetimeIndex) -> pd.DataFrame` | `all_metrics` with per-ticker pre-computed returns + beta | ~63 |
| `_merge_metrics_with_events` | `(df: pd.DataFrame, all_metrics: pd.DataFrame, ticker_col: str) -> pd.DataFrame` | `merged` with metrics joined to each event | ~23 |
| `_compute_final_momentum_features` | `(merged: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame` | `out` with 3 final columns filled | ~20 |

Main function becomes a linear sequence of 6 calls (~35 lines).

### C3. `run_all_robustness` (241 lines)

**File:** `backtest/robustness.py:926`

Signature:
```python
def run_all_robustness(
    features_path: Path,
    universe_name: str = "sp500",
    price_cache_dir: Path = PRICE_CACHE_DIR,
    shares_cache_dir: Path = SHARES_CACHE_DIR,
    output_dir: Path | None = None,
    features: list[str] | None = None,
    skip_mcap: bool = False,
) -> RobustnessResult:
```

Extract each of the 9 numbered sections to a module-level `_run_*()` helper:

| Function | Purpose |
|----------|---------|
| `_run_subperiod_ic(df, horizons, signal_types, features, output_dir) -> pd.DataFrame` | Section 1 |
| `_run_subperiod_quintile(df, horizons, signal_types, features, output_dir) -> pd.DataFrame` | Section 2 |
| `_run_sector_neutral(df, horizons, output_dir) -> pd.DataFrame` | Section 3 |
| `_run_mcap_buckets(df, horizons, shares_cache_dir, output_dir) -> pd.DataFrame` | Section 4 |
| `_run_weighting_comparison(df, horizons, output_dir) -> pd.DataFrame` | Section 5 |
| `_run_ofat_quantile(df, horizons, output_dir) -> pd.DataFrame` | Section 6 |
| `_run_ofat_cost(df, horizons, output_dir) -> pd.DataFrame` | Section 7 |
| `_run_bootstrap(df, horizons, output_dir) -> dict` | Section 8 |
| `_run_beta_window_ic(df, horizons, output_dir) -> pd.DataFrame` | Section 9 |

Main function becomes ~30 lines of sequential calls populating `RobustnessResult`.

### C4. `run_walk_forward` (155 lines) + `_run_one_fold` (168 lines)

**File:** `backtest/model.py`

**`run_walk_forward`** signature:
```python
def run_walk_forward(
    df: pd.DataFrame,
    model_name: str = "ridge",
    tier: str = "enhanced",
    horizons: list[int] | None = None,
    hparams_dir: Path | None = None,
    audit_dir: Path | None = None,
    availability_col: str = "availability_date",
    universe_name: str = "all",
    signal_type: str = "all",
) -> dict[int, WalkForwardResult]:
```

Extract:
- `_build_oos_predictions_df(fold_results: list[FoldResult], horizon: int, model_name: str) -> pd.DataFrame` — concatenates per-fold predictions into one OOS DataFrame
- `_write_walk_forward_audit(oos_df: pd.DataFrame, per_fold: list[FoldResult], horizon: int, model_name: str, audit_dir: Path) -> None` — writes parquet + manifest + sample-size audit

**`_run_one_fold`** signature:
```python
def _run_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    model_name: str,
    hparams: dict[str, Any],
    horizon: int,
    tier: str,
    availability_col: str,
    feature_names: list[str] | None = None,
) -> FoldResult | None:
```

Extract:
- `_purge_and_sort(df, fold, horizon, availability_col) -> tuple[np.ndarray, np.ndarray]` — G11 purge, train sort. Returns `(train_idx, test_idx)` or raises early-return signal (empty fold).
- `_fit_and_audit_model(X_train, y_train, X_test, model_name, hparams, tier, horizon, fold, feature_names) -> tuple[Any, list[str], float, list[str]]` — impute, scale, select (stretch), fit, audit call stack, validate fold boundaries. Returns `(model, fit_violations, fit_time, selected_features)`.

### C5. `compute_forward_returns` (150 lines)

**File:** `backtest/splits.py:216`

Signature:
```python
def compute_forward_returns(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    horizons: Sequence[int] = HORIZONS,
    entry_date_col: str = "availability_date",
) -> pd.DataFrame:
```
Returns DataFrame with `forward_return_{h}d` (float64) and `target_available_date_{h}d` (datetime64[ns]) columns.

Extract:
- `_prepare_events(df, entry_date_col) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]` — pre-allocate result arrays, build per-ticker event groups. Returns `(event_dates, event_tickers, events_df)`.
- `_compute_ticker_forward_returns(tkr_str, ev, price_cache_dir, h_list) -> tuple[int, dict[int, float], dict[int, pd.Timestamp]]` — per-ticker loop body. Returns `(n_with_prices, ret_map, date_map)`.
- `_merge_ticker_results(ret_arrays, date_arrays, h_list, n_with_prices, ret_map, date_map, idx) -> int` — copy per-ticker results into pre-allocated arrays. Returns updated `n_with_prices`.

**Cache invalidation fix — do BEFORE extraction:**

The current cache signature at `splits.py:172` hashes `inspect.getsource(compute_forward_returns)`. After extraction, changes to helper functions won't alter that source string. Add a `FORWARD_RETURNS_CACHE_VERSION` constant and include it in the cache signature:

```python
# Near the top of splits.py, near HORIZONS
FORWARD_RETURNS_CACHE_VERSION = 1  # bump when compute_forward_returns or its helpers change
```

In `_build_cache_signature` (~line 172), replace:
```python
source = inspect.getsource(compute_forward_returns)
source_hash = int(hashlib.sha256(source.encode()).hexdigest()[:16], 16)
```
with:
```python
version_bytes = str(FORWARD_RETURNS_CACHE_VERSION).encode()
source = inspect.getsource(compute_forward_returns)
combined = version_bytes + source
source_hash = int(hashlib.sha256(combined).hexdigest()[:16], 16)
```

The version approach is simpler and less error-prone than maintaining a list of helper source hashes. Any future helper change requires bumping the integer — impossible to miss in review.

### C6. `main` functions

**`portfolio.py:1107`** — `def main() -> None`, 229 lines:

Extract:
- `_normalize_signals(signals_path: Path, signal_type: str) -> pd.DataFrame` — load parquet, detect format (OOS `[df_index, y_pred]` vs standard `[date, ticker, score]`), normalize column names, return standard-format DataFrame
- `_persist_results(result: PortfolioResult, output_dir: Path, tag: str | None, sim_config: dict) -> None` — write daily returns, weights, trade log, violations, coverage, gap accounting, capacity metrics, summary JSON

**`model.py:727`** — `def main() -> None`, 171 lines:

Extract:
- `_load_and_filter_features(args) -> pd.DataFrame` — read parquet, compute forward returns if needed, filter to universe + signal_type
- `_run_tuning_if_needed(args, feat_df) -> None` — per-model hparam tuning (skips models with existing frozen params unless `--tune-only`)
- `_run_walk_forward_all_models(args, feat_df) -> None` — iterate models, call `run_walk_forward`, write OOS predictions

### C7. `run_streaming_vs_batch_test` (176 lines) + `run_all_audits` (159 lines)

**File:** `features/audit.py`

**`run_streaming_vs_batch_test`** at line 130:
```python
def run_streaming_vs_batch_test(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    n_sample_dates: int | None = None,
    include_momentum: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
```

Extract the three clear sections:
- `_batch_build(df, price_cache_dir, tier, include_momentum) -> tuple[pd.DataFrame, pd.Index, pd.Index]` — compute features on full sample, partition column names. Returns `(batch, strict_cols, xsectional_cols)`.
- `_streaming_replay(df, batch, sampled_dates, price_cache_dir, tier, include_momentum, rtol, atol, strict_cols, xsectional_cols) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]` — per-date streaming loop. Returns `(all_strict_mm, all_xsec_mm)`.
- `_compare_results(all_strict_mm, all_xsec_mm, strict_cols, xsectional_cols, n_rows) -> tuple[bool, pd.DataFrame, dict[str, Any]]` — assemble mismatch table + summary stats.

**`run_all_audits`** at line 1104:
```python
def run_all_audits(
    signals_path: Path = SIGNALS_PARQUET,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    small_only: bool = False,
    full_dates: int = 15,
) -> dict[str, Any]:
```

This function already delegates each assertion to a standalone function call. No further extraction needed — it is already well-decomposed. Drop C7 for `run_all_audits`.

---

## Implementation Order

0. **Freeze baseline outputs** before any refactoring:
   ```bash
   # Forward returns
   python -c "
   from backtest.splits import compute_forward_returns
   import pandas as pd
   df = pd.read_parquet('results/features_enhanced.parquet').head(500)
   fwd = compute_forward_returns(df)
   fwd.to_parquet('/tmp/baseline_forward_returns.parquet')
   "
   # Quintile bucket returns
   python -m backtest.quintile --features results/features_enhanced.parquet --universe sp500 \
       --output /tmp/baseline_quintile/
   # Single-feature IC
   python -m backtest.single_feature_ic --features results/features_enhanced.parquet --universe sp500 \
       --output /tmp/baseline_ic/
   # Portfolio simulation (on a small fixture)
   python -m backtest.portfolio --signals <fixture> --features results/features_enhanced.parquet \
       --universe sp500 --cadence weekly --lookback 5 --output-dir /tmp/baseline_portfolio/
   # Failed-ticker CSVs
   cp data/cache/*_failed_tickers.csv /tmp/baseline_failed_prices.csv  # if exists
   cp data/cache/shares_failed_tickers.csv /tmp/baseline_failed_shares.csv  # if exists
   ```
1. **Create `data/_utils.py`** → verify importable. Test manifest + failed-log + backoff helpers on synthetic inputs.
2. **Create `backtest/_stats.py`** → verify importable. Test each stats helper on small DataFrames.
3. **Data-layer consolidation** (B1) → compare failed-log CSV and manifest JSON against baseline.
4. **Backtest consolidation** (B2, B3, B4) → compare forward returns, IC, bucket returns, and equity curves against baseline.
5. **Fix `FORWARD_RETURNS_CACHE_VERSION` in splits.py** → bump to 1, update cache signature code.
6. **Split large functions** (C1→C7, ordered by impact) → after each split, compare relevant fixture output against baseline.
7. **Final cleanup** — remove unused imports, verify all CLI entry points.

## Verification

- **Per step:** `python -m compileall <modified_files>` passes.
- **Per consolidation:** import the shared module from all consumers.
- **Per split:** diff outputs before/after against baseline fixtures.
- **Cache safety:** after C5, bump `FORWARD_RETURNS_CACHE_VERSION` whenever any forward-return helper changes.
- **Failed logs:** `diff /tmp/baseline_failed_prices.csv <regenerated>` is empty.
- **Stats helpers:** compare bucket returns, equity curves, portfolio stats, and drawdown values before/after B3.
- **End-to-end:** `python run_all.py --dry-run --tier both` enumerates all sub-tasks correctly.
- **Smoke test:** `python run_all.py --from-phase 1 --stop-at-phase 1` completes.

The most critical single verification is running `PortfolioSimulator` on a known fixture before/after the C1 split — the gap-resolution logic in `_compute_daily_pnl` is the most complex code in the project.
