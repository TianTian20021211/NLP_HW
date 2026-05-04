# C2. Split `features/engineer.py::compute_momentum_features`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Current behavior must be preserved:
- T is `MOSTIMPORTANTDATEUTC`, not `entry_date`.
- Anchor is the trading day strictly before T.
- Rolling beta ends at T-5 via `.shift(4)`.
- Sector returns use median daily returns.
- Per-ticker `merge_asof` uses 10-calendar-day tolerance.

## C2.1 `_initialize_momentum_output`

```python
def _initialize_momentum_output(index: pd.Index) -> pd.DataFrame:
    """Create the three-column all-NaN momentum output frame."""
```

Parameters:
- `index`: output index.

Return: DataFrame with:
- `pre_event_ret_21d`
- `pre_event_ret_21d_sector_rel`
- `pre_event_idio_resid_5d`

## C2.2 `_load_ticker_price_cache`

```python
def _load_ticker_price_cache(
    unique_tickers: np.ndarray,
    price_cache_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], pd.DatetimeIndex]:
    """Load per-ticker prices, daily returns, and the union trading calendar."""
```

Return:
- `price_cache`
- `daily_ret_cache`
- `calendar`

## C2.3 `_build_ticker_sector_map`

```python
def _build_ticker_sector_map(
    df: pd.DataFrame,
    ticker_col: str,
) -> dict[str, str]:
    """Map ticker to first observed sector using one groupby pass."""
```

Return: `{ticker: sector}`.

## C2.4 `_build_sector_return_metrics`

```python
def _build_sector_return_metrics(
    daily_ret_cache: dict[str, pd.Series],
    ticker_to_sector: dict[str, str],
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Build sector median daily returns and 5d/21d sector return metrics."""
```

Return:
- `sector_daily_rets`: sector -> daily median return Series.
- `sector_metrics`: sector -> DataFrame with `sector_ret_5d`,
  `sector_ret_21d`.

## C2.5 `_build_event_anchor_frame`

```python
def _build_event_anchor_frame(
    df: pd.DataFrame,
    ticker_col: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build event rows with _orig_idx, ticker, parsed T, and anchor_date."""
```

Return: DataFrame with columns:
- `_orig_idx`
- `ticker_col`
- `_t`
- `anchor_date`

## C2.6 `_precompute_ticker_momentum_metrics`

```python
def _precompute_ticker_momentum_metrics(
    price_cache: dict[str, pd.DataFrame],
    daily_ret_cache: dict[str, pd.Series],
    sector_daily_rets: dict[str, pd.Series],
    sector_metrics: dict[str, pd.DataFrame],
    ticker_to_sector: dict[str, str],
    ticker_col: str,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Precompute stock returns, sector returns, and shifted rolling beta by ticker/date."""
```

Return: DataFrame with at least:
- `ticker_col`
- `date`
- `stock_ret_5d`
- `stock_ret_21d`
- `sector_ret_5d`
- `sector_ret_21d`
- `beta`

## C2.7 `_merge_momentum_metrics`

```python
def _merge_momentum_metrics(
    events: pd.DataFrame,
    all_metrics: pd.DataFrame,
    ticker_col: str,
) -> pd.DataFrame:
    """Per-ticker merge_asof from event anchor_date to precomputed metrics."""
```

Return: merged event/metric DataFrame.

## C2.8 `_fill_momentum_output`

```python
def _fill_momentum_output(
    out: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """Fill the three final momentum columns from merged metrics."""
```

Return: `out`.
