# C5. Split `backtest/splits.py::compute_forward_returns`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

## C5.1 Cache Version And Source Hash

After extraction, bump:

```python
FORWARD_RETURNS_CACHE_VERSION = 2
```

Add:

```python
def _build_forward_returns_source_hash() -> int:
    """Hash compute_forward_returns plus all extracted helper sources and cache version."""
```

Parameters: none.

Return: deterministic integer hash.

Implementation:

```python
funcs = [
    compute_forward_returns,
    _empty_forward_return_frame,
    _prepare_forward_return_events,
    _load_forward_return_price_metrics,
    _compute_ticker_forward_return_matches,
    _fill_forward_return_arrays,
]
source_blob = "\n\n".join(inspect.getsource(fn) for fn in funcs)
source_blob += f"\nCACHE_VERSION={FORWARD_RETURNS_CACHE_VERSION}"
return int(hashlib.sha256(source_blob.encode()).hexdigest()[:16], 16)
```

Then ``_build_cache_signature(features_path: Path, price_cache_dir: Path,
horizons: Sequence[int], entry_date_col: str, df: pd.DataFrame) ->
ForwardReturnsCacheSignature`` uses ``_build_forward_returns_source_hash()``
and also stores ``len(df)`` and a deterministic row-order fingerprint
(``hash(tuple(df.index[:1000]))``) so that the cache automatically
misses when the caller passes a filtered/subset DataFrame with the same
``features_path``.  On cache hit the receiving code asserts that the
cached fingerprint matches the callerʼs ``df`` and raises ``ValueError``
on mismatch — this catches any case where the signature fields happen to
collide for different DataFrames.

## C5.2 `_empty_forward_return_frame`

```python
def _empty_forward_return_frame(
    index: pd.Index,
    horizons: Sequence[int],
    ret_arrays: dict[int, np.ndarray],
    date_arrays: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Build the standard forward-return result DataFrame."""
```

Return: DataFrame indexed like the input with
`forward_return_{h}d` and `target_available_date_{h}d`.

## C5.3 `_prepare_forward_return_events`

```python
def _prepare_forward_return_events(
    df: pd.DataFrame,
    ticker_col: str,
    entry_date_col: str,
) -> pd.DataFrame:
    """Build per-row event metadata for forward-return matching."""
```

Return: DataFrame with:
- `_orig_pos`
- `_ticker`
- `_entry_date`

Drop rows with missing ticker or entry date.

## C5.4 `_load_forward_return_price_metrics`

```python
def _load_forward_return_price_metrics(
    path: Path,
    horizons: Sequence[int],
) -> pd.DataFrame | None:
    """Load one ticker price file and precompute forward returns by price row."""
```

Return: metrics DataFrame with:
- `_merge_date`
- `_price_pos`
- `forward_return_{h}d`
- `target_available_date_{h}d`

Return `None` for missing/empty/unusable prices.

## C5.5 `_compute_ticker_forward_return_matches`

```python
def _compute_ticker_forward_return_matches(
    ticker: str,
    events: pd.DataFrame,
    price_cache_dir: Path,
    horizons: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Merge one ticker's events to entry prices and return matched arrays."""
```

Return:
- `orig_pos`: original row positions with valid entries/returns for at least
  one horizon.
- `ret_by_h`: horizon -> return array aligned to `orig_pos`.
- `date_by_h`: horizon -> target date array aligned to `orig_pos`.

Rules:
- Entry match uses `pd.merge_asof(events, metrics, left_on="_entry_date", right_on="_merge_date", direction="forward")`.
- Entry gap must be between 0 and 3 business days inclusive.
- Missing exits remain NaN/NaT.

## C5.6 `_fill_forward_return_arrays`

```python
def _fill_forward_return_arrays(
    ret_arrays: dict[int, np.ndarray],
    date_arrays: dict[int, np.ndarray],
    orig_pos: np.ndarray,
    ret_by_h: dict[int, np.ndarray],
    date_by_h: dict[int, np.ndarray],
    horizons: Sequence[int],
) -> None:
    """Copy one ticker's matched return arrays into the preallocated output arrays."""
```

Return: `None`.

## C5.7 New `compute_forward_returns()` Shape

`compute_forward_returns(df: pd.DataFrame, price_cache_dir: Path = PRICE_CACHE_DIR, horizons: Sequence[int] = HORIZONS, entry_date_col: str = "availability_date") -> pd.DataFrame` should:

1. Allocate arrays.
2. Return empty result if price directory is missing.
3. Build events with `_prepare_forward_return_events`.
4. Loop by ticker, calling `_compute_ticker_forward_return_matches`.
5. Fill arrays with `_fill_forward_return_arrays`.
6. Return `_empty_forward_return_frame(df.index, h_list, ret_arrays, date_arrays)`.
