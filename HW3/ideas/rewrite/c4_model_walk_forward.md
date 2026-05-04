# C4. Split `backtest/model.py::_run_one_fold` and `run_walk_forward`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

## C4.1 `FoldSample`

```python
@dataclass
class FoldSample:
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    target_col: str
    target_date_col: str
```

Return: dataclass constructor returns `FoldSample`.

## C4.2 `_prepare_fold_sample`

```python
def _prepare_fold_sample(
    df: pd.DataFrame,
    fold: Fold,
    horizon: int,
    availability_col: str,
    feature_names: list[str],
) -> FoldSample | None:
    """Apply G11 purge, sort training by availability date, drop NaN targets, and build X/y."""
```

Return: `FoldSample`, or `None` when the fold has too few valid rows.

Rules:
- Use `purge_train_for_horizon`.
- Sort training rows by `availability_col`.
- Keep DataFrame indices as `DatetimeIndex` so fit monitoring records dates.

## C4.3 `_fit_select_predict_with_audit`

```python
def _fit_select_predict_with_audit(
    sample: FoldSample,
    fold: Fold,
    model_name: str,
    hparams: dict[str, Any],
    tier: str,
    horizon: int,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str] | None, float, list[str], list[dict[str, Any]]] | None:
    """Impute, scale, optionally select stretch features, fit model, predict, and audit fit calls.

    Returns ``None`` when the stretch inner purged CV is empty (can happen for
    small folds with many NaN targets after G11 purge).  Callers must handle
    this by skipping the fold.
    """
```

Return (when not ``None``):
- `y_pred`
- `selected_features`
- `fit_time_s`
- `fit_violations`
- `fit_log`

Rules:
- Wrap imputation, scaling, LassoCV, and model fit in `monitor_fit_calls()`.
- Validate fit logs with `assert_fit_callstack`.
- Validate fold boundaries with `assert_fold_boundaries`.
- When ``tier == "stretch"`` and the inner purged time-series split produces
  no valid folds, return ``None`` immediately (before calling any model fit
  routine).

## C4.4 `_build_fold_result`

```python
def _build_fold_result(
    fold: Fold,
    horizon: int,
    model_name: str,
    sample: FoldSample,
    y_pred: np.ndarray,
    selected_features: list[str] | None,
    fit_time_s: float,
    fit_violations: list[str],
    fit_log: list[dict[str, Any]],
) -> FoldResult:
    """Pack fold outputs into FoldResult."""
```

Return: `FoldResult`.

## C4.5 `_build_oos_predictions_df`

```python
def _build_oos_predictions_df(
    df: pd.DataFrame,
    fold_results: list[FoldResult],
    horizon: int,
    model_name: str,
    tier: str,
    universe_name: str,
    signal_type: str,
) -> tuple[pd.DataFrame | None, float, float]:
    """Concatenate fold predictions and compute aggregate OOS IC/MSE."""
```

Return:
- OOS predictions DataFrame or `None`.
- `oos_ic`.
- `oos_mse`.

OOS DataFrame columns must stay:
- `df_index`
- `horizon`
- `model`
- `tier`
- `universe`
- `signal_type`
- `y_pred`
- `y_true`

## C4.6 `_write_oos_predictions`

```python
def _write_oos_predictions(
    oos_df: pd.DataFrame | None,
    audit_dir: Path,
    model_name: str,
    tier: str,
    universe_name: str,
    signal_type: str,
    horizon: int,
) -> Path | None:
    """Write one horizon's OOS prediction parquet using the existing filename scheme."""
```

Return: written path, or `None`.

## C4.7 `_write_walk_forward_audit_outputs`

```python
def _write_walk_forward_audit_outputs(
    df: pd.DataFrame,
    folds: list[Fold],
    results: dict[int, WalkForwardResult],
    model_name: str,
    tier: str,
    horizons: list[int],
    audit_dir: Path,
    universe_name: str,
    signal_type: str,
) -> None:
    """Write fold manifest and sample-size audit files."""
```

Return: `None`.

## C4.8 New `_run_one_fold()` Shape

`_run_one_fold(df: pd.DataFrame, fold: Fold, model_name: str, hparams: dict[str, Any], horizon: int, tier: str, availability_col: str, feature_names: list[str] | None = None) -> FoldResult | None` becomes:

1. `sample = _prepare_fold_sample(df, fold, horizon, availability_col, feature_names)`
2. If `sample is None`, return `None`.
3. `result = _fit_select_predict_with_audit(sample, fold, model_name, hparams, tier, horizon, feature_names)`
4. If `result is None` (stretch inner CV empty), return `None`.
5. `y_pred, selected, fit_time, violations, fit_log = result`
6. Return `_build_fold_result(fold, horizon, model_name, sample, y_pred, selected, fit_time, violations, fit_log)`.

## C4.9 New `run_walk_forward()` Shape

`run_walk_forward(df: pd.DataFrame, model_name: str, tier: str = "enhanced", horizons: list[int] | None = None, hparams_dir: Path | None = None, audit_dir: Path | None = None, availability_col: str = "availability_date", universe_name: str = "all", signal_type: str = "all") -> dict[int, WalkForwardResult]` should:

1. Generate folds once.
2. Cache `feature_names` once.
3. For each horizon, load frozen hparams.
4. Run `_run_one_fold` across folds.
5. Use `_build_oos_predictions_df`.
6. Use `_write_oos_predictions`.
7. Use `_write_walk_forward_audit_outputs` once after all horizons.
