# C6. Split CLI `main()` Functions

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

## C6.1 `backtest/portfolio.py::main`

Add:

```python
def _normalize_signals(
    signals_path: Path,
    features_path: Path | None,
    signal_type: str,
    date_col: str,
) -> pd.DataFrame:
    """Load signals or OOS predictions and return standard [date, ticker, score] columns."""
```

Return: normalized signals DataFrame.

```python
def _portfolio_output_suffix(
    universe: str,
    cadence: str,
    lookback: int,
    tag: str | None,
    signals: pd.DataFrame,
) -> str:
    """Build the existing output filename suffix."""
```

Return: suffix string.

```python
def _weights_history_to_frame(
    weights_history: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Convert rebalance-date keyed weights history to tall DataFrame."""
```

Return: DataFrame with `[ticker, raw_weight, rebalance_date]`.

```python
def _cohort_weights_history_to_frame(
    cohort_weights_history: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Convert raw cohort weights history to tall DataFrame."""
```

Return: DataFrame with `[cohort_date, ticker, raw_weight, rebalance_date]`.

```python
def _persist_portfolio_results(
    result: PortfolioResult,
    output_dir: Path,
    audit_dir: Path,
    suffix: str,
    cost_bps: float,
    price_cache_dir: Path,
    tag: str | None,
) -> Path:
    """Write portfolio, audit, capacity, and summary artifacts."""
```

Return: path to summary JSON.

## C6.2 `backtest/model.py::main`

Add:

```python
def _load_features_with_forward_returns(args: argparse.Namespace) -> pd.DataFrame:
    """Load feature parquet and ensure requested forward returns are present."""
```

Return: feature DataFrame.

```python
def _models_to_run(model_arg: str) -> list[str]:
    """Expand 'all' into ridge/lightgbm/xgboost."""
```

Return: model-name list.

```python
def _tune_missing_hparams(
    args: argparse.Namespace,
    feat_df: pd.DataFrame,
    models_to_run: list[str],
) -> None:
    """Tune missing or requested frozen hyperparameters for all horizons."""
```

Return: `None`.

```python
def _run_walk_forward_models(
    args: argparse.Namespace,
    feat_df: pd.DataFrame,
    models_to_run: list[str],
) -> None:
    """Run walk-forward and write fit audit logs for all selected models."""
```

Return: `None`.
