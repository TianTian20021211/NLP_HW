# Code Restructuring Plan - Reviewed Implementation Checklist

This plan is for a behavior-preserving refactor of the current NLP_HW
backtesting codebase. The goal is readability and stability, not new research
logic. The code must still satisfy the project requirements in
`docs/requirement.md`, the execution plan in `ideas/plan.md`, and the report
promises in `docs/report.md`.

## Module Index

Per-module reconstruction details live in the files below. The cross-cutting
rules in this file (Non-Negotiable Rules, Keep As-Is, Fixes To The Previous
Draft, Testing Strategy, Specific Regression Risks To Watch) apply to every
per-module plan.

| Section | Topic                                                               | File                                                       |
| ------- | ------------------------------------------------------------------- | ---------------------------------------------------------- |
| A1      | New shared module `data/_utils.py`                                  | [a1_data_utils.md](a1_data_utils.md)                       |
| A2      | New shared module `backtest/_stats.py`                              | [a2_backtest_stats.md](a2_backtest_stats.md)               |
| B1      | Data loader consolidation (`data/load_*.py`)                        | [b1_data_loaders.md](b1_data_loaders.md)                   |
| B2      | Remove redundant universe filter wrappers                           | [b2_universe_filter.md](b2_universe_filter.md)             |
| B3      | Replace duplicated stats helpers                                    | [b3_stats_helpers.md](b3_stats_helpers.md)                 |
| B4      | Add `ensure_forward_returns`                                        | [b4_ensure_forward_returns.md](b4_ensure_forward_returns.md) |
| C1      | Split `backtest/portfolio.py::PortfolioSimulator.run`               | [c1_portfolio_simulator.md](c1_portfolio_simulator.md)     |
| C2      | Split `features/engineer.py::compute_momentum_features`             | [c2_momentum_features.md](c2_momentum_features.md)         |
| C3      | Split `backtest/robustness.py::run_all_robustness`                  | [c3_robustness.md](c3_robustness.md)                       |
| C4      | Split `backtest/model.py::_run_one_fold` and `run_walk_forward`     | [c4_model_walk_forward.md](c4_model_walk_forward.md)       |
| C5      | Split `backtest/splits.py::compute_forward_returns`                 | [c5_forward_returns.md](c5_forward_returns.md)             |
| C6      | Split CLI `main()` functions                                        | [c6_cli_main.md](c6_cli_main.md)                           |
| C7      | Split `features/audit.py::run_streaming_vs_batch_test`              | [c7_streaming_audit.md](c7_streaming_audit.md)             |

## Non-Negotiable Rules

- Preserve all public CLI contracts and output file names unless this document
  explicitly calls out a required correctness fix.
- Preserve row count, row order, and index alignment whenever adding columns to
  a DataFrame.
- Do not change the no-look-ahead rules: `availability_date`, label purge,
  PIT universe filtering, 3-trading-day quote lookup, and delayed-entry PnL
  gating must behave exactly as they do now.
- ``ensure_forward_returns`` and ``get_forward_returns_cached`` must only be
  called on the full, unfiltered DataFrame as loaded from the features parquet.
  Calling on a filtered/subset DataFrame will hit a joblib cache keyed only on
  ``features_path`` and return forward returns for a different row set —
  producing silent row misalignment.  The cache signature now includes
  ``len(df)`` and a row-order fingerprint to detect this at call time.
- Keep helper modules as leaf modules. `data/_utils.py` must not import
  `data.load_*`; `backtest/_stats.py` must not import other `backtest.*`
  modules.
- Keep audit variants separate where they intentionally cross-check production
  logic. In particular, do not unify the Fenwick percentile implementation in
  `features/engineer.py` with the audit implementations in
  `features/audit.py`.
- Preserve existing output schemas. For example, quintile statistics currently
  use `n_months`; the shared stats helper must keep that key.

## Keep As-Is

These functions are long but still coherent, performance-sensitive, or useful
as independent cross-checks:

- `data/load_universes.py::build_sp500_pit(current: pd.DataFrame, changes: pd.DataFrame, start: dt.date = dt.date(2009, 1, 1), end: dt.date | None = None) -> tuple[pd.DataFrame, list[CoverageGap]]`
  - Reverse-replay algorithm; keep as one conceptual unit.
- `data/load_signals.py::stream_chunks(csv_path: Path, main_cols: list[str], slim_cols: list[str], chunksize: int = CHUNKSIZE) -> dict`
  - Streaming CSV to Parquet pipeline; avoid splitting around Arrow writes.
- `data/load_prices.py::_download_batch(tickers, start, end, base_sleep, max_retries) -> list[FetchResult]`
  - Keep the yfinance retry integration intact; only extract tiny utilities
    around it.
- `features/engineer.py::_strict_historical_percentile(values: pd.Series, history_dates: pd.Series, cutoff_dates: pd.Series) -> np.ndarray`
  and `features/audit.py::_strict_historical_percentile_queries(history_values: np.ndarray, history_dates: np.ndarray, query_values: np.ndarray, cutoff_dates: np.ndarray) -> np.ndarray`
  - They are intentionally separate production/audit implementations.
- `features/engineer.py::compute_qoq_deltas(df) -> pd.DataFrame`
  and `features/engineer.py::compute_4q_trend(df) -> pd.Series`
  - Sort/shift/date-block logic is tightly coupled and already readable.
- `backtest/robustness.py::assign_market_cap_buckets(df: pd.DataFrame, price_cache_dir: Path = PRICE_CACHE_DIR, shares_cache_dir: Path = SHARES_CACHE_DIR, universe_name: str = "sp500", coverage_warn_threshold: float = 0.85) -> pd.DataFrame`
  - Vectorized PIT market-cap logic; splitting it would force too many
    intermediate arguments and increase bug risk.
- `backtest/portfolio.py::_build_cohort_weights(scores, long_frac, n_min_positions) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Single transformation pipeline with important dollar-neutral semantics.
- `backtest/robustness.py::block_bootstrap(returns, block_size, n_boot, seed, periods_per_year) -> dict[str, float]`
  - Single clear bootstrap loop.

## Fixes To The Previous Draft

The earlier draft had several implementation hazards. This version fixes them:

- `PortfolioSimulator.run()` extraction now carries `universe_coverage`,
  `cal_pos_by_date`, and `rebal_dates` through the helper signatures.
- `SimulationState.weights_history` and `cohort_weights_history` are typed as
  `dict[pd.Timestamp, pd.DataFrame]`, matching `PortfolioResult` and
  `compute_capacity_metrics()`.
- `SimulationState.long_gap_records` is a
  `dict[pd.Timestamp, dict[str, float]]`, matching `_audit_long_gap_recovery()`.
- The daily-loop turnover field is `today_turnover`, not an ambiguous cumulative
  `turnover`.
- `compute_momentum_features()` helpers return both `sector_daily_rets` and
  `sector_metrics`; beta needs the former.
- Forward-return cache invalidation hashes the extracted helper sources, not
  only `compute_forward_returns()`.
- `ensure_forward_returns()` checks both return columns and target-date columns,
  avoids duplicate columns, and preserves the caller's index.
- Shared portfolio stats preserve the existing `n_months` key.

---

# Implementation Order

## Step 0 - Freeze Baselines

Before refactoring, create small reproducible baselines under
`tests/fixtures/rewrite_baseline/` when the artifacts are small enough to keep
with the repo. If an artifact is too large, store it under
`results/audit/rewrite_baseline/` and record the path in the manifest. Do not
use `/tmp` as the only copy of a baseline.

Required baseline artifacts:
- Forward returns on first 500 feature rows.
- IC summary for SP500.
- Quintile summary and equity curves for SP500.
- A small portfolio fixture covering normal fills, delayed entry, missing
  entry, missing exit, and long quote gap.
- Current failed-ticker CSVs:
  - `data/cache/prices/failed_tickers.csv`
  - `data/cache/shares/failed_tickers.csv`
- `manifest.json` containing the git SHA, command lines, Python/package
  versions, input artifact paths, input sizes/mtimes, input content hashes when
  practical, output hashes, and `created_at`.

## Testing Strategy

The rewrite uses a **layered strategy** matching the risk profile of each
change type.

### Layer 1: Unit Tests For New Shared Modules (TDD)

Write tests before implementation for the two new leaf modules. These are
pure functions with clear inputs and outputs — ideal for traditional TDD.
All file-writing tests must use `tmp_path`; unit tests must not read or write
the real `data/cache/`, `results/`, or user baseline directories.

- **`tests/test_data_utils.py`** covers `data/_utils.py`:
  - `test_fetch_result_constructor` — all field combinations, default error
  - `test_us_ticker_re_valid` — valid tickers (AAPL, BRK.B, A, ABC-D)
  - `test_us_ticker_re_invalid` — empty, numeric, lowercase, too long
  - `test_setup_logger_returns_logger` — correct name, level configurable
  - `test_suppress_yfinance_logging` — no-op when yfinance not imported (the
    function must handle this gracefully)
  - `test_load_manifest_missing_file` — returns default dict
  - `test_load_manifest_valid_file` — roundtrip with `save_manifest`
  - `test_save_manifest_stamps_updated_at` — key present after write
  - `test_save_manifest_thread_safety` — no corruption under concurrent writes
  - `test_exponential_backoff` — base case, max cap, attempt=0 through attempt=5
  - `test_load_failed_log_missing` — returns empty standard-schema frame
  - `test_load_failed_log_roundtrip` — save then load
  - `test_apply_failed_log_result_success_clears_ticker` — shares semantics
  - `test_apply_failed_log_result_empty_appends` — shares semantics, one row
    per ticker
  - `test_apply_failed_log_result_failure_replaces_previous_failure` — shares
    semantics, one latest failure row per ticker
  - `test_apply_failed_log_result_failure_then_success_clears` — shares
    semantics, later success removes stale failure
  - `test_apply_failed_log_result_success_then_failure_records_failure` —
    shares semantics, later non-success is retained
  - `test_apply_failed_log_results_success_clears` — price semantics
  - `test_apply_failed_log_results_empty_appends_no_dedup` — price semantics
  - `test_apply_failed_log_results_same_ticker_success_and_failure` — price
    semantics, successes clear stale rows and same-batch non-success rows are
    appended
  - `test_update_failed_log_result_locked_read_apply_write` — file roundtrip
    under the optional lock
  - `test_update_failed_log_results_locked_batch_read_apply_write` — batch file
    roundtrip under the optional lock

- **`tests/test_backtest_stats.py`** covers `backtest/_stats.py`:
  - `test_spearman_perfect_positive` — linear arrays
  - `test_spearman_perfect_negative` — reversed arrays
  - `test_spearman_with_ties` — rank-average behavior
  - `test_spearman_pairwise_nan_compatibility` — pandas-compatible pairwise
    NaN dropping
  - `test_spearman_insufficient_pairs_nan` — returns NaN when too few valid
    paired observations remain
  - `test_make_median_imputer_returns_simple_imputer` — strategy check
  - `test_dedup_latest_per_ticker` — basic dedup, custom column names
  - `test_dedup_latest_per_ticker_empty` — returns empty
  - `test_bucket_returns_empty_input` — returns standard-schema empty frame
  - `test_bucket_returns_basic` — synthetic data with known bucket assignments
  - `test_bucket_returns_min_samples` — small group drops below-threshold buckets
  - `test_bucket_returns_qcut_duplicate_edges` — preserves current
    `duplicates="drop"` behavior
  - `test_build_equity_curves_full_bucket_columns` — all expected column names
    present when all buckets are observed
  - `test_build_equity_curves_sparse_bucket_columns` — only observed bucket
    columns are created when qcut drops buckets
  - `test_build_equity_curves_cumulative` — cum buckets are correct cumulative
    products
  - `test_build_equity_curves_long_short` — L/S = long_only + short_only
  - `test_max_drawdown_from_equity_basic` — known peak/trough
  - `test_max_drawdown_from_equity_empty` — returns NaN
  - `test_max_drawdown_from_equity_no_drawdown` — monotonic up returns 0
  - `test_portfolio_stats_long_short` — all expected keys present
  - `test_portfolio_stats_count_key` — uses `n_months` by default
  - `test_portfolio_stats_empty_equity` — returns `{}`
  - `test_portfolio_stats_missing_leg` — returns `{}`
  - `test_portfolio_stats_one_row_equity` — returns `{}`
  - `test_portfolio_stats_zero_vol_sharpe_nan` — zero volatility gives NaN
    Sharpe while preserving the other keys

### Layer 2: Golden/Snapshot Tests For Split Functions

For `compute_forward_returns`, `PortfolioSimulator.run`,
`compute_momentum_features`, and `run_all_robustness`:

- **Step 0 freezes real baselines** into the persistent baseline directory and
  writes the manifest described above.
- Each split is verified by comparing pre/post output with
  `pd.testing.assert_frame_equal` or equivalent.
- Golden comparisons must check row count, row order, index alignment, schemas,
  and dtypes where those are part of the public artifact contract.
- Any intentional correctness fix must have a named diff note in the verification
  log before the new output replaces the old baseline.
- No hand-written test fixtures — the baselines are the source of truth.

### Layer 3: CLI Main() Splits — Compile-Only

No unit tests. These extract code blocks from `main()` into helper
functions without logic changes. Verification is `python -m compileall`.

---

## Step 1 - Add Shared Modules (TDD)

See [a1_data_utils.md](a1_data_utils.md) and [a2_backtest_stats.md](a2_backtest_stats.md).

0. Confirm the test runner is available with `python -m pytest --version`. If
   it is missing, add/install `pytest` in the project environment before
   writing the red tests.
1. Write `tests/test_data_utils.py` per Layer 1 above. (Tests fail: red.)
2. Write `tests/test_backtest_stats.py` per Layer 1 above. (Tests fail: red.)
3. Create `data/_utils.py` implementing A1.1–A1.8. (Tests pass: green.)
4. Create `backtest/_stats.py` implementing A2.2–A2.8. (Tests pass: green.)
5. Run:

```bash
python -m pytest --version
python -m pytest tests/test_data_utils.py tests/test_backtest_stats.py -v
python -m compileall data/_utils.py backtest/_stats.py
```

## Step 2 - Data Consolidation

See [b1_data_loaders.md](b1_data_loaders.md). Apply B1 one file at a time:

1. `data/load_prices.py`
2. `data/load_shares.py`
3. `data/load_universes.py`
4. `data/load_signals.py`

Verification after each file:

```bash
python -m compileall <modified-file>
```

Then run synthetic manifest and failed-log checks:
- one success clears a previous failure.
- one share failure replaces the previous row for that ticker.
- one price failure appends while successes clear stale rows.
- same-ticker success/failure edge cases preserve the documented shares and
  price failed-log semantics.

## Step 3 - Backtest Consolidation

Apply [B2](b2_universe_filter.md), [B3](b3_stats_helpers.md), and
[B4](b4_ensure_forward_returns.md).

Verification:
- `python -m compileall backtest`
- Compare baseline forward returns with `pd.testing.assert_frame_equal`.
- Compare IC and quintile baseline outputs.
- Confirm summary columns still include `n_months`.

## Step 4 - Split Large Functions

Apply in this order:

1. [C5 forward returns](c5_forward_returns.md).
2. [C1 portfolio simulator](c1_portfolio_simulator.md).
3. [C2 momentum features](c2_momentum_features.md).
4. [C4 model walk-forward](c4_model_walk_forward.md).
5. [C3 robustness orchestrator](c3_robustness.md).
6. [C6 CLI main functions](c6_cli_main.md).
7. [C7 streaming audit](c7_streaming_audit.md).

Reasoning:
- C5 affects many callers and cache safety.
- C1 is the highest-risk behavior split, so do it before broad cleanup.
- C2/C4/C3/C6/C7 are mostly internal structure once shared helpers exist.

Verification after each split:
- `python -m compileall <modified-file>`
- Run the relevant baseline comparison.

## Step 5 - Final Verification

Required final checks:

```bash
python -m compileall data features backtest reports run_all.py
python run_all.py --dry-run --tier enhanced --from-phase 5 --stop-at-phase 5
```

Recommended smoke checks:
- `python -m features.audit --small-only`
- `python -m backtest.single_feature_ic --features results/features_enhanced.parquet --universe sp500`
- `python -m backtest.quintile --features results/features_enhanced.parquet --universe sp500`
- Portfolio fixture before/after equality on:
  - daily returns
  - weights
  - trade log
  - gap accounting
  - universe coverage

## Specific Regression Risks To Watch

- `ensure_forward_returns()` must not create duplicate return columns.
- Forward-return cache must invalidate after helper changes.
- Portfolio delayed entries must not contribute PnL before `actual_entry_date`.
- Portfolio `weights_history` must remain `dict[pd.Timestamp, pd.DataFrame]`.
- Long-gap records must remain keyed by `(gap_date, ticker)` through recovery.
- Stats summaries must keep `n_months` unless a caller explicitly asks for a
  different count key.
- `filter_to_universe()` tolerance must remain auto-detected by the shared
  universe module; do not hard-code 65 days for SP500.
- Any helper used by model fitting must preserve pandas date indices until
  after `monitor_fit_calls()` records fit ranges.
