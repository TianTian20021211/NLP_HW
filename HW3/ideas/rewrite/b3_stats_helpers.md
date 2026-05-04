# B3. Replace Duplicated Stats Helpers

Part of [Part B — Behavior-Preserving Consolidation](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Use `backtest._stats` in:

- `backtest/model.py`: `make_median_imputer`, `spearman`
- `backtest/splits.py`: `make_median_imputer`, `spearman`
- `backtest/quintile.py`: `bucket_returns`, `build_equity_curves`,
  `portfolio_stats`
- `backtest/robustness.py`: `dedup_latest_per_ticker`, `bucket_returns`,
  `build_equity_curves`, `portfolio_stats`, `max_drawdown_from_equity`
- `backtest/single_feature_ic.py`: `dedup_latest_per_ticker`

Do not replace:
- `backtest/robustness.py::score_weighted_bucket_returns(df: pd.DataFrame, feature_col: str, return_col: str, n_buckets: int = 5) -> pd.DataFrame`
- `backtest/robustness.py::_top_n_bucket_returns(df: pd.DataFrame, feature_col: str, return_col: str, n_stocks: int) -> pd.DataFrame`
- `backtest/robustness.py::_build_equity_curves_2leg(monthly: pd.DataFrame) -> pd.DataFrame`

Those are different algorithms.
