# B2. Remove Redundant Universe Filter Wrappers

Part of [Part B — Behavior-Preserving Consolidation](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Delete these wrappers:

```python
def _filter_to_universe(
    df: pd.DataFrame,
    universe_name: str,
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    raise NotImplementedError
```

Replace call sites with:

```python
from backtest.universe import filter_to_universe

df = filter_to_universe(df, universe_name)
```

This is safe because `backtest.universe.filter_to_universe` already defaults to
`date_col="call_entry_date"` and `ticker_col="BESTTICKER"`.

Also change `backtest/robustness.py::run_all_robustness(features_path: Path, universe_name: str = "sp500", price_cache_dir: Path = PRICE_CACHE_DIR, shares_cache_dir: Path = SHARES_CACHE_DIR, output_dir: Path | None = None, features: list[str] | None = None, skip_mcap: bool = False) -> RobustnessResult` from:

```python
from backtest.quintile import _filter_to_universe
```

to:

```python
from backtest.universe import filter_to_universe
```
