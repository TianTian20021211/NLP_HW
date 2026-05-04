# C1. Split `backtest/portfolio.py::PortfolioSimulator.run`

Part of [Part C — Split Large Functions](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

Current risk: `run()` mixes input validation, PIT filtering, price loading,
rebalance construction, quote lookup, daily PnL, long-gap audit, and result
assembly. Split it, but keep all existing semantics.

## C1.1 `SimulationState`

Add near `PortfolioResult`:

```python
@dataclass
class SimulationState:
    current_weights: dict[str, float] = field(default_factory=dict)
    prev_weights: dict[str, float] = field(default_factory=dict)
    current_entry_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    today_turnover: float = 0.0
    daily_records: list[dict[str, Any]] = field(default_factory=list)
    trade_records: list[dict[str, Any]] = field(default_factory=list)
    long_gap_records: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    weights_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    cohort_weights_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
```

Return: dataclass constructor returns `SimulationState`.

## C1.2 `_empty_result`

```python
def _empty_result(
    universe_coverage: pd.DataFrame | None = None,
    trade_records: list[dict[str, Any]] | None = None,
    trade_log_violations: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Build a standard empty PortfolioResult with stable schemas."""
```

Parameters:
- `universe_coverage`: optional coverage DataFrame.
- `trade_records`: optional already collected trade records.
- `trade_log_violations`: optional validation result.

Return: `PortfolioResult`.

## C1.3 `_validate_and_filter_signals`

```python
def _validate_and_filter_signals(
    self,
    signals: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    """Normalize signal columns, date bounds, and PIT universe membership."""
```

Parameters:
- `signals`: DataFrame with `[date, ticker, score]`.
- `start`: optional simulation start.
- `end`: optional simulation end.

Return:
- filtered `signals`
- normalized `start`
- normalized `end`
- `n_before` row count before PIT filter

Rules:
- Preserve current `date` coercion and `ticker` string conversion.
- Call `filter_to_universe(signals, self.universe_name, date_col="date", ticker_col="ticker")`.
- Keep only `_in_universe`.

## C1.4 `_collect_price_tickers`

```python
def _collect_price_tickers(
    self,
    signals: pd.DataFrame,
) -> tuple[set[str], set[str], set[str]]:
    """Collect traded tickers, coverage tickers, and union price tickers."""
```

Parameters:
- `signals`: PIT-filtered signals.

Return:
- `tickers`: tickers appearing in signals.
- `coverage_tickers`: all historical PIT members for coverage audit.
- `price_tickers`: union of the two.

## C1.5 `_load_calendar_prices_and_coverage`

```python
def _load_calendar_prices_and_coverage(
    self,
    price_tickers: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int,
    cadence: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, int], pd.DatetimeIndex, pd.DataFrame]:
    """Load prices, close matrix, trading calendar, rebalance dates, and coverage."""
```

Parameters:
- `price_tickers`: union of traded and coverage tickers.
- `start`, `end`: simulation bounds.
- `lookback`: rebalance lookback length in trading days.
- `cadence`: `daily | weekly | monthly`.

Return:
- `close_matrix`
- `calendar`
- `cal_pos_by_date`
- `rebal_dates`
- `universe_coverage`

Rules:
- Keep current 30-calendar-day post-end buffer.
- Set `self._price_table` and `self._calendar` as today.

## C1.6 `_rebalance_lookback_start`

```python
def _rebalance_lookback_start(
    calendar: pd.DatetimeIndex,
    cal_pos_by_date: dict[pd.Timestamp, int],
    date: pd.Timestamp,
    lookback: int,
) -> pd.Timestamp:
    """Return the first eligible signal date for a rebalance date."""
```

Parameters:
- `calendar`: trading calendar.
- `cal_pos_by_date`: date-to-position map.
- `date`: rebalance date.
- `lookback`: trailing trading-day lookback.

Return: `pd.Timestamp`.

## C1.7 `_lookup_tradeable_entries`

```python
def _lookup_tradeable_entries(
    self,
    agg_weights_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
    planned_entry_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], dict[str, float], list[dict[str, Any]]]:
    """Find entry quotes, remove untradeable names, and rescale gross to 2.0."""
```

Parameters:
- `agg_weights_df`: `[ticker, raw_weight]` from `_build_cohort_weights`.
- `close_matrix`: wide price matrix.
- `calendar`: list of trading dates.
- `cal_pos_by_date`: date-to-position map.
- `planned_entry_date`: rebalance date.

Return:
- tradeable/rescaled `agg_weights_df`
- `entry_dates`
- `entry_prices`
- `skip_trade_records`

Rules:
- Use `_next_valid_quote(close_matrix, ticker, planned_entry_date, calendar, cal_pos_by_date, max_forward_days=3)`.
- For missing entry, append `skip_no_entry_quote`.
- Rescale remaining gross exposure to `2.0`.

## C1.8 `_append_entered_trade_records`

```python
def _append_entered_trade_records(
    self,
    state: SimulationState,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
    rebal_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
    lookback: int,
    entry_dates: dict[str, pd.Timestamp],
    entry_prices: dict[str, float],
) -> None:
    """Append trade records for successfully entered positions."""
```

Parameters: current simulation state, price context, rebalance context, entry
date/price maps.

Return: `None`; mutates `state.trade_records`.

Rules:
- Planned exit is next rebalance date.
- Exit quote lookup uses `_next_valid_quote(close_matrix, ticker, planned_exit, calendar, cal_pos_by_date, max_forward_days=3)`.
- Missing exit gets `right_censored_no_exit_quote`.
- Normal trades use `skip_reason = None`.

## C1.9 `_process_rebalance`

```python
def _process_rebalance(
    self,
    state: SimulationState,
    signals: pd.DataFrame,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
    rebal_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
    lookback: int,
    long_frac: float,
) -> None:
    """Build weights for one rebalance date and update state."""
```

Return: `None`; mutates:
- `state.current_weights`
- `state.current_entry_dates`
- `state.prev_weights`
- `state.today_turnover`
- `state.weights_history`
- `state.cohort_weights_history`
- `state.trade_records`

## C1.10 `_compute_daily_pnl`

```python
def _compute_daily_pnl(
    self,
    state: SimulationState,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    did_rebalance: bool,
) -> None:
    """Compute one daily PnL row with delayed-entry and gap accounting."""
```

Parameters: current state, price context, current and next trading date, and
rebalance flag.

Return: `None`; appends one row to `state.daily_records`.

Rules:
- Active weights include only names with `actual_entry_date <= date`.
- Valid next-day quote contributes normal return.
- One- and two-day forward-fill gaps contribute zero daily return and are
  accounted in weight/count columns.
- Longer gaps are added to `state.long_gap_records`.

## C1.11 `_apply_long_gap_recovery`

```python
def _apply_long_gap_recovery(
    self,
    daily_df: pd.DataFrame,
    state: SimulationState,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 30-trading-day long-gap recovery audit and update daily_df."""
```

Return:
- updated `daily_df`
- `gap_records_df`

## C1.12 `_post_process_result`

```python
def _post_process_result(
    self,
    state: SimulationState,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
    universe_coverage: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cadence: str,
    lookback: int,
    long_frac: float,
    transaction_cost_bps: float,
) -> PortfolioResult:
    """Validate trade log, apply gap recovery, and return PortfolioResult."""
```

Return: `PortfolioResult`.

Rules:
- Always call `validate_trade_log()` when the trade log is non-empty.
- Preserve generic config keys exactly: `cadence`, `lookback`, `long_frac`,
  `transaction_cost_bps`, `start`, `end`, `universe`.

## C1.13 New `run()` Shape

```python
def run(
    self,
    signals: pd.DataFrame,
    cadence: str = "weekly",
    lookback: int = 5,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    long_frac: float = 0.2,
    transaction_cost_bps: float = 5.0,
) -> PortfolioResult:
    signals, start, end, n_before = self._validate_and_filter_signals(
        signals, start, end
    )
    if signals.empty:
        return _empty_result()

    tickers, coverage_tickers, price_tickers = self._collect_price_tickers(signals)
    close_matrix, calendar, cal_pos_by_date, rebal_dates, universe_coverage = (
        self._load_calendar_prices_and_coverage(
            price_tickers, start, end, lookback, cadence
        )
    )

    state = SimulationState()
    cal_list = list(calendar)
    rebal_set = set(rebal_dates)
    for i in progress(range(len(cal_list) - 1), desc=f"Simulating ({cadence})", unit="day"):
        date = cal_list[i]
        next_date = cal_list[i + 1]
        if date < start:
            continue
        if date > end:
            break
        state.today_turnover = 0.0
        did_rebalance = date in rebal_set
        if did_rebalance:
            self._process_rebalance(
                state, signals, close_matrix, cal_list, cal_pos_by_date,
                rebal_dates, date, lookback, long_frac
            )
        self._compute_daily_pnl(
            state, close_matrix, cal_list, cal_pos_by_date,
            date, next_date, did_rebalance
        )

    return self._post_process_result(
        state, close_matrix, cal_list, cal_pos_by_date, universe_coverage,
        start, end, cadence, lookback, long_frac, transaction_cost_bps
    )
```
