# B1. Data Loader Consolidation

Part of [Part B — Behavior-Preserving Consolidation](main.md#module-index). The
cross-cutting rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is,
Fixes To The Previous Draft, Testing Strategy, Specific Regression Risks To
Watch) apply to this plan.

## `data/load_prices.py`

Imports to add:

```python
from data._utils import (
    FetchResult,
    US_TICKER_RE,
    exponential_backoff,
    load_manifest,
    save_manifest,
    setup_logger,
    suppress_yfinance_logging,
    update_failed_log_results,
)
```

Remove local imports no longer needed: `json`, `re`, `dataclass`.

Specific changes:
- Delete local `FetchResult`.
- Delete local `_US_TICKER_RE`; use `US_TICKER_RE`.
- Replace logging boilerplate with `log = setup_logger("load_prices")`.
- Replace local `load_manifest()` and `save_manifest()`.
- Replace local `update_failed_log(results)` or keep as:

```python
def update_failed_log(results: list[FetchResult]) -> None:
    update_failed_log_results(PRICE_FAILED_TICKERS, results, _FAILED_LOG_LOCK)
```

- Replace yfinance logger suppression with `suppress_yfinance_logging()`.
- Replace `min(MAX_BACKOFF, base_sleep * (2 ** attempt))` with
  `exponential_backoff(attempt, base_sleep, MAX_BACKOFF)`.

## `data/load_shares.py`

Imports to add:

```python
from data._utils import (
    FetchResult,
    apply_failed_log_result,
    exponential_backoff,
    load_failed_log,
    load_manifest,
    save_failed_log,
    save_manifest,
    setup_logger,
    suppress_yfinance_logging,
)
```

Remove local imports no longer needed: `json`, `dataclass`.

Specific changes:
- Delete local `FetchResult`.
- Delete local manifest helpers.
- Delete local failed-log helper definitions.
- Replace logging boilerplate with `log = setup_logger("load_shares")`.
- In `_yf_shares`, call `suppress_yfinance_logging()` after importing
  yfinance.
- Preserve the current in-memory `failed_log` batching in `run()`.

## `data/load_universes.py`

Imports to add:

```python
from data._utils import setup_logger
```

Specific changes:
- Replace logging boilerplate with `log = setup_logger("load_universes")`.

## `data/load_signals.py`

Imports to add:

```python
from data._utils import setup_logger
```

Specific changes:
- Replace logging boilerplate with `log = setup_logger("load_signals")`.
