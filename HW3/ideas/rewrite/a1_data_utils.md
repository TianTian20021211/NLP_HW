# A1. `data/_utils.py`

Part of [Part A — New Shared Modules](main.md#module-index). The cross-cutting
rules in [main.md](main.md) (Non-Negotiable Rules, Keep As-Is, Fixes To The
Previous Draft, Testing Strategy, Specific Regression Risks To Watch) apply to
this plan.

Create `data/_utils.py` as a leaf utility module for data loaders.

## A1.1 Imports

```python
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import pandas as pd
```

## A1.2 `FetchResult`

```python
@dataclass
class FetchResult:
    ticker: str
    status: str
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None = None
```

Parameters: dataclass fields only.

Return: constructor returns a `FetchResult`.

Rules:
- `status` vocabulary stays `success | empty | error`.
- Replace identical dataclasses in `data/load_prices.py` and
  `data/load_shares.py`.

## A1.3 `US_TICKER_RE`

```python
US_TICKER_RE: Pattern[str] = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")
```

Return: compiled regex constant.

Caller changes:
- `data/load_prices.py::collect_tickers(use_signals: bool = False) -> list[str]`
  imports `US_TICKER_RE` and keeps the existing `.str.match(...)` behavior.

## A1.4 `setup_logger`

```python
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure process logging and return a named logger."""
```

Parameters:
- `name`: logger name.
- `level`: root logging level.

Return: `logging.Logger`.

Caller changes:
- `data/load_prices.py`: `log = setup_logger("load_prices")`
- `data/load_shares.py`: `log = setup_logger("load_shares")`
- `data/load_universes.py`: `log = setup_logger("load_universes")`
- `data/load_signals.py`: `log = setup_logger("load_signals")`

Do not change `backtest.*` logging in this step.

## A1.5 `suppress_yfinance_logging`

```python
def suppress_yfinance_logging() -> None:
    """Silence yfinance and peewee after lazy yfinance import."""
```

Parameters: none.

Return: `None`.

Caller changes:
- In `data/load_prices.py::_download_batch(tickers: list[str], start: str, end: str | None, base_sleep: float, max_retries: int) -> list[FetchResult]`, call immediately after
  `import yfinance as yf`.
- In `data/load_shares.py::_yf_shares(ticker: str, start: str, end: str | None) -> pd.DataFrame`, call immediately after
  `import yfinance as yf`.

## A1.6 Manifest Helpers

```python
def load_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest or return {'updated_at': None, 'tickers': {}}."""
```

Parameters:
- `path`: manifest JSON path.

Return: manifest dictionary.

```python
def save_manifest(
    manifest: dict[str, Any],
    path: Path,
    lock: threading.Lock | None = None,
) -> None:
    """Stamp updated_at and write manifest JSON."""
```

Parameters:
- `manifest`: mutable manifest dictionary.
- `path`: destination JSON path.
- `lock`: optional lock for thread-safe writes.

Return: `None`.

Caller changes:
- `data/load_prices.py`:
  - `load_manifest()` becomes `load_manifest(PRICE_MANIFEST)`.
  - `save_manifest(manifest)` becomes
    `save_manifest(manifest, PRICE_MANIFEST, lock=_MANIFEST_LOCK)`.
- `data/load_shares.py`:
  - `load_manifest()` becomes `load_manifest(SHARES_MANIFEST)`.
  - `save_manifest(manifest)` becomes
    `save_manifest(manifest, SHARES_MANIFEST)`.

## A1.7 Failed-Log Helpers

```python
FAILED_LOG_COLUMNS: list[str] = [
    "ticker", "status", "rows", "first_date", "last_date", "error",
]
```

```python
def load_failed_log(path: Path) -> pd.DataFrame:
    """Read a failed-ticker CSV, or return an empty standard-schema frame."""
```

Parameters:
- `path`: CSV path.

Return: `pd.DataFrame` with `FAILED_LOG_COLUMNS`.

```python
def save_failed_log(df: pd.DataFrame, path: Path) -> None:
    """Write the failed-ticker DataFrame to CSV."""
```

Parameters:
- `df`: failed-log DataFrame.
- `path`: CSV path.

Return: `None`.

```python
def apply_failed_log_result(
    existing: pd.DataFrame,
    result: FetchResult,
) -> pd.DataFrame:
    """Apply one fetch result; shares semantics: one latest row per ticker."""
```

Parameters:
- `existing`: current failed-log DataFrame.
- `result`: one fetch result.

Return: new DataFrame. Previous rows for `result.ticker` are removed; a
`success` clears the ticker; `empty` or `error` appends one latest row.

```python
def apply_failed_log_results(
    existing: pd.DataFrame,
    results: list[FetchResult],
) -> pd.DataFrame:
    """Apply a batch of fetch results; price semantics: append batch misses, clear successes."""
```

Parameters:
- `existing`: current failed-log DataFrame.
- `results`: batch results.

Return: new DataFrame. Successful tickers are cleared. New non-success rows are
appended without deduping repeated failures, preserving current price-loader
semantics.

```python
def update_failed_log_result(
    path: Path,
    result: FetchResult,
    lock: threading.Lock | None = None,
) -> None:
    """Locked read-apply-write for one result."""
```

Parameters:
- `path`: CSV path.
- `result`: one fetch result.
- `lock`: optional lock guarding the full read/modify/write sequence.

Return: `None`.

```python
def update_failed_log_results(
    path: Path,
    results: list[FetchResult],
    lock: threading.Lock | None = None,
) -> None:
    """Locked read-apply-write for a batch of results."""
```

Parameters:
- `path`: CSV path.
- `results`: batch results.
- `lock`: optional lock guarding the full read/modify/write sequence.

Return: `None`.

Caller changes:
- `data/load_prices.py::update_failed_log(results)` can either be deleted and
  replaced at call sites with:
  `update_failed_log_results(PRICE_FAILED_TICKERS, results, _FAILED_LOG_LOCK)`,
  or kept as a thin wrapper around that call.
- `data/load_shares.py` should remove local `load_failed_log`,
  `save_failed_log`, and `apply_failed_log_result`. In the run loop it may keep
  the current in-memory `failed_log` batching pattern:
  `failed_log = apply_failed_log_result(failed_log, res)` followed by
  `save_failed_log(failed_log, SHARES_FAILED_TICKERS)`.

## A1.8 `exponential_backoff`

```python
def exponential_backoff(attempt: int, base: float, max_val: float) -> float:
    """Return min(max_val, base * 2**attempt)."""
```

Parameters:
- `attempt`: zero-based retry attempt.
- `base`: base sleep seconds.
- `max_val`: maximum sleep seconds.

Return: `float`.

Caller changes:
- `data/load_prices.py::_download_batch(tickers: list[str], start: str, end: str | None, base_sleep: float, max_retries: int) -> list[FetchResult]`
- `data/load_shares.py::fetch_ticker(ticker: str, start: str = DEFAULT_START, end: str | None = None, base_sleep: float = BASE_SLEEP, max_retries: int = MAX_RETRIES) -> FetchResult`
