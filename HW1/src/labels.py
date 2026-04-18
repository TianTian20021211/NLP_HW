"""Forward returns for each ``(ticker, call_date)`` call — no look-ahead.

Part II §2.4 spec:

* ``T`` = first trading day on/after ``call_date`` in the ticker's prices.
* ``entry_date`` = trading day **after** ``T`` (PDF: call happens at
  ``T`` close, earliest tradable point is ``T+1``).
* ``exit_k_date`` = ``entry_date`` shifted forward by ``k`` trading days
  for ``k ∈ {1, 5, 21, 63}``. ``raw_k = Close_{exit_k} / Close_{entry}`` - 1.
* ``spy_k`` uses the ticker's ``entry_date`` / ``exit_k_date`` directly on
  the SPY series — those are already confirmed ticker-trading-days, and
  SPY missing a Close on either day drops that horizon only.
* ``excess_k = raw_k - spy_k``. NaNs happen per-horizon (boundary or SPY
  miss); a single row can have valid +5d but NaN +21d.

Only Close is available, so entry is Close-to-Close — intraday / open
prices are explicitly out of scope (Part II §2.11).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .aggregate import load_call_record
from .cache_keys import stage_key, write_sidecar
from .io_paths import ensure_dirs, error, labels_path
from .prices import BENCHMARK_TICKER, load_prices


HORIZONS = (1, 5, 21, 63)


def compute_labels(
    calls: Sequence[tuple[str, str]],
    benchmark: str = BENCHMARK_TICKER,
    write: bool = True,
) -> pd.DataFrame:
    """Build the labels parquet from call records + cached prices.

    Returns a DataFrame keyed ``(ticker, call_date)`` with columns
    ``entry_date`` + ``raw_{k}d`` / ``excess_{k}d`` for k in
    :data:`HORIZONS`. Rows with a missing ``call_date`` are dropped and
    not emitted (Part II §2.4). Per-horizon NaN is allowed and recorded
    in the QC ``label_nan_rate[horizon]`` counter.
    """
    bench_df = load_prices(benchmark)
    prices_cache: dict[str, pd.DataFrame] = {}

    rows: list[dict] = []
    for ticker, quarter in calls:
        rec = load_call_record(ticker, quarter, variant="finbert")
        call_date = rec.get("call_date")
        if call_date is None:
            continue
        if ticker not in prices_cache:
            prices_cache[ticker] = load_prices(ticker)
        row = _label_row(ticker, call_date, prices_cache[ticker], bench_df)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        error("compute_labels: produced empty frame; every call had call_date=None?")
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    if df.duplicated(["ticker", "call_date"]).any():
        error("labels frame has duplicate (ticker, call_date) keys")
    if write:
        ensure_dirs()
        path = labels_path()
        df.to_parquet(path, index=False)
        write_sidecar(path, "labels")
    return df


def _label_row(ticker: str, call_date: str, stock: pd.DataFrame, bench: pd.DataFrame) -> dict:
    """Compute one row of labels for a given ticker/call."""
    call_ts = pd.to_datetime(call_date)
    stock_dates = stock["Date"]
    idx_T = stock_dates.searchsorted(call_ts, side="left")
    if idx_T >= len(stock_dates):
        return _nan_row(ticker, call_date, entry_date=None)

    entry_idx = idx_T + 1
    if entry_idx >= len(stock_dates):
        return _nan_row(ticker, call_date, entry_date=None)

    entry_date = stock_dates.iloc[entry_idx]
    entry_close = float(stock["Close"].iloc[entry_idx])
    if not entry_close or entry_close != entry_close:
        return _nan_row(ticker, call_date, entry_date=entry_date.date().isoformat())

    spy_entry = _spy_close_on(bench, entry_date)

    row: dict = {
        "ticker": ticker,
        "call_date": call_date,
        "entry_date": entry_date.date().isoformat(),
    }
    for k in HORIZONS:
        exit_idx = entry_idx + k
        if exit_idx >= len(stock_dates):
            row[f"raw_{k}d"] = float("nan")
            row[f"excess_{k}d"] = float("nan")
            continue
        exit_close = float(stock["Close"].iloc[exit_idx])
        exit_date = stock_dates.iloc[exit_idx]
        if not exit_close or exit_close != exit_close:
            row[f"raw_{k}d"] = float("nan")
            row[f"excess_{k}d"] = float("nan")
            continue
        raw_k = exit_close / entry_close - 1.0
        row[f"raw_{k}d"] = raw_k
        spy_exit = _spy_close_on(bench, exit_date)
        if spy_entry is None or spy_exit is None:
            row[f"excess_{k}d"] = float("nan")
        else:
            row[f"excess_{k}d"] = raw_k - (spy_exit / spy_entry - 1.0)
    return row


def _spy_close_on(bench: pd.DataFrame, day: pd.Timestamp) -> float | None:
    """Return SPY Close on the exact ``day`` or ``None`` if not a SPY trading day."""
    hits = bench.loc[bench["Date"] == day, "Close"]
    if hits.empty:
        return None
    v = float(hits.iloc[0])
    if not v or v != v:
        return None
    return v


def _nan_row(ticker: str, call_date: str, entry_date: str | None) -> dict:
    row: dict = {"ticker": ticker, "call_date": call_date, "entry_date": entry_date}
    for k in HORIZONS:
        row[f"raw_{k}d"] = float("nan")
        row[f"excess_{k}d"] = float("nan")
    return row


def label_nan_rate(df: pd.DataFrame) -> dict[str, float]:
    """Per-horizon fraction of rows whose ``excess_{k}d`` is NaN.

    Uses ``excess`` rather than ``raw`` because a valid ``raw_k`` but NaN
    ``excess_k`` (SPY-miss case) is still a drop from the backtest's
    perspective.
    """
    out: dict[str, float] = {}
    n = len(df)
    if n == 0:
        return {f"{k}d": 0.0 for k in HORIZONS}
    for k in HORIZONS:
        col = f"excess_{k}d"
        out[f"{k}d"] = float(df[col].isna().mean())
    return out
