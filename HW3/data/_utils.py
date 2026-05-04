"""Leaf utility module for data loaders.

This is a LEAF module — it must NOT import from ``data.load_*`` or any
other project module.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string with Z suffix."""
    try:
        return dt.datetime.now(dt.UTC).isoformat() + "Z"
    except AttributeError:
        return dt.datetime.utcnow().isoformat() + "Z"  # pragma: no cover - py<3.12


# ---------------------------------------------------------------------------
# A1.2 FetchResult
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    ticker: str
    status: str  # "success" | "empty" | "error"
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# A1.3 US_TICKER_RE
# ---------------------------------------------------------------------------

US_TICKER_RE: Pattern[str] = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")


# ---------------------------------------------------------------------------
# A1.4 setup_logger
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure process logging and return a named logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# A1.5 suppress_yfinance_logging
# ---------------------------------------------------------------------------

def suppress_yfinance_logging() -> None:
    """Silence yfinance and peewee after lazy yfinance import."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# A1.6 Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest or return {'updated_at': None, 'tickers': {}}."""
    if path.exists():
        return json.loads(path.read_text())
    return {"updated_at": None, "tickers": {}}


def save_manifest(
    manifest: dict[str, Any],
    path: Path,
    lock: threading.Lock | None = None,
) -> None:
    """Stamp updated_at and write manifest JSON atomically.

    Uses a temporary file + Path.replace() to avoid corrupt manifests
    if the process crashes mid-write.
    """
    manifest["updated_at"] = _utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock:
        lock.acquire()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write(json.dumps(manifest, indent=2, sort_keys=True))
            tmp_path = Path(tf.name)
        tmp_path.replace(path)
    except BaseException:
        if "tmp_path" in dir():
            tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if lock:
            lock.release()


# ---------------------------------------------------------------------------
# A1.7 Failed-log helpers
# ---------------------------------------------------------------------------

FAILED_LOG_COLUMNS: list[str] = [
    "ticker", "status", "rows", "first_date", "last_date", "error",
]


def load_failed_log(path: Path) -> pd.DataFrame:
    """Read a failed-ticker CSV, or return an empty standard-schema frame."""
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=FAILED_LOG_COLUMNS)


def save_failed_log(df: pd.DataFrame, path: Path) -> None:
    """Write the failed-ticker DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def apply_failed_log_result(
    existing: pd.DataFrame,
    result: FetchResult,
) -> pd.DataFrame:
    """Apply one fetch result; shares semantics: one latest row per ticker."""
    existing = existing[existing["ticker"] != result.ticker].copy()
    if result.status != "success":
        row = pd.DataFrame([{
            "ticker": result.ticker,
            "status": result.status,
            "rows": str(result.rows),
            "first_date": result.first_date or "",
            "last_date": result.last_date or "",
            "error": result.error or "",
        }])
        existing = pd.concat([existing, row], ignore_index=True)
    return existing


def apply_failed_log_results(
    existing: pd.DataFrame,
    results: list[FetchResult],
) -> pd.DataFrame:
    """Apply a batch of fetch results; price semantics: append batch misses, clear successes."""
    succeeded = {r.ticker for r in results if r.status == "success"}
    if succeeded:
        existing = existing[~existing["ticker"].isin(succeeded)]

    new_rows = [
        {
            "ticker": r.ticker,
            "status": r.status,
            "rows": str(r.rows),
            "first_date": r.first_date or "",
            "last_date": r.last_date or "",
            "error": r.error or "",
        }
        for r in results
        if r.status != "success"
    ]
    if new_rows:
        existing = pd.concat(
            [existing, pd.DataFrame(new_rows, columns=FAILED_LOG_COLUMNS)],
            ignore_index=True,
        )
    return existing


def update_failed_log_result(
    path: Path,
    result: FetchResult,
    lock: threading.Lock | None = None,
) -> None:
    """Locked read-apply-write for one result."""
    if lock:
        lock.acquire()
    try:
        existing = load_failed_log(path)
        new = apply_failed_log_result(existing, result)
        save_failed_log(new, path)
    finally:
        if lock:
            lock.release()


def update_failed_log_results(
    path: Path,
    results: list[FetchResult],
    lock: threading.Lock | None = None,
) -> None:
    """Locked read-apply-write for a batch of results."""
    if lock:
        lock.acquire()
    try:
        existing = load_failed_log(path)
        new = apply_failed_log_results(existing, results)
        save_failed_log(new, path)
    finally:
        if lock:
            lock.release()


# ---------------------------------------------------------------------------
# A1.8 exponential_backoff
# ---------------------------------------------------------------------------

def exponential_backoff(attempt: int, base: float, max_val: float) -> float:
    """Return min(max_val, jittered_base * 2**attempt) with random jitter."""
    jittered_base = base + random.uniform(0, base * 0.5)
    return min(max_val, jittered_base * 2 ** attempt)
