"""Phase 1.3 - Daily adj_close + volume per ticker (yfinance).

Plan rules (`ideas/plan.md` 1.3):
- Collect tickers from universe x historical members -> unique set.
- yfinance keyed by `BESTTICKER`; pull adjusted close + volume.
- Rate limit and exponential backoff. Resumable manifest at
  `data/cache/prices/_manifest.json`.
- **Do not** rotate request headers to bypass rate limits.
- Maintain `data/cache/prices/failed_tickers.csv` (success/partial/empty).
- Output: `data/cache/prices/{ticker}.parquet` (date, adj_close, volume).

Performance notes:
- Batch download (up to 50 tickers per yfinance call) reduces HTTP round-trips
  by ~50x compared to one-at-a-time.
- Optional `--workers` flag parallelises batches across threads.
- `collect_tickers()` filters signal-side BESTTICKER to US-format symbols only
  (1-5 letters, optionally with a single dash/dot suffix like BRK-B).
  Purely numeric / Chinese / Japanese codes are discarded before touching the
  network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.config import (
    PRICE_CACHE_DIR,
    PRICE_FAILED_TICKERS,
    PRICE_MANIFEST,
    SIGNALS_PARQUET,
    SIGNALS_SLIM_PARQUET,
    UNIVERSE_CACHE_DIR,
    UNIVERSE_NAMES,
    ensure_dirs,
    set_global_seed,
)
from data.progress import progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("load_prices")

# yfinance logs "possibly delisted; no timezone found" at ERROR for every
# delisted or invalid ticker, even though our code handles those gracefully
# and returns FetchResult("empty", ...). Suppression is applied *after*
# the lazy import inside _download_batch() because yfinance re-configures
# its own handler on import.

DEFAULT_START = "2009-01-01"
BATCH_SIZE = 50
MAX_WORKERS = 4
BASE_SLEEP = 1.0
MAX_BACKOFF = 120.0
MAX_RETRIES = 3

# Ticker pattern: 1-5 uppercase letters, optional dot/dash + letter suffix
_US_TICKER_RE = re.compile(r"^[A-Z]{1,5}([\.\-][A-Z])?$")

_MANIFEST_LOCK = threading.Lock()
_FAILED_LOG_LOCK = threading.Lock()


@dataclass
class FetchResult:
    ticker: str
    status: str  # "success" | "empty" | "error"
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Ticker discovery
# ---------------------------------------------------------------------------

def collect_tickers(use_signals: bool = False) -> list[str]:
    """Union of universe historical members and optionally valid signal tickers.

    Signal-side tickers are filtered to US-format symbols (1-5 letters,
    optionally with a single dash/dot suffix). Purely numeric codes (Chinese
    A-shares, Japanese codes, etc.) are discarded before any network call.
    """
    tickers: set[str] = set()

    for u in UNIVERSE_NAMES:
        path = UNIVERSE_CACHE_DIR / f"{u}_pit.parquet"
        if not path.exists():
            log.warning("universe parquet missing: %s", path)
            continue
        df = pd.read_parquet(path, columns=["ticker"])
        tickers.update(df["ticker"].astype(str).str.upper().str.strip())

    if use_signals:
        sig_path = (
            SIGNALS_SLIM_PARQUET if SIGNALS_SLIM_PARQUET.exists() else SIGNALS_PARQUET
        )
        if sig_path.exists():
            df = pd.read_parquet(sig_path, columns=["BESTTICKER"])
            raw = df["BESTTICKER"].dropna().astype(str).str.upper().str.strip()
            valid_mask = raw.str.match(_US_TICKER_RE)
            n_total = len(raw)
            n_valid = valid_mask.sum()
            log.info(
                "signal tickers: %d total, %d US-format kept, %d skipped",
                n_total,
                n_valid,
                n_total - n_valid,
            )
            tickers.update(raw[valid_mask].unique())
        else:
            log.warning("signals parquet missing; skipping signal-side tickers")

    cleaned: set[str] = set()
    for t in tickers:
        t = t.strip().upper()
        if not t or t == "NAN":
            continue
        cleaned.add(t)
    return sorted(cleaned)


# ---------------------------------------------------------------------------
# Manifest (resumable)
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if PRICE_MANIFEST.exists():
        return json.loads(PRICE_MANIFEST.read_text())
    return {"updated_at": None, "tickers": {}}


def save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = dt.datetime.utcnow().isoformat() + "Z"
    PRICE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_LOCK:
        PRICE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def update_failed_log(results: list[FetchResult]) -> None:
    """Persist non-success rows; clear them when a ticker later succeeds."""
    columns = ["ticker", "status", "rows", "first_date", "last_date", "error"]
    PRICE_FAILED_TICKERS.parent.mkdir(parents=True, exist_ok=True)

    with _FAILED_LOG_LOCK:
        if PRICE_FAILED_TICKERS.exists():
            existing = pd.read_csv(PRICE_FAILED_TICKERS, dtype=str)
        else:
            existing = pd.DataFrame(columns=columns)

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
                [existing, pd.DataFrame(new_rows, columns=columns)],
                ignore_index=True,
            )

        existing.to_csv(PRICE_FAILED_TICKERS, index=False)


# ---------------------------------------------------------------------------
# yfinance batch download
# ---------------------------------------------------------------------------

def _parse_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extract (date, adj_close, volume) for one ticker from a download result.

    Handles both MultiIndex columns (batch download) and single-level columns
    (single-ticker or only one ticker returned).
    """
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(0)
        if "Close" not in levels:
            return None
        try:
            close = df["Close"][ticker]
            volume = df["Volume"][ticker]
        except KeyError:
            return None
    else:
        close = df["Close"]
        volume = df["Volume"]

    close = close.dropna()
    if close.empty:
        return None

    volume = volume.reindex(close.index)

    out = pd.DataFrame({
        "date": pd.to_datetime(close.index).date,
        "adj_close": close.astype("float64").values,
        "volume": volume.astype("float64").values,
    })
    out = out.dropna(subset=["adj_close"]).reset_index(drop=True)
    return out if len(out) > 0 else None


def _download_batch(
    tickers: list[str],
    start: str,
    end: str | None,
    base_sleep: float,
    max_retries: int,
) -> list[FetchResult]:
    """Download one batch of tickers from yfinance.

    Each batch is a single yfinance call. Individual tickers that don't exist
    or were delisted are returned as ``empty`` (not ``error``).
    ``error`` is reserved for network/API failures that survived all retries.
    """
    import yfinance as yf

    # yfinance re-configures its logger on import — silence it post-import.
    # Delisted/invalid tickers are handled gracefully by our parser; we don't
    # need yfinance screaming "possibly delisted; no timezone found" at ERROR.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)

    ticker_str = " ".join(tickers)
    last_err: str | None = None

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker_str,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            last_err = str(exc)
            backoff = min(MAX_BACKOFF, base_sleep * (2 ** attempt))
            log.warning(
                "batch [%s…] failed (%d/%d): %s — retrying in %.1fs",
                tickers[0],
                attempt + 1,
                max_retries,
                last_err[:120],
                backoff,
            )
            time.sleep(backoff)
            continue

        time.sleep(base_sleep + random.uniform(0, base_sleep))

        results: list[FetchResult] = []
        for t in tickers:
            sub = _parse_single_ticker(df, t)
            if sub is not None:
                out_path = PRICE_CACHE_DIR / f"{t}.parquet"
                sub.to_parquet(out_path, index=False)
                results.append(FetchResult(
                    t, "success", len(sub),
                    str(sub["date"].iloc[0]),
                    str(sub["date"].iloc[-1]),
                ))
            else:
                results.append(FetchResult(t, "empty", 0, None, None))
        return results

    # All retries exhausted
    log.error("batch [%s…] FAILED after %d retries: %s",
              tickers[0], max_retries, last_err or "unknown")
    return [
        FetchResult(t, "error", 0, None, None, error=last_err)
        for t in tickers
    ]


# ---------------------------------------------------------------------------
# Cached-freshness check
# ---------------------------------------------------------------------------

def is_cached_fresh(
    ticker: str,
    manifest: dict,
    end: str | None = None,
) -> bool:
    """True if the result is recent enough to skip re-fetching.

    - ``success``: parquet must exist and last_date within 5 days of end.
    - ``empty``: cached for 90 days (delisted/invalid tickers won't come back,
      so re-fetching every run is wasteful).
    - ``error``: never cached (transient failures should be retried).
    """
    entry = manifest.get("tickers", {}).get(ticker)
    if not entry:
        return False

    status = entry.get("status", "")
    target = dt.date.fromisoformat(end) if end else dt.date.today()

    if status == "success":
        path = PRICE_CACHE_DIR / f"{ticker}.parquet"
        if not path.exists():
            return False
        last_date = entry.get("last_date")
        if not last_date:
            return False
        last = dt.date.fromisoformat(last_date)
        return (target - last).days <= 5

    if status == "empty":
        fetched = entry.get("fetched_at")
        if not fetched:
            return False
        fetched_date = dt.date.fromisoformat(fetched[:10])
        return (target - fetched_date).days <= 90

    return False


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def run(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    sleep: float = BASE_SLEEP,
    workers: int = MAX_WORKERS,
    limit: int | None = None,
) -> dict:
    set_global_seed()
    ensure_dirs()
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = collect_tickers()
    if limit is not None:
        tickers = tickers[:limit]

    log.info("ticker universe: %d", len(tickers))

    manifest = load_manifest()

    # Split into fresh (skip) and stale (fetch)
    stale: list[str] = []
    counts = {"success": 0, "empty": 0, "error": 0, "skipped_cached": 0}

    for t in tickers:
        if is_cached_fresh(t, manifest, end=end):
            counts["skipped_cached"] += 1
        else:
            stale.append(t)

    log.info(
        "cached=%d  to_fetch=%d",
        counts["skipped_cached"],
        len(stale),
    )

    if not stale:
        log.info("nothing to fetch — all tickers are fresh")
        return counts

    # Build batches
    batches = [
        stale[i:i + BATCH_SIZE]
        for i in range(0, len(stale), BATCH_SIZE)
    ]
    log.info(
        "%d batches (size=%d, workers=%d)",
        len(batches),
        BATCH_SIZE,
        workers,
    )

    def _process_batch(batch: list[str]) -> list[FetchResult]:
        return _download_batch(batch, start, end, sleep, MAX_RETRIES)

    if workers <= 1:
        # Sequential — simpler logging, works with tqdm
        for completed, batch in enumerate(
            progress(batches, desc="price batches", unit="batch"),
            start=1,
        ):
            batch_results = _process_batch(batch)
            for r in batch_results:
                manifest.setdefault("tickers", {})[r.ticker] = {
                    "status": r.status,
                    "rows": r.rows,
                    "first_date": r.first_date,
                    "last_date": r.last_date,
                    "error": r.error,
                    "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
                }
                counts[r.status] = counts.get(r.status, 0) + 1
            update_failed_log(batch_results)
            if completed % 5 == 0:
                save_manifest(manifest)
    else:
        # Parallel — tqdm still works but we update per-future
        batch_bar = progress(
            total=len(batches), desc="price batches", unit="batch",
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_process_batch, b): i
                for i, b in enumerate(batches)
            }
            completed = 0
            for future in as_completed(future_map):
                batch_results = future.result()
                for r in batch_results:
                    manifest.setdefault("tickers", {})[r.ticker] = {
                        "status": r.status,
                        "rows": r.rows,
                        "first_date": r.first_date,
                        "last_date": r.last_date,
                        "error": r.error,
                        "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
                    }
                    counts[r.status] = counts.get(r.status, 0) + 1
                update_failed_log(batch_results)
                completed += 1
                batch_bar.update(1)
                batch_bar.set_postfix(counts)
                if completed % 10 == 0:
                    save_manifest(manifest)
        batch_bar.close()

    save_manifest(manifest)
    log.info("done. counts=%s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="yfinance prices loader")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sleep", type=float, default=BASE_SLEEP)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="disable batch parallelism (workers=1)",
    )
    parser.add_argument(
        "--include-signal-tickers", action="store_true",
        help="also fetch non-universe tickers that appear in the signals file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only fetch the first N tickers (smoke test)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="explicit ticker list, e.g. --only AAPL MSFT",
    )
    args = parser.parse_args()

    tickers = args.only
    if tickers is None and args.include_signal_tickers:
        tickers = collect_tickers(use_signals=True)

    run(
        tickers=tickers,
        start=args.start,
        end=args.end,
        sleep=args.sleep,
        workers=1 if args.no_parallel else args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
