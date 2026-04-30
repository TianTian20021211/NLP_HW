"""Phase 1.4 - Historical shares outstanding (market-cap buckets only).

Plan rules (`ideas/plan.md` 1.4):
- Prefer Yahoo historical shares outstanding (`Ticker.get_shares_full`).
- Mark missing series instead of imputing; coverage logged in
  `results/audit/marketcap_capacity_coverage.csv`.
- If coverage on any (universe, year) is below 70%, downgrade market-cap
  buckets to qualitative robustness only and flag in red in the PDF.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.config import (
    AUDIT_DIR,
    SHARES_CACHE_DIR,
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
log = logging.getLogger("load_shares")


SHARES_MANIFEST = SHARES_CACHE_DIR / "_manifest.json"
SHARES_FAILED_TICKERS = SHARES_CACHE_DIR / "failed_tickers.csv"
COVERAGE_CSV = AUDIT_DIR / "marketcap_capacity_coverage.csv"

DEFAULT_START = "2009-01-01"
BASE_SLEEP = 0.6
MAX_BACKOFF = 60.0
MAX_RETRIES = 4
COVERAGE_FLOOR = 0.70


@dataclass
class FetchResult:
    ticker: str
    status: str  # "success" | "empty" | "error"
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if SHARES_MANIFEST.exists():
        return json.loads(SHARES_MANIFEST.read_text())
    return {"updated_at": None, "tickers": {}}


def save_manifest(manifest: dict) -> None:
    from data.config import utc_now_iso
    manifest["updated_at"] = utc_now_iso()
    SHARES_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SHARES_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def update_failed_log(result: FetchResult) -> None:
    """Keep one latest non-success row per ticker; clear stale rows on success."""
    columns = ["ticker", "status", "rows", "first_date", "last_date", "error"]
    SHARES_FAILED_TICKERS.parent.mkdir(parents=True, exist_ok=True)
    if SHARES_FAILED_TICKERS.exists():
        existing = pd.read_csv(SHARES_FAILED_TICKERS, dtype=str)
    else:
        existing = pd.DataFrame(columns=columns)

    existing = apply_failed_log_result(existing, result)
    existing.to_csv(SHARES_FAILED_TICKERS, index=False)


def load_failed_log() -> pd.DataFrame:
    columns = ["ticker", "status", "rows", "first_date", "last_date", "error"]
    if SHARES_FAILED_TICKERS.exists():
        return pd.read_csv(SHARES_FAILED_TICKERS, dtype=str)
    return pd.DataFrame(columns=columns)


def save_failed_log(existing: pd.DataFrame) -> None:
    SHARES_FAILED_TICKERS.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(SHARES_FAILED_TICKERS, index=False)


def apply_failed_log_result(existing: pd.DataFrame, result: FetchResult) -> pd.DataFrame:
    """Apply one fetch result to an in-memory failed ticker log."""
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


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def _yf_shares(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch via `Ticker.get_shares_full`. Returns (date, shares) frame."""
    import yfinance as yf

    yt = yf.Ticker(ticker)
    series = None
    if hasattr(yt, "get_shares_full"):
        try:
            series = yt.get_shares_full(start=start, end=end)
        except Exception as e:
            log.debug("get_shares_full(%s) raised: %s", ticker, e)

    if series is None or len(series) == 0:
        return pd.DataFrame()

    if isinstance(series, pd.DataFrame):
        col = "Shares" if "Shares" in series.columns else series.columns[0]
        series = series[col]

    df = pd.DataFrame({
        "date": pd.to_datetime(series.index).date,
        "shares": series.astype("float64").values,
    })
    df = df.dropna(subset=["shares"])
    df = df[df["shares"] > 0]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    base_sleep: float = BASE_SLEEP,
    max_retries: int = MAX_RETRIES,
) -> FetchResult:
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            df = _yf_shares(ticker, start=start, end=end)
            time.sleep(base_sleep + random.uniform(0, base_sleep))
            if df.empty:
                return FetchResult(ticker, "empty", 0, None, None)
            out = SHARES_CACHE_DIR / f"{ticker}.parquet"
            df.to_parquet(out, index=False)
            return FetchResult(
                ticker,
                "success",
                len(df),
                str(df["date"].iloc[0]),
                str(df["date"].iloc[-1]),
            )
        except Exception as e:
            last_err = repr(e)
            backoff = min(MAX_BACKOFF, base_sleep * (2 ** attempt))
            log.warning(
                "shares %s failed (%d/%d): %s -> sleep %.1fs",
                ticker, attempt + 1, max_retries, e, backoff,
            )
            time.sleep(backoff)
    return FetchResult(ticker, "error", 0, None, None, error=last_err)


# ---------------------------------------------------------------------------
# Universe ticker discovery
# ---------------------------------------------------------------------------

def universe_tickers() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for u in progress(UNIVERSE_NAMES, desc="read universe PIT", unit="universe"):
        path = UNIVERSE_CACHE_DIR / f"{u}_pit.parquet"
        if not path.exists():
            log.warning("universe parquet missing: %s", path)
            out[u] = set()
            continue
        df = pd.read_parquet(path, columns=["ticker"])
        out[u] = set(df["ticker"].astype(str).str.upper().unique())
    return out


def union_tickers(per_universe: dict[str, set[str]]) -> list[str]:
    s: set[str] = set()
    for v in per_universe.values():
        s.update(v)
    return sorted(s)


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def _yearly_pit_members(universe: str) -> dict[int, set[str]]:
    """Load the PIT parquet for *universe* and return per-year member sets.

    For each calendar year that has at least one snapshot, the member set is
    the union of all tickers appearing on snapshots within that year.  Years
    with no snapshots are omitted from the result.
    """
    path = UNIVERSE_CACHE_DIR / f"{universe}_pit.parquet"
    if not path.exists():
        return {}

    pit_df = pd.read_parquet(path)
    if pit_df.empty:
        return {}

    year_series = pd.to_datetime(pit_df["date"]).dt.year.astype(int)

    result: dict[int, set[str]] = {}
    for yr, grp in pit_df.groupby(year_series):
        result[int(yr)] = set(grp["ticker"].astype(str).str.upper().unique())
    return result


def write_coverage(per_universe: dict[str, set[str]], manifest: dict) -> None:
    """For each (universe, year) compute fraction of tickers with shares
    coverage on that year. Flag any (u, y) below COVERAGE_FLOOR.

    The denominator is the union of PIT members on snapshots within that
    calendar year only (not all of history).  Years with no PIT snapshots
    get an explicit missing/empty row.
    """
    rows: list[dict] = []

    cached: dict[str, tuple[dt.date, dt.date]] = {}
    for tkr, info in manifest.get("tickers", {}).items():
        if info.get("status") != "success":
            continue
        if not info.get("first_date") or not info.get("last_date"):
            continue
        cached[tkr] = (
            dt.date.fromisoformat(info["first_date"]),
            dt.date.fromisoformat(info["last_date"]),
        )

    today_year = dt.date.today().year
    universe_names = sorted(per_universe.keys())

    for u in progress(universe_names, desc="shares coverage", unit="universe"):
        year_members = _yearly_pit_members(u)

        for year in range(2010, today_year + 1):
            year_start = dt.date(year, 1, 1)
            year_end = dt.date(year, 12, 31)
            members = year_members.get(year)

            if members is None:
                # No PIT snapshots in this year -> explicit empty row
                rows.append({
                    "universe": u,
                    "year": year,
                    "members": 0,
                    "covered": 0,
                    "coverage_ratio": 0.0,
                    "below_floor": True,
                })
                continue

            covered = 0
            for t in members:
                rng = cached.get(t)
                if not rng:
                    continue
                if rng[0] <= year_end and rng[1] >= year_start:
                    covered += 1

            ratio = covered / len(members)
            rows.append({
                "universe": u,
                "year": year,
                "members": len(members),
                "covered": covered,
                "coverage_ratio": round(ratio, 4),
                "below_floor": ratio < COVERAGE_FLOOR,
            })

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COVERAGE_CSV, index=False)
    flagged = sum(1 for r in rows if r["below_floor"])
    log.info("coverage report: %s (%d (universe,year) below %d%%)",
             COVERAGE_CSV, flagged, int(COVERAGE_FLOOR * 100))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def is_cached_fresh(ticker: str, manifest: dict, end: str | None = None) -> bool:
    out = SHARES_CACHE_DIR / f"{ticker}.parquet"
    if not out.exists():
        return False
    entry = manifest.get("tickers", {}).get(ticker)
    if not entry or entry.get("status") != "success":
        return False
    last_date = entry.get("last_date")
    if not last_date:
        return False
    last = dt.date.fromisoformat(last_date)
    target = dt.date.fromisoformat(end) if end else dt.date.today()
    return (target - last).days <= 14


def run(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    sleep: float = BASE_SLEEP,
    limit: int | None = None,
) -> dict:
    from data.config import utc_now_iso
    set_global_seed()
    ensure_dirs()
    SHARES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    per_u = universe_tickers()
    if tickers is None:
        tickers = union_tickers(per_u)
    if limit is not None:
        tickers = tickers[:limit]
    log.info("ticker universe: %d", len(tickers))

    manifest = load_manifest()
    failed_log = load_failed_log()
    counts = {"success": 0, "empty": 0, "error": 0, "skipped_cached": 0}

    ticker_bar = progress(tickers, desc="shares tickers", unit="ticker")
    for i, t in enumerate(ticker_bar, 1):
        if is_cached_fresh(t, manifest, end=end):
            counts["skipped_cached"] += 1
            failed_log = apply_failed_log_result(
                failed_log,
                FetchResult(t, "success", 0, None, None),
            )
            ticker_bar.set_postfix(counts)
            continue
        res = fetch_ticker(t, start=start, end=end, base_sleep=sleep)
        manifest.setdefault("tickers", {})[t] = {
            "status": res.status,
            "rows": res.rows,
            "first_date": res.first_date,
            "last_date": res.last_date,
            "error": res.error,
            "fetched_at": utc_now_iso(),
        }
        counts[res.status] = counts.get(res.status, 0) + 1
        failed_log = apply_failed_log_result(failed_log, res)
        ticker_bar.set_postfix(counts)

        if i % 50 == 0:
            save_manifest(manifest)
            save_failed_log(failed_log)
            log.info("progress %d/%d  counts=%s", i, len(tickers), counts)

    save_manifest(manifest)
    save_failed_log(failed_log)
    write_coverage(per_u, manifest)
    log.info("done. counts=%s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="yfinance shares-outstanding loader")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sleep", type=float, default=BASE_SLEEP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    run(
        tickers=args.only,
        start=args.start,
        end=args.end,
        sleep=args.sleep,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
