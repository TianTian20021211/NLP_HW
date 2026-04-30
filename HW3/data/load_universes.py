"""Phase 1.2 - Point-in-time universe membership for SP500 / SP1500 / RU3K.

Plan rules (`ideas/plan.md` 1.2):
- SP500: Wikipedia historical constituent changes + current constituents
  -> add/remove ledger -> daily PIT parquet.
- SP1500: iShares `IJH` (SP400) + `IJR` (SP600) monthly holdings + SP500
  PIT, monthly frequency.
- RU3K: iShares `IWV` monthly holdings, monthly frequency.
- Raw snapshots: `data/universe_raw/`.
- Compiled artifacts: `data/cache/universes/{sp500,sp1500,ru3k}_pit.parquet`.
- Any segment that cannot be recovered automatically -> write to
  `results/audit/universe_coverage_gaps.csv`. **Never** silently patch gaps;
  **never** fall back to the current snapshot.
- `members_at(universe, date)`:
    - raise directly when `date > today`,
    - return an empty set when `date < min(snapshot)`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from data.config import (
    AUDIT_DIR,
    UNIVERSE_CACHE_DIR,
    UNIVERSE_RAW_DIR,
    ensure_dirs,
    set_global_seed,
)
from data.progress import progress


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("load_universes")


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

ISHARES_IJH_URL = (
    "https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IJH_holdings&dataType=fund"
)
ISHARES_IJR_URL = (
    "https://www.ishares.com/us/products/239774/ishares-core-sp-smallcap-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IJR_holdings&dataType=fund"
)
ISHARES_IWV_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)

USER_AGENT = "Mozilla/5.0 (compatible; HW3-research/1.0)"

COVERAGE_GAPS_CSV = AUDIT_DIR / "universe_coverage_gaps.csv"


@dataclass
class CoverageGap:
    universe: str
    period_start: str
    period_end: str
    reason: str


# ---------------------------------------------------------------------------
# SP500 - Wikipedia ledger -> daily PIT parquet
# ---------------------------------------------------------------------------

def _http_get(url: str, retries: int = 3, sleep: float = 2.0) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            log.warning("GET %s failed (%d/%d): %s", url, attempt + 1, retries, e)
            time.sleep(sleep * (2 ** attempt))
    raise RuntimeError(f"could not fetch {url}: {last_err}")


def fetch_sp500_wiki(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Wikipedia SP500 page; persist raw HTML; return (current, changes)."""
    html_path = raw_dir / "sp500_wiki.html"
    if not html_path.exists():
        html_path.write_bytes(_http_get(SP500_WIKI_URL))

    tables = pd.read_html(io.BytesIO(html_path.read_bytes()))
    if len(tables) < 2:
        raise RuntimeError(
            "expected at least 2 tables on the SP500 Wikipedia page "
            "(current constituents + historical changes)"
        )
    current = tables[0].copy()
    changes = tables[1].copy()
    return current, changes


def _flatten_wiki_column(column: object) -> str:
    """Flatten a pandas-read Wikipedia header while dropping duplicate levels."""
    if not isinstance(column, tuple):
        return str(column).strip()

    parts: list[str] = []
    seen: set[str] = set()
    for part in column:
        text = str(part).strip()
        if not text or text.lower() == "nan" or text.startswith("Unnamed:"):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return "_".join(parts)


def _normalize_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia changes table has a 2-row header. Flatten + clean."""
    changes = changes.copy()
    changes.columns = [_flatten_wiki_column(c) for c in changes.columns]

    rename = {}
    for c in changes.columns:
        cl = c.lower()
        if "date" in cl:
            rename[c] = "date"
        elif "added" in cl and ("ticker" in cl or "symbol" in cl):
            rename[c] = "added_ticker"
        elif "removed" in cl and ("ticker" in cl or "symbol" in cl):
            rename[c] = "removed_ticker"
    changes = changes.rename(columns=rename)

    required = {"date", "added_ticker", "removed_ticker"}
    missing = sorted(required - set(changes.columns))
    if missing:
        raise RuntimeError(
            "could not normalize SP500 Wikipedia changes table; "
            f"missing columns={missing}; parsed columns={list(changes.columns)}"
        )

    keep = [c for c in ("date", "added_ticker", "removed_ticker") if c in changes.columns]
    changes = changes[keep].copy()
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"]).reset_index(drop=True)
    return changes


def build_sp500_pit(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    start: dt.date = dt.date(2009, 1, 1),
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, list[CoverageGap]]:
    """Replay Wikipedia changes backwards from current membership.

    Output schema: ``date, ticker``. Daily granularity, every business day in
    the range gets one row per current member.
    """
    end = end or dt.date.today()
    gaps: list[CoverageGap] = []

    ticker_col = next(
        (c for c in current.columns if str(c).lower().startswith("symbol")),
        None,
    )
    if ticker_col is None:
        raise RuntimeError("could not find a Symbol column in SP500 current constituents")
    today_set = set(current[ticker_col].astype(str).str.replace(".", "-", regex=False))

    changes = _normalize_changes(changes)

    # Walk changes from most recent to oldest, undoing each event so the
    # reconstructed set matches the membership *before* that change.
    membership_history: list[tuple[pd.Timestamp, set[str]]] = []
    today_ts = pd.Timestamp(end)
    membership_history.append((today_ts, set(today_set)))

    state = set(today_set)
    change_rows = list(changes.sort_values("date", ascending=False).iterrows())
    for _, row in progress(change_rows, desc="sp500 changes", unit="change"):
        d = pd.Timestamp(row["date"])
        added = str(row.get("added_ticker", "")).strip()
        removed = str(row.get("removed_ticker", "")).strip()
        if added and added != "nan":
            state.discard(added.replace(".", "-"))
        if removed and removed != "nan":
            state.add(removed.replace(".", "-"))
        membership_history.append((d - pd.Timedelta(days=1), set(state)))

    membership_history.sort(key=lambda x: x[0])

    if membership_history and membership_history[0][0].date() > start:
        gaps.append(CoverageGap(
            universe="sp500",
            period_start=str(start),
            period_end=str(membership_history[0][0].date()),
            reason="Wikipedia changes table does not extend back this far",
        ))

    rows: list[tuple[dt.date, str]] = []
    bdates = pd.bdate_range(start=start, end=end)
    snapshots = sorted(membership_history, key=lambda x: x[0])

    j = 0
    current_state: set[str] = set()
    for d in progress(bdates, desc="sp500 date grid", unit="date"):
        while j < len(snapshots) and snapshots[j][0] <= d:
            current_state = snapshots[j][1]
            j += 1
        if not current_state:
            continue
        d_date = d.date()
        rows.extend((d_date, t) for t in current_state)

    pit = pd.DataFrame(rows, columns=["date", "ticker"])
    pit["date"] = pd.to_datetime(pit["date"]).dt.date
    return pit, gaps


# ---------------------------------------------------------------------------
# iShares monthly snapshot loaders (IJH / IJR / IWV)
# ---------------------------------------------------------------------------

def fetch_ishares_holdings(symbol: str, url: str, raw_dir: Path) -> Path:
    """Fetch the *current* iShares holdings CSV and persist it dated today.

    iShares does not expose a historical-snapshot API; the operator is
    expected to run this loader monthly so a chain of dated snapshots
    accumulates in `data/universe_raw/`.
    """
    today = dt.date.today().isoformat()
    out = raw_dir / f"{symbol}_{today}.csv"
    if out.exists():
        log.info("%s snapshot already cached: %s", symbol, out.name)
        return out
    out.write_bytes(_http_get(url))
    log.info("%s snapshot saved: %s", symbol, out.name)
    return out


def parse_ishares_csv(path: Path) -> pd.DataFrame:
    """Locate the holdings table inside an iShares CSV.

    iShares CSVs prepend ~10 fund-level metadata rows before the actual
    table. The header row is the first one starting with ``Ticker,``.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Ticker,")),
        None,
    )
    if header_idx is None:
        raise RuntimeError(f"could not locate Ticker header in {path}")
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Ticker"].notna()].copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    df = df[df["Ticker"].str.match(r"^[A-Z0-9\-]+$")]
    return df


def collect_ishares_snapshots(symbol: str, raw_dir: Path) -> pd.DataFrame:
    """Aggregate every dated snapshot of one iShares product into a long table.

    Filename convention: ``{symbol}_YYYY-MM-DD.csv``. Returns a frame with
    columns ``snapshot_date`` (month-end of the snapshot), ``ticker``.
    """
    pattern = re.compile(rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    rows: list[pd.DataFrame] = []
    files = sorted(raw_dir.glob(f"{symbol}_*.csv"))
    for f in progress(files, desc=f"{symbol} snapshots", unit="file"):
        m = pattern.match(f.name)
        if not m:
            continue
        snap_date = pd.Timestamp(m.group(1))
        try:
            df = parse_ishares_csv(f)
        except Exception as e:
            log.warning("failed to parse %s: %s", f, e)
            continue
        rows.append(pd.DataFrame({
            "snapshot_date": (snap_date + pd.offsets.MonthEnd(0)).date(),
            "ticker": df["Ticker"],
        }))
    if not rows:
        return pd.DataFrame(columns=["snapshot_date", "ticker"])
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def expand_to_month_grid(
    snapshots: pd.DataFrame,
    universe: str,
    start: dt.date = dt.date(2010, 1, 1),
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, list[CoverageGap]]:
    """Forward-fill monthly snapshots to a contiguous month-end grid.

    Months without any snapshot at or before them get logged as gaps in the
    audit file; **never** filled with the current snapshot.
    """
    end = end or dt.date.today()
    gaps: list[CoverageGap] = []
    if snapshots.empty:
        gaps.append(CoverageGap(
            universe=universe,
            period_start=str(start),
            period_end=str(end),
            reason="no iShares snapshots available",
        ))
        return pd.DataFrame(columns=["date", "ticker"]), gaps

    snap_by_date: dict[dt.date, set[str]] = {}
    for d, grp in snapshots.groupby("snapshot_date"):
        snap_by_date[d] = set(grp["ticker"].astype(str))
    sorted_dates = sorted(snap_by_date)

    grid = pd.date_range(start=start, end=end, freq="ME").date
    rows: list[tuple[dt.date, str]] = []
    earliest = sorted_dates[0]

    for d in progress(grid, desc=f"{universe} month grid", unit="month"):
        if d < earliest:
            gaps.append(CoverageGap(
                universe=universe,
                period_start=str(d),
                period_end=str(d),
                reason=f"no iShares snapshot at or before {d}",
            ))
            continue
        eligible = [s for s in sorted_dates if s <= d]
        snap = snap_by_date[eligible[-1]]
        rows.extend((d, t) for t in snap)

    return pd.DataFrame(rows, columns=["date", "ticker"]), gaps


# ---------------------------------------------------------------------------
# Combined loaders
# ---------------------------------------------------------------------------

def write_pit(df: pd.DataFrame, universe: str) -> Path:
    out = UNIVERSE_CACHE_DIR / f"{universe}_pit.parquet"
    if df.empty:
        # Still write an empty placeholder so downstream `members_at` can
        # see "no snapshot" cleanly.
        empty = pd.DataFrame({"date": pd.Series(dtype="object"),
                              "ticker": pd.Series(dtype="object")})
        empty.to_parquet(out, index=False)
    else:
        df.to_parquet(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))
    return out


def write_coverage_gaps(gaps: list[CoverageGap]) -> None:
    if not gaps:
        return
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["universe", "period_start", "period_end", "reason"]
    new = pd.DataFrame(
        [[g.universe, g.period_start, g.period_end, g.reason] for g in gaps],
        columns=columns,
    )
    if COVERAGE_GAPS_CSV.exists():
        existing = pd.read_csv(COVERAGE_GAPS_CSV, dtype=str)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates().sort_values(columns).reset_index(drop=True)
    else:
        combined = new
    combined.to_csv(COVERAGE_GAPS_CSV, index=False)
    log.info("wrote %d unique coverage gaps to %s", len(combined), COVERAGE_GAPS_CSV)


def _tag_component_gaps(gaps: list[CoverageGap], component: str) -> list[CoverageGap]:
    return [
        CoverageGap(
            universe=g.universe,
            period_start=g.period_start,
            period_end=g.period_end,
            reason=f"{component}: {g.reason}",
        )
        for g in gaps
    ]


def build_sp500(raw_dir: Path) -> list[CoverageGap]:
    current, changes = fetch_sp500_wiki(raw_dir)
    pit, gaps = build_sp500_pit(current, changes)
    write_pit(pit, "sp500")
    return gaps


def build_sp1500(raw_dir: Path) -> list[CoverageGap]:
    monthly_start = dt.date(2010, 1, 1)
    fetch_ishares_holdings("IJH", ISHARES_IJH_URL, raw_dir)
    fetch_ishares_holdings("IJR", ISHARES_IJR_URL, raw_dir)
    ijh = collect_ishares_snapshots("IJH", raw_dir)
    ijr = collect_ishares_snapshots("IJR", raw_dir)
    ijh_pit, ijh_gaps = expand_to_month_grid(ijh, universe="sp1500", start=monthly_start)
    ijr_pit, ijr_gaps = expand_to_month_grid(ijr, universe="sp1500", start=monthly_start)

    sp500_path = UNIVERSE_CACHE_DIR / "sp500_pit.parquet"
    if not sp500_path.exists():
        raise RuntimeError("sp500 PIT must exist before building sp1500")
    sp500 = pd.read_parquet(sp500_path)
    sp500["snapshot_date"] = (
        pd.to_datetime(sp500["date"]) + pd.offsets.MonthEnd(0)
    ).dt.date
    sp500_monthly = (
        sp500[["snapshot_date", "ticker"]]
        .rename(columns={"snapshot_date": "date"})
        .drop_duplicates()
    )
    sp500_monthly = sp500_monthly[
        (sp500_monthly["date"] >= monthly_start)
        & (sp500_monthly["date"] <= dt.date.today())
    ]

    pit = pd.concat([ijh_pit, ijr_pit, sp500_monthly], ignore_index=True)
    pit = pit.drop_duplicates()
    write_pit(pit, "sp1500")
    return _tag_component_gaps(ijh_gaps, "IJH") + _tag_component_gaps(ijr_gaps, "IJR")


def build_ru3k(raw_dir: Path) -> list[CoverageGap]:
    fetch_ishares_holdings("IWV", ISHARES_IWV_URL, raw_dir)
    iwv = collect_ishares_snapshots("IWV", raw_dir)
    pit, gaps = expand_to_month_grid(iwv, universe="ru3k")
    write_pit(pit, "ru3k")
    return gaps


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

_PIT_CACHE: dict[str, pd.DataFrame] = {}


def _load_pit(universe: str) -> pd.DataFrame:
    if universe in _PIT_CACHE:
        return _PIT_CACHE[universe]
    path = UNIVERSE_CACHE_DIR / f"{universe}_pit.parquet"
    if not path.exists():
        raise FileNotFoundError(f"PIT parquet missing for {universe}: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    _PIT_CACHE[universe] = df
    return df


def members_at(universe: str, date: dt.date) -> set[str]:
    """Return the set of tickers in `universe` as of `date`.

    Plan rule (1.2):
    - Raise if `date > today`.
    - Return empty set if `date < min(snapshot)`.
    - Otherwise return the latest snapshot at or before `date`.
    """
    if isinstance(date, (pd.Timestamp, dt.datetime)):
        date = date.date()
    if date > dt.date.today():
        raise ValueError(f"members_at({universe}, {date}): date > today")

    df = _load_pit(universe)
    if df.empty:
        return set()
    min_date = df["date"].min()
    if date < min_date:
        return set()

    eligible = df[df["date"] <= date]
    if eligible.empty:
        return set()
    last_date = eligible["date"].max()
    return set(eligible.loc[eligible["date"] == last_date, "ticker"].astype(str))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run() -> None:
    set_global_seed()
    ensure_dirs()
    raw_dir = UNIVERSE_RAW_DIR

    all_gaps: list[CoverageGap] = []
    all_gaps += build_sp500(raw_dir)
    all_gaps += build_sp1500(raw_dir)
    all_gaps += build_ru3k(raw_dir)

    write_coverage_gaps(all_gaps)
    log.info("done. coverage_gaps=%d", len(all_gaps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIT universe parquet files")
    parser.add_argument(
        "--only",
        choices=("sp500", "sp1500", "ru3k", "all"),
        default="all",
    )
    args = parser.parse_args()

    set_global_seed()
    ensure_dirs()
    raw_dir = UNIVERSE_RAW_DIR

    gaps: list[CoverageGap] = []
    if args.only in ("sp500", "all"):
        gaps += build_sp500(raw_dir)
    if args.only in ("sp1500", "all"):
        gaps += build_sp1500(raw_dir)
    if args.only in ("ru3k", "all"):
        gaps += build_ru3k(raw_dir)
    write_coverage_gaps(gaps)


if __name__ == "__main__":
    main()
