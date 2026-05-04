"""Phase 4 — Walk-Forward Splits and Frozen Hyperparameters.

Provides:
- Forward return computation (from price data via merge_asof)
- Walk-forward fold generation (2020Q1 → 2026Q2, expanding window)
- Label availability purge (G11)
- Hyperparameter tuning on 2010-2019 sample, frozen to JSON
- Forward-return caching with joblib.Memory + explicit dependency signatures
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
from joblib import Memory
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

from data.config import PRICE_CACHE_DIR, RESULTS_DIR, AUDIT_DIR, SEED, utc_now_iso
from data.progress import progress

from backtest._stats import make_median_imputer, spearman

log = logging.getLogger("backtest.splits")

HORIZONS: list[int] = [1, 3, 5, 10, 20]


@lru_cache(maxsize=1)
def _cached_read_features(path: str) -> pd.DataFrame:
    """Read features parquet with LRU caching (maxsize=1)."""
    return pd.read_parquet(path)


def read_feature_columns(
    path: Path,
    columns: Iterable[str],
    *,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Read only requested columns that exist in a feature parquet.

    Phase 5 often needs a small metadata/short-list slice from the stretch
    feature table. Loading all columns can inflate a 447 MB parquet file into
    10+ GB RSS, so callers should use this helper instead of full reads.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = list(dict.fromkeys(required_columns))
    missing_required = [c for c in required if c not in available]
    if missing_required:
        raise KeyError(f"{path} is missing required columns: {missing_required}")

    requested = list(dict.fromkeys([*required, *columns]))
    read_cols = [c for c in requested if c in available]
    return pd.read_parquet(path, columns=read_cols)

TUNING_START = pd.Timestamp("2010-01-01")
TUNING_END = pd.Timestamp("2019-12-31")
WF_START = pd.Timestamp("2020-01-01")
WF_END = pd.Timestamp("2026-06-30")

# Columns that must NOT enter the feature matrix.
ID_COLS = {
    "BESTTICKER", "SECTOR", "availability_date", "call_entry_date",
    "ingest_entry_date", "MOSTIMPORTANTDATEUTC", "INGESTDATEUTC",
    "SignalType", "QTR_YEAR", "call_hour_utc", "SECTOR_GICS",
    "_orig_df_index", "_in_universe",
}
# Forward-return / target-date columns added by compute_forward_returns.
_RETURN_COL_RE = re.compile(r"^forward_return_\d+d$")
_TARGET_DATE_COL_RE = re.compile(r"^target_available_date_\d+d$")
_RAW_RETURN_COL_RE = re.compile(r"^Return_\d+d$")
_RIGHT_CENSORED_COL_RE = re.compile(r"^right_censored_\d+d$")


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

AVAILABILITY_RULE_VERSION = "v1"
"""Version identifier for the availability-date computation rule.
Bump this when the rule changes to invalidate cached forward returns.
"""

# Cache directory for forward returns (used by both the old parquet cache
# and the new joblib.Memory + signature cache).
FORWARD_RETURNS_CACHE_DIR: Path = RESULTS_DIR / "cache"

# Hyperparameter storage: tier x model x horizon granularity.
HPARAMS_DIR: Path = RESULTS_DIR / "hparams"

# ---------------------------------------------------------------------------
# Forward-return cache infrastructure (joblib.Memory + explicit signatures)
# ---------------------------------------------------------------------------

FORWARD_RETURNS_CACHE_VERSION = 3
"""Schema version for the forward-returns cache signature.
Bump this when the cache signature schema changes (e.g., new dependency
fields added) to force a full recomputation.
"""

MAX_ENTRY_GAP_BDAYS = 5
"""Maximum business-day roll-forward from signal availability to first quote."""


@dataclass(frozen=True)
class ForwardReturnsCacheSignature:
    """Explicit dependency signature for forward-returns cache invalidation.

    Every field that could affect the forward-return computation is captured
    so that joblib.Memory auto-invalidates when any dependency changes.
    """

    features_path: str
    """Absolute path to the features parquet file."""
    features_size: int
    """File size in bytes when the signature was built."""
    features_mtime: float
    """Last modification timestamp (seconds since epoch)."""
    price_manifest_path: str
    """Absolute path to ``data/cache/prices/_manifest.json``."""
    price_manifest_size: int
    """File size in bytes."""
    price_manifest_mtime: float
    """Last modification timestamp."""
    price_manifest_hash: str
    """SHA-256 hex digest of the price manifest file contents."""
    horizons: tuple[int, ...]
    """Sorted tuple of requested return horizons."""
    entry_date_col: str
    """Column used as the entry date for forward returns."""
    source_hash: int
    """SHA-256 hash of ``compute_forward_returns`` and all helper sources —
    captures code changes in the forward-return computation itself."""
    cache_version: int = FORWARD_RETURNS_CACHE_VERSION
    """Schema version — bump to force full cache invalidation."""
    len_df: int = 0
    """Number of rows in the DataFrame whose forward returns were computed.
    Used for cache collision detection."""
    row_fingerprint: int = 0
    """Deterministic hash of the first 1000 index values for cache collision
    detection. Catches cases where a filtered/subset DataFrame would
    otherwise produce a cache hit for the wrong row set."""


# Shared joblib.Memory instance for the forward-returns cache.
# joblib stores cached results in FORWARD_RETURNS_CACHE_DIR/joblib/...
_forward_returns_memory = Memory(
    location=str(FORWARD_RETURNS_CACHE_DIR / "joblib"),
    verbose=0,
)


@_forward_returns_memory.cache(ignore=["df"])
def _compute_forward_returns_cached(
    signature: ForwardReturnsCacheSignature,
    df: pd.DataFrame,
    price_cache_dir: Path,
) -> pd.DataFrame:
    """Compute forward returns keyed by *signature*; cached by joblib.Memory.

    The ``df`` argument is excluded from the cache key (via ``ignore=["df"]``)
    because it is fully determined by the dependency fields in *signature*.
    When any dependency changes, *signature* changes, and joblib recomputes.
    """
    return compute_forward_returns(
        df,
        price_cache_dir,
        horizons=list(signature.horizons),
        entry_date_col=signature.entry_date_col,
    )


def _build_cache_signature(
    features_path: Path,
    price_cache_dir: Path,
    horizons: Sequence[int],
    entry_date_col: str,
    df: pd.DataFrame,
) -> ForwardReturnsCacheSignature:
    """Build a dependency signature for forward-returns caching.

    Gathers file metadata and content hashes for every input that affects
    the forward-return computation, plus a row-position fingerprint of *df*
    to detect when a filtered/subset DataFrame is passed to the cache.
    """
    # Features file info
    fs = features_path.stat() if features_path.exists() else None
    features_size = fs.st_size if fs else 0
    features_mtime = fs.st_mtime if fs else 0.0

    # Price manifest file info + content hash
    _sync_price_cache_manifest(price_cache_dir)
    price_manifest_path = price_cache_dir / "_manifest.json"
    if price_manifest_path.exists():
        pms = price_manifest_path.stat()
        manifest_bytes = price_manifest_path.read_bytes()
        price_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        price_manifest_size = pms.st_size
        price_manifest_mtime = pms.st_mtime
    else:
        price_manifest_hash = ""
        price_manifest_size = 0
        price_manifest_mtime = 0.0

    # Source-code hash for the core computation and all extracted helpers
    try:
        source_hash = _build_forward_returns_source_hash()
    except (OSError, TypeError):
        source_hash = 0

    # DataFrame fingerprint for cache collision detection
    len_df = len(df)
    row_fingerprint = int(hashlib.sha256(str(tuple(df.index[:1000])).encode()).hexdigest()[:16], 16)

    return ForwardReturnsCacheSignature(
        features_path=str(features_path.resolve()),
        features_size=features_size,
        features_mtime=features_mtime,
        price_manifest_path=str(price_manifest_path.resolve()),
        price_manifest_size=price_manifest_size,
        price_manifest_mtime=price_manifest_mtime,
        price_manifest_hash=price_manifest_hash,
        horizons=tuple(sorted(horizons)),
        entry_date_col=entry_date_col,
        source_hash=source_hash,
        len_df=len_df,
        row_fingerprint=row_fingerprint,
    )


def _write_forward_returns_manifest(
    signature: ForwardReturnsCacheSignature,
) -> Path:
    """Write a human-readable manifest JSON recording cache inputs."""
    manifest = {
        "cache_version": FORWARD_RETURNS_CACHE_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "features_path": signature.features_path,
        "features_size": signature.features_size,
        "features_mtime": signature.features_mtime,
        "price_manifest_path": signature.price_manifest_path,
        "price_manifest_size": signature.price_manifest_size,
        "price_manifest_mtime": signature.price_manifest_mtime,
        "price_manifest_hash": signature.price_manifest_hash,
        "horizons": list(signature.horizons),
        "entry_date_col": signature.entry_date_col,
        "source_hash": signature.source_hash,
    }
    path = FORWARD_RETURNS_CACHE_DIR / "forward_returns_manifest.json"
    FORWARD_RETURNS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _sync_price_cache_manifest(price_cache_dir: Path) -> None:
    """Force a lightweight manifest update from current price parquet files.

    The external loaders own the rich ticker metadata in ``_manifest.json``.
    This helper only adds a deterministic fingerprint over current parquet
    names, sizes, and mtimes before forward-return cache signatures are built.
    """
    price_cache_dir = Path(price_cache_dir)
    if not price_cache_dir.exists():
        return

    manifest_path = price_cache_dir / "_manifest.json"
    entries: list[str] = []
    for path in sorted(price_cache_dir.glob("*.parquet")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")

    fingerprint = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    if manifest.get("_price_files_fingerprint") == fingerprint:
        return

    manifest["_price_files_fingerprint"] = fingerprint
    manifest["_price_files_count"] = len(entries)
    manifest["_price_files_synced_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _build_forward_returns_source_hash() -> int:
    """Hash compute_forward_returns plus all helper/orchestrator sources and cache version."""
    funcs = [
        compute_forward_returns,
        _empty_forward_return_frame,
        _prepare_forward_return_events,
        _load_forward_return_price_metrics,
        _compute_ticker_forward_return_matches,
        _fill_forward_return_arrays,
        _build_cache_signature,
        _sync_price_cache_manifest,
        get_forward_returns_cached,
        ensure_forward_returns,
    ]
    source_blob = "\n\n".join(inspect.getsource(fn) for fn in funcs)
    source_blob += f"\nCACHE_VERSION={FORWARD_RETURNS_CACHE_VERSION}"
    source_blob += f"\nMAX_ENTRY_GAP_BDAYS={MAX_ENTRY_GAP_BDAYS}"
    return int(hashlib.sha256(source_blob.encode()).hexdigest()[:16], 16)


def _empty_forward_return_frame(
    index: pd.Index,
    horizons: Sequence[int],
    ret_arrays: dict[int, np.ndarray],
    date_arrays: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Build the standard forward-return result DataFrame from pre-allocated arrays.

    Returns a DataFrame indexed like *index* with ``forward_return_{h}d`` and
    ``target_available_date_{h}d`` for each horizon.
    """
    result = pd.DataFrame(index=index)
    for hh in horizons:
        result[f"forward_return_{hh}d"] = ret_arrays[hh]
        result[f"target_available_date_{hh}d"] = date_arrays[hh]
        result[f"right_censored_{hh}d"] = pd.isna(date_arrays[hh])
    return result


def _prepare_forward_return_events(
    df: pd.DataFrame,
    ticker_col: str,
    entry_date_col: str,
) -> pd.DataFrame:
    """Build per-row event metadata for forward-return matching.

    Returns a DataFrame with ``_orig_pos``, ``_ticker``, and ``_entry_date``
    columns. Rows with missing ticker or entry date are dropped.
    """
    events = pd.DataFrame({
        "_orig_pos": np.arange(len(df), dtype=np.int64),
        "_ticker": df[ticker_col].values,
        "_entry_date": pd.to_datetime(
            df[entry_date_col], errors="coerce"
        ).to_numpy(dtype="datetime64[ns]"),
    })
    events = events[
        events["_ticker"].notna() & events["_entry_date"].notna()
    ].copy()
    if not events.empty:
        events["_ticker"] = events["_ticker"].astype(str)
    return events


def _load_forward_return_price_metrics(
    path: Path,
    horizons: Sequence[int],
) -> pd.DataFrame | None:
    """Load one ticker price file and precompute forward returns by price row.

    Returns a metrics DataFrame with ``_merge_date``, ``_price_pos``,
    ``forward_return_{h}d``, and ``target_available_date_{h}d``, or None
    for missing/empty/unusable prices.
    """
    if not path.exists():
        return None

    try:
        px = pd.read_parquet(path, columns=["date", "adj_close", "volume"])
    except Exception:
        px = pd.read_parquet(path)
        if "volume" not in px.columns:
            px["volume"] = np.nan
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px[
        px["date"].notna()
        & px["adj_close"].notna()
        & (px["adj_close"] > 0)
        & px["volume"].notna()
        & (px["volume"] > 0)
    ].sort_values("date")
    if px.empty:
        return None

    dates = px["date"].to_numpy(dtype="datetime64[ns]")
    closes = px["adj_close"].to_numpy(dtype="float64")
    metrics = pd.DataFrame({
        "_merge_date": dates,
        "_price_pos": np.arange(len(dates), dtype=np.int64),
    })
    for h in horizons:
        returns = np.full(len(closes), np.nan, dtype="float64")
        target_dates = np.full(
            len(dates), np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        if len(closes) > h:
            returns[:-h] = closes[h:] / closes[:-h] - 1.0
            target_dates[:-h] = dates[h:]
        metrics[f"forward_return_{h}d"] = returns
        metrics[f"target_available_date_{h}d"] = target_dates

    return metrics


def _compute_ticker_forward_return_matches(
    ticker: str,
    events: pd.DataFrame,
    price_cache_dir: Path,
    horizons: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Merge one ticker's events to entry prices and return matched arrays.

    Loads the ticker's price data via ``_load_forward_return_price_metrics``,
    sorts events, then merges with ``pd.merge_asof`` (direction="forward").
    Entry gap must be between 0 and 5 business days inclusive.
    Missing exits remain NaN/NaT.

    Returns (orig_pos, ret_by_h, date_by_h) where:
    - orig_pos: original row positions with valid entries/returns for at least
      one horizon.
    - ret_by_h: horizon -> return array aligned to orig_pos.
    - date_by_h: horizon -> target date array aligned to orig_pos.

    Returns None when no price data or no valid matches are found.
    """
    path = price_cache_dir / f"{ticker}.parquet"
    metrics = _load_forward_return_price_metrics(path, horizons)
    if metrics is None:
        return None

    h_list = list(horizons)
    ev = events.sort_values("_entry_date")
    ev["_entry_date"] = ev["_entry_date"].astype("datetime64[ns]")

    mg = pd.merge_asof(
        ev, metrics,
        left_on="_entry_date", right_on="_merge_date",
        direction="forward",
    )

    # Entry gap check
    entry_days_np = mg["_entry_date"].to_numpy(dtype="datetime64[D]")
    entry_days_m = pd.to_datetime(mg["_merge_date"], errors="coerce").to_numpy(
        dtype="datetime64[D]"
    )
    entry_gap_bdays = np.full(len(mg), np.nan, dtype="float64")
    has_entry = ~pd.isna(entry_days_m)
    entry_gap_bdays[has_entry] = np.busday_count(
        entry_days_np[has_entry],
        entry_days_m[has_entry],
    )

    price_pos_notna = mg["_price_pos"].notna().to_numpy()
    entry_valid = (
        price_pos_notna
        & np.isfinite(entry_gap_bdays)
        & (entry_gap_bdays >= 0)
        & (entry_gap_bdays <= MAX_ENTRY_GAP_BDAYS)
    )

    # Build all_valid mask (union across horizons)
    all_valid = np.zeros(len(mg), dtype=bool)
    for h in h_list:
        valid = entry_valid & mg[f"forward_return_{h}d"].notna().to_numpy()
        all_valid |= valid

    if not all_valid.any():
        return None

    orig_pos = mg["_orig_pos"].to_numpy(dtype=np.int64)[all_valid]
    all_valid_indices = np.where(all_valid)[0]

    ret_by_h: dict[int, np.ndarray] = {}
    date_by_h: dict[int, np.ndarray] = {}
    for h in h_list:
        ret_col = f"forward_return_{h}d"
        date_col = f"target_available_date_{h}d"

        valid = entry_valid & mg[ret_col].notna().to_numpy()

        arr_ret = np.full(len(orig_pos), np.nan, dtype="float64")
        arr_date = np.full(len(orig_pos), np.datetime64("NaT"), dtype="datetime64[ns]")

        if valid.any():
            valid_indices = np.where(valid)[0]
            pos_in_all = np.searchsorted(all_valid_indices, valid_indices)

            valid_ret = mg[ret_col].to_numpy(dtype="float64")[valid_indices]
            valid_date = pd.to_datetime(
                mg[date_col], errors="coerce"
            ).to_numpy(dtype="datetime64[ns]")[valid_indices]

            arr_ret[pos_in_all] = valid_ret
            arr_date[pos_in_all] = valid_date

        ret_by_h[h] = arr_ret
        date_by_h[h] = arr_date

    return orig_pos, ret_by_h, date_by_h


def _fill_forward_return_arrays(
    ret_arrays: dict[int, np.ndarray],
    date_arrays: dict[int, np.ndarray],
    orig_pos: np.ndarray,
    ret_by_h: dict[int, np.ndarray],
    date_by_h: dict[int, np.ndarray],
    horizons: Sequence[int],
) -> None:
    """Copy one ticker's matched return arrays into the preallocated output arrays."""
    for h in horizons:
        ret_arrays[h][orig_pos] = ret_by_h[h]
        date_arrays[h][orig_pos] = date_by_h[h]


def compute_forward_returns(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    horizons: Sequence[int] = HORIZONS,
    entry_date_col: str = "availability_date",
) -> pd.DataFrame:
    """Compute forward returns for every event row.

    For each event:
    1. Find next valid close on/after ``entry_date_col`` (merge_asof forward).
    2. For each horizon *h*, ``close[t+h] / close[t] - 1`` using the ticker's
       own price history.
    3. ``target_available_date_h`` = date of the exit close.

    .. important::
        Forward returns are anchored on *entry_date_col* (default
        ``availability_date``).  When these returns serve as model targets,
        the model learns to predict returns starting at signal-availability
        time.  In :class:`backtest.portfolio.PortfolioSimulator`, however,
        signals are collected in a lookback window and traded at the next
        rebalance date, which is always >= *entry_date_col*.  This creates a
        systematic delay between the target horizon and the actual holding
        period.

        Consequence: walk-forward OOS IC/MSE is an **upper bound** on
        achievable signal quality.  The portfolio simulator's own post-cost
        Sharpe (which uses actual entry/exit dates) is the ground truth.

    Parameters
    ----------
    entry_date_col:
        Column used as the entry date for forward returns. Defaults to
        ``"availability_date"`` (the unified operational availability date).
        Use ``"call_entry_date"`` for forward returns timed from the earnings
        call publication date.

    Returns a DataFrame indexed like *df* with columns
    ``forward_return_{h}d`` (float64) and ``target_available_date_{h}d``
    (datetime64[ns]). NaN/NaT where the entry or exit bar is unavailable.
    """
    h_list = list(horizons)
    n = len(df)
    ticker_col = "BESTTICKER" if "BESTTICKER" in df.columns else "ticker"

    ret_arrays = {
        h: np.full(n, np.nan, dtype="float64")
        for h in h_list
    }
    date_arrays = {
        h: np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        for h in h_list
    }

    if not price_cache_dir.exists():
        log.warning("price cache dir %s missing", price_cache_dir)
        return _empty_forward_return_frame(df.index, h_list, ret_arrays, date_arrays)

    events = _prepare_forward_return_events(df, ticker_col, entry_date_col)
    if events.empty:
        return _empty_forward_return_frame(df.index, h_list, ret_arrays, date_arrays)

    n_with_prices = 0
    event_groups = events.groupby("_ticker", sort=False)
    for tkr_str, ev in progress(
        event_groups,
        total=events["_ticker"].nunique(),
        desc="forward returns",
        unit="tkr",
    ):
        result = _compute_ticker_forward_return_matches(
            tkr_str, ev, price_cache_dir, h_list,
        )
        if result is not None:
            n_with_prices += 1
            orig_pos, ret_by_h, date_by_h = result
            _fill_forward_return_arrays(
                ret_arrays, date_arrays, orig_pos, ret_by_h, date_by_h, h_list,
            )

    if n_with_prices == 0:
        log.warning("no price data available — forward returns are all NaN")
    return _empty_forward_return_frame(df.index, h_list, ret_arrays, date_arrays)


# ---------------------------------------------------------------------------
# Feature-column helpers
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature-column names, excluding IDs, returns, and
    target-availability date columns.

    The model matrix is numeric-only so timestamp/object identifiers cannot
    accidentally leak into fitting or fail during ``to_numpy(dtype=float)``.
    """
    excluded = ID_COLS.copy()
    feature_cols: list[str] = []
    for c in df.columns:
        if (
            c in excluded
            or _RETURN_COL_RE.match(c)
            or _TARGET_DATE_COL_RE.match(c)
            or _RAW_RETURN_COL_RE.match(c)
            or _RIGHT_CENSORED_COL_RE.match(c)
        ):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    return feature_cols


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    """A single walk-forward fold.

    ``train_indices`` / ``test_indices`` are positional indices into the
    DataFrame passed to ``generate_folds``.
    """
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray
    # per-horizon → latest target_available_date in purged training set
    max_train_target_date: dict[int, pd.Timestamp] = field(default_factory=dict)


def _quarter_boundaries(
    start: pd.Timestamp = WF_START,
    end: pd.Timestamp = WF_END,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    qs = pd.date_range(start=start, end=end, freq="QE")
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for q_end in qs:
        q_start = ((q_end - pd.DateOffset(months=3))
                   .normalize() + pd.Timedelta(days=1))
        boundaries.append((q_start.normalize(), q_end.normalize()))
    return boundaries


def generate_folds(
    df: pd.DataFrame,
    start: pd.Timestamp = WF_START,
    end: pd.Timestamp = WF_END,
    availability_col: str = "availability_date",) -> list[Fold]:
    """Generate walk-forward folds (2020Q1 through 2026Q2).

    Each fold:
    - **test**: events whose *availability_col* is inside the quarter.
    - **train**: events with *availability_col* < test_start and
      ``call_entry_date >= 2010-01-01``.

    G11 (label availability purge) is **not** applied here; it is applied
    per-horizon by ``purge_train_for_horizon``.

    Parameters
    ----------
    availability_col:
        Column used as the PIT availability date. Defaults to
        ``"availability_date"`` (the unified operational availability date).
        ``"call_entry_date"`` (derived from MOSTIMPORTANTDATEUTC, the actual
        earnings call publication time) is available for backward
        compatibility.
    """
    call_entry = pd.to_datetime(df["call_entry_date"].to_numpy())
    avail_date = pd.to_datetime(df[availability_col].to_numpy())

    quarters = _quarter_boundaries(start, end)
    folds: list[Fold] = []

    for fold_id, (q_start, q_end) in enumerate(quarters):
        test_mask = (
            (avail_date >= q_start.to_datetime64())
            & (avail_date <= q_end.to_datetime64())
        )
        test_idx = np.flatnonzero(test_mask)
        if len(test_idx) == 0:
            log.info("fold %d (%s → %s): 0 test events, skipped",
                     fold_id, q_start.date(), q_end.date())
            continue

        train_mask = (
            (avail_date < q_start.to_datetime64())
            & (call_entry >= TUNING_START.to_datetime64())
        )
        train_idx = np.flatnonzero(train_mask)

        folds.append(Fold(
            fold_id=fold_id,
            train_start=TUNING_START,
            train_end=q_start - pd.Timedelta(days=1),
            test_start=q_start,
            test_end=q_end,
            train_indices=train_idx,
            test_indices=test_idx,
        ))

    log.info("generated %d folds (availability_col=%s)", len(folds), availability_col)
    return folds


def purge_train_for_horizon(
    fold: Fold,
    df: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Purge training indices so no row's forward-return realisation leaks
    past *fold.train_end* (G11).

    Rows whose ``target_available_date_{h}d`` is NaT (right-censored or no
    price data) or > ``fold.train_end`` are removed. The remaining indices
    are returned, and the maximum surviving target date is recorded in
    ``fold.max_train_target_date`` for the audit trail.
    """
    date_col = f"target_available_date_{horizon}d"
    if df[date_col].dtype == "datetime64[ns]":
        target_dates = df[date_col].to_numpy()
    else:
        target_dates = pd.to_datetime(df[date_col].values)
    train_end_ns = fold.train_end.to_datetime64()

    valid = (
        pd.notna(target_dates)
        & (target_dates <= train_end_ns)
    )
    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[fold.train_indices] = True
    keep_mask &= valid

    purged = np.flatnonzero(keep_mask)
    if len(purged) > 0:
        fold.max_train_target_date[horizon] = pd.Timestamp(
            target_dates[purged].max()
        )
    return purged


# ---------------------------------------------------------------------------
# Tuning sample
# ---------------------------------------------------------------------------

def get_tuning_sample(
    df: pd.DataFrame,
    target_col: str,
    target_avail_col: str,
    availability_col: str = "availability_date",) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tuning arrays for the 2010-01 → 2019-12 tuning period.

    Filters:
    - *availability_col* between 2010-01-01 and 2019-12-31
    - ``target_available_date_h <= 2019-12-31`` (no partial-period leakage)
    - finite target

    Parameters
    ----------
    availability_col:
        Column for PIT availability. Defaults to ``"availability_date"``
        (the unified operational availability date).
    """
    avail = pd.to_datetime(df[availability_col], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    call_entry = pd.to_datetime(df["call_entry_date"], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    target_avail = pd.to_datetime(df[target_avail_col], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    y_vals = df[target_col].to_numpy(dtype="float64")

    mask = (
        (avail >= TUNING_START.to_datetime64())
        & (avail <= TUNING_END.to_datetime64())
        & (call_entry >= TUNING_START.to_datetime64())
        & pd.notna(target_avail)
        & (target_avail <= TUNING_END.to_datetime64())
        & np.isfinite(y_vals)
    )
    idx = np.flatnonzero(mask)
    if len(idx):
        order = np.argsort(avail[idx], kind="mergesort")
        idx = idx[order]

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].iloc[idx].to_numpy(dtype="float64")
    y = y_vals[idx].astype("float64")

    log.info("tuning sample: %d rows, %d features", len(idx), X.shape[1])
    return X, y, idx, avail[idx], target_avail[idx]


# ---------------------------------------------------------------------------
# TimeSeriesSplit helper
# ---------------------------------------------------------------------------

def _time_series_splits(
    feature_dates: np.ndarray,
    target_available_dates: np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate label-purged TimeSeriesSplit train/val index pairs.

    The tuning sample is sorted by feature availability date. For each inner
    validation block, training rows are kept only when both the feature and
    realized target were available before the validation block starts.
    """
    from sklearn.model_selection import TimeSeriesSplit

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    base = np.arange(len(feature_dates))
    for tr, vl in TimeSeriesSplit(n_splits=n_splits).split(base):
        val_start = feature_dates[vl].min()
        keep = (
            (feature_dates[tr] < val_start)
            & (target_available_dates[tr] < val_start)
        )
        tr_purged = tr[keep]
        if len(tr_purged) == 0:
            log.warning("inner CV split dropped: empty purged training fold")
            continue
        splits.append((tr_purged, vl))
    return splits


def _preprocess_train_val(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on an inner-CV training fold and transform val."""
    X_train = X[train_idx]
    X_val = X[val_idx]

    imp = make_median_imputer()
    X_train = imp.fit_transform(X_train)
    X_val = imp.transform(X_val)

    if scale:
        from sklearn.preprocessing import StandardScaler

        scl = StandardScaler()
        X_train = scl.fit_transform(X_train)
        X_val = scl.transform(X_val)

    return X_train, X_val


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def _fit_lgbm_trial(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> float:
    import lightgbm as lgb

    m = lgb.LGBMRegressor(**params, random_state=seed, verbose=-1, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    return spearman(pred, y_val)


def _fit_xgb_trial(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> float:
    import xgboost as xgb

    m = xgb.XGBRegressor(**params, random_state=seed, verbosity=0, n_jobs=1)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    return spearman(pred, y_val)


def _fit_ridge_alpha(
    alpha: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Fit Ridge with given alpha and return Spearman IC."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha, random_state=SEED)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    return spearman(pred, y_val)


def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Grid-search Ridge alpha.  Spearman IC is the objective."""
    from joblib import Parallel, delayed

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    scores_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}

    for tr, vl in progress(cv_splits, desc="ridge tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=True)
        fold_scores: list[float] = Parallel(n_jobs=-1)(
            delayed(_fit_ridge_alpha)(a, X_tr, y[tr], X_vl, y[vl])
            for a in alphas
        )
        for alpha, score in zip(alphas, fold_scores):
            scores_by_alpha[alpha].append(score)

    best_alpha: float | None = None
    best_score = -np.inf
    for alpha, scores in scores_by_alpha.items():
        mean_ic = float(np.mean(scores))
        log.info("  alpha=%.4f  cv_ic=%.6f", alpha, mean_ic)
        if mean_ic > best_score:
            best_score = mean_ic
            best_alpha = alpha

    return {"alpha": best_alpha, "cv_ic": best_score}


def tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Random-search LightGBM hyperparameters.  30 trials, Spearman IC."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed — returning defaults")
        return {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.05,
                "num_leaves": 31, "min_child_samples": 50,
                "subsample": 0.8, "colsample_bytree": 0.8, "cv_ic": float("nan")}

    space = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    rng = np.random.RandomState(SEED)
    trials = _random_param_combos(space, 30, rng)

    from joblib import Parallel, delayed

    scores_by_trial: list[list[float]] = [[] for _ in trials]
    for tr, vl in progress(cv_splits, desc="lgbm tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=False)
        fold_scores: list[float] = Parallel(n_jobs=-1)(
            delayed(_fit_lgbm_trial)(params, X_tr, y[tr], X_vl, y[vl], SEED)
            for params in trials
        )
        for i, score in enumerate(fold_scores):
            scores_by_trial[i].append(score)

    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    for params, scores in zip(trials, scores_by_trial):
        mean_ic = float(np.mean(scores))
        if mean_ic > best_score:
            best_score = mean_ic
            best_params = dict(params)

    if best_params is None:
        best_params = {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.05,
                       "num_leaves": 31, "min_child_samples": 50}
    best_params["cv_ic"] = best_score
    return best_params


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Random-search XGBoost hyperparameters.  30 trials, Spearman IC."""
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("xgboost not installed — returning defaults")
        return {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 10.0, "cv_ic": float("nan")}

    space = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 10.0, 100.0],
    }
    rng = np.random.RandomState(SEED)
    trials = _random_param_combos(space, 30, rng)

    from joblib import Parallel, delayed

    scores_by_trial: list[list[float]] = [[] for _ in trials]
    for tr, vl in progress(cv_splits, desc="xgb tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=False)
        fold_scores: list[float] = Parallel(n_jobs=-1)(
            delayed(_fit_xgb_trial)(params, X_tr, y[tr], X_vl, y[vl], SEED)
            for params in trials
        )
        for i, score in enumerate(fold_scores):
            scores_by_trial[i].append(score)

    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    for params, scores in zip(trials, scores_by_trial):
        mean_ic = float(np.mean(scores))
        if mean_ic > best_score:
            best_score = mean_ic
            best_params = dict(params)

    if best_params is None:
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8}
    best_params["cv_ic"] = best_score
    return best_params


def _random_param_combos(
    space: dict[str, list[Any]],
    n: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """Sample *n* random hyperparameter combinations from *space*."""
    keys = list(space)
    combos: list[dict[str, Any]] = []
    for _ in range(n):
        combo = {k: _json_safe(rng.choice(space[k])) for k in keys}
        combos.append(combo)
    return combos


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar containers into JSON-native objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value



# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def tune_all_models(
    df: pd.DataFrame,
    target_col: str,
    target_avail_col: str,
    output_dir: Path | None = None,
    availability_col: str = "availability_date",
    models: Sequence[str] | None = None,
    tier: str = "enhanced",
    horizon: int = 5,
) -> dict[str, dict[str, Any]]:
    """Tune Ridge / LightGBM / XGBoost on 2010–2019 data.

    Returns ``{model_name: best_params}`` and writes
    ``{output_dir}/{tier}/h{horizon}d/frozen_hparams_{model}.json``.
    """
    output_dir = output_dir or HPARAMS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, _, feature_dates, target_dates = get_tuning_sample(
        df, target_col, target_avail_col, availability_col=availability_col
    )
    if len(y) == 0:
        raise RuntimeError("tuning sample is empty — check your data")

    cv_splits = _time_series_splits(feature_dates, target_dates, n_splits=5)
    if not cv_splits:
        raise RuntimeError("all inner CV splits were empty after label purge")
    log.info("inner CV: %d purged splits  train sizes=%s  val sizes=%s",
             len(cv_splits), [len(tr) for tr, _ in cv_splits],
             [len(vl) for _, vl in cv_splits])

    results: dict[str, dict[str, Any]] = {}
    model_fns = {
        "ridge": tune_ridge,
        "lightgbm": tune_lightgbm,
        "xgboost": tune_xgboost,
    }
    names = list(models) if models is not None else list(model_fns)

    for name in names:
        if name not in model_fns:
            raise ValueError(f"unknown model: {name}")
        fn = model_fns[name]
        t0 = time.time()
        log.info("tuning %s …", name)
        params = fn(X, y, cv_splits)
        log.info("%s done (%.0fs): %s", name, time.time() - t0, params)
        results[name] = params
        _write_frozen_hparams(name, params, output_dir, tier=tier, horizon=horizon)

    return results


# ---------------------------------------------------------------------------
# Frozen-hparams persistence
# ---------------------------------------------------------------------------

def _write_frozen_hparams(
    model_name: str,
    params: dict[str, Any],
    output_dir: Path,
    tier: str = "enhanced",
    horizon: int = 5,
) -> Path:
    """Write frozen hyperparameters to a tier x horizon-specific path.

    Path: ``{output_dir}/{tier}/h{horizon}d/frozen_hparams_{model_name}.json``.
    """
    path = output_dir / tier / f"h{horizon}d" / f"frozen_hparams_{model_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _json_safe(params)
    path.write_text(json.dumps(clean, indent=2))
    log.info("wrote %s", path)

    # Update the hparams manifest
    _update_hparams_manifest(output_dir, tier, horizon, model_name, clean)
    return path


def load_frozen_hparams(
    model_name: str,
    hparams_dir: Path | None = None,
    tier: str = "enhanced",
    horizon: int = 5,
) -> dict[str, Any]:
    """Load frozen hyperparameters from a tier x horizon-specific path.

    Path: ``{hparams_dir}/{tier}/h{horizon}d/frozen_hparams_{model_name}.json``.
    """
    d = hparams_dir or HPARAMS_DIR
    path = d / tier / f"h{horizon}d" / f"frozen_hparams_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"frozen hparams not found: {path}")
    return json.loads(path.read_text())


def _update_hparams_manifest(
    hparams_dir: Path,
    tier: str,
    horizon: int,
    model_name: str,
    params: dict[str, Any],
) -> Path:
    """Append an entry to ``results/hparams/hparams_manifest.json``.

    Creates the file with metadata if it does not yet exist.
    """
    manifest_path = hparams_dir / "hparams_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "entries": [],
            "created_at": utc_now_iso(),
        }

    # Source fingerprint: git commit hash (best-effort).
    git_hash = ""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    entry: dict[str, Any] = {
        "model": model_name,
        "tier": tier,
        "horizon": f"{horizon}d",
        "cv_ic": _json_safe(params.get("cv_ic", None)),
        "git_hash": git_hash,
        "created_at": utc_now_iso(),
    }
    # Include sanitised params (exclude cv_ic which is already top-level).
    clean_params = _json_safe({k: v for k, v in params.items() if k != "cv_ic"})
    entry["params"] = clean_params

    manifest["entries"].append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2))
    log.info("updated hparams manifest: %s", manifest_path)
    return manifest_path


def write_hparams_manifest(
    hparams_dir: Path | None = None,
) -> Path:
    """Explicitly write or update the full hparams manifest.

    Scans the hparams directory for all frozen_hparams_*.json files and
    rebuilds the manifest. Useful for batch updates after tuning all models.
    """
    d = hparams_dir or HPARAMS_DIR
    if not d.exists():
        raise FileNotFoundError(f"hparams dir not found: {d}")

    entries: list[dict[str, Any]] = []
    # Iterate over all tier/horizon/model combinations.
    for tier_path in sorted(d.iterdir()):
        if not tier_path.is_dir() or tier_path.name.startswith("."):
            continue
        tier = tier_path.name
        for horizon_path in sorted(tier_path.iterdir()):
            if not horizon_path.is_dir():
                continue
            # Extract horizon number from "h{N}d" directory name.
            m = re.match(r"^h(\d+)d$", horizon_path.name)
            if not m:
                continue
            horizon = int(m.group(1))
            for json_path in sorted(horizon_path.glob("frozen_hparams_*.json")):
                model_name = json_path.stem.replace("frozen_hparams_", "")
                params = json.loads(json_path.read_text())
                entry: dict[str, Any] = {
                    "model": model_name,
                    "tier": tier,
                    "horizon": f"{horizon}d",
                    "cv_ic": params.pop("cv_ic", None),
                }
                entry["params"] = params
                entries.append(entry)

    manifest: dict[str, Any] = {
        "entries": entries,
        "created_at": utc_now_iso(),
        "n_entries": len(entries),
    }
    manifest_path = d / "hparams_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2))
    log.info("wrote hparams manifest: %s (%d entries)", manifest_path, len(entries))
    return manifest_path


# ---------------------------------------------------------------------------
# Fold audit manifest
# ---------------------------------------------------------------------------

def write_fold_manifest(
    folds: list[Fold],
    output_path: Path | None = None,
    model: str = "",
    tier: str = "",
    horizons: list[int] | None = None,
) -> Path:
    """Persist fold metadata to ``results/audit/fold_manifest.parquet``.

    When *model* and *tier* are provided they are included as columns in
    every row so the manifest identifies which model/tier combination produced
    the fold boundaries.  *horizons* controls which target-date columns appear.
    """
    output_path = output_path or (AUDIT_DIR / "fold_manifest.parquet")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    h_list = horizons or HORIZONS
    rows = []
    for f in folds:
        row: dict[str, Any] = {
            "fold_id": f.fold_id,
            "train_start": str(f.train_start.date()),
            "train_end": str(f.train_end.date()),
            "test_start": str(f.test_start.date()),
            "test_end": str(f.test_end.date()),
            "n_train": len(f.train_indices),
            "n_test": len(f.test_indices),
        }
        if model:
            row["model"] = model
        if tier:
            row["tier"] = tier
        for h in h_list:
            d = f.max_train_target_date.get(h)
            row[f"max_train_target_date_{h}d"] = str(d.date()) if d else ""
        rows.append(row)

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("wrote fold manifest: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Sample-size audit
# ---------------------------------------------------------------------------

def write_sample_size_audit(
    df: pd.DataFrame,
    folds: list[Fold],
    output_path: Path | None = None,
    universe: str = "",
    signal_type: str = "",
) -> Path:
    """Write per-quarter event / tradeable / censored counts to parquet.

    For each fold the test events are broken down by calendar quarter with
    columns: *n_events*, *n_tradeable*, *censored_{h}d*, and *low_sample_flag*
    (True when *n_events* < 100).

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame (must contain ``call_entry_date``,
        ``forward_return_{h}d``, and ``right_censored_{h}d`` columns).
    folds : list[Fold]
        Walk-forward folds.
    output_path : Path, optional
        Parquet output path. Defaults to ``AUDIT_DIR / sample_size_by_quarter.parquet``.
    universe : str
        Universe label (e.g. ``"sp500"``).
    signal_type : str
        Signal type (e.g. ``"Total"``).
    """
    output_path = output_path or (AUDIT_DIR / "sample_size_by_quarter.parquet")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in folds:
        test_df = df.iloc[f.test_indices].copy()
        test_df["_quarter"] = pd.to_datetime(test_df["call_entry_date"]).dt.to_period("Q")

        for qtr, qtr_df in test_df.groupby("_quarter"):
            n_events = len(qtr_df)
            n_tradeable = int(qtr_df["forward_return_1d"].notna().sum())
            censored: dict[str, int] = {}
            for h in HORIZONS:
                rc_col = f"right_censored_{h}d"
                if rc_col in qtr_df.columns:
                    n_censored = int(qtr_df[rc_col].sum())
                else:
                    dcol = f"target_available_date_{h}d"
                    if dcol in qtr_df.columns:
                        targets = pd.to_datetime(qtr_df[dcol].values)
                        n_censored = int(targets.isna().sum())
                    else:
                        n_censored = -1
                censored[f"censored_{h}d"] = n_censored

            row = {
                "universe": universe,
                "signal_type": signal_type,
                "fold_id": f.fold_id,
                "quarter": str(qtr),
                "n_events": n_events,
                "n_tradeable": n_tradeable,
                "low_sample_flag": n_events < 100,
                **censored,
            }
            rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_parquet(output_path, index=False)
    log.info("wrote sample size audit: %s  (universe=%s, signal_type=%s)",
             output_path, universe, signal_type)
    return output_path


# ---------------------------------------------------------------------------
# Forward-return cache orchestration
# ---------------------------------------------------------------------------

# Note: FORWARD_RETURNS_CACHE_DIR is defined above near the cache
# infrastructure, alongside ``ForwardReturnsCacheSignature`` and the
# ``_compute_forward_returns_cached`` function that both use it.


def _forward_returns_cache_path(
    features_path: Path,
    horizons: Sequence[int],
    entry_date_col: str = "availability_date",
) -> Path:
    """Return the joblib-managed cache subdirectory for forward returns.

    This path is informational; ``joblib.Memory`` manages its own cache
    structure inside this directory.  The path is deterministic (same inputs
    produce the same subdirectory) for manual inspection.
    """
    h_str = "_".join(str(h) for h in sorted(horizons))
    stem = Path(features_path).stem
    return (
        FORWARD_RETURNS_CACHE_DIR
        / "joblib"
        / f"forward_returns_{stem}_h{h_str}_{entry_date_col}_{AVAILABILITY_RULE_VERSION}"
    )


def get_forward_returns_cached(
    features_path: Path,
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    horizons: Sequence[int] = HORIZONS,
    entry_date_col: str = "availability_date",
) -> pd.DataFrame:
    """Return forward returns for *df*, using ``joblib.Memory`` caching with
    explicit dependency signatures for automatic invalidation.

    The cache is invalidated when any of the following change:
    - Features parquet file (size or mtime)
    - Price manifest file (size, mtime, or SHA-256 content hash)
    - Source code of ``compute_forward_returns`` (via ``inspect.getsource``)
    - ``ForwardReturnsCacheSignature`` schema version
    - Requested horizons or ``entry_date_col``

    The returned DataFrame has the same row count and order as *df* and
    contains exactly the columns produced by ``compute_forward_returns``.

    Parameters
    ----------
    entry_date_col:
        Column used as the entry date. Passed through to
        ``compute_forward_returns`` and included in the cache key.
    """
    features_path = Path(features_path)

    # Build the dependency signature (includes df fingerprint)
    signature = _build_cache_signature(
        features_path, price_cache_dir, horizons, entry_date_col, df,
    )

    # Call the joblib-cached function.  joblib compares the signature fields
    # to decide cache hit vs miss.  The ``df`` argument is excluded from the
    # cache key (``ignore=["df"]``) because it is fully determined by the
    # dependency fields in *signature*.
    fwd = _compute_forward_returns_cached(signature, df, price_cache_dir)

    # Defense-in-depth: verify the cached result matches the caller's DataFrame.
    # This catches any case where the signature fields happen to collide for
    # different DataFrames (virtually impossible but we check anyway).
    if len(df) != signature.len_df:
        raise ValueError(
            f"DataFrame length mismatch in cached forward returns: "
            f"caller passed {len(df)} rows, cache built for {signature.len_df} rows. "
            "This likely means ensure_forward_returns was called on a "
            "filtered/subset DataFrame, which is not supported."
        )
    df_fprint = int(hashlib.sha256(str(tuple(df.index[:1000])).encode()).hexdigest()[:16], 16)
    if df_fprint != signature.row_fingerprint:
        raise ValueError(
            f"DataFrame row fingerprint mismatch in cached forward returns. "
            f"Caller fingerprint: {df_fprint}, cached: {signature.row_fingerprint}. "
            "Cache collided for different DataFrames with the same features_path."
        )

    # Write the human-readable manifest (always, even on cache hit)
    _write_forward_returns_manifest(signature)
    log.info(
        "forward returns: ready (signature=%s)", signature
    )

    return fwd


def ensure_forward_returns(
    df: pd.DataFrame,
    features_path: Path,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    horizons: Sequence[int] | None = None,
    entry_date_col: str = "availability_date",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return df with requested forward returns and target-available dates.

    CRITICAL: This function must ONLY be called on the full, unfiltered
    DataFrame loaded directly from the features parquet file. Never call
    it on a subset (e.g. after universe filtering or row slicing). The
    underlying get_forward_returns_cached uses features_path as the primary
    cache key and excludes df from the joblib key, so calling with a
    filtered/subset DataFrame with the same features_path will return cached
    forward returns for the full row set — causing silent row misalignment and
    forward-return contamination. To reinforce this, the cache signature
    includes len(df) and a deterministic row-position fingerprint; a
    size/order mismatch between the cached and caller df raises ValueError.

    Parameters
    ----------
    df:
        Full, unfiltered feature DataFrame.
    features_path:
        Feature parquet path used by the cache signature.
    price_cache_dir:
        Per-ticker price parquet directory.
    horizons:
        Requested forward-return horizons. Defaults to global HORIZONS.
    entry_date_col:
        Column used as the entry date for forward returns.
    overwrite:
        When True, recompute even if all requested columns exist.

    Returns
    -------
    DataFrame with requested forward_return_{h}d and target_available_date_{h}d
    columns present. Preserves row order and index.
    """
    if horizons is None:
        horizons = HORIZONS

    # Build the set of column names required for the given horizons
    required_cols: list[str] = []
    for h in horizons:
        required_cols.append(f"forward_return_{h}d")
        required_cols.append(f"target_available_date_{h}d")

    # If all required columns exist and overwrite is False, return unchanged
    if not overwrite:
        missing = [c for c in required_cols if c not in df.columns]
        if not missing:
            return df

    # Compute forward returns from cache
    fwd = get_forward_returns_cached(
        features_path, df, price_cache_dir, horizons, entry_date_col,
    )

    # Drop stale requested columns, then assign new values aligned to row order
    drop_cols = [c for c in required_cols if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    for col in fwd.columns:
        out[col] = fwd[col].to_numpy()
    return out
