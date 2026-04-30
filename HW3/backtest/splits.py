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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
from joblib import Memory
import numpy as np
import pandas as pd

from data.config import PRICE_CACHE_DIR, RESULTS_DIR, AUDIT_DIR, SEED, utc_now_iso
from data.progress import progress

log = logging.getLogger("backtest.splits")

HORIZONS: list[int] = [1, 3, 5, 10, 20]

TUNING_START = pd.Timestamp("2010-01-01")
TUNING_END = pd.Timestamp("2019-12-31")
WF_START = pd.Timestamp("2020-01-01")
WF_END = pd.Timestamp("2026-06-30")

# Columns that must NOT enter the feature matrix.
ID_COLS = {
    "BESTTICKER", "SECTOR", "availability_date", "call_entry_date",
    "ingest_entry_date", "MOSTIMPORTANTDATEUTC", "INGESTDATEUTC",
    "SignalType", "QTR_YEAR", "call_hour_utc", "SECTOR_GICS",
    "_orig_df_index",
}
# Forward-return / target-date columns added by compute_forward_returns.
_RETURN_COL_RE = re.compile(r"^forward_return_\d+d$")
_TARGET_DATE_COL_RE = re.compile(r"^target_available_date_\d+d$")
_RAW_RETURN_COL_RE = re.compile(r"^Return_\d+d$")


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

FORWARD_RETURNS_CACHE_VERSION = 1
"""Schema version for the forward-returns cache signature.
Bump this when the cache signature schema changes (e.g., new dependency
fields added) to force a full recomputation.
"""


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
    """``hash(inspect.getsource(compute_forward_returns))`` — captures code
    changes in the forward-return computation itself."""
    cache_version: int = FORWARD_RETURNS_CACHE_VERSION
    """Schema version — bump to force full cache invalidation."""


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
) -> ForwardReturnsCacheSignature:
    """Build a dependency signature for forward-returns caching.

    Gathers file metadata and content hashes for every input that affects
    the forward-return computation.
    """
    # Features file info
    fs = features_path.stat() if features_path.exists() else None
    features_size = fs.st_size if fs else 0
    features_mtime = fs.st_mtime if fs else 0.0

    # Price manifest file info + content hash
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

    # Source-code hash for the core computation function
    try:
        source = inspect.getsource(compute_forward_returns)
        source_hash = int(hashlib.sha256(source.encode()).hexdigest()[:16], 16)
    except (OSError, TypeError):
        source_hash = 0

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

    def _result_frame() -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for hh in h_list:
            result[f"forward_return_{hh}d"] = ret_arrays[hh]
            result[f"target_available_date_{hh}d"] = date_arrays[hh]
        return result

    if not price_cache_dir.exists():
        log.warning("price cache dir %s missing", price_cache_dir)
        return _result_frame()

    # ------------------------------------------------------------------
    # Build per-ticker event groups once. This avoids scanning the full event
    # table for every ticker and also keeps only one price history in memory.
    # ------------------------------------------------------------------
    events = pd.DataFrame({
        "_orig_pos": np.arange(n, dtype=np.int64),
        "_ticker": df[ticker_col].values,
        "_entry_date": pd.to_datetime(
            df[entry_date_col], errors="coerce"
        ).to_numpy(dtype="datetime64[ns]"),
    })
    events = events[
        events["_ticker"].notna() & events["_entry_date"].notna()
    ].copy()
    if events.empty:
        return _result_frame()
    events["_ticker"] = events["_ticker"].astype(str)

    # ------------------------------------------------------------------
    # Build per-ticker forward-return tables, merge_asof with events
    # ------------------------------------------------------------------
    n_with_prices = 0
    event_groups = events.groupby("_ticker", sort=False)
    for tkr_str, ev in progress(
        event_groups,
        total=events["_ticker"].nunique(),
        desc="forward returns",
        unit="tkr",
    ):
        path = price_cache_dir / f"{tkr_str}.parquet"
        if not path.exists():
            continue

        px = pd.read_parquet(path, columns=["date", "adj_close"])
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px[px["date"].notna() & px["adj_close"].notna()]
        px = px[px["adj_close"] > 0].sort_values("date")
        if px.empty:
            continue

        n_with_prices += 1
        ev = ev.sort_values("_entry_date").copy()
        ev["_entry_date"] = ev["_entry_date"].astype("datetime64[ns]")

        dates = px["date"].to_numpy(dtype="datetime64[ns]")
        closes = px["adj_close"].to_numpy(dtype="float64")
        metrics = pd.DataFrame({
            "_merge_date": dates,
            "_price_pos": np.arange(len(dates), dtype=np.int64),
        })
        for h in h_list:
            returns = np.full(len(closes), np.nan, dtype="float64")
            target_dates = np.full(
                len(dates), np.datetime64("NaT"), dtype="datetime64[ns]"
            )
            if len(closes) > h:
                returns[:-h] = closes[h:] / closes[:-h] - 1.0
                target_dates[:-h] = dates[h:]
            metrics[f"forward_return_{h}d"] = returns
            metrics[f"target_available_date_{h}d"] = target_dates

        mg = pd.merge_asof(
            ev, metrics,
            left_on="_entry_date", right_on="_merge_date",
            direction="forward",
        )
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
        entry_valid = (
            mg["_price_pos"].notna()
            & np.isfinite(entry_gap_bdays)
            & (entry_gap_bdays >= 0)
            & (entry_gap_bdays <= 3)
        )

        for h in h_list:
            ret_col = f"forward_return_{h}d"
            date_col = f"target_available_date_{h}d"
            valid = entry_valid & mg[ret_col].notna()
            if not valid.any():
                continue
            orig_pos = mg.loc[valid, "_orig_pos"].to_numpy(dtype=np.int64)
            ret_arrays[h][orig_pos] = mg.loc[valid, ret_col].to_numpy(
                dtype="float64"
            )
            date_arrays[h][orig_pos] = pd.to_datetime(
                mg.loc[valid, date_col], errors="coerce"
            ).to_numpy(dtype="datetime64[ns]")

    if n_with_prices == 0:
        log.warning("no price data available — forward returns are all NaN")
    return _result_frame()


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
    - **test**: events whose ``call_entry_date`` is inside the quarter.
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
    df = df.reset_index(drop=True)
    call_entry = pd.to_datetime(df["call_entry_date"].values)
    avail_date = pd.to_datetime(df[availability_col].values)

    quarters = _quarter_boundaries(start, end)
    folds: list[Fold] = []

    for fold_id, (q_start, q_end) in enumerate(quarters):
        test_mask = (
            (call_entry >= q_start.to_datetime64())
            & (call_entry <= q_end.to_datetime64())
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


def _make_median_imputer() -> Any:
    from sklearn.impute import SimpleImputer

    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older scikit-learn fallback
        return SimpleImputer(strategy="median")


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

    imp = _make_median_imputer()
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

def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Grid-search Ridge alpha.  Spearman IC is the objective."""
    from sklearn.linear_model import Ridge

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    scores_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}

    for tr, vl in progress(cv_splits, desc="ridge tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=True)
        for alpha in alphas:
            m = Ridge(alpha=alpha, random_state=SEED)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_vl)
            scores_by_alpha[alpha].append(_spearman(pred, y[vl]))

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

    scores_by_trial: list[list[float]] = [[] for _ in trials]
    for tr, vl in progress(cv_splits, desc="lgbm tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=False)
        for i, params in enumerate(trials):
            m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1, n_jobs=1)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_vl)
            scores_by_trial[i].append(_spearman(pred, y[vl]))

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

    scores_by_trial: list[list[float]] = [[] for _ in trials]
    for tr, vl in progress(cv_splits, desc="xgb tuning", unit="fold"):
        X_tr, X_vl = _preprocess_train_val(X, tr, vl, scale=False)
        for i, params in enumerate(trials):
            m = xgb.XGBRegressor(**params, random_state=SEED, verbosity=0, n_jobs=1)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_vl)
            scores_by_trial[i].append(_spearman(pred, y[vl]))

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
        combo = {k: rng.choice(space[k]) for k in keys}
        combos.append(combo)
    return combos


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-d arrays."""
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


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
    # Sanitise numpy types for JSON
    clean = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    path.write_text(json.dumps(clean, indent=2))
    log.info("wrote %s", path)

    # Update the hparams manifest
    _update_hparams_manifest(output_dir, tier, horizon, model_name, params)
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
        "cv_ic": params.get("cv_ic", None),
        "git_hash": git_hash,
        "created_at": utc_now_iso(),
    }
    # Include sanitised params (exclude cv_ic which is already top-level).
    clean_params = {k: v for k, v in params.items() if k != "cv_ic"}
    entry["params"] = clean_params

    manifest["entries"].append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
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
    manifest_path.write_text(json.dumps(manifest, indent=2))
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
) -> Path:
    """Write per-quarter event / tradeable / censored counts to parquet."""
    output_path = output_path or (AUDIT_DIR / "sample_size_by_quarter.parquet")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    call_entry = pd.to_datetime(df["call_entry_date"].values)

    rows = []
    for f in folds:
        n_test = len(f.test_indices)
        # count censored (right-censored targets) per horizon
        censored: dict[str, int] = {}
        for h in HORIZONS:
            dcol = f"target_available_date_{h}d"
            if dcol in df.columns:
                targets = pd.to_datetime(df[dcol].iloc[f.test_indices].values)
                n_censored = int((targets.isna()).sum())
            else:
                n_censored = -1
            censored[f"censored_{h}d"] = n_censored

        row = {
            "fold_id": f.fold_id,
            "quarter_start": str(f.test_start.date()),
            "quarter_end": str(f.test_end.date()),
            "n_test_events": n_test,
            "n_train_events": len(f.train_indices),
            **censored,
        }
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_parquet(output_path, index=False)
    log.info("wrote sample size audit: %s", output_path)
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

    # Build the dependency signature
    signature = _build_cache_signature(
        features_path, price_cache_dir, horizons, entry_date_col
    )

    # Call the joblib-cached function.  joblib compares the signature fields
    # to decide cache hit vs miss.  The ``df`` argument is excluded from the
    # cache key (``ignore=["df"]``) because it is fully determined by the
    # dependency fields in *signature*.
    fwd = _compute_forward_returns_cached(signature, df, price_cache_dir)

    # Write the human-readable manifest (always, even on cache hit)
    _write_forward_returns_manifest(signature)
    log.info(
        "forward returns: ready (signature=%s)", signature
    )

    return fwd
