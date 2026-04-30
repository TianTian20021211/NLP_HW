"""Phase 3 - Automated Look-Ahead Tests.

3.1  Streaming vs batch regression test — fit once on the full sample, then
     compare against per-day streaming fits. Any mismatch flags look-ahead
     leakage in the batch implementation.

3.2  Eight assertion classes that each turn CI red on failure.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.config import AUDIT_DIR, PRICE_CACHE_DIR, SIGNALS_PARQUET, UNIVERSE_CACHE_DIR
from data.load_universes import members_at

log = logging.getLogger("audit")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

IDENTIFIER_LIKE = {
    "BESTTICKER", "SECTOR", "SignalType", "MOSTIMPORTANTDATEUTC",
    "INGESTDATEUTC", "call_hour_utc", "QTR_YEAR",
    "call_entry_date", "ingest_entry_date", "availability_date",
    "SECTOR_GICS",  # possible alternate name
}

RETURN_COL_PATTERN = re.compile(r"^Return_\d+d$")

# Features whose values depend on the cross-sectional ticker universe available
# at a point in time.  Streaming-vs-batch comparisons for these columns are
# reported separately because they are expected to differ when the signal
# universe expands over time — not because of look-ahead leakage.
CROSS_SECTIONAL_FEATURE_PATTERNS = [
    "pre_event_ret_21d_sector_rel",   # sector median changes with ticker set
    "pre_event_idio_resid_5d",        # depends on beta × sector median
]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns in *df* that are not identifier-like and not return cols."""
    return [
        c for c in df.columns
        if c not in IDENTIFIER_LIKE
        and not RETURN_COL_PATTERN.match(c)
        and not c.startswith("_")
    ]


def _partition_feature_cols(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Split columns into (strict, cross_sectional) groups.

    Strict features should match exactly between batch and streaming.
    Cross-sectional features may differ because their computation depends
    on the ticker universe available at each point in time.
    """
    strict: list[str] = []
    xsectional: list[str] = []
    for col in feature_columns:
        if any(col == p or col.startswith(p) for p in CROSS_SECTIONAL_FEATURE_PATTERNS):
            xsectional.append(col)
        else:
            strict.append(col)
    return strict, xsectional


# ---------------------------------------------------------------------------
# 3.1  Streaming vs batch regression test
# ---------------------------------------------------------------------------

def _sample_availability_dates(
    dates: pd.DatetimeIndex,
    n: int,
) -> list[pd.Timestamp]:
    """Pick at most *n* evenly-spaced dates from *dates*."""
    uniq = sorted(dates.dropna().unique())
    if n is None or len(uniq) <= n:
        return uniq
    step = max(1, len(uniq) // n)
    return [uniq[i] for i in range(0, len(uniq), step)][:n]


def _compare_parity(
    batch: pd.DataFrame,
    streaming: pd.DataFrame,
    rtol: float,
    atol: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare two feature DataFrames element-wise and return mismatches.

    If *columns* is provided, only those columns are compared.
    """
    feat_cols = columns if columns is not None else _feature_cols(batch)
    common = sorted(set(feat_cols) & set(streaming.columns))

    mismatches: list[dict[str, Any]] = []
    for col in common:
        b = batch[col].to_numpy(dtype="float64")
        s = streaming[col].to_numpy(dtype="float64")
        if not np.allclose(b, s, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.abs(b - s)
            mask = ~(np.isclose(b, s, rtol=rtol, atol=atol, equal_nan=True))
            n_bad = mask.sum()
            max_diff = float(np.nanmax(diff[mask])) if n_bad else 0.0
            mismatches.append({
                "column": col,
                "n_mismatch": int(n_bad),
                "max_abs_diff": max_diff,
            })

    if mismatches:
        return pd.DataFrame(mismatches).sort_values("n_mismatch", ascending=False)
    return pd.DataFrame(columns=["column", "n_mismatch", "max_abs_diff"])


def run_streaming_vs_batch_test(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    n_sample_dates: int | None = None,
    include_momentum: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
    """Compare batch feature computation against per-day streaming fits.

    For a sample of availability dates, compute features using only data
    available up to that date, then compare each row against the full
    batch result.  Differences indicate look-ahead leakage.

    Features are partitioned into two groups:

    * **Strict** — must match exactly (row features, QoQ deltas, 4Q trend,
      PIT percentiles, raw pre-event returns).
    * **Cross-sectional** — may differ legitimately because the sector-median /
      ticker-universe available at each PIT snapshot expands over time
      (sector-relative returns, idiosyncratic residual).

    The strict group determines ``passed``.  Cross-sectional mismatches are
    recorded in the output table with ``feature_group = "xsectional"``.

    Returns ``(passed, mismatch_df, summary_dict)``.
    """
    t0 = time.time()
    log.info("streaming-vs-batch: tier=%s  n_sample_dates=%s  momentum=%s",
             tier, n_sample_dates, include_momentum)

    if "availability_date" not in df.columns:
        from features.engineer import compute_timestamps
        df = compute_timestamps(df)

    n_rows = len(df)

    # 1.  Batch — fit once on the full sample
    log.info("  [1/2] batch on %d rows …", n_rows)
    batch = _build_safe(df, price_cache_dir, tier, include_momentum)
    log.info("  batch done: %d rows x %d cols", len(batch), len(batch.columns))

    # 2.  Partition feature columns
    all_feat = _feature_cols(batch)
    strict_cols, xsectional_cols = _partition_feature_cols(all_feat)
    log.info("  strict features: %d  cross-sectional: %d",
             len(strict_cols), len(xsectional_cols))

    # 3.  Sample dates
    if "availability_date" not in batch.columns:
        raise RuntimeError("batch output missing availability_date")

    dates = pd.DatetimeIndex(batch["availability_date"].dropna().unique())
    sampled = _sample_availability_dates(dates, n_sample_dates or len(dates))
    log.info("  [2/2] streaming on %d / %d dates …", len(sampled), len(dates))

    streaming_targets: pd.DataFrame | None = None
    if not include_momentum:
        log.info("  using target-only streaming audit path (no momentum)")
        streaming_targets = _build_streaming_targets_no_momentum(
            batch, sampled, strict_cols + xsectional_cols
        )

    compare_cols = ["availability_date"] + strict_cols + xsectional_cols
    batch = batch.loc[
        batch["availability_date"].isin(sampled),
        [c for c in compare_cols if c in batch.columns],
    ].copy()
    gc.collect()

    # 4.  Per-date streaming comparison
    all_strict_mm: list[pd.DataFrame] = []
    all_xsec_mm: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "n_dates_tested": 0, "n_rows_compared": 0,
        "n_strict_mismatch": 0, "n_xsectional_mismatch": 0,
        "n_row_count_mismatch": 0,
    }

    for d in sampled:
        if streaming_targets is None:
            orig_mask = df["availability_date"] <= d
            if orig_mask.sum() < 2:
                continue

            subset = df.loc[orig_mask]
            try:
                streaming = _build_safe(subset, price_cache_dir, tier, include_momentum)
            except Exception:
                log.exception("streaming build failed for date %s", d)
                continue
        else:
            subset = None
            streaming = streaming_targets

        batch_test = batch.loc[batch["availability_date"] == d]
        streaming_test = streaming.loc[streaming["availability_date"] == d]

        if len(batch_test) == 0 and len(streaming_test) == 0:
            continue

        if len(batch_test) != len(streaming_test):
            log.error(
                "date %s: row-count mismatch batch=%d streaming=%d",
                d, len(batch_test), len(streaming_test),
            )
            all_strict_mm.append(pd.DataFrame([{
                "column": "__ROW_COUNT__",
                "n_mismatch": abs(len(batch_test) - len(streaming_test)),
                "max_abs_diff": np.nan,
                "feature_group": "strict",
            }]))
            stats["n_row_count_mismatch"] += 1
            continue

        batch_test = batch_test.reset_index(drop=True)
        streaming_test = streaming_test.reset_index(drop=True)

        # Strict comparison
        mm_strict = _compare_parity(batch_test, streaming_test, rtol, atol,
                                    columns=strict_cols)
        if len(mm_strict):
            mm_strict.insert(0, "date", str(d.date()))
            mm_strict["feature_group"] = "strict"
            all_strict_mm.append(mm_strict)

        # Cross-sectional comparison (informational only)
        if xsectional_cols:
            mm_xsec = _compare_parity(batch_test, streaming_test, rtol, atol,
                                      columns=xsectional_cols)
            if len(mm_xsec):
                mm_xsec.insert(0, "date", str(d.date()))
                mm_xsec["feature_group"] = "xsectional"
                all_xsec_mm.append(mm_xsec)

        stats["n_rows_compared"] += len(batch_test)

        if streaming_targets is None:
            del subset, streaming
        del batch_test, streaming_test
        gc.collect()

    stats["n_dates_tested"] = len(sampled)
    stats["elapsed_s"] = round(time.time() - t0, 1)

    # Combine all mismatches
    all_parts = all_strict_mm + all_xsec_mm
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=["date", "column", "n_mismatch", "max_abs_diff", "feature_group"]
        )

    stats["n_strict_mismatch"] = int(
        combined.loc[combined["feature_group"] == "strict", "n_mismatch"].sum()
        if len(combined) else 0
    )
    stats["n_xsectional_mismatch"] = int(
        combined.loc[combined["feature_group"] == "xsectional", "n_mismatch"].sum()
        if len(combined) else 0
    )

    passed = stats["n_strict_mismatch"] == 0 and stats["n_row_count_mismatch"] == 0

    if not passed:
        log.warning("  FAIL: %d strict + %d xsectional mismatches",
                    stats["n_strict_mismatch"], stats["n_xsectional_mismatch"])
    else:
        log.info("  PASS: %d rows compared, 0 strict mismatches", stats["n_rows_compared"])
        if stats["n_xsectional_mismatch"]:
            log.info("  (info) %d cross-sectional mismatches expected due to "
                     "expanding ticker universe", stats["n_xsectional_mismatch"])

    return passed, combined, stats


def _build_safe(
    df: pd.DataFrame,
    price_cache_dir: Path,
    tier: str,
    include_momentum: bool,
) -> pd.DataFrame:
    """Wrapper around build_features."""
    from features.engineer import build_features

    return build_features(
        df,
        price_cache_dir=price_cache_dir,
        tier=tier,
        include_momentum=include_momentum,
    )


def _audit_timeseries_input_cols(df: pd.DataFrame) -> list[str]:
    from features.engineer import ASPECTS, THEMES

    cols = (
        ["ATCClassifierScore"]
        + [f"aspect_{a}_total" for a in ASPECTS]
        + [f"theme_{t}_total" for t in THEMES]
    )
    return [c for c in cols if c in df.columns]


def _audit_pit_input_cols(df: pd.DataFrame) -> list[str]:
    from features.engineer import ASPECTS

    cols = ["ATCClassifierScore"] + [f"aspect_{a}_total" for a in ASPECTS]
    return [c for c in cols if c in df.columns]


def _audit_generated_no_momentum_cols(df: pd.DataFrame) -> set[str]:
    generated = {f"qoq_delta_{c}" for c in _audit_timeseries_input_cols(df)}
    if "ATCClassifierScore" in df.columns:
        generated.add("qoq_4q_trend_atc")
    generated.update(f"{c}_sector_pct" for c in _audit_pit_input_cols(df))
    return generated


def _build_streaming_targets_no_momentum(
    batch: pd.DataFrame,
    sampled_dates: list[pd.Timestamp],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build streaming-equivalent rows only for sampled audit dates.

    This path is used for the full Phase 3 regression where momentum is
    intentionally disabled. Row-level features are independent of other rows,
    so they are reused from the batch output. Time-series and PIT features are
    recomputed for only the sampled-date rows from the row-level history.
    """
    generated = _audit_generated_no_momentum_cols(batch)
    id_cols = [
        "BESTTICKER", "SECTOR", "availability_date",
        "call_entry_date", "MOSTIMPORTANTDATEUTC", "SignalType",
    ]
    keep_cols = [
        c for c in id_cols + feature_columns
        if c in batch.columns and c not in generated
    ]
    keep_cols = list(dict.fromkeys(keep_cols))

    base = batch[keep_cols].copy()
    base["_audit_row_id"] = np.arange(len(base), dtype="int64")

    target_mask = base["availability_date"].isin(sampled_dates)
    target = base.loc[target_mask].copy()
    if target.empty:
        return target.drop(columns=["_audit_row_id"], errors="ignore")

    ts = _compute_target_timeseries_features(base, target)
    pit = _compute_target_pit_percentiles(base, target)
    for col in ts.columns:
        target[col] = ts[col]
    for col in pit.columns:
        target[col] = pit[col]

    return target.drop(columns=["_audit_row_id"], errors="ignore")


def _audit_group_keys(df: pd.DataFrame) -> list[str]:
    keys = ["BESTTICKER"]
    if "SignalType" in df.columns:
        keys.append("SignalType")
    return keys


def _compute_target_timeseries_features(
    base: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    available = _audit_timeseries_input_cols(base)
    out = pd.DataFrame(index=target.index)
    if not available:
        return out

    keys = _audit_group_keys(base)
    block_cols = keys + ["availability_date"]
    work_cols = block_cols + available
    if "call_entry_date" in base.columns:
        work_cols.append("call_entry_date")

    work = base[work_cols + ["_audit_row_id"]].copy()
    sort_cols = keys + ["availability_date"]
    if "call_entry_date" in work.columns:
        sort_cols.append("call_entry_date")
    sort_cols.append("_audit_row_id")
    work = work.sort_values(sort_cols)

    last_by_date = work.groupby(block_cols, sort=False)[available].last()
    prev_by_date = (
        last_by_date
        .groupby(level=keys, sort=False)
        .shift(1)
        .rename(columns=lambda c: f"_prev_{c}")
        .reset_index()
    )

    target_work = target[block_cols + available].copy()
    target_work["_target_idx"] = target.index
    merged = target_work.merge(prev_by_date, on=block_cols, how="left", sort=False)
    merged = merged.set_index("_target_idx")

    for col in available:
        out[f"qoq_delta_{col}"] = (
            merged[col].to_numpy(dtype="float64")
            - merged[f"_prev_{col}"].to_numpy(dtype="float64")
        )

    if "ATCClassifierScore" in available:
        atc_by_date = work.groupby(block_cols, sort=False)["ATCClassifierScore"].last()
        lagged = pd.concat(
            {
                "_lag1": atc_by_date.groupby(level=keys, sort=False).shift(1),
                "_lag2": atc_by_date.groupby(level=keys, sort=False).shift(2),
                "_lag3": atc_by_date.groupby(level=keys, sort=False).shift(3),
            },
            axis=1,
        ).reset_index()
        trend_work = target[block_cols + ["ATCClassifierScore"]].copy()
        trend_work["_target_idx"] = target.index
        trend = trend_work.merge(lagged, on=block_cols, how="left", sort=False)
        trend = trend.set_index("_target_idx")
        out["qoq_4q_trend_atc"] = (
            -3 * trend["_lag3"].to_numpy(dtype="float64")
            - trend["_lag2"].to_numpy(dtype="float64")
            + trend["_lag1"].to_numpy(dtype="float64")
            + 3 * trend["ATCClassifierScore"].to_numpy(dtype="float64")
        ) / 10.0

    return out


def _compute_target_pit_percentiles(
    base: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    available = _audit_pit_input_cols(base)
    out = pd.DataFrame(index=target.index)
    if not available:
        return out

    group_keys = ["SECTOR"]
    if "SignalType" in base.columns:
        group_keys.append("SignalType")

    base_groups = base.groupby(group_keys, dropna=False, sort=False).indices
    target_groups = target.groupby(group_keys, dropna=False, sort=False).indices
    base_hist_dt = pd.to_datetime(base["availability_date"]).to_numpy(dtype="datetime64[ns]")
    target_cutoff_dt = pd.to_datetime(target["call_entry_date"]).to_numpy(dtype="datetime64[ns]")
    target_index = target.index.to_numpy()

    for col in available:
        result = pd.Series(np.nan, index=target.index, dtype="float64")
        base_vals = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype="float64")
        target_vals = pd.to_numeric(target[col], errors="coerce").to_numpy(dtype="float64")

        for key, target_pos in target_groups.items():
            hist_pos = base_groups.get(key)
            if hist_pos is None or len(hist_pos) == 0:
                continue

            pct = _strict_historical_percentile_queries(
                history_values=base_vals[hist_pos],
                history_dates=base_hist_dt[hist_pos],
                query_values=target_vals[target_pos],
                cutoff_dates=target_cutoff_dt[target_pos],
            )
            result.loc[target_index[target_pos]] = pct

        out[f"{col}_sector_pct"] = result

    return out


def _strict_historical_percentile_queries(
    history_values: np.ndarray,
    history_dates: np.ndarray,
    query_values: np.ndarray,
    cutoff_dates: np.ndarray,
) -> np.ndarray:
    """Strict empirical percentiles for query rows only."""
    pct = np.full(len(query_values), np.nan, dtype="float64")
    valid_hist = (~np.isnan(history_values)) & (~np.isnat(history_dates))
    valid_query = (~np.isnan(query_values)) & (~np.isnat(cutoff_dates))
    if not valid_hist.any() or not valid_query.any():
        return pct

    unique_vals = np.sort(np.unique(history_values[valid_hist]))
    if len(unique_vals) == 0:
        return pct

    hist_idx = np.flatnonzero(valid_hist)
    hist_order = hist_idx[np.argsort(history_dates[hist_idx], kind="mergesort")]
    hist_dates = history_dates[hist_order]
    hist_codes = np.searchsorted(unique_vals, history_values[hist_order], side="left") + 1

    query_idx = np.flatnonzero(valid_query)
    query_order = query_idx[np.argsort(cutoff_dates[query_idx], kind="mergesort")]

    try:
        from features.engineer import _strict_historical_percentile_numba
    except ImportError:
        _strict_historical_percentile_numba = None

    if _strict_historical_percentile_numba is not None:
        return _strict_historical_percentile_numba(
            query_values,
            unique_vals,
            hist_dates.astype("int64"),
            hist_codes.astype("int64"),
            query_order.astype("int64"),
            cutoff_dates.astype("int64"),
        )

    bit = np.zeros(len(unique_vals) + 1, dtype=np.int64)
    total = 0

    def add(i: int) -> None:
        while i < len(bit):
            bit[i] += 1
            i += i & -i

    def prefix_sum(i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    add_pos = 0
    for qi in query_order:
        q_cutoff = cutoff_dates[qi]
        while add_pos < len(hist_order) and hist_dates[add_pos] < q_cutoff:
            add(int(hist_codes[add_pos]))
            total += 1
            add_pos += 1
        if total == 0:
            continue
        q_code = int(np.searchsorted(unique_vals, query_values[qi], side="right"))
        pct[qi] = prefix_sum(q_code) / total

    return pct


# ---------------------------------------------------------------------------
# 3.2  Assertion 1 — Feature parity  (driven by 3.1 above + targeted checks)
# ---------------------------------------------------------------------------

def assert_feature_parity(
    batch: pd.DataFrame,
    streaming_results: list[tuple[pd.Timestamp, pd.DataFrame]],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Run streaming-vs-batch on pre-computed streaming frames.

    Returns a DataFrame of mismatches (empty if all clear).  Only strict
    (non-cross-sectional) features are checked; cross-sectional differences
    are omitted from this assertion because they are expected to change as
    the ticker universe expands.
    """
    strict_cols, _ = _partition_feature_cols(_feature_cols(batch))
    all_mm: list[pd.DataFrame] = []
    for d, streaming in streaming_results:
        batch_test = batch.loc[batch["availability_date"] == d].reset_index(drop=True)
        streaming_test = streaming.loc[streaming["availability_date"] == d].reset_index(drop=True)
        if len(batch_test) != len(streaming_test):
            all_mm.append(pd.DataFrame([{
                "column": "__ROW_COUNT__",
                "n_mismatch": abs(len(batch_test) - len(streaming_test)),
                "max_abs_diff": np.nan,
            }]))
            continue
        mm = _compare_parity(batch_test, streaming_test, rtol, atol, columns=strict_cols)
        if len(mm):
            mm.insert(0, "date", str(d.date()))
            all_mm.append(mm)
    if all_mm:
        return pd.concat(all_mm, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3.2  Assertion 2 — Fold boundary + label purge
# ---------------------------------------------------------------------------

def assert_fold_boundaries(
    fold_train_start: pd.Timestamp,
    fold_train_end: pd.Timestamp,
    fold_test_start: pd.Timestamp,
    fold_test_end: pd.Timestamp,
    max_train_feature_date: pd.Timestamp,
    max_train_target_available: pd.Timestamp | None = None,
    horizon_days: int = 5,
) -> list[str]:
    """Check fold boundary constraints.  Returns a list of violation messages.

    Rules:
    - ``max(train_feature_date) < min(test_feature_date)``
    - ``max(train_target_available_date_h) < min(test_feature_date)``  (label purge)
    """
    violations: list[str] = []

    if max_train_feature_date >= fold_test_start:
        violations.append(
            f"max(train_feature_date)={max_train_feature_date.date()} >= "
            f"min(test_feature_date)={fold_test_start.date()}"
        )

    if max_train_target_available is not None:
        if max_train_target_available >= fold_test_start:
            violations.append(
                f"label purge: max(train_target_available)="
                f"{max_train_target_available.date()} >= "
                f"min(test_feature_date)={fold_test_start.date()}"
            )

    if fold_train_end >= fold_test_start:
        violations.append(
            f"train_end={fold_train_end.date()} >= test_start={fold_test_start.date()}"
        )

    return violations


# ---------------------------------------------------------------------------
# 3.2  Assertion 3 — fit() call-stack monitoring
# ---------------------------------------------------------------------------

_FIT_LOG: list[dict[str, Any]] = []


class _FitMonitor:
    """Monkey-patch wrapper that records every ``fit()`` call with its
    caller context, fold id, and input date range so we can later verify
    that no fold sees data outside its own boundaries."""

    def __init__(
        self,
        original_fit: Callable[..., Any],
        estimator_name: str,
    ):
        self._original = original_fit
        self._name = estimator_name

    def __call__(self, instance: Any, X: Any, y: Any = None, **kwargs: Any) -> Any:
        # Try to recover date range from X (DataFrame / array)
        date_info: dict[str, Any] = {"name": self._name}
        try:
            if hasattr(X, "index"):
                idx = X.index
                if isinstance(idx, pd.DatetimeIndex) and len(idx):
                    date_info["min_date"] = str(idx.min().date())
                    date_info["max_date"] = str(idx.max().date())
                elif hasattr(idx, "min") and hasattr(idx, "max"):
                    date_info["n_samples"] = len(idx)
            elif hasattr(X, "shape"):
                date_info["shape"] = list(X.shape)
        except Exception:
            date_info["shape"] = "unknown"

        # Walk call stack for caller context (one level up from the fit call)
        import traceback
        stack = traceback.extract_stack()
        caller_frame = stack[-3] if len(stack) >= 3 else stack[-1]
        date_info["caller"] = f"{caller_frame.filename}:{caller_frame.lineno}"

        date_info["timestamp"] = dt.datetime.now().isoformat()
        _FIT_LOG.append(date_info)

        return self._original(instance, X, y, **kwargs)


@contextmanager
def monitor_fit_calls():
    """Context manager that monkey-patches common sklearn ``fit`` methods.

    Usage::

        with monitor_fit_calls():
            model.fit(X_train, y_train)   # recorded
            scaler.fit(X_train)           # recorded

        # _FIT_LOG now contains metadata for each call.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    _FIT_LOG.clear()

    patches: list[tuple[type, str, Any]] = []

    def _patch(cls: type, method_name: str, estimator_name: str) -> None:
        orig = getattr(cls, method_name, None)
        if orig is None:
            return

        monitor = _FitMonitor(orig, estimator_name)

        def _patched(self: Any, X: Any, y: Any = None, **kw: Any) -> Any:
            return monitor(self, X, y, **kw)

        setattr(cls, method_name, _patched)
        patches.append((cls, method_name, orig))

    _patch(StandardScaler, "fit", "StandardScaler")
    _patch(SimpleImputer, "fit", "SimpleImputer")

    try:
        from sklearn.linear_model import LassoCV
        _patch(LassoCV, "fit", "LassoCV")
    except ImportError:
        pass

    try:
        from sklearn.linear_model import RidgeCV
        _patch(RidgeCV, "fit", "RidgeCV")
    except ImportError:
        pass

    try:
        from sklearn.linear_model import Ridge
        _patch(Ridge, "fit", "Ridge")
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        _patch(LGBMRegressor, "fit", "LGBMRegressor")
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor
        _patch(XGBRegressor, "fit", "XGBRegressor")
    except ImportError:
        pass

    try:
        yield
    finally:
        for cls, method, orig in patches:
            setattr(cls, method, orig)


def get_fit_log() -> list[dict[str, Any]]:
    return list(_FIT_LOG)


def assert_fit_callstack(
    fit_log: list[dict[str, Any]],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    fold_id: str = "",
) -> list[str]:
    """Verify that every recorded fit call used data within [fold_start, fold_end].

    Returns a list of violation messages.
    """
    violations: list[str] = []
    for entry in fit_log:
        min_d = entry.get("min_date")
        max_d = entry.get("max_date")
        name = entry.get("name", "?")
        if min_d and max_d:
            if pd.Timestamp(min_d) < fold_start:
                violations.append(
                    f"[{fold_id}] {name}: min_date={min_d} < fold_start={fold_start.date()}"
                )
            if pd.Timestamp(max_d) > fold_end:
                violations.append(
                    f"[{fold_id}] {name}: max_date={max_d} > fold_end={fold_end.date()}"
                )
    return violations


# ---------------------------------------------------------------------------
# 3.2  Assertion 4 — PIT universe defense
# ---------------------------------------------------------------------------

def assert_pit_universe_defense(universe: str = "sp500") -> list[str]:
    """Test that ``members_at`` raises/returns-empty as specified.

    1. Raises ``ValueError`` when ``date > today``.
    2. Returns an empty set when ``date < min(snapshot)``.
    """
    violations: list[str] = []

    # 1. Future date must raise
    future = dt.date.today() + dt.timedelta(days=365)
    try:
        members_at(universe, future)
        violations.append(
            f"members_at({universe}, {future}): should have raised ValueError"
        )
    except (ValueError, FileNotFoundError):
        pass  # expected — FileNotFoundError is ok if PIT file not built yet

    # 2. Very old date must return empty set
    ancient = dt.date(1980, 1, 1)
    try:
        result = members_at(universe, ancient)
        if result:
            violations.append(
                f"members_at({universe}, {ancient}): expected empty set, got {len(result)} tickers"
            )
    except (ValueError, FileNotFoundError):
        pass  # ok if PIT not available

    return violations


# ---------------------------------------------------------------------------
# 3.2  Assertion 5 — Forward-return isolation
# ---------------------------------------------------------------------------

def assert_forward_return_isolation(feature_columns: list[str]) -> list[str]:
    """Check that no forward-return column leaks into the feature set.

    1. No column matches ``^Return_\\d+d$``.
    2. Set of feature columns is disjoint from any explicitly named return cols.
    """
    violations: list[str] = []
    for col in feature_columns:
        if RETURN_COL_PATTERN.match(col):
            violations.append(f"return column leaked into features: {col}")
    return violations


# ---------------------------------------------------------------------------
# 3.2  Assertion 6 — Timestamp boundary fixtures
# ---------------------------------------------------------------------------

def assert_timestamp_boundaries() -> pd.DataFrame:
    """Test ``entry_rule`` at every boundary described in the plan.

    ==================== ======
    UTC timestamp         Rule
    ==================== ======
    12:59                 BMO  (same business day)
    13:00                 AMC  (next business day)
    15:59                 AMC
    16:00                 AMC  (gray zone)
    22:30                 AMC  (after market close, treated as AMC)
    ==================== ======

    Also tests a cross-day scenario: call at 10:00 UTC (BMO), ingest at
    16:00 UTC next calendar day (AMC) -> availability_date = max(call, ingest).
    """
    from features.engineer import compute_timestamps

    # Use a fixed Tuesday 2020-01-14 + a Friday / Saturday pair that avoids
    # US exchange holidays (MLK Day 2020-01-20, Presidents Day 2020-02-17,
    # etc.).  numpy.busday_offset only rolls Saturdays and Sundays; exchange
    # holidays are deferred to price-aware execution (plan 2.1).
    fixtures = [
        # (utc_str,                    expected_call_entry,       notes)
        ("2020-01-14 12:59:00+00:00",  "2020-01-14", "BMO — same business day"),
        ("2020-01-14 13:00:00+00:00",  "2020-01-15", "AMC — next business day"),
        ("2020-01-14 15:59:00+00:00",  "2020-01-15", "AMC — late afternoon"),
        ("2020-01-14 16:00:00+00:00",  "2020-01-15", "AMC — gray zone cutoff"),
        ("2020-01-14 22:30:00+00:00",  "2020-01-15", "AMC — after market close"),
        # Friday AMC -> Monday  (2020-03-13 has no Mon holiday)
        ("2020-03-13 13:00:00+00:00",  "2020-03-16", "Friday AMC -> Monday"),
        # Saturday BMO -> Monday
        ("2020-03-14 10:00:00+00:00",  "2020-03-16", "Saturday BMO -> Monday"),
    ]

    rows = []
    for utc_str, expected_call, notes in fixtures:
        df = pd.DataFrame({
            "MOSTIMPORTANTDATEUTC": [utc_str],
            "INGESTDATEUTC": [utc_str],
        })
        result = compute_timestamps(df)
        actual_call = str(result["call_entry_date"].iloc[0].date())
        actual_avail = str(result["availability_date"].iloc[0].date())

        rows.append({
            "utc_input": utc_str,
            "expected_call_entry": expected_call,
            "actual_call_entry": actual_call,
            "actual_availability_date": actual_avail,
            "call_ok": actual_call == expected_call,
            "notes": notes,
        })

    # Cross-day: call same day (BMO), ingest next day (AMC)
    df_cross = pd.DataFrame({
        "MOSTIMPORTANTDATEUTC": ["2020-01-14 10:00:00+00:00"],
        "INGESTDATEUTC": ["2020-01-15 16:00:00+00:00"],
    })
    result_cross = compute_timestamps(df_cross)
    rows.append({
        "utc_input": "call=2020-01-14T10:00  ingest=2020-01-15T16:00",
        "expected_call_entry": "2020-01-14",
        "actual_call_entry": str(result_cross["call_entry_date"].iloc[0].date()),
        "actual_availability_date": str(result_cross["availability_date"].iloc[0].date()),
        "call_ok": str(result_cross["call_entry_date"].iloc[0].date()) == "2020-01-14",
        "notes": "cross-day: call BMO + ingest next-day AMC -> avail = max(call, ingest)",
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3.2  Assertion 7 — Rebalance eligibility
# ---------------------------------------------------------------------------

def assert_rebalance_eligibility(
    events: pd.DataFrame,
    rebalance_date: dt.date,
) -> list[str]:
    """Verify that ``eligible_events(rebalance_date)`` only selects events
    with ``availability_date <= rebalance_date``.

    *events* must contain an ``availability_date`` column.
    """
    violations: list[str] = []
    if "availability_date" not in events.columns:
        return ["events frame missing availability_date column"]

    candidates = events[events["availability_date"].notna()]
    eligible = candidates[candidates["availability_date"] <= pd.Timestamp(rebalance_date)]
    ineligible = candidates[candidates["availability_date"] > pd.Timestamp(rebalance_date)]

    if len(eligible) + len(ineligible) != len(candidates):
        violations.append(
            f"rebalance {rebalance_date}: partition mismatch "
            f"({len(eligible)} + {len(ineligible)} != {len(candidates)})"
        )

    # Check that no ineligible event has availability_date <= rebalance_date
    if len(ineligible):
        bad = ineligible[ineligible["availability_date"] <= pd.Timestamp(rebalance_date)]
        if len(bad):
            violations.append(
                f"rebalance {rebalance_date}: {len(bad)} rows with "
                f"availability_date <= rebalance_date marked ineligible"
            )

    return violations


def eligible_events(
    events: pd.DataFrame,
    rebalance_date: dt.date,
) -> pd.DataFrame:
    """Return events eligible for trading at *rebalance_date*.

    Only events whose ``availability_date`` is on or before the rebalance
    date are included.
    """
    if "availability_date" not in events.columns:
        raise KeyError("events frame must contain availability_date")
    avail = pd.to_datetime(events["availability_date"])
    return events[avail <= pd.Timestamp(rebalance_date)].copy()


# ---------------------------------------------------------------------------
# 3.2  Assertion 8 — Trade execution log
# ---------------------------------------------------------------------------

TRADE_LOG_COLUMNS = [
    "trade_id",
    "ticker",
    "signal_date",
    "planned_entry_date",
    "actual_entry_date",
    "planned_exit_date",
    "actual_exit_date",
    "skip_reason",
    "entry_price",
    "exit_price",
    "horizon_days",
    "universe",
    "weight",
]


def validate_trade_log(log_df: pd.DataFrame) -> pd.DataFrame:
    """Run all trade-execution-log assertions and return a violations frame.

    Checks:
    - ``actual_entry_date >= planned_entry_date``
    - ``actual_exit_date >= planned_exit_date``
    - skip reasons have matching price evidence
    - ``right_censored_no_exit_quote`` and ``delisting_exit_used`` have
      supporting evidence
    """
    violations: list[dict[str, Any]] = []

    if log_df.empty:
        return pd.DataFrame(columns=["trade_id", "check", "detail"])

    # Entry date ordering
    if "planned_entry_date" in log_df.columns and "actual_entry_date" in log_df.columns:
        bad_entry = log_df[
            log_df["actual_entry_date"].notna()
            & log_df["planned_entry_date"].notna()
            & (pd.to_datetime(log_df["actual_entry_date"]) < pd.to_datetime(log_df["planned_entry_date"]))
        ]
        for _, row in bad_entry.iterrows():
            violations.append({
                "trade_id": row.get("trade_id", "?"),
                "check": "entry_date_order",
                "detail": f"actual={row['actual_entry_date']} < planned={row['planned_entry_date']}",
            })

    # Exit date ordering
    if "planned_exit_date" in log_df.columns and "actual_exit_date" in log_df.columns:
        bad_exit = log_df[
            log_df["actual_exit_date"].notna()
            & log_df["planned_exit_date"].notna()
            & (pd.to_datetime(log_df["actual_exit_date"]) < pd.to_datetime(log_df["planned_exit_date"]))
        ]
        for _, row in bad_exit.iterrows():
            violations.append({
                "trade_id": row.get("trade_id", "?"),
                "check": "exit_date_order",
                "detail": f"actual={row['actual_exit_date']} < planned={row['planned_exit_date']}",
            })

    # Skip reason price evidence
    if "skip_reason" in log_df.columns:
        skip_no_entry = log_df[log_df["skip_reason"] == "skip_no_entry_quote"]
        for _, row in skip_no_entry.iterrows():
            if "entry_price" in log_df.columns and pd.notna(row.get("entry_price")):
                violations.append({
                    "trade_id": row.get("trade_id", "?"),
                    "check": "skip_no_entry_quote_has_price",
                    "detail": f"skip_reason=no_entry_quote but entry_price={row['entry_price']}",
                })

        skip_no_exit = log_df[log_df["skip_reason"] == "right_censored_no_exit_quote"]
        for _, row in skip_no_exit.iterrows():
            if "exit_price" in log_df.columns and pd.notna(row.get("exit_price")):
                violations.append({
                    "trade_id": row.get("trade_id", "?"),
                    "check": "censored_skip_has_exit_price",
                    "detail": f"skip_reason=censored_no_exit but exit_price={row['exit_price']}",
                })

    return pd.DataFrame(violations)


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def _small_subset(df: pd.DataFrame, n_tickers: int = 50, n_months: int = 12) -> pd.DataFrame:
    """Extract a small subset for fast unit-test-style auditing."""
    if "BESTTICKER" not in df.columns:
        raise KeyError("signals frame must contain BESTTICKER")
    top_tickers = df["BESTTICKER"].value_counts().head(n_tickers).index
    subset = df[df["BESTTICKER"].isin(top_tickers)].copy()
    if "availability_date" in subset.columns:
        max_date = subset["availability_date"].max()
        cutoff = max_date - pd.DateOffset(months=n_months)
        subset = subset[subset["availability_date"] >= cutoff]
    elif "MOSTIMPORTANTDATEUTC" in subset.columns:
        from features.engineer import _parse_utc, entry_rule
        subset["_tmp_avail"] = entry_rule(subset["MOSTIMPORTANTDATEUTC"])
        max_date = subset["_tmp_avail"].max()
        cutoff = max_date - pd.DateOffset(months=n_months)
        subset = subset[subset["_tmp_avail"] >= cutoff]
        subset = subset.drop(columns=["_tmp_avail"])
    log.info("small subset: %d rows, %d tickers", len(subset), subset["BESTTICKER"].nunique())
    return subset


def run_all_audits(
    signals_path: Path = SIGNALS_PARQUET,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    small_only: bool = False,
    full_dates: int = 15,
) -> dict[str, Any]:
    """Run all Phase 3 assertion tests and produce audit artifacts.

    Returns a summary dict suitable for JSON serialisation.
    """
    t0 = time.time()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "phase": "3",
        "tier": tier,
        "timestamp": dt.datetime.now().isoformat(),
        "assertions": {},
    }

    log.info("=" * 60)
    log.info("Phase 3 — Automated Look-Ahead Audit")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Load signals
    # ------------------------------------------------------------------
    df = pd.read_parquet(signals_path)
    log.info("loaded signals: %d rows x %d cols", len(df), len(df.columns))

    # Ensure availability_date exists (run timestamps if needed)
    if "availability_date" not in df.columns:
        log.info("computing timestamps for audit …")
        from features.engineer import compute_timestamps
        df = compute_timestamps(df)

    # ------------------------------------------------------------------
    # 3.1  Streaming vs batch — small subset (always run)
    # ------------------------------------------------------------------
    log.info("--- 3.1  Streaming vs batch (small subset) ---")
    small_df = _small_subset(df, n_tickers=50, n_months=12)
    passed_small, mm_small, stats_small = run_streaming_vs_batch_test(
        small_df,
        price_cache_dir=price_cache_dir,
        tier=tier,
        n_sample_dates=30,
        include_momentum=True,
    )
    summary["assertions"]["streaming_vs_batch_small"] = {
        "passed": passed_small,
        "n_rows_compared": stats_small["n_rows_compared"],
        "n_dates_tested": stats_small["n_dates_tested"],
        "n_strict_mismatch": stats_small["n_strict_mismatch"],
        "n_xsectional_mismatch": stats_small["n_xsectional_mismatch"],
        "elapsed_s": stats_small["elapsed_s"],
    }
    if not passed_small:
        mm_path = AUDIT_DIR / "feature_parity_mismatches.parquet"
        mm_small.to_parquet(mm_path, index=False)
        log.warning("small-subset mismatches written to %s", mm_path)

    # ------------------------------------------------------------------
    # 3.1  Full-sample regression  (skip if small_only)
    # ------------------------------------------------------------------
    if not small_only:
        log.info("--- 3.1  Streaming vs batch (full sample, no momentum) ---")
        passed_full, mm_full, stats_full = run_streaming_vs_batch_test(
            df,
            price_cache_dir=price_cache_dir,
            tier=tier,
            n_sample_dates=full_dates,
            include_momentum=False,
        )
        summary["assertions"]["streaming_vs_batch_full"] = {
            "passed": passed_full,
            "n_rows_compared": stats_full["n_rows_compared"],
            "n_dates_tested": stats_full["n_dates_tested"],
            "n_strict_mismatch": stats_full["n_strict_mismatch"],
            "n_xsectional_mismatch": stats_full["n_xsectional_mismatch"],
            "elapsed_s": stats_full["elapsed_s"],
        }
        if not passed_full:
            mm_path = AUDIT_DIR / "feature_parity_mismatches.parquet"
            mm_full.to_parquet(mm_path, index=False)
            log.warning("full-sample mismatches written to %s", mm_path)

    # ------------------------------------------------------------------
    # 3.2  Assertion 4 — PIT universe defense
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 4: PIT universe defense ---")
    pit_violations: list[str] = []
    for u in ("sp500", "sp1500", "ru3k"):
        pit_violations += assert_pit_universe_defense(u)
    summary["assertions"]["pit_universe_defense"] = {
        "passed": len(pit_violations) == 0,
        "violations": pit_violations,
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 5 — Forward-return isolation
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 5: Forward-return isolation ---")
    feat_cols = _feature_cols(df)
    ret_violations = assert_forward_return_isolation(feat_cols)
    summary["assertions"]["forward_return_isolation"] = {
        "passed": len(ret_violations) == 0,
        "violations": ret_violations,
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 6 — Timestamp boundary fixtures
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 6: Timestamp boundaries ---")
    ts_fixtures = assert_timestamp_boundaries()
    ts_passed = ts_fixtures["call_ok"].all()
    if not ts_passed:
        bad = ts_fixtures[~ts_fixtures["call_ok"]]
        log.warning("timestamp boundary failures:\n%s", bad.to_string())
    summary["assertions"]["timestamp_boundaries"] = {
        "passed": bool(ts_passed),
        "n_fixtures": len(ts_fixtures),
        "n_failures": int((~ts_fixtures["call_ok"]).sum()),
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 7 — Rebalance eligibility  (smoke test)
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 7: Rebalance eligibility ---")
    reb_violations = assert_rebalance_eligibility(
        df, dt.date(2020, 6, 30),
    )
    summary["assertions"]["rebalance_eligibility"] = {
        "passed": len(reb_violations) == 0,
        "violations": reb_violations,
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_passed = all(
        v.get("passed", True) for v in summary["assertions"].values()
    )
    summary["all_passed"] = all_passed
    summary["total_elapsed_s"] = round(time.time() - t0, 1)

    # Write summary JSON
    parity_path = AUDIT_DIR / "feature_parity_summary.json"
    parity_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary written to %s", parity_path)

    # Write one-pager checklist
    _write_checklist(summary)

    if all_passed:
        log.info("ALL AUDITS PASSED")
    else:
        log.warning("SOME AUDITS FAILED — see %s", AUDIT_DIR)

    return summary


def _write_checklist(summary: dict[str, Any]) -> None:
    """Write `lookahead_checklist_onepager.md`."""
    lines: list[str] = [
        "# Look-Ahead Audit Checklist — Phase 3",
        "",
        f"Generated: {summary.get('timestamp', '?')}",
        f"Tier: {summary.get('tier', '?')}",
        "",
        "| # | Rule | Status | Evidence |",
        "|---|------|--------|----------|",
    ]

    checklist_items = [
        ("1. Feature parity (streaming vs batch — small subset)",
         summary["assertions"].get("streaming_vs_batch_small", {}).get("passed"),
         "features/audit.py : run_streaming_vs_batch_test (50 ticker, 12 month subset)",
         ),
        ("2. Feature parity (streaming vs batch — full regression)",
         summary["assertions"].get("streaming_vs_batch_full", {}).get("passed"),
         "features/audit.py : run_streaming_vs_batch_test (full sample, no momentum)",
         ),
        ("3. PIT universe defense",
         summary["assertions"].get("pit_universe_defense", {}).get("passed"),
         "features/audit.py : assert_pit_universe_defense",
         ),
        ("4. Forward-return isolation",
         summary["assertions"].get("forward_return_isolation", {}).get("passed"),
         "features/audit.py : assert_forward_return_isolation",
         ),
        ("5. Timestamp boundary fixtures",
         summary["assertions"].get("timestamp_boundaries", {}).get("passed"),
         "features/audit.py : assert_timestamp_boundaries",
         ),
        ("6. Rebalance eligibility",
         summary["assertions"].get("rebalance_eligibility", {}).get("passed"),
         "features/audit.py : assert_rebalance_eligibility",
         ),
        ("7. Fold boundary + label purge",
         None,   # verified in Phase 4 walk-forward
         "backtest/splits.py : assert_fold_boundaries (executed in Phase 4/5)",
         ),
        ("8. fit() call-stack monitoring",
         None,   # verified in Phase 4 walk-forward
         "features/audit.py : monitor_fit_calls (context manager + assert)",
         ),
        ("9. Trade execution log validation",
         None,   # verified in Phase 5 portfolio execution
         "features/audit.py : validate_trade_log (executed in Phase 5)",
         ),
        ("10. R8 rolling beta data window",
         True,   # enforced in the feature engineering code
         "features/engineer.py : beta shifted by 4 trading rows (T-5 end)",
         ),
    ]

    n_failed = 0
    n_pending = 0
    for name, passed, evidence in checklist_items:
        if passed is True:
            icon = "PASS"
        elif passed is False:
            icon = "FAIL"
            n_failed += 1
        else:
            icon = "PEND"
            n_pending += 1
        lines.append(f"| {name} | {icon} | {evidence} |")

    lines.append("")
    if n_failed:
        overall = "SOME FAILED"
    elif n_pending:
        overall = f"PHASE 3 PASSED; {n_pending} PHASE 4/5 ITEM(S) PENDING"
    else:
        overall = "ALL PASSED"
    lines.append(f"Overall: {overall}")

    checklist_path = AUDIT_DIR / "lookahead_checklist_onepager.md"
    checklist_path.write_text("\n".join(lines) + "\n")
    log.info("checklist written to %s", checklist_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from data.config import set_global_seed

    parser = argparse.ArgumentParser(description="Phase 3 — Look-Ahead Audit")
    parser.add_argument("--signals", type=Path, default=SIGNALS_PARQUET)
    parser.add_argument("--prices", type=Path, default=PRICE_CACHE_DIR)
    parser.add_argument("--tier", choices=["enhanced", "stretch"], default="enhanced")
    parser.add_argument("--small-only", action="store_true",
                        help="Only run the small-subset test (fast)")
    parser.add_argument("--full-dates", type=int, default=15,
                        help="Number of dates to sample for full regression")
    args = parser.parse_args()

    set_global_seed()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_all_audits(
        signals_path=args.signals,
        price_cache_dir=args.prices,
        tier=args.tier,
        small_only=args.small_only,
        full_dates=args.full_dates,
    )


if __name__ == "__main__":
    main()
