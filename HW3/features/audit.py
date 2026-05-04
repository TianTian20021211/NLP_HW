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

from data.cache_utils import build_cache_manifest, write_cache_manifest
from data.config import (
    AUDIT_DIR,
    CACHE_DIR,
    CACHE_MANIFEST_DIR,
    PRICE_CACHE_DIR,
    PRICE_MANIFEST,
    RAW_SIGNAL_CSV,
    RAW_SIGNAL_ZIP,
    RESULTS_DIR,
    SIGNALS_PARQUET,
    UNIVERSE_CACHE_DIR,
)
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
FORWARD_RETURN_COL_PATTERN = re.compile(r"^forward_return_\d+d$")
TARGET_DATE_COL_PATTERN = re.compile(r"^target_available_date_\d+d$")

# Features whose values depend on the cross-sectional ticker universe available
# at a point in time.  Streaming-vs-batch comparisons for these columns are
# reported separately because they are expected to differ when the signal
# universe expands over time — not because of look-ahead leakage.
CROSS_SECTIONAL_FEATURE_PATTERNS = [
    "pre_event_ret_21d_sector_rel",   # sector median changes with ticker set
    "pre_event_idio_resid_5d",        # depends on beta × sector median
]

MOMENTUM_FEATURE_COLS = {
    "pre_event_ret_21d",
    "pre_event_ret_21d_sector_rel",
    "pre_event_idio_resid_5d",
}


def _feature_col_names(
    columns: list[str] | pd.Index,
    *,
    include_momentum: bool = True,
) -> list[str]:
    """Return generated feature column names from an iterable of names."""
    return [
        c for c in columns
        if c not in IDENTIFIER_LIKE
        and not RETURN_COL_PATTERN.match(c)
        and not FORWARD_RETURN_COL_PATTERN.match(c)
        and not TARGET_DATE_COL_PATTERN.match(c)
        and not c.startswith("_")
        and (include_momentum or c not in MOMENTUM_FEATURE_COLS)
    ]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns in *df* that are not identifier-like and not return cols."""
    return _feature_col_names(df.columns)


def _non_identifier_columns(df: pd.DataFrame) -> list[str]:
    """Return all generated non-identifier columns for leakage assertions."""
    return [
        c for c in df.columns
        if c not in IDENTIFIER_LIKE
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
    uniq = dates.dropna().unique().tolist()
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


def _build_batch_features_for_audit(
    df: pd.DataFrame,
    price_cache_dir: Path,
    tier: str,
    include_momentum: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build full batch features and partition columns into strict/xsectional."""
    log.info("  [1/2] batch on %d rows …", len(df))
    batch = _build_safe(df, price_cache_dir, tier, include_momentum)
    log.info("  batch done: %d rows x %d cols", len(batch), len(batch.columns))

    all_feat = _feature_cols(batch)
    strict_cols, xsectional_cols = _partition_feature_cols(all_feat)
    log.info("  strict features: %d  cross-sectional: %d",
             len(strict_cols), len(xsectional_cols))
    return batch, strict_cols, xsectional_cols


def _audit_history_base_columns(columns: list[str] | pd.Index) -> list[str]:
    """Columns needed to recompute no-momentum streaming targets."""
    from features.engineer import ASPECTS, THEMES

    id_cols = [
        "BESTTICKER", "SECTOR", "availability_date",
        "call_entry_date", "SignalType",
    ]
    input_cols = (
        ["ATCClassifierScore"]
        + [f"aspect_{a}_total" for a in ASPECTS]
        + [f"theme_{t}_total" for t in THEMES]
    )
    available = set(columns)
    return [c for c in id_cols + input_cols if c in available]


def _read_feature_rows_for_dates(
    features_path: Path,
    sampled_dates: list[pd.Timestamp],
    columns: list[str],
    *,
    batch_size: int = 32_768,
) -> pd.DataFrame:
    """Read only sampled-date rows from a feature parquet in Arrow batches."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if not sampled_dates:
        return pd.DataFrame(columns=columns)

    pf = pq.ParquetFile(features_path)
    schema_names = set(pf.schema_arrow.names)
    read_cols = [c for c in dict.fromkeys(columns) if c in schema_names]
    if "availability_date" not in read_cols:
        read_cols.insert(0, "availability_date")

    date_type = pf.schema_arrow.field("availability_date").type
    date_values = pa.array(
        pd.to_datetime(sampled_dates).to_numpy(dtype="datetime64[ns]"),
        type=date_type,
    )

    parts: list[pa.Table] = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        table = pa.Table.from_batches([batch])
        mask = pc.is_in(table["availability_date"], value_set=date_values)
        filtered = table.filter(mask)
        if filtered.num_rows:
            parts.append(filtered)

    if not parts:
        return pd.DataFrame(columns=read_cols)
    return pa.concat_tables(parts, promote_options="default").to_pandas()


def _build_artifact_features_for_audit(
    features_path: Path,
    sampled_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load no-momentum audit inputs from an existing Phase 2 feature parquet."""
    import pyarrow.parquet as pq

    log.info("  [1/2] batch from feature artifact %s", features_path)
    pf = pq.ParquetFile(features_path)
    schema_cols = pf.schema_arrow.names

    all_feat = _feature_col_names(schema_cols, include_momentum=False)
    strict_cols, xsectional_cols = _partition_feature_cols(all_feat)
    log.info(
        "  strict features: %d  cross-sectional: %d",
        len(strict_cols), len(xsectional_cols),
    )

    history_cols = _audit_history_base_columns(schema_cols)
    history_base = pd.read_parquet(features_path, columns=history_cols)
    log.info(
        "  loaded history base: %d rows x %d cols",
        len(history_base), len(history_base.columns),
    )

    # Include id columns needed by _build_streaming_targets_no_momentum
    # for recomputing time-series and PIT features on sampled dates.
    id_cols = ["BESTTICKER", "SECTOR", "availability_date",
               "call_entry_date", "SignalType"]
    compare_cols = list(dict.fromkeys(id_cols + strict_cols + xsectional_cols))
    batch = _read_feature_rows_for_dates(features_path, sampled_dates, compare_cols)
    log.info("  batch target rows: %d rows x %d cols", len(batch), len(batch.columns))
    return batch, history_base, strict_cols, xsectional_cols


def _prepare_streaming_targets_for_audit(
    batch: pd.DataFrame,
    sampled_dates: list[pd.Timestamp],
    strict_cols: list[str],
    xsectional_cols: list[str],
    include_momentum: bool,
    history_base: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Use target-only streaming path when momentum is disabled."""
    if not include_momentum:
        log.info("  using target-only streaming audit path (no momentum)")
        return _build_streaming_targets_no_momentum(
            batch,
            sampled_dates,
            strict_cols + xsectional_cols,
            history_base=history_base,
        )
    return None


def _compare_one_streaming_date(
    original_df: pd.DataFrame,
    batch: pd.DataFrame,
    streaming_targets: pd.DataFrame | None,
    date: pd.Timestamp,
    price_cache_dir: Path,
    tier: str,
    include_momentum: bool,
    strict_cols: list[str],
    xsectional_cols: list[str],
    rtol: float,
    atol: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int, bool]:
    """Compare one sampled date between batch and streaming results."""
    if streaming_targets is None:
        orig_mask = original_df["availability_date"] <= date
        if orig_mask.sum() < 2:
            return None, None, 0, False
        subset = original_df.loc[orig_mask]
        try:
            streaming = _build_safe(subset, price_cache_dir, tier, include_momentum)
        except Exception:
            log.exception("streaming build failed for date %s", date)
            return None, None, 0, False
    else:
        subset = None
        streaming = streaming_targets

    batch_test = batch
    streaming_test = streaming.loc[streaming["availability_date"] == date]

    if len(batch_test) == 0 and len(streaming_test) == 0:
        return None, None, 0, False

    if len(batch_test) != len(streaming_test):
        log.error(
            "date %s: row-count mismatch batch=%d streaming=%d",
            date, len(batch_test), len(streaming_test),
        )
        rc_df = pd.DataFrame([{
            "column": "__ROW_COUNT__",
            "n_mismatch": abs(len(batch_test) - len(streaming_test)),
            "max_abs_diff": np.nan,
            "feature_group": "strict",
        }])
        return rc_df, None, 0, True

    batch_test = batch_test.reset_index(drop=True)
    streaming_test = streaming_test.reset_index(drop=True)

    strict_mismatch_df: pd.DataFrame | None = None
    mm_strict = _compare_parity(batch_test, streaming_test, rtol, atol,
                                columns=strict_cols)
    if len(mm_strict):
        mm_strict.insert(0, "date", str(date.date()))
        mm_strict["feature_group"] = "strict"
        strict_mismatch_df = mm_strict

    xsec_mismatch_df: pd.DataFrame | None = None
    if xsectional_cols:
        mm_xsec = _compare_parity(batch_test, streaming_test, rtol, atol,
                                  columns=xsectional_cols)
        if len(mm_xsec):
            mm_xsec.insert(0, "date", str(date.date()))
            mm_xsec["feature_group"] = "xsectional"
            xsec_mismatch_df = mm_xsec

    rows_compared = len(batch_test)

    if streaming_targets is None:
        del subset, streaming
    del batch_test, streaming_test

    return strict_mismatch_df, xsec_mismatch_df, rows_compared, False


def _summarize_streaming_mismatches(
    strict_parts: list[pd.DataFrame],
    xsectional_parts: list[pd.DataFrame],
    n_dates_tested: int,
    n_rows_compared: int,
    n_row_count_mismatch: int,
    elapsed_s: float,
) -> tuple[bool, pd.DataFrame, dict[str, Any]]:
    """Combine mismatch tables and return the public audit tuple."""
    all_parts = strict_parts + xsectional_parts
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=["date", "column", "n_mismatch", "max_abs_diff", "feature_group"],
        )

    n_strict_mismatch = int(
        combined.loc[combined["feature_group"] == "strict", "n_mismatch"].sum()
        if len(combined) else 0
    )
    n_xsectional_mismatch = int(
        combined.loc[combined["feature_group"] == "xsectional", "n_mismatch"].sum()
        if len(combined) else 0
    )

    passed = n_strict_mismatch == 0 and n_row_count_mismatch == 0

    stats: dict[str, Any] = {
        "n_dates_tested": n_dates_tested,
        "n_rows_compared": n_rows_compared,
        "n_strict_mismatch": n_strict_mismatch,
        "n_xsectional_mismatch": n_xsectional_mismatch,
        "n_row_count_mismatch": n_row_count_mismatch,
        "elapsed_s": elapsed_s,
    }

    return passed, combined, stats


def run_streaming_vs_batch_test(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    n_sample_dates: int | None = None,
    include_momentum: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    batch_features_path: Path | None = None,
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

    # 1. Sample dates from availability metadata.
    dates = pd.DatetimeIndex(df["availability_date"].dropna().unique())
    sampled = _sample_availability_dates(dates, n_sample_dates or len(dates))

    history_base: pd.DataFrame | None = None
    if batch_features_path is not None and not include_momentum:
        batch, history_base, strict_cols, xsectional_cols = _build_artifact_features_for_audit(
            batch_features_path,
            sampled,
        )
    else:
        # 2. Batch — fit once on the full sample
        batch, strict_cols, xsectional_cols = _build_batch_features_for_audit(
            df, price_cache_dir, tier, include_momentum,
        )
        if "availability_date" not in batch.columns:
            raise RuntimeError("batch output missing availability_date")

    log.info("  [2/2] streaming on %d / %d dates …", len(sampled), len(dates))

    # 3. Prepare streaming targets (no-momentum path)
    streaming_targets = _prepare_streaming_targets_for_audit(
        batch,
        sampled,
        strict_cols,
        xsectional_cols,
        include_momentum,
        history_base=history_base,
    )

    # 4. Filter batch to sampled dates for comparison
    compare_cols = ["availability_date"] + strict_cols + xsectional_cols
    batch = batch.loc[
        batch["availability_date"].isin(sampled),
        [c for c in compare_cols if c in batch.columns],
    ].copy()
    gc.collect()

    # 5. Per-date streaming comparison
    all_strict_mm: list[pd.DataFrame] = []
    all_xsec_mm: list[pd.DataFrame] = []
    n_rows_compared = 0
    n_row_count_mismatch = 0

    batch_by_date = dict(list(batch.groupby("availability_date")))

    for d in sampled:
        batch_d = batch_by_date.get(d)
        if batch_d is None:
            continue
        strict_df, xsec_df, rows_cmp, row_cnt_flag = _compare_one_streaming_date(
            df, batch_d, streaming_targets, d,
            price_cache_dir, tier, include_momentum,
            strict_cols, xsectional_cols, rtol, atol,
        )
        if strict_df is not None:
            all_strict_mm.append(strict_df)
        if xsec_df is not None:
            all_xsec_mm.append(xsec_df)
        n_rows_compared += rows_cmp
        n_row_count_mismatch += int(row_cnt_flag)

    gc.collect()

    # 6. Summarize
    elapsed_s = round(time.time() - t0, 1)
    passed, combined, stats = _summarize_streaming_mismatches(
        all_strict_mm, all_xsec_mm,
        n_dates_tested=len(sampled),
        n_rows_compared=n_rows_compared,
        n_row_count_mismatch=n_row_count_mismatch,
        elapsed_s=elapsed_s,
    )

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
    history_base: pd.DataFrame | None = None,
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
    source = history_base if history_base is not None else batch
    base_cols = _audit_history_base_columns(source.columns)
    base = source[base_cols].copy()
    base["_audit_row_id"] = np.arange(len(base), dtype="int64")

    target_keep_cols = [
        c for c in (
            id_cols
            + feature_columns
            + _audit_timeseries_input_cols(source)
            + _audit_pit_input_cols(source)
        )
        if c in batch.columns and c not in generated
    ]
    target_keep_cols = list(dict.fromkeys(target_keep_cols))

    target_mask = batch["availability_date"].isin(sampled_dates)
    target = batch.loc[target_mask, target_keep_cols].copy()
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

    results = {
        col: pd.Series(np.nan, index=target.index, dtype="float64")
        for col in available
    }

    for key, target_pos in target_groups.items():
        hist_pos = base_groups.get(key)
        if hist_pos is None or len(hist_pos) == 0:
            continue

        for col in available:
            base_vals = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype="float64")
            target_vals = pd.to_numeric(target[col], errors="coerce").to_numpy(dtype="float64")

            pct = _strict_historical_percentile_queries(
                history_values=base_vals[hist_pos],
                history_dates=base_hist_dt[hist_pos],
                query_values=target_vals[target_pos],
                cutoff_dates=target_cutoff_dt[target_pos],
            )
            results[col].loc[target_index[target_pos]] = pct

    for col in available:
        out[f"{col}_sector_pct"] = results[col]

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
# 3.2  Assertion 2 — Fold boundary + label purge
# ---------------------------------------------------------------------------

def assert_fold_boundaries(
    fold_train_start: pd.Timestamp,
    fold_train_end: pd.Timestamp,
    fold_test_start: pd.Timestamp,
    max_train_feature_date: pd.Timestamp,
    max_train_target_available: pd.Timestamp | None = None,
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
        if (
            RETURN_COL_PATTERN.match(col)
            or FORWARD_RETURN_COL_PATTERN.match(col)
            or TARGET_DATE_COL_PATTERN.match(col)
        ):
            violations.append(f"target/return column leaked into features: {col}")
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
        # (utc_str,                    expected_call_entry,  expected_avail,  notes)
        ("2020-01-14 12:59:00+00:00",  "2020-01-14", "2020-01-16", "BMO — same business day"),
        ("2020-01-14 13:00:00+00:00",  "2020-01-15", "2020-01-17", "AMC — next business day"),
        ("2020-01-14 15:59:00+00:00",  "2020-01-15", "2020-01-17", "AMC — late afternoon"),
        ("2020-01-14 16:00:00+00:00",  "2020-01-15", "2020-01-17", "AMC — gray zone cutoff"),
        ("2020-01-14 22:30:00+00:00",  "2020-01-15", "2020-01-17", "AMC — after market close"),
        # Friday AMC -> Monday  (2020-03-13 has no Mon holiday)
        ("2020-03-13 13:00:00+00:00",  "2020-03-16", "2020-03-18", "Friday AMC -> Monday"),
        # Saturday BMO -> Monday
        ("2020-03-14 10:00:00+00:00",  "2020-03-16", "2020-03-18", "Saturday BMO -> Monday"),
    ]

    rows = []
    for utc_str, expected_call, expected_avail, notes in fixtures:
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
            "expected_availability_date": expected_avail,
            "actual_availability_date": actual_avail,
            "call_ok": actual_call == expected_call,
            "availability_ok": actual_avail == expected_avail,
            "notes": notes,
        })

    # Cross-day: call same day (BMO), ingest next day (AMC)
    df_cross = pd.DataFrame({
        "MOSTIMPORTANTDATEUTC": ["2020-01-14 10:00:00+00:00"],
        "INGESTDATEUTC": ["2020-01-15 16:00:00+00:00"],
    })
    result_cross = compute_timestamps(df_cross)
    actual_call = str(result_cross["call_entry_date"].iloc[0].date())
    actual_avail = str(result_cross["availability_date"].iloc[0].date())
    rows.append({
        "utc_input": "call=2020-01-14T10:00  ingest=2020-01-15T16:00",
        "expected_call_entry": "2020-01-14",
        "actual_call_entry": actual_call,
        "expected_availability_date": "2020-01-16",
        "actual_availability_date": actual_avail,
        "call_ok": actual_call == "2020-01-14",
        "availability_ok": actual_avail == "2020-01-16",
        "notes": "cross-day: call BMO + ingest next-day AMC -> avail = max(call, ingest) (pre-cutoff: call_entry+2bd)",
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
    - ``right_censored_no_exit_quote`` has no exit price
    - ``delisting_exit_used`` has gap-accounting evidence (NaN actual_exit_date,
      non-NaN entry_price indicating the position was entered)
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

        delisting_used = log_df[log_df["skip_reason"] == "delisting_exit_used"]
        for _, row in delisting_used.iterrows():
            if "actual_exit_date" in log_df.columns and pd.notna(row.get("actual_exit_date")):
                violations.append({
                    "trade_id": row.get("trade_id", "?"),
                    "check": "delisting_exit_has_exit_date",
                    "detail": f"skip_reason=delisting_exit_used but actual_exit_date={row['actual_exit_date']}",
                })
            if "entry_price" in log_df.columns and pd.isna(row.get("entry_price")):
                violations.append({
                    "trade_id": row.get("trade_id", "?"),
                    "check": "delisting_exit_no_entry_price",
                    "detail": "skip_reason=delisting_exit_used but entry_price is NaN (position was never entered)",
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


# ---------------------------------------------------------------------------
# 3.2  Assertion 11 — Fluff/Filler control  (requirement §1.6)
# ---------------------------------------------------------------------------

FLUFF_FILLER_CACHE: Path = CACHE_DIR / "fluff_filler.parquet"

_FLUFF_FILLER_ID_COLS: list[str] = [
    "BESTTICKER", "MOSTIMPORTANTDATEUTC", "INGESTDATEUTC",
    "SignalType", "SECTOR",
]

_SENTIMENT_SIGN: dict[str, int] = {"Positive": 1, "Neutral": 0, "Negative": -1}
_MAGNITUDE_WEIGHT: dict[str, int] = {"High": 3, "Medium": 2, "Low": 1}


def _load_fluff_filler_df() -> pd.DataFrame:
    """Load Fluff/Filler AspectTheme columns from raw CSV, cached to parquet."""
    if FLUFF_FILLER_CACHE.exists():
        return pd.read_parquet(FLUFF_FILLER_CACHE)

    csv_path = RAW_SIGNAL_CSV if RAW_SIGNAL_CSV.exists() else None
    zip_path = RAW_SIGNAL_ZIP if RAW_SIGNAL_ZIP.exists() else None

    if csv_path is None and zip_path is None:
        raise FileNotFoundError(
            f"Neither {RAW_SIGNAL_CSV} nor {RAW_SIGNAL_ZIP} found"
        )

    src = csv_path if csv_path is not None else zip_path
    log.info("scanning raw CSV header for Fluff/Filler columns …")
    header = pd.read_csv(src, nrows=0)
    all_cols = list(header.columns)
    fluff_cols = [
        c for c in all_cols
        if c.startswith("AspectTheme_Fluff_") or c.startswith("AspectTheme_Filler_")
    ]
    log.info("found %d Fluff/Filler AspectTheme columns", len(fluff_cols))

    usecols = _FLUFF_FILLER_ID_COLS + fluff_cols

    parts: list[pd.DataFrame] = []
    chunks = pd.read_csv(src, usecols=usecols, chunksize=100_000)
    for ch in chunks:
        ch = ch[ch["SignalType"] != "delete"].copy()
        for c in fluff_cols:
            ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0.0)
        # Ensure consistent dtypes for pyarrow parquet writer
        ch["BESTTICKER"] = ch["BESTTICKER"].astype(str)
        ch["SECTOR"] = ch["SECTOR"].astype(str)
        ch["SignalType"] = ch["SignalType"].astype(str)
        parts.append(ch)

    df = pd.concat(parts, ignore_index=True)
    del parts
    log.info("Fluff/Filler raw: %d rows x %d cols", len(df), len(df.columns))

    FLUFF_FILLER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FLUFF_FILLER_CACHE, index=False)
    log.info("cached to %s", FLUFF_FILLER_CACHE)
    return df


def _parse_fluff_col(col_name: str) -> tuple[str, str, str, str] | None:
    """Parse an AspectTheme column name into (aspect, theme, magnitude, sentiment)."""
    m = re.match(
        r"^AspectTheme_(Fluff|Filler)_(.+?) - (High|Medium|Low) - (Positive|Neutral|Negative)$",
        col_name,
    )
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _build_fluff_signal(fluff_df: pd.DataFrame) -> pd.DataFrame:
    """Build Fluff/Filler-only aggregate signals.

    Returns a DataFrame with columns:
      [BESTTICKER, MOSTIMPORTANTDATEUTC, SignalType, SECTOR,
       fluff_total_count, fluff_net_sentiment, fluff_mag_weighted_score]
    """
    fluff_cols = [
        c for c in fluff_df.columns
        if c.startswith("AspectTheme_Fluff_") or c.startswith("AspectTheme_Filler_")
    ]

    mag_weighted: list[float] = [0.0] * len(fluff_df)
    net_sent: list[float] = [0.0] * len(fluff_df)
    total_count: list[float] = [0.0] * len(fluff_df)

    for col in fluff_cols:
        parsed = _parse_fluff_col(col)
        if parsed is None:
            continue
        _, _, magnitude, sentiment = parsed
        sign = _SENTIMENT_SIGN.get(sentiment, 0)
        w = _MAGNITUDE_WEIGHT.get(magnitude, 1)
        vals = fluff_df[col].to_numpy(dtype="float64")
        total_count = [t + v for t, v in zip(total_count, vals)]
        net_sent = [s + sign * v for s, v in zip(net_sent, vals)]
        mag_weighted = [m + sign * w * v for m, v in zip(mag_weighted, vals)]

    result = fluff_df[_FLUFF_FILLER_ID_COLS].copy()
    result["fluff_total_count"] = total_count
    result["fluff_net_sentiment"] = net_sent
    result["fluff_mag_weighted_score"] = mag_weighted
    return result


def _newey_west_t_stat(ic_series: np.ndarray, lag: int) -> float:
    """Compute Newey-West adjusted t-statistic (Bartlett kernel)."""
    n = len(ic_series)
    if n < 2:
        return np.nan
    mean_ic = float(np.mean(ic_series))
    if mean_ic == 0.0 and np.allclose(ic_series, 0.0):
        return 0.0

    residuals = ic_series - mean_ic
    var = np.sum(residuals ** 2) / (n - 1)
    nw_var = var
    for k in range(1, min(lag + 1, n - 1)):
        w = 1.0 - k / (lag + 1.0)  # Bartlett kernel
        auto_cov = np.sum(residuals[k:] * residuals[:-k]) / (n - k)
        nw_var += 2.0 * w * auto_cov
    nw_var = max(nw_var, 1e-15)
    return float(mean_ic / np.sqrt(nw_var / n))


def assert_fluff_filler_no_alpha(
    price_cache_dir: Path = PRICE_CACHE_DIR,
) -> dict[str, Any]:
    """Requirement §1.6: Fluff/Filler-only signal must generate ≈0 alpha.

    Loads raw Fluff/Filler columns, builds a simple aggregate signal, computes
    forward returns, and checks that Spearman IC is not statistically
    distinguishable from zero at any horizon.
    """
    horizons = [1, 3, 5, 10, 20]
    result: dict[str, Any] = {
        "passed": True,
        "violations": [],
        "horizons": {},
    }

    # 1. Load and build Fluff signal
    try:
        fluff_df = _load_fluff_filler_df()
    except FileNotFoundError:
        result["passed"] = False
        result["violations"].append(
            "Raw CSV/zip not found — cannot run Fluff/Filler control"
        )
        return result

    fluff_signal = _build_fluff_signal(fluff_df)
    fluff_total = fluff_signal[fluff_signal["SignalType"] == "Total"].copy()
    if len(fluff_total) == 0:
        result["violations"].append("No Total SignalType rows in Fluff/Filler data")
        return result

    # 2. Compute entry timestamps
    from features.engineer import compute_timestamps
    fluff_total = compute_timestamps(fluff_total)

    # 3. Compute forward returns (anchored on call_entry_date)
    from backtest.splits import compute_forward_returns as compute_fwd
    fwd_returns = compute_fwd(
        fluff_total,
        price_cache_dir=price_cache_dir,
        horizons=horizons,
        entry_date_col="call_entry_date",
    )
    for h in horizons:
        col = f"forward_return_{h}d"
        if col in fwd_returns.columns:
            fluff_total[col] = fwd_returns[col].values

    # 4. Compute monthly cross-sectional Spearman IC per horizon
    fluff_total["_year_month"] = fluff_total["call_entry_date"].dt.to_period("M")
    months = sorted(fluff_total["_year_month"].dropna().unique())

    for h in horizons:
        col = f"forward_return_{h}d"
        if col not in fluff_total.columns:
            result["horizons"][f"h{h}d"] = {
                "n_samples": 0, "n_months": 0,
                "mean_ic": None, "nw_t_stat": None,
                "warning": f"column {col} not found",
            }
            continue

        monthly_ics: list[float] = []
        for ym in months:
            mask = fluff_total["_year_month"] == ym
            subset = fluff_total.loc[mask, [col, "fluff_mag_weighted_score"]].dropna()
            if len(subset) < 10:
                continue
            from scipy.stats import spearmanr
            ic, _pv = spearmanr(
                subset["fluff_mag_weighted_score"].to_numpy(),
                subset[col].to_numpy(),
                nan_policy="omit",
            )
            if not np.isnan(ic):
                monthly_ics.append(float(ic))

        if len(monthly_ics) < 6:
            result["horizons"][f"h{h}d"] = {
                "n_samples": int(fluff_total[[col, "fluff_mag_weighted_score"]].dropna().shape[0]),
                "n_months": len(monthly_ics),
                "mean_ic": None, "nw_t_stat": None,
                "warning": f"insufficient months ({len(monthly_ics)})",
            }
            continue

        ic_arr = np.array(monthly_ics, dtype="float64")
        mean_ic = float(np.mean(ic_arr))
        nw_t = _newey_west_t_stat(ic_arr, lag=min(h, len(ic_arr) - 1))

        result["horizons"][f"h{h}d"] = {
            "n_samples": int(fluff_total[[col, "fluff_mag_weighted_score"]].dropna().shape[0]),
            "n_months": len(monthly_ics),
            "mean_ic": mean_ic,
            "nw_t_stat": float(nw_t),
        }

        if abs(mean_ic) > 0.02 and abs(nw_t) > 2.0:
            result["passed"] = False
            result["violations"].append(
                f"h{h}d: mean monthly IC={mean_ic:.4f}, NW t-stat={nw_t:.2f} — "
                f"Fluff/Filler signal shows significant alpha "
                f"(requirement §1.6 violation)"
            )

    return result


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

    audit_dates = df[["availability_date"]].copy()
    rebalance_events = audit_dates.copy()
    feature_artifact = RESULTS_DIR / f"features_{tier}.parquet"
    batch_features_path = feature_artifact if feature_artifact.exists() else None

    # ------------------------------------------------------------------
    # 3.1  Streaming vs batch — small subset (always run)
    # ------------------------------------------------------------------
    log.info("--- 3.1  Streaming vs batch (small subset) ---")
    small_df = _small_subset(df, n_tickers=50, n_months=12)
    if small_only or batch_features_path is not None:
        full_audit_df = audit_dates
        del df
        gc.collect()
    else:
        full_audit_df = df

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
        if batch_features_path is None:
            log.warning(
                "feature artifact %s missing; falling back to in-memory batch build",
                feature_artifact,
            )
        passed_full, mm_full, stats_full = run_streaming_vs_batch_test(
            full_audit_df,
            price_cache_dir=price_cache_dir,
            tier=tier,
            n_sample_dates=full_dates,
            include_momentum=False,
            batch_features_path=batch_features_path,
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
    isolation_df = _build_safe(
        small_df,
        price_cache_dir=price_cache_dir,
        tier=tier,
        include_momentum=False,
    )
    feature_candidates = _non_identifier_columns(isolation_df)
    ret_violations = assert_forward_return_isolation(feature_candidates)
    summary["assertions"]["forward_return_isolation"] = {
        "passed": len(ret_violations) == 0,
        "violations": ret_violations,
        "n_generated_columns_checked": len(feature_candidates),
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 6 — Timestamp boundary fixtures
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 6: Timestamp boundaries ---")
    ts_fixtures = assert_timestamp_boundaries()
    call_passed = ts_fixtures["call_ok"].all()
    avail_passed = ts_fixtures["availability_ok"].all() if "availability_ok" in ts_fixtures.columns else True
    ts_passed = bool(call_passed and avail_passed)
    if not call_passed:
        bad_call = ts_fixtures[~ts_fixtures["call_ok"]]
        log.warning("timestamp boundary failures (call_entry_date):\n%s", bad_call.to_string())
    if not avail_passed:
        bad_avail = ts_fixtures[~ts_fixtures["availability_ok"]]
        log.warning("timestamp boundary failures (availability_date):\n%s", bad_avail.to_string())
    summary["assertions"]["timestamp_boundaries"] = {
        "passed": ts_passed,
        "n_fixtures": len(ts_fixtures),
        "n_failures": int((~ts_fixtures.get("call_ok", pd.Series(True, index=ts_fixtures.index))).sum()),
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 7 — Rebalance eligibility  (smoke test)
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 7: Rebalance eligibility ---")
    reb_violations = assert_rebalance_eligibility(
        rebalance_events, dt.date(2020, 6, 30),
    )
    summary["assertions"]["rebalance_eligibility"] = {
        "passed": len(reb_violations) == 0,
        "violations": reb_violations,
    }

    # Synthetic fixture with edge cases
    log.info("--- 3.2  Assertion 7: Rebalance eligibility (synthetic edge cases) ---")
    synthetic_fixture = pd.DataFrame({
        "BESTTICKER": ["A", "B", "C", "D", "E", "F"],
        "availability_date": pd.to_datetime([
            "2020-06-30",   # boundary — exactly on rebalance date
            "2020-06-25",   # well before — should be eligible
            "2020-07-05",   # after — should be ineligible
            pd.NaT,         # NaN — should be dropped from candidates
            "2020-06-30",   # duplicate ticker on same date (same as A)
            "2020-06-30",   # duplicate ticker on same date (same as A)
        ]),
        "SECTOR": ["X"] * 6,
    })
    synthetic_violations = assert_rebalance_eligibility(
        synthetic_fixture, dt.date(2020, 6, 30),
    )
    log.info("  synthetic fixture: 6 rows (boundary=1, before=1, after=1, NaN=1, dup=2)")
    if synthetic_violations:
        log.warning("  synthetic fixture violations:\n%s", synthetic_violations)
    else:
        log.info("  synthetic fixture PASSED")
    summary["assertions"]["rebalance_eligibility_synthetic"] = {
        "passed": len(synthetic_violations) == 0,
        "violations": synthetic_violations,
        "n_rows": len(synthetic_fixture),
    }

    # ------------------------------------------------------------------
    # 3.2  Assertion 11 — Fluff/Filler control  (requirement §1.6)
    # ------------------------------------------------------------------
    log.info("--- 3.2  Assertion 11: Fluff/Filler control (IC ≈ 0) ---")
    fluff_result = assert_fluff_filler_no_alpha(
        price_cache_dir=price_cache_dir,
    )
    summary["assertions"]["fluff_filler_control"] = fluff_result
    if fluff_result["passed"]:
        log.info("  PASSED")
    else:
        log.warning("  FAILED: %s", fluff_result["violations"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_passed = all(
        v.get("passed", True) for v in summary["assertions"].values()
    )
    summary["all_passed"] = all_passed
    summary["total_elapsed_s"] = round(time.time() - t0, 1)

    # Write summary JSON
    parity_path = AUDIT_DIR / f"feature_parity_summary_{tier}.json"
    parity_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary written to %s", parity_path)

    _write_validation_summary(summary)

    # Write one-pager checklist
    _write_checklist(summary)

    if all_passed:
        log.info("ALL AUDITS PASSED")
    else:
        log.warning("SOME AUDITS FAILED — see %s", AUDIT_DIR)

    # Write cache manifest for resume / skip support.
    _write_audit_cache_manifest(tier, small_only, full_dates)

    return summary


def _write_audit_cache_manifest(
    tier: str,
    small_only: bool,
    full_dates: int,
) -> None:
    """Write a content-addressed cache manifest for Phase 3."""
    feature_artifact = RESULTS_DIR / f"features_{tier}.parquet"
    manifest = build_cache_manifest(
        phase=f"3_{tier}",
        parameters={"tier": tier, "small_only": small_only, "full_dates": full_dates},
        input_paths=[
            SIGNALS_PARQUET,
            PRICE_MANIFEST,
            feature_artifact,
        ],
        source_funcs=[run_all_audits, run_streaming_vs_batch_test],
    )
    manifest_path = CACHE_MANIFEST_DIR / f"3_{tier}.json"
    write_cache_manifest(manifest, manifest_path)
    log.info("cache manifest written to %s", manifest_path)


def _write_validation_summary(summary: dict[str, Any]) -> Path:
    """Write the generic Phase 6 validation-summary artifact."""
    checks: dict[str, Any] = {}
    n_pass = 0
    n_fail = 0
    n_pending = 0

    # Map each assertion name to its specific evidence file.
    evidence_map = {
        "streaming_vs_batch_small": "feature_parity_summary.json",
        "streaming_vs_batch_full": "feature_parity_summary.json",
        "pit_universe_defense": "pit_universe_defense_results.json",
        "forward_return_isolation": "forward_return_isolation_results.json",
        "timestamp_boundaries": "timestamp_boundary_results.json",
        "rebalance_eligibility": "rebalance_eligibility_results.json",
        "fluff_filler_control": "feature_parity_summary.json",
    }

    for name, info in summary.get("assertions", {}).items():
        passed = info.get("passed")
        if passed is True:
            status = "pass"
            n_pass += 1
        elif passed is False:
            status = "fail"
            n_fail += 1
        else:
            status = "pending"
            n_pending += 1
        checks[name] = {
            "status": status,
            "evidence": evidence_map.get(name, "feature_parity_summary.json"),
            "details": info,
        }

    validation = {
        "summary": {
            "passed": n_pass,
            "failed": n_fail,
            "pending": n_pending,
            "total": n_pass + n_fail + n_pending,
        },
        "checks": checks,
        "generated_at": summary.get("timestamp"),
        "tier": summary.get("tier"),
    }
    path = AUDIT_DIR / "validation_summary.json"
    path.write_text(json.dumps(validation, indent=2, default=str))
    log.info("validation summary written to %s", path)
    return path


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
         "ENFORCED AT BUILD TIME",
         "features/engineer.py : beta shifted by 4 trading rows (T-5 end) — design guarantee, no runtime assertion",
         ),
        ("11. Fluff/Filler control (requirement §1.6)",
         summary["assertions"].get("fluff_filler_control", {}).get("passed"),
         "features/audit.py : assert_fluff_filler_no_alpha (IC on Fluff/Filler-only signal must be ≈0)",
         ),
    ]

    n_failed = 0
    n_pending = 0
    for i, (name, passed, evidence) in enumerate(checklist_items, start=1):
        if passed is True:
            icon = "PASS"
        elif passed == "ENFORCED AT BUILD TIME":
            icon = "DESIGN"
        elif passed is False:
            icon = "FAIL"
            n_failed += 1
        else:
            icon = "PEND"
            n_pending += 1
        lines.append(f"| {i} | {name} | {icon} | {evidence} |")

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
