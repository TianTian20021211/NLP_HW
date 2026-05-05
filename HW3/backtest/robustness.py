"""Phase 5.5 — Robustness Checks.

Subperiod analysis, sector neutralization, market-cap buckets, block bootstrap,
ATC-score-weighted portfolios, and OFAT sensitivity.

Usage::

    python -m backtest.robustness \\
        --features results/features_enhanced.parquet \\
        --universe sp500
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.config import (
    CACHE_MANIFEST_DIR,
    PRICE_CACHE_DIR,
    RESULTS_DIR,
    SHARES_CACHE_DIR,
    UNIVERSE_CACHE_DIR,
    UNIVERSE_NAMES,
)
from data.progress import progress

from backtest._stats import (
    bucket_returns,
    build_equity_curves,
    dedup_latest_per_ticker,
    max_drawdown_from_equity,
    portfolio_stats,
)
from data.cache_utils import build_cache_manifest, write_cache_manifest

log = logging.getLogger("backtest.robustness")

HORIZONS: list[int] = [1, 3, 5, 10, 20]
SIGNAL_TYPES: list[str] = ["Total", "CEO", "CFO", "Analysts", "Executives"]

# Subperiod boundaries
SUBPERIODS: dict[str, tuple[str, str]] = {
    "pre_2020": ("2010-01-01", "2019-12-31"),
    "2020_2022": ("2020-01-01", "2022-12-31"),
    "2023_2026": ("2023-01-01", "2026-06-30"),
}


# ===================================================================
# 1. Subperiod analysis
# ===================================================================


def subperiod_label(ts: pd.Timestamp) -> str:
    """Return the subperiod label for a timestamp."""
    if ts < pd.Timestamp("2020-01-01"):
        return "pre_2020"
    if ts < pd.Timestamp("2023-01-01"):
        return "2020_2022"
    return "2023_2026"


def _monthly_ic_by_subperiod(
    sub: pd.DataFrame,
    feature_col: str,
    return_col: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute monthly IC with subperiod labels."""
    mask = sub[feature_col].notna() & sub[return_col].notna()
    sub = sub[mask]
    records: list[dict[str, Any]] = []
    for (month,), gdf in sub.groupby(["year_month"], observed=True):
        if len(gdf) < min_samples:
            continue
        gdf = dedup_latest_per_ticker(gdf)
        if len(gdf) < min_samples:
            continue
        ic, _ = spearmanr(gdf[feature_col].values, gdf[return_col].values)
        records.append(
            {
                "year_month": month,
                "subperiod": subperiod_label(gdf["call_entry_date"].iloc[0]),
                "ic": ic if not np.isnan(ic) else np.nan,
                "n_samples": len(gdf),
            }
        )
    return pd.DataFrame(records)


# Module-level global for ProcessPoolExecutor workers (cf. quintile.py pattern).
_ROBUSTNESS_DFS: dict[str, pd.DataFrame] = {}


def _subperiod_ic_worker(args: tuple) -> list[dict[str, Any]]:
    feat, horizon, sig_type = args
    signal_dfs = _ROBUSTNESS_DFS
    ret_col = f"forward_return_{horizon}d"
    ic_df = _monthly_ic_by_subperiod(signal_dfs[sig_type], feat, ret_col)
    if ic_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for sp in SUBPERIODS:
        sp_ic = ic_df[ic_df["subperiod"] == sp]["ic"].dropna()
        n = len(sp_ic)
        rows.append(
            {
                "feature": feat,
                "horizon": horizon,
                "signal_type": sig_type,
                "subperiod": sp,
                "n_periods": n,
                "mean_ic": float(sp_ic.mean()) if n > 0 else np.nan,
                "ic_std": float(sp_ic.std(ddof=1)) if n > 1 else np.nan,
                "t_stat": float(sp_ic.mean() / sp_ic.std(ddof=1) * np.sqrt(n))
                if n > 1 and sp_ic.std(ddof=1) > 0
                else np.nan,
                "hit_rate": float((sp_ic > 0).mean()) if n > 0 else np.nan,
            }
        )
    return rows


def _subperiod_quintile_worker(args: tuple) -> list[dict[str, Any]]:
    feat, horizon, sig_type = args
    signal_dfs = _ROBUSTNESS_DFS
    ret_col = f"forward_return_{horizon}d"
    sub = signal_dfs[sig_type]
    mask = sub[feat].notna() & sub[ret_col].notna()
    sub = sub[mask]
    rows: list[dict[str, Any]] = []
    for sp, (sp_start, sp_end) in SUBPERIODS.items():
        sp_df = sub[
            (sub["call_entry_date"] >= pd.Timestamp(sp_start))
            & (sub["call_entry_date"] <= pd.Timestamp(sp_end))
        ]
        monthly_ret = bucket_returns(
            sp_df, feat, ret_col, n_buckets=5,
            group_col="bucket_date",
            date_col="availability_date",
        )
        if monthly_ret.empty:
            continue
        eq = build_equity_curves(monthly_ret, n_buckets=5, group_col="bucket_date")
        lsp_col = "long_short"
        rets = eq[lsp_col].dropna() if lsp_col in eq.columns else pd.Series(dtype=float)
        n_m = len(rets)
        rows.append(
            {
                "feature": feat,
                "horizon": horizon,
                "signal_type": sig_type,
                "subperiod": sp,
                "n_months": n_m,
                "ann_return": float(rets.mean() * (252 / horizon)) if n_m > 1 else np.nan,
                "ann_vol": float(rets.std() * np.sqrt(252 / horizon)) if n_m > 1 else np.nan,
                "sharpe": float(rets.mean() / rets.std() * np.sqrt(252 / horizon))
                if n_m > 1 and rets.std() > 0
                else np.nan,
                "max_drawdown": (
                    float(max_drawdown_from_equity(eq[f"cum_{lsp_col}"]))
                    if f"cum_{lsp_col}" in eq.columns
                    else np.nan
                ),
            }
        )
    return rows


def run_subperiod_ic(
    signal_dfs: dict[str, pd.DataFrame],
    features: list[str] | None = None,
    output_dir: Path | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Recompute IC split by subperiod for all feature x horizon x SignalType combos.

    Returns a DataFrame with columns:
    ``[feature, horizon, signal_type, subperiod, n_periods, mean_ic, ic_std, t_stat, hit_rate]``.
    """
    global _ROBUSTNESS_DFS
    if features is None:
        from backtest.single_feature_ic import SHORT_LIST_FEATURES as features

    first_df = next(iter(signal_dfs.values()))
    features = [f for f in features if f in first_df.columns]

    combos = list(itertools.product(features, HORIZONS, SIGNAL_TYPES))
    _ROBUSTNESS_DFS = signal_dfs
    _n_jobs = n_jobs if n_jobs else min(os.cpu_count() or 4, 6)

    if _n_jobs > 1:
        with ProcessPoolExecutor(max_workers=_n_jobs) as ex:
            futures = [ex.submit(_subperiod_ic_worker, c) for c in combos]
            chunk_rows = []
            for f in progress(as_completed(futures), total=len(combos), desc="Subperiod IC"):
                chunk_rows.append(f.result())
    else:
        chunk_rows = []
        for combo in progress(combos, desc="Subperiod IC"):
            chunk_rows.append(_subperiod_ic_worker(combo))

    rows: list[dict[str, Any]] = []
    for chunk in chunk_rows:
        rows.extend(chunk)

    result = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_dir / "robustness_subperiod_ic.parquet")
    return result


def run_subperiod_quintile(
    signal_dfs: dict[str, pd.DataFrame],
    features: list[str] | None = None,
    output_dir: Path | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Recompute quintile L/S spread by subperiod.

    Uses T-0 daily bucketing (group_col=bucket_date) to match the main
    quintile analysis in ``backtest/quintile.py``.
    """
    global _ROBUSTNESS_DFS
    if features is None:
        from backtest.single_feature_ic import SHORT_LIST_FEATURES as features

    first_df = next(iter(signal_dfs.values()))
    features = [f for f in features if f in first_df.columns]

    for st in SIGNAL_TYPES:
        signal_dfs[st]["bucket_date"] = pd.to_datetime(
            signal_dfs[st]["availability_date"], errors="coerce"
        ).dt.normalize()

    _ROBUSTNESS_DFS = signal_dfs
    combos = list(itertools.product(features, HORIZONS, SIGNAL_TYPES))
    _n_jobs = n_jobs if n_jobs else min(os.cpu_count() or 4, 6)

    if _n_jobs > 1:
        with ProcessPoolExecutor(max_workers=_n_jobs) as ex:
            futures = [ex.submit(_subperiod_quintile_worker, c) for c in combos]
            chunk_rows = []
            for f in progress(as_completed(futures), total=len(combos), desc="Subperiod quintile"):
                chunk_rows.append(f.result())
    else:
        chunk_rows = []
        for combo in progress(combos, desc="Subperiod quintile"):
            chunk_rows.append(_subperiod_quintile_worker(combo))

    rows: list[dict[str, Any]] = []
    for chunk in chunk_rows:
        rows.extend(chunk)

    result = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_dir / "robustness_subperiod_quintile.parquet")
    return result


# ===================================================================
# 2. Sector neutralization (signal-stage within-sector ranking)
# ===================================================================


def sector_neutral_quintile(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int = 5,
    min_per_bucket: int = 5,
) -> pd.DataFrame:
    """Quintile buckets formed within each GICS sector, then merged across sectors.

    The plan requires "rank quintiles inside each GICS sector, then merge
    long/short legs" (plan 3.6). This means each sector contributes equal
    numbers to each quintile bucket.
    """
    mask = df[feature_col].notna() & df[return_col].notna() & df["SECTOR"].notna()
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["year_month", "bucket", "ret", "n"])

    records: list[dict[str, Any]] = []
    for (month, sector), gdf in sub.groupby(
        ["year_month", "SECTOR"], observed=True
    ):
        gdf = dedup_latest_per_ticker(gdf)
        if len(gdf) < n_buckets * min_per_bucket:
            continue
        try:
            gdf["_bucket"] = pd.qcut(
                gdf[feature_col], q=n_buckets, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        buckets = gdf.groupby("_bucket")[return_col]
        for b_idx, b_ret in buckets.mean().items():
            records.append(
                {
                    "year_month": month,
                    "sector": sector,
                    "bucket": int(b_idx),
                    "ret": float(b_ret),
                    "n": int(buckets.size()[b_idx]),
                }
            )
    # Merge across sectors: average bucket returns across sectors per month
    result = pd.DataFrame(records)
    if result.empty:
        return result
    # Aggregate: mean across sectors for each (month, bucket)
    result = result.groupby(["year_month", "bucket"], as_index=False).agg(
        {"ret": "mean", "n": "sum"}
    )
    return result


# ===================================================================
# 3. Market-cap buckets
# ===================================================================


def _load_shares_lookup(
    tickers: set[str],
    target_date: pd.Timestamp,
    shares_dir: Path = SHARES_CACHE_DIR,
) -> dict[str, float]:
    """Return ``{ticker: shares_outstanding}`` for the most recent date <= *target_date*.

    Retained for compatibility with any external callers; the vectorized
    ``assign_market_cap_buckets`` no longer routes through this helper.
    """
    result: dict[str, float] = {}
    for tkr in tickers:
        path = shares_dir / f"{tkr}.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path, columns=["date", "shares"])
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s[s["date"] <= target_date]
        if s.empty:
            continue
        row = s.loc[s["date"].idxmax()]
        if row["shares"] > 0:
            result[tkr] = float(row["shares"])
    return result


def _load_ticker_history(
    tickers: set[str],
    cache_dir: Path,
    value_col: str,
    require_positive: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read each ticker's parquet *once* and return ``(dates, values)`` dicts.

    Both arrays are sorted ascending by date. Used by
    ``assign_market_cap_buckets`` to amortize parquet I/O across all months.
    """
    dates_map: dict[str, np.ndarray] = {}
    values_map: dict[str, np.ndarray] = {}
    for tkr in tickers:
        if pd.isna(tkr):
            continue
        tkr = str(tkr)
        path = cache_dir / f"{tkr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["date", value_col])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        mask = df["date"].notna() & df[value_col].notna()
        if require_positive:
            mask &= df[value_col] > 0
        df = df[mask]
        if df.empty:
            continue
        df = df.sort_values("date")
        dates_map[tkr] = df["date"].to_numpy(dtype="datetime64[ns]")
        values_map[tkr] = df[value_col].to_numpy(dtype="float64")
    return dates_map, values_map


def assign_market_cap_buckets(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    shares_cache_dir: Path = SHARES_CACHE_DIR,
    universe_name: str = "sp500",
    coverage_warn_threshold: float = 0.85,
) -> pd.DataFrame:
    """Add ``_mcap_bucket`` column to *df*.

    Buckets: ``mega`` (top 10%), ``large`` (10-40%), ``mid`` (40-70%),
    ``small`` (bottom 30%). Events without shares or price data are labelled
    ``unknown``.

    PIT semantics:
    * universe membership from the latest PIT snapshot on or before event date
    * price / shares as-of ``< event_date`` (strictly left-exclusive, T-1)
    * cross-sectional quantile boundaries computed inside the same-day universe.

    The inner per-date loop uses pre-built aligned matrices so every per-ticker
    ``np.searchsorted`` call is eliminated in favour of a single vectorised
    indexing operation per date.
    """
    universe_path = UNIVERSE_CACHE_DIR / f"{universe_name}_pit.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(universe_path, columns=["date", "ticker"])
        if len(universe) > 0:
            universe["date"] = pd.to_datetime(universe["date"], errors="coerce").astype(
                "datetime64[ns]"
            )
            universe["ticker"] = universe["ticker"].astype(str)
            universe = universe.dropna(subset=["date", "ticker"]).drop_duplicates()
    else:
        universe = pd.DataFrame(columns=["date", "ticker"])

    df = df.copy()
    df["_mcap_bucket"] = pd.Series("unknown", index=df.index, dtype="object")
    df["_mcap_coverage"] = np.nan

    if universe.empty:
        log.warning("market-cap buckets: universe %s is empty or missing", universe_name)
        return df

    snapshots = {
        pd.Timestamp(d): set(g["ticker"].astype(str))
        for d, g in universe.groupby("date", sort=True)
    }
    snapshot_dates = pd.DatetimeIndex(sorted(snapshots))

    all_tickers: set[str] = set(universe["ticker"].astype(str).unique())
    log.info(
        "market-cap buckets: pre-loading price/shares for %d PIT members",
        len(all_tickers),
    )
    px_dates, px_close = _load_ticker_history(
        all_tickers, price_cache_dir, "adj_close", require_positive=True
    )
    sh_dates, sh_shares = _load_ticker_history(
        all_tickers, shares_cache_dir, "shares", require_positive=True
    )

    # ---- Build aligned (n_tickers x n_dates) matrices -------------------
    # Collect every ticker that appears in BOTH price and shares data.
    covered = sorted(set(px_dates) & set(sh_dates))
    if not covered:
        log.warning("market-cap buckets: no ticker has both price and shares data")
        return df
    ticker_to_idx = {t: i for i, t in enumerate(covered)}
    n_tickers = len(covered)

    # Common date axis: sorted union of all price dates.
    all_date_sets: list[np.ndarray] = []
    for tkr in covered:
        all_date_sets.append(px_dates[tkr])
    common_dates = np.unique(np.concatenate(all_date_sets))
    common_dates = np.sort(common_dates)
    n_dates = len(common_dates)
    date_to_pos = {d: i for i, d in enumerate(common_dates)}

    # Build and forward-fill price matrix.
    px_mat = np.full((n_tickers, n_dates), np.nan, dtype=np.float64)
    for tkr in covered:
        i = ticker_to_idx[tkr]
        for d, v in zip(px_dates[tkr], px_close[tkr]):
            pos = date_to_pos.get(d)
            if pos is not None:
                px_mat[i, pos] = v
    px_df = pd.DataFrame(px_mat).ffill(axis=1)
    px_mat = px_df.to_numpy(dtype=np.float64)
    del px_df

    # Build and forward-fill shares matrix.
    sh_mat = np.full((n_tickers, n_dates), np.nan, dtype=np.float64)
    for tkr in covered:
        i = ticker_to_idx[tkr]
        for d, v in zip(sh_dates[tkr], sh_shares[tkr]):
            pos = date_to_pos.get(d)
            if pos is not None:
                sh_mat[i, pos] = v
    sh_df = pd.DataFrame(sh_mat).ffill(axis=1)
    sh_mat = sh_df.to_numpy(dtype=np.float64)
    del sh_df

    log.info(
        "market-cap buckets: built (%d x %d) aligned matrices, %.1f MB",
        n_tickers, n_dates,
        (px_mat.nbytes + sh_mat.nbytes) / (1024 * 1024),
    )

    # ---- Per-date loop (vectorised inner) -------------------------------
    bucket_arr = df["_mcap_bucket"].to_numpy().copy()
    coverage_arr = df["_mcap_coverage"].to_numpy().copy()
    bestticker_arr = df["BESTTICKER"].astype("object").to_numpy()
    call_entry_arr = pd.to_datetime(df["call_entry_date"], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )

    valid_dates = ~np.isnat(call_entry_arr)
    valid_pos = np.flatnonzero(valid_dates)
    if valid_pos.size == 0:
        df["_mcap_bucket"] = bucket_arr
        df["_mcap_coverage"] = coverage_arr
        return df

    date_values = call_entry_arr[valid_pos]
    order = np.argsort(date_values, kind="mergesort")
    sorted_dates = date_values[order]
    sorted_pos = valid_pos[order]
    unique_dates, starts = np.unique(sorted_dates, return_index=True)
    warned_months: set[pd.Period] = set()

    for date_i, event_date_np in enumerate(
        progress(unique_dates, desc="Market-cap buckets", unit="date")
    ):
        start_i = starts[date_i]
        end_i = starts[date_i + 1] if date_i + 1 < len(starts) else len(sorted_pos)
        event_idx = sorted_pos[start_i:end_i]
        event_ts = pd.Timestamp(event_date_np)

        snap_pos = snapshot_dates.searchsorted(event_ts, side="right") - 1
        if snap_pos < 0:
            continue

        members = snapshots.get(pd.Timestamp(snapshot_dates[snap_pos]), set())
        if not members:
            continue

        # T-1 date position on common axis.
        date_pos = int(np.searchsorted(common_dates, event_date_np, side="left")) - 1
        if date_pos < 0:
            continue

        # Vectorised: all tickers' T-1 values at once.
        px_t1 = px_mat[:, date_pos]
        sh_t1 = sh_mat[:, date_pos]

        # Map PIT members to matrix rows.
        member_indices = np.array(
            [ticker_to_idx[t] for t in members if t in ticker_to_idx],
            dtype=np.intp,
        )
        if len(member_indices) == 0:
            continue

        member_px = px_t1[member_indices]
        member_sh = sh_t1[member_indices]
        valid_mask = np.isfinite(member_px) & np.isfinite(member_sh)
        mcaps_vals = member_px[valid_mask] * member_sh[valid_mask]

        if len(mcaps_vals) == 0:
            continue

        p10 = float(np.quantile(mcaps_vals, 0.9))
        p40 = float(np.quantile(mcaps_vals, 0.6))
        p70 = float(np.quantile(mcaps_vals, 0.3))

        # Per-row bucket assignment (vectorised).
        row_tickers = bestticker_arr[event_idx]
        row_indices = np.array(
            [ticker_to_idx.get(str(t), -1) if pd.notna(t) else -1 for t in row_tickers],
            dtype=np.intp,
        )
        known = row_indices >= 0
        row_mcap = np.full(len(row_tickers), np.nan, dtype=np.float64)
        if known.any():
            row_mcap[known] = px_t1[row_indices[known]] * sh_t1[row_indices[known]]
        row_known = ~np.isnan(row_mcap)
        conds = [
            row_known & (row_mcap >= p10),
            row_known & (row_mcap >= p40),
            row_known & (row_mcap >= p70),
            row_known,
        ]
        bucket_arr[event_idx] = np.select(conds, ["mega", "large", "mid", "small"], default="unknown")

        n_in_universe = len(members)
        n_covered = int(valid_mask.sum())
        coverage = n_covered / n_in_universe if n_in_universe > 0 else 1.0
        coverage_arr[event_idx] = coverage
        if coverage < coverage_warn_threshold:
            month_key = event_ts.to_period("M")
            if month_key not in warned_months:
                warned_months.add(month_key)
                log.warning(
                    "Market-cap coverage %.1f%% for %s (< %.0f%%); "
                    "first warned date=%s",
                    coverage * 100,
                    month_key,
                    coverage_warn_threshold * 100,
                    event_ts.date(),
                )

    df["_mcap_bucket"] = bucket_arr
    df["_mcap_coverage"] = coverage_arr
    return df


def mcap_bucket_quintile(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
) -> pd.DataFrame:
    """Compute quintile L/S spread within each market-cap bucket."""
    if "_mcap_bucket" not in df.columns:
        raise ValueError("Run assign_market_cap_buckets() first")

    records: list[dict[str, Any]] = []
    for bucket_name in ["mega", "large", "mid", "small"]:
        bucket_df = df[df["_mcap_bucket"] == bucket_name]
        if bucket_df.empty:
            continue
        monthly = bucket_returns(bucket_df, feature_col, return_col, n_buckets=5)
        eq = build_equity_curves(monthly, n_buckets=5)
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna()
        n_m = len(rets)
        records.append(
            {
                "mcap_bucket": bucket_name,
                "n_months": n_m,
                "ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
                "sharpe": float(rets.mean() / rets.std() * np.sqrt(12))
                if n_m > 1 and rets.std() > 0
                else np.nan,
            }
        )
    return pd.DataFrame(records)


# ===================================================================
# 4. Block bootstrap
# ===================================================================


def block_bootstrap(
    returns: np.ndarray,
    block_size: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Monthly block bootstrap for mean return and Sharpe ratio 95% CI.

    Parameters
    ----------
    returns:
        1-d array of periodic returns (e.g. monthly or daily).
    block_size:
        Number of observations per block (21 trading days ≈ 1 month).
    n_boot:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    periods_per_year:
        Annualization factor matching the *frequency of ``returns``*.
        Use 252 for daily returns, 12 for monthly returns.  Sharpe is
        scaled by ``sqrt(periods_per_year)``; a mismatched value
        overstates or understates the reported Sharpe CI.

    Returns
    -------
    Dict with ``mean_ci_low, mean_ci_high, sharpe_ci_low, sharpe_ci_high``.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < block_size:
        return {
            "mean_ci_low": np.nan,
            "mean_ci_high": np.nan,
            "sharpe_ci_low": np.nan,
            "sharpe_ci_high": np.nan,
        }

    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot)
    boot_sharpes = np.empty(n_boot)

    sqrt_ppy = np.sqrt(periods_per_year)
    for boot_i in range(n_boot):
        block_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Build sample from blocks via concatenation, then truncate to n.
        blocks = [returns[bi: bi + block_size] for bi in block_indices]
        sample = np.concatenate(blocks)[:n]
        boot_means[boot_i] = float(sample.mean())
        ann_r = float(sample.mean()) * periods_per_year
        ann_v = float(sample.std()) * sqrt_ppy
        boot_sharpes[boot_i] = ann_r / ann_v if ann_v > 0 else np.nan

    boot_sharpes = boot_sharpes[np.isfinite(boot_sharpes)]

    return {
        "mean_ci_low": float(np.percentile(boot_means, 2.5)),
        "mean_ci_high": float(np.percentile(boot_means, 97.5)),
        "sharpe_ci_low": float(np.percentile(boot_sharpes, 2.5))
        if len(boot_sharpes) > 0
        else np.nan,
        "sharpe_ci_high": float(np.percentile(boot_sharpes, 97.5))
        if len(boot_sharpes) > 0
        else np.nan,
    }


# ===================================================================
# 5. ATC-score-weighted portfolios
# ===================================================================


def score_weighted_bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Weight positions by absolute score value within each bucket.

    Equal-weight bucketing is the main convention; this is a robustness
    check to test whether weighting by signal strength adds value.
    """
    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["year_month", "bucket", "ret", "n"])

    records: list[dict[str, Any]] = []
    for month, gdf in sub.groupby("year_month", observed=True):
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
        if len(gdf) < n_buckets * 3:
            continue
        try:
            gdf["_bucket"] = pd.qcut(
                gdf[feature_col], q=n_buckets, labels=False, duplicates="drop"
            )
        except ValueError:
            continue

        # Score-weighted: weight each position by |feature_value|
        for b_idx in range(n_buckets):
            bdf = gdf[gdf["_bucket"] == b_idx]
            if bdf.empty:
                continue
            abs_score = bdf[feature_col].abs()
            total_abs = abs_score.sum()
            if total_abs == 0:
                w = pd.Series(1.0 / len(bdf), index=bdf.index)
            else:
                w = abs_score / total_abs
            w_ret = float((w * bdf[return_col]).sum())
            records.append(
                {
                    "year_month": month,
                    "bucket": int(b_idx),
                    "ret": w_ret,
                    "n": len(bdf),
                }
            )
    return pd.DataFrame(records)


# ===================================================================
# 6. OFAT Sensitivity
# ===================================================================


def ofat_quantile_cutoff(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    cutoffs: list[int] | None = None,
) -> pd.DataFrame:
    """Vary the number of stocks in long/short legs.

    Parameters
    ----------
    cutoffs:
        Number of stocks in each leg. Default mirrors the plan:
        ``[5, 10, 20, 50, 100]`` (top-5 = ~decile, top-10, top-20, top-50, top-100).
    """
    if cutoffs is None:
        cutoffs = [5, 10, 20, 50, 100]

    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask].copy()
    sub["year_month"] = sub["call_entry_date"].dt.to_period("M")

    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        monthly = _top_n_bucket_returns(sub, feature_col, return_col, cutoff)
        eq = _build_equity_curves_2leg(monthly)
        if eq.empty:
            continue
        rets = eq["long_short"].dropna()
        n_m = len(rets)
        rows.append(
            {
                "cutoff": cutoff,
                "n_months": n_m,
                "ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
                "sharpe": float(rets.mean() / rets.std() * np.sqrt(12))
                if n_m > 1 and rets.std() > 0
                else np.nan,
                "max_drawdown": float(max_drawdown_from_equity(eq["cum_long_short"]))
                if "cum_long_short" in eq.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _top_n_bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_stocks: int,
) -> pd.DataFrame:
    """Long top-N, short bottom-N by feature value."""
    records: list[dict[str, Any]] = []
    for month, gdf in df.groupby("year_month", observed=True):
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
        if len(gdf) < 2 * n_stocks:
            continue
        gdf = gdf.sort_values(feature_col)
        short_ret = gdf.head(n_stocks)[return_col].mean()
        long_ret = gdf.tail(n_stocks)[return_col].mean()
        records.append(
            {
                "year_month": month,
                "long_only": float(long_ret),
                "short_only": float(-short_ret),
                "long_short": float(long_ret - short_ret),
            }
        )
    return pd.DataFrame(records)


def _build_equity_curves_2leg(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build equity curves from a DataFrame with long_short column."""
    if monthly.empty:
        return pd.DataFrame()
    eq = monthly.set_index("year_month").sort_index()
    for leg in ["long_only", "short_only", "long_short"]:
        if leg in eq.columns:
            eq[f"cum_{leg}"] = (1 + eq[leg]).cumprod()
    return eq


def ofat_transaction_cost(
    daily_returns: pd.DataFrame,
    cost_levels_bps: list[float] | None = None,
) -> pd.DataFrame:
    """Recompute post-cost Sharpe at different transaction cost levels.

    Parameters
    ----------
    daily_returns:
        DataFrame with ``[pnl, turnover]`` columns from ``PortfolioResult.daily_returns``.
    cost_levels_bps:
        One-way transaction costs in bps. Default: ``[3, 5, 7, 10]``.
    """
    if cost_levels_bps is None:
        cost_levels_bps = [3.0, 5.0, 7.0, 10.0]

    rows: list[dict[str, Any]] = []
    for cost_bps in cost_levels_bps:
        cost_per_day = cost_bps / 10_000.0 * daily_returns["turnover"]
        post_pnl = daily_returns["pnl"] - cost_per_day
        ann_ret = float(post_pnl.mean() * 252)
        ann_vol = float(post_pnl.std() * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rows.append(
            {
                "cost_bps": cost_bps,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe_post_cost": sharpe,
            }
        )
    return pd.DataFrame(rows)


def compare_weighting_schemes(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Compare equal-weight vs ATC-score-weighted quintile portfolios.

    Returns a DataFrame with one row per weighting scheme.
    """
    # Equal-weight
    ew = bucket_returns(df, feature_col, return_col, n_buckets)
    ew_eq = build_equity_curves(ew, n_buckets)
    # Score-weighted
    sw = score_weighted_bucket_returns(df, feature_col, return_col, n_buckets)
    sw_eq = build_equity_curves(sw, n_buckets)

    rows: list[dict[str, Any]] = []
    for scheme, eq in [("equal_weight", ew_eq), ("score_weighted", sw_eq)]:
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna()
        n_m = len(rets)
        rows.append(
            {
                "weighting": scheme,
                "n_months": n_m,
                "ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
                "sharpe": float(rets.mean() / rets.std() * np.sqrt(12))
                if n_m > 1 and rets.std() > 0
                else np.nan,
                "max_drawdown": float(max_drawdown_from_equity(eq["cum_long_short"]))
                if "cum_long_short" in eq.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ===================================================================
# Main orchestrator
# ===================================================================


@dataclass
class RobustnessResult:
    subperiod_ic: pd.DataFrame = field(default_factory=pd.DataFrame)
    subperiod_quintile: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_neutral_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    mcap_bucket_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    weighting_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofat_quantile: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofat_cost: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofat_lookback: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_purge_gap: pd.DataFrame = field(default_factory=pd.DataFrame)
    beta_window: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_ci: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def _load_robustness_features(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load features, ensure forward returns, filter PIT universe, and add year_month."""
    from backtest.splits import ensure_forward_returns, read_feature_columns
    from backtest.universe import filter_to_universe

    log.info("Loading features from %s", features_path)
    required_cols = [
        "SignalType", "call_entry_date", "availability_date",
        "BESTTICKER", "SECTOR", "MOSTIMPORTANTDATEUTC",
    ]
    df = read_feature_columns(
        features_path,
        [*required_cols, *features],
        required_columns=required_cols,
    )

    log.info("Ensuring forward returns")
    df = ensure_forward_returns(
        df, features_path, price_cache_dir,
        entry_date_col="availability_date",
    )

    log.info("Filtering to universe %s", universe_name)
    df = filter_to_universe(df, universe_name)
    df = df[df["_in_universe"]].copy()
    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    available_features = [f for f in features if f in df.columns]
    log.info("Features available: %s", available_features)

    needed_cols = [
        "SignalType", "year_month", "call_entry_date", "availability_date",
        "BESTTICKER", "SECTOR", "MOSTIMPORTANTDATEUTC",
    ] + available_features
    for h in HORIZONS:
        needed_cols.append(f"forward_return_{h}d")
        needed_cols.append(f"target_available_date_{h}d")
    needed_cols = list(dict.fromkeys(c for c in needed_cols if c in df.columns))
    df = df.loc[:, needed_cols].copy()
    gc.collect()

    return df, available_features


def _run_subperiod_ic_section(
    signal_dfs: dict[str, pd.DataFrame],
    features: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run and persist subperiod IC robustness."""
    log.info("=== Subperiod IC ===")
    return run_subperiod_ic(signal_dfs, features, output_dir)


def _run_subperiod_quintile_section(
    signal_dfs: dict[str, pd.DataFrame],
    features: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run and persist subperiod quintile robustness."""
    log.info("=== Subperiod quintile ===")
    return run_subperiod_quintile(signal_dfs, features, output_dir)


def _run_sector_neutral_section(
    total_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare raw vs sector-neutral ATC quintiles for Total across horizons."""
    log.info("=== Sector neutralization ===")
    sn_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        sn = sector_neutral_quintile(total_df, "ATCClassifierScore", ret_col, n_buckets=5)
        eq = build_equity_curves(sn, n_buckets=5)
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna()
        n_m = len(rets)
        raw = bucket_returns(total_df, "ATCClassifierScore", ret_col, n_buckets=5)
        raw_eq = build_equity_curves(raw, n_buckets=5)
        raw_rets = (
            raw_eq["long_short"].dropna()
            if not raw_eq.empty and "long_short" in raw_eq.columns
            else pd.Series(dtype=float)
        )
        sn_rows.append(
            {
                "horizon": horizon,
                "sector_neutral_sharpe": float(rets.mean() / rets.std() * np.sqrt(12))
                if n_m > 1 and rets.std() > 0
                else np.nan,
                "sector_neutral_ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
                "raw_sharpe": float(raw_rets.mean() / raw_rets.std() * np.sqrt(12))
                if len(raw_rets) > 1 and raw_rets.std() > 0
                else np.nan,
                "n_months_sn": n_m,
                "n_months_raw": len(raw_rets),
            }
        )
    result = pd.DataFrame(sn_rows)
    if output_dir:
        result.to_parquet(output_dir / "robustness_sector_neutral.parquet")
    return result


def _run_mcap_bucket_section(
    df: pd.DataFrame,
    universe_name: str,
    price_cache_dir: Path,
    shares_cache_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run market-cap bucket robustness for horizons 1, 5, and 20."""
    log.info("=== Market-cap buckets ===")
    df_mcap = assign_market_cap_buckets(
        df, price_cache_dir, shares_cache_dir, universe_name
    )
    mcap_rows: list[dict[str, Any]] = []
    for horizon in [1, 5, 20]:
        ret_col = f"forward_return_{horizon}d"
        bucket_df = mcap_bucket_quintile(df_mcap, "ATCClassifierScore", ret_col)
        if not bucket_df.empty:
            bucket_df["horizon"] = horizon
            mcap_rows.append(bucket_df)
    result = pd.concat(mcap_rows, ignore_index=True) if mcap_rows else pd.DataFrame()
    if output_dir:
        result.to_parquet(output_dir / "robustness_mcap_buckets.parquet")
    return result


def _run_weighting_section(
    total_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare equal-weight and score-weighted ATC quintiles."""
    log.info("=== Weighting scheme ===")
    wt_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        comp = compare_weighting_schemes(
            total_df,
            "ATCClassifierScore",
            ret_col,
        )
        comp["horizon"] = horizon
        wt_rows.append(comp)
    result = pd.concat(wt_rows, ignore_index=True) if wt_rows else pd.DataFrame()
    if output_dir:
        result.to_parquet(output_dir / "robustness_weighting.parquet")
    return result


def _run_ofat_quantile_section(
    total_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run top-N cutoff sensitivity for ATCClassifierScore / Total."""
    log.info("=== OFAT quantile cutoff ===")
    ofat_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        cutoff_df = ofat_quantile_cutoff(total_df, "ATCClassifierScore", ret_col)
        if not cutoff_df.empty:
            cutoff_df["horizon"] = horizon
            ofat_rows.append(cutoff_df)
    result = pd.concat(ofat_rows, ignore_index=True) if ofat_rows else pd.DataFrame()
    if output_dir:
        result.to_parquet(output_dir / "robustness_ofat_quantile.parquet")
    return result


def _run_ofat_cost_section(
    universe_name: str,
    output_dir: Path,
    portfolio_dir: Path = RESULTS_DIR / "portfolio",
) -> pd.DataFrame:
    """Load weekly 5d portfolio returns if available and run cost sensitivity."""
    log.info("=== OFAT transaction cost ===")
    portfolio_candidates = [
        portfolio_dir / f"daily_returns_{universe_name}_lightgbm_weekly_5d.parquet",
        portfolio_dir / f"daily_returns_{universe_name}_weekly_5d.parquet",
        *sorted(portfolio_dir.glob(f"daily_returns_{universe_name}_*_weekly_5d.parquet")),
    ]
    portfolio_path = next((p for p in portfolio_candidates if p.exists()), None)
    if portfolio_path is not None:
        dr = pd.read_parquet(portfolio_path)
        result = ofat_transaction_cost(dr)
        if output_dir:
            result.to_parquet(output_dir / "robustness_ofat_cost.parquet")
        return result
    else:
        log.warning(
            "No weekly 5d portfolio daily returns found in %s — skipping cost OFAT",
            portfolio_dir,
        )
        return pd.DataFrame()


def _baseline_total_signals(total_df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical Total ATC signal frame for portfolio robustness."""
    required = {"availability_date", "BESTTICKER", "ATCClassifierScore"}
    missing = required - set(total_df.columns)
    if missing:
        log.warning("Cannot build baseline signals; missing columns: %s", sorted(missing))
        return pd.DataFrame(columns=["date", "ticker", "score"])
    signals = total_df[["availability_date", "BESTTICKER", "ATCClassifierScore"]].copy()
    signals = signals.rename(
        columns={
            "availability_date": "date",
            "BESTTICKER": "ticker",
            "ATCClassifierScore": "score",
        }
    )
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    signals = signals.dropna(subset=["date", "ticker", "score"])
    signals["ticker"] = signals["ticker"].astype(str)
    return signals


def _summary_row(summary: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    row = dict(metadata)
    for key, value in summary.items():
        if isinstance(value, (int, float, np.integer, np.floating)) or pd.isna(value):
            row[key] = float(value) if pd.notna(value) else np.nan
    return row


def _run_portfolio_combined_section(
    total_df: pd.DataFrame,
    universe_name: str,
    price_cache_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerun portfolio simulation across OFAT cadence/lookback + weekly timing.

    Both modules share one PortfolioSimulator instance and the weekly_5d_monday
    run is computed only once.
    """
    log.info("=== Portfolio robustness (OFAT lookback + weekly timing) ===")
    signals = _baseline_total_signals(total_df)
    if signals.empty:
        empty_df = pd.DataFrame()
        empty_df.to_parquet(output_dir / "robustness_ofat_lookback.parquet")
        empty_df.to_parquet(output_dir / "robustness_weekly_timing.parquet")
        return empty_df, empty_df

    from backtest.portfolio import PortfolioSimulator

    sim = PortfolioSimulator(price_dir=price_cache_dir, universe_name=universe_name)

    # Build all unique combos.  weekly_5_monday appears in both OFAT and
    # timing — run it once and share the result.
    combos: list[dict[str, Any]] = []
    # OFAT lookbacks
    for cadence, lookbacks in [("daily", [1, 3]), ("weekly", [3, 5, 10]), ("monthly", [15, 21, 30])]:
        for lookback in lookbacks:
            combos.append({"cadence": cadence, "lookback": lookback, "weekly_day": "monday", "source": "ofat"})
    # Weekly timing (friday only — monday_5d already covered above)
    combos.append({"cadence": "weekly", "lookback": 5, "weekly_day": "friday", "source": "timing"})

    ofat_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for combo in combos:
        result = sim.run(
            signals,
            cadence=combo["cadence"],
            lookback=combo["lookback"],
            transaction_cost_bps=5.0,
            weekly_day=combo["weekly_day"],
        )
        row = _summary_row(
            result.summary(cost_bps=5.0),
            cadence=combo["cadence"],
            lookback=combo["lookback"],
            weekly_day=combo["weekly_day"] if combo["cadence"] == "weekly" else None,
            n_rebalance_dates=len(result.weights_history),
        )
        if combo["source"] == "ofat":
            ofat_rows.append(row)
        if combo["cadence"] == "weekly" and combo["lookback"] == 5:
            timing_rows.append(row)

    ofat_df = pd.DataFrame(ofat_rows)
    timing_df = pd.DataFrame(timing_rows)
    ofat_df.to_parquet(output_dir / "robustness_ofat_lookback.parquet")
    timing_df.to_parquet(output_dir / "robustness_weekly_timing.parquet")
    return ofat_df, timing_df


def _run_label_purge_gap_section(
    df: pd.DataFrame,
    output_dir: Path,
    availability_col: str = "availability_date",
) -> pd.DataFrame:
    """Audit training sample retained under extra label-purge gaps.

    The main model already applies the G11 target-availability purge at each
    fold. This quantifies extra rows removed by tightening that cutoff by
    0, 5, and 21 business days; gap=0 is the negative-control row.
    """
    log.info("=== Label purge gap sensitivity ===")
    from backtest.splits import generate_folds

    folds = generate_folds(df, availability_col=availability_col)
    rows: list[dict[str, Any]] = []
    for gap_bdays in [0, 5, 21]:
        for horizon in HORIZONS:
            target_col = f"target_available_date_{horizon}d"
            target_dates = pd.to_datetime(df[target_col], errors="coerce")
            for fold in folds:
                cutoff = fold.train_end - pd.offsets.BDay(gap_bdays)
                base_idx = fold.train_indices
                valid = target_dates.iloc[base_idx].notna()
                kept = valid & (target_dates.iloc[base_idx] <= cutoff)
                rows.append(
                    {
                        "gap_bdays": gap_bdays,
                        "horizon": horizon,
                        "fold_id": fold.fold_id,
                        "train_end": fold.train_end,
                        "effective_target_cutoff": pd.Timestamp(cutoff),
                        "candidate_rows": int(len(base_idx)),
                        "target_available_rows": int(valid.sum()),
                        "kept_rows": int(kept.sum()),
                        "purged_rows": int(valid.sum() - kept.sum()),
                        "negative_control": gap_bdays == 0,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_parquet(output_dir / "robustness_label_purge_gap.parquet")
    if not out.empty and (out["gap_bdays"] == 0).any():
        log.warning("Label purge gap=0 is a negative control, not a deployment setting")
    return out


def _run_bootstrap_section(
    total_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Run quarterly-block bootstrap on monthly ATC decile L/S returns.

    CRITICAL: ``block_bootstrap`` defaults to ``periods_per_year=252`` (daily).
    This function must pass ``periods_per_year=12`` because it operates on
    monthly decile L/S returns from ``build_equity_curves``. Omitting the
    explicit argument would annualize the Sharpe ratio with sqrt(252) instead
    of sqrt(12), inflating reported Sharpe CIs by approximately 4.6x.
    """
    log.info("=== Block bootstrap ===")
    bootstrap_results: dict[str, Any] = {}
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        monthly = bucket_returns(total_df, "ATCClassifierScore", ret_col, n_buckets=10)
        eq = build_equity_curves(monthly, n_buckets=10)
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna().values
        ci = block_bootstrap(
            rets,
            block_size=1,
            n_boot=2000,
            periods_per_year=12,
        )
        ci["horizon"] = horizon
        ci["n_months"] = len(rets)
        bootstrap_results[f"h{horizon}d"] = ci
    if output_dir:
        json_path = output_dir / "robustness_bootstrap_ci.json"
        json_path.write_text(json.dumps(bootstrap_results, indent=2, default=str))
    return bootstrap_results


def _run_beta_window_section(
    signal_dfs: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    universe_name: str,
    price_cache_dir: Path,
    output_dir: Path,
    tier: str = "enhanced",
) -> pd.DataFrame:
    """Recompute R8 idiosyncratic residual IC under beta window variants.

    beta_window=60 reuses the existing df column (already computed in Phase 2).
    beta_window=40/90 are computed on demand and cached to parquet.
    """
    log.info("=== R8 beta window IC check ===")
    r8_rows: list[dict[str, Any]] = []
    baseline_signs: dict[tuple[int, str], float] = {}

    from features.engineer import compute_momentum_features

    cache_dir = RESULTS_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for beta_window in [60, 40, 90]:
        if beta_window == 60:
            resid = df["pre_event_idio_resid_5d"]
        else:
            cache_path = cache_dir / f"momentum_beta_{beta_window}_{universe_name}_{tier}.parquet"
            if cache_path.exists():
                momentum = pd.read_parquet(cache_path)
                resid = momentum["pre_event_idio_resid_5d"]
                log.info("Beta window %d: loaded from cache", beta_window)
            else:
                momentum = compute_momentum_features(
                    df,
                    price_cache_dir=price_cache_dir,
                    beta_window=beta_window,
                    beta_min_periods=30,
                    beta_lag=4,
                )
                resid = momentum["pre_event_idio_resid_5d"]
                momentum[["pre_event_idio_resid_5d"]].to_parquet(cache_path)
                log.info("Beta window %d: computed and cached", beta_window)

        for horizon in HORIZONS:
            ret_col = f"forward_return_{horizon}d"
            for sig_type in SIGNAL_TYPES:
                sub = signal_dfs[sig_type]
                sub_resid = resid.loc[sub.index]
                mask = sub_resid.notna() & sub[ret_col].notna()
                valid = sub[mask]
                if len(valid) < 30:
                    continue
                ic, _ = spearmanr(
                    sub_resid.loc[valid.index].values,
                    valid[ret_col].values,
                )
                sign_key = (horizon, sig_type)
                if beta_window == 60:
                    baseline_signs[sign_key] = np.sign(ic) if np.isfinite(ic) else np.nan
                baseline_sign = baseline_signs.get(sign_key, np.nan)
                current_sign = np.sign(ic) if np.isfinite(ic) else np.nan
                r8_rows.append(
                    {
                        "beta_window": beta_window,
                        "horizon": horizon,
                        "signal_type": sig_type,
                        "ic": ic,
                        "n_events": len(valid),
                        "ic_sign_flips": bool(
                            np.isfinite(baseline_sign)
                            and np.isfinite(current_sign)
                            and current_sign != baseline_sign
                        ),
                    }
                )
    r8_df = pd.DataFrame(r8_rows)
    if output_dir:
        r8_df.to_parquet(output_dir / "robustness_r8_beta_window.parquet")
    return r8_df


# -------------------------------------------------------------------
# Section-name mapping for cache manifests
# -------------------------------------------------------------------

_SECTION_MANIFEST_CONFIG: dict = {
    _run_subperiod_ic_section: ("subperiod_ic", False),
    _run_subperiod_quintile_section: ("subperiod_quintile", False),
    _run_sector_neutral_section: ("sector_neutral", False),
    _run_weighting_section: ("weighting", False),
    _run_ofat_quantile_section: ("ofat_quantile", False),
    _run_ofat_cost_section: ("ofat_cost", False),
    _run_portfolio_combined_section: ("portfolio_combined", False),
    _run_label_purge_gap_section: ("label_purge_gap", False),
    _run_bootstrap_section: ("bootstrap", False),
    _run_beta_window_section: ("beta_window", False),
    _run_mcap_bucket_section: ("mcap_buckets", True),
}


def _write_section_manifest(
    section_name: str,
    universe_name: str,
    features_path: Path,
    source_funcs: list,
    price_manifest: Path,
    shares_manifest: Path | None = None,
    tier: str = "enhanced",
) -> None:
    """Write a cache manifest recording inputs for a completed robustness section."""
    phase = f"5d_{section_name}_{tier}_{universe_name}"
    input_paths = [features_path, price_manifest]
    if shares_manifest is not None:
        input_paths.append(shares_manifest)
    manifest = build_cache_manifest(
        phase=phase,
        parameters={"universe_name": universe_name, "section": section_name},
        input_paths=input_paths,
        source_funcs=source_funcs,
    )
    write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"{phase}.json")


def _write_section_manifests(
    func,
    universe_name: str,
    features_path: Path,
    tier: str = "enhanced",
) -> None:
    """Dispatch manifest writes for the section tied to *func*."""
    info = _SECTION_MANIFEST_CONFIG.get(func)
    if info is None:
        return
    section_name, needs_shares = info
    price_manifest = PRICE_CACHE_DIR / "_manifest.json"
    shares_manifest = SHARES_CACHE_DIR / "_manifest.json" if needs_shares else None
    sub_sections = (
        ["ofat_lookback", "weekly_timing"]
        if section_name == "portfolio_combined"
        else [section_name]
    )
    for sub_name in sub_sections:
        _write_section_manifest(
            section_name=sub_name,
            universe_name=universe_name,
            features_path=features_path,
            source_funcs=[func],
            price_manifest=price_manifest,
            shares_manifest=shares_manifest,
            tier=tier,
        )


def run_all_robustness(
    features_path: Path,
    universe_name: str = "sp500",
    price_cache_dir: Path = PRICE_CACHE_DIR,
    shares_cache_dir: Path = SHARES_CACHE_DIR,
    output_dir: Path | None = None,
    features: list[str] | None = None,
    skip_mcap: bool = False,
    tier: str = "enhanced",
    n_jobs: int = 0,
) -> RobustnessResult:
    """Run all Phase 5.5 robustness checks.

    Parameters
    ----------
    features_path:
        Path to enhanced features parquet (must have forward returns joined).
    universe_name:
        Which universe to analyse.
    skip_mcap:
        Skip market-cap bucket analysis (slow, requires shares data).
    """
    from backtest.single_feature_ic import SHORT_LIST_FEATURES

    if features is None:
        features = [f for f in SHORT_LIST_FEATURES if f != "dummy"]

    output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, available_features = _load_robustness_features(
        features_path, universe_name, price_cache_dir, features,
    )

    # Pre-split by SignalType so downstream sections avoid 700+ redundant
    # boolean-mask filter passes over the full dataframe.
    signal_dfs = {st: df[df["SignalType"] == st].copy() for st in SIGNAL_TYPES}
    total_df = signal_dfs["Total"]

    result = RobustnessResult()

    # Sections as (func, args, attr_name_or_None)
    # None attr means side-effect-only (file written by func itself)
    _n_jobs = n_jobs if n_jobs > 0 else 3
    sections: list[tuple[Any, tuple, str | None]] = [
        (_run_subperiod_ic_section, (signal_dfs, available_features, output_dir), "subperiod_ic"),
        (_run_subperiod_quintile_section, (signal_dfs, available_features, output_dir), "subperiod_quintile"),
        (_run_sector_neutral_section, (total_df, output_dir), "sector_neutral_summary"),
        (_run_weighting_section, (total_df, output_dir), "weighting_comparison"),
        (_run_ofat_quantile_section, (total_df, output_dir), "ofat_quantile"),
        (_run_ofat_cost_section, (universe_name, output_dir), "ofat_cost"),
        (_run_portfolio_combined_section, (total_df, universe_name, price_cache_dir, output_dir), None),
        (_run_label_purge_gap_section, (df, output_dir), "label_purge_gap"),
        (_run_bootstrap_section, (total_df, output_dir), "bootstrap_ci"),
        (_run_beta_window_section, (signal_dfs, df, universe_name, price_cache_dir, output_dir, tier), "beta_window"),
    ]
    if not skip_mcap:
        sections.append(
            (_run_mcap_bucket_section, (df, universe_name, price_cache_dir, shares_cache_dir, output_dir), "mcap_bucket_summary")
        )

    if _n_jobs > 1:
        with ProcessPoolExecutor(max_workers=_n_jobs) as ex:
            future_map = {}
            for func, fargs, attr in sections:
                future_map[ex.submit(func, *fargs)] = (func, attr)
            for f in as_completed(future_map):
                func, attr = future_map[f]
                try:
                    val = f.result()
                except Exception as exc:
                    log.error("Robustness section %s failed: %s", attr, exc)
                    raise
                if attr is not None:
                    setattr(result, attr, val)
                _write_section_manifests(func, universe_name, features_path, tier=tier)
    else:
        for func, fargs, attr in sections:
            val = func(*fargs)
            if attr is not None:
                setattr(result, attr, val)
            _write_section_manifests(func, universe_name, features_path, tier=tier)

    result.config = {
        "universe": universe_name,
        "features": available_features,
        "horizons": HORIZONS,
        "signal_types": SIGNAL_TYPES,
        "subperiods": list(SUBPERIODS.keys()),
        "ofat_lookbacks": {
            "daily": [1, 3],
            "weekly": [3, 5, 10],
            "monthly": [15, 21, 30],
        },
        "weekly_timing": ["monday", "friday"],
        "label_purge_gap_bdays": [0, 5, 21],
        "beta_windows": [40, 60, 90],
    }

    return result


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Robustness checks (Phase 5.5)")
    p.add_argument(
        "--features",
        type=Path,
        default=RESULTS_DIR / "features_enhanced.parquet",
    )
    p.add_argument(
        "--universe",
        choices=["sp500", "sp1500", "ru3k", "all"],
        default="all",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "robustness",
    )
    p.add_argument("--price-cache", type=Path, default=PRICE_CACHE_DIR)
    p.add_argument("--shares-cache", type=Path, default=SHARES_CACHE_DIR)
    p.add_argument(
        "--skip-mcap",
        action="store_true",
        help="Skip market-cap bucket analysis (slow)",
    )
    p.add_argument("--tier", type=str, default="enhanced",
                   help="Feature tier name for cache manifest isolation")
    p.add_argument("--n-jobs", type=int, default=0,
                   help="Parallel workers for robustness sections (0=memory-safe auto)")
    args = p.parse_args()

    universes = list(UNIVERSE_NAMES) if args.universe == "all" else [args.universe]
    for univ in universes:
        log.info("==== Robustness checks for %s ====", univ)
        run_all_robustness(
            features_path=args.features,
            universe_name=univ,
            price_cache_dir=args.price_cache,
            shares_cache_dir=args.shares_cache,
            output_dir=args.output_dir,
            skip_mcap=args.skip_mcap,
            tier=args.tier,
            n_jobs=args.n_jobs,
        )
        log.info("Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
