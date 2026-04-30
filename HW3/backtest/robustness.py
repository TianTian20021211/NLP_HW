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
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.config import (
    PRICE_CACHE_DIR,
    RESULTS_DIR,
    SHARES_CACHE_DIR,
    UNIVERSE_CACHE_DIR,
    UNIVERSE_NAMES,
)
from data.progress import progress

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
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    signal_type: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute monthly IC with subperiod labels."""
    sub = df[df["SignalType"] == signal_type]
    mask = sub[feature_col].notna() & sub[return_col].notna()
    sub = sub[mask]
    records: list[dict[str, Any]] = []
    for (month,), gdf in sub.groupby(["year_month"], observed=True):
        if len(gdf) < min_samples:
            continue
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
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


def run_subperiod_ic(
    df: pd.DataFrame,
    features: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Recompute IC split by subperiod for all feature × horizon × SignalType combos.

    Returns a DataFrame with columns:
    ``[feature, horizon, signal_type, subperiod, n_periods, mean_ic, ic_std, t_stat, hit_rate]``.
    """
    if features is None:
        from backtest.single_feature_ic import SHORT_LIST_FEATURES as features

    features = [f for f in features if f in df.columns]
    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    rows: list[dict[str, Any]] = []
    total = len(features) * len(HORIZONS) * len(SIGNAL_TYPES)

    for feat, horizon, sig_type in progress(
        itertools.product(features, HORIZONS, SIGNAL_TYPES),
        total=total,
        desc="Subperiod IC",
    ):
        ret_col = f"forward_return_{horizon}d"
        ic_df = _monthly_ic_by_subperiod(df, feat, ret_col, sig_type)
        if ic_df.empty:
            continue
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

    result = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_dir / "robustness_subperiod_ic.parquet")
    return result


def run_subperiod_quintile(
    df: pd.DataFrame,
    features: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Recompute quintile L/S spread by subperiod."""
    if features is None:
        from backtest.single_feature_ic import SHORT_LIST_FEATURES as features

    features = [f for f in features if f in df.columns]
    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    rows: list[dict[str, Any]] = []
    total = len(features) * len(HORIZONS) * len(SIGNAL_TYPES)

    for feat, horizon, sig_type in progress(
        itertools.product(features, HORIZONS, SIGNAL_TYPES),
        total=total,
        desc="Subperiod quintile",
    ):
        ret_col = f"forward_return_{horizon}d"
        sub = df[df["SignalType"] == sig_type]
        mask = sub[feat].notna() & sub[ret_col].notna()
        sub = sub[mask]
        for sp, (sp_start, sp_end) in SUBPERIODS.items():
            sp_df = sub[
                (sub["call_entry_date"] >= pd.Timestamp(sp_start))
                & (sub["call_entry_date"] <= pd.Timestamp(sp_end))
            ]
            monthly_ret = _monthly_bucket_returns(sp_df, feat, ret_col, n_buckets=5)
            if monthly_ret.empty:
                continue
            eq = _build_equity_curves(monthly_ret, n_buckets=5)
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
                    "ann_return": float(rets.mean() * 12) if n_m > 1 else np.nan,
                    "ann_vol": float(rets.std() * np.sqrt(12)) if n_m > 1 else np.nan,
                    "sharpe": float(rets.mean() / rets.std() * np.sqrt(12))
                    if n_m > 1 and rets.std() > 0
                    else np.nan,
                    "max_drawdown": float(_max_drawdown(eq[lsp_col])) if lsp_col in eq.columns else np.nan,
                }
            )

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
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
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

    Implementation note: each ticker's price/shares history is read from
    parquet exactly once at the start, then ``np.searchsorted`` and
    ``np.select`` are used per-event-date for as-of lookups and bucket
    assignment.  This replaces the previous per-month-per-ticker parquet
    reads + per-row ``df.loc`` assigns and is roughly 10× faster on a
    full SP500 run while preserving the intended PIT semantics:

    * universe membership from the latest PIT snapshot on or before event date
    * price as-of ``< event_date`` (left-exclusive, i.e. T-1)
    * shares as-of ``< event_date`` (left-exclusive, i.e. T-1)
    * cross-sectional quantile boundaries computed inside the same-day universe.

    Returns *df* with ``_mcap_bucket`` and ``_mcap_coverage`` columns added.
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

    # Pre-load every PIT universe ticker's price + shares history exactly once.
    # Bucket thresholds must be formed from the same-day universe, not only
    # from tickers that happened to have an event that month.
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
        event_np = np.datetime64(event_ts)

        mcaps: dict[str, float] = {}
        for tkr in members:
            pd_arr = px_dates.get(tkr)
            sd_arr = sh_dates.get(tkr)
            if pd_arr is None or sd_arr is None:
                continue
            # Price and shares strictly before event date (T-1 semantics).
            pos_p = int(np.searchsorted(pd_arr, event_np, side="left")) - 1
            if pos_p < 0:
                continue
            pos_s = int(np.searchsorted(sd_arr, event_np, side="left")) - 1
            if pos_s < 0:
                continue
            mcaps[tkr] = float(sh_shares[tkr][pos_s] * px_close[tkr][pos_p])

        if not mcaps:
            continue

        mcap_series = pd.Series(mcaps).sort_values()
        p10 = mcap_series.quantile(0.9)
        p40 = mcap_series.quantile(0.6)
        p70 = mcap_series.quantile(0.3)

        # Vectorized per-row bucket assignment.
        row_tickers = bestticker_arr[event_idx]
        row_mcap = np.fromiter(
            (mcaps.get(str(t), np.nan) if pd.notna(t) else np.nan for t in row_tickers),
            dtype="float64",
            count=len(row_tickers),
        )
        known = ~np.isnan(row_mcap)
        # np.select picks the first True per row, so order matters.
        conds = [
            known & (row_mcap >= p10),
            known & (row_mcap >= p40),
            known & (row_mcap >= p70),
            known,
        ]
        choices = ["mega", "large", "mid", "small"]
        bucket_arr[event_idx] = np.select(conds, choices, default="unknown")

        n_in_universe = len(members)
        n_covered = len(mcaps)
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
        monthly = _monthly_bucket_returns(bucket_df, feature_col, return_col, n_buckets=5)
        eq = _build_equity_curves(monthly, n_buckets=5)
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
    boot_means: list[float] = []
    boot_sharpes: list[float] = []

    sqrt_ppy = np.sqrt(periods_per_year)
    for _ in range(n_boot):
        block_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample: list[float] = []
        for bi in block_indices:
            sample.extend(returns[bi : bi + block_size].tolist())
        sample = np.array(sample[:n])
        boot_means.append(float(sample.mean()))
        ann_r = float(sample.mean()) * periods_per_year
        ann_v = float(sample.std()) * sqrt_ppy
        boot_sharpes.append(ann_r / ann_v if ann_v > 0 else np.nan)

    boot_means = np.array(boot_means)
    boot_sharpes = np.array(boot_sharpes)
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
    sub = df[mask]
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
                "max_drawdown": float(_max_drawdown(eq["long_short"]))
                if "long_short" in eq.columns
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
    ew = _monthly_bucket_returns(df, feature_col, return_col, n_buckets)
    ew_eq = _build_equity_curves(ew, n_buckets)
    # Score-weighted
    sw = score_weighted_bucket_returns(df, feature_col, return_col, n_buckets)
    sw_eq = _build_equity_curves(sw, n_buckets)

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
                "max_drawdown": float(_max_drawdown(eq["long_short"]))
                if "long_short" in eq.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ===================================================================
# Shared helpers (mirror quintile.py for independence)
# ===================================================================


def _monthly_bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int,
    min_per_bucket: int = 5,
) -> pd.DataFrame:
    """Form equal-size buckets per month."""
    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask]
    records: list[dict[str, Any]] = []
    for month, gdf in sub.groupby("year_month", observed=True):
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
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
                    "bucket": int(b_idx),
                    "ret": float(b_ret),
                    "n": int(buckets.size()[b_idx]),
                }
            )
    return pd.DataFrame(records)


def _build_equity_curves(bucket_returns: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    """Build cumulative equity curves from monthly bucket returns."""
    if bucket_returns.empty:
        return pd.DataFrame()
    piv = bucket_returns.pivot_table(
        index="year_month", columns="bucket", values="ret", aggfunc="mean"
    )
    piv.columns = [f"bucket_{int(c)}" for c in piv.columns]
    piv = piv.sort_index()
    for col in piv.columns:
        piv[f"cum_{col}"] = (1 + piv[col]).cumprod()
    top = f"bucket_{n_buckets - 1}"
    bot = "bucket_0"
    if top in piv.columns and bot in piv.columns:
        piv["long_only"] = piv[top]
        piv["short_only"] = -piv[bot]
        piv["long_short"] = piv[top] - piv[bot]
        for leg in ["long_only", "short_only", "long_short"]:
            piv[f"cum_{leg}"] = (1 + piv[leg]).cumprod()
    return piv


def _max_drawdown(cum: pd.Series) -> float:
    """Maximum drawdown from a cumulative return series."""
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()) if len(dd) > 0 else np.nan


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
    bootstrap_ci: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def run_all_robustness(
    features_path: Path,
    universe_name: str = "sp500",
    price_cache_dir: Path = PRICE_CACHE_DIR,
    shares_cache_dir: Path = SHARES_CACHE_DIR,
    output_dir: Path | None = None,
    features: list[str] | None = None,
    skip_mcap: bool = False,
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
    from backtest.splits import get_forward_returns_cached
    from backtest.single_feature_ic import SHORT_LIST_FEATURES
    from backtest.quintile import _filter_to_universe

    if features is None:
        features = [f for f in SHORT_LIST_FEATURES if f != "dummy"]

    output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    # Compute forward returns if needed (uses on-disk cache when valid).
    has_returns = any(c.startswith("forward_return_") for c in df.columns)
    if not has_returns:
        log.info("Computing forward returns (cached)")
        fwd = get_forward_returns_cached(features_path, df, price_cache_dir, entry_date_col="availability_date")
        for col in fwd.columns:
            df[col] = fwd[col]

    log.info("Filtering to universe %s", universe_name)
    df = _filter_to_universe(df, universe_name)
    df = df[df["_in_universe"]].copy()
    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    available_features = [f for f in features if f in df.columns]
    log.info("Features available: %s", available_features)

    result = RobustnessResult()

    # ---- 1. Subperiod IC ----
    log.info("=== Subperiod IC ===")
    result.subperiod_ic = run_subperiod_ic(df, available_features, output_dir)

    # ---- 2. Subperiod quintile ----
    log.info("=== Subperiod quintile ===")
    result.subperiod_quintile = run_subperiod_quintile(df, available_features, output_dir)

    # ---- 3. Sector neutralization (ATCClassifierScore, Total, h=5d) ----
    log.info("=== Sector neutralization ===")
    sn_rows: list[dict[str, Any]] = []
    total_df = df[df["SignalType"] == "Total"].copy()
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        sn = sector_neutral_quintile(total_df, "ATCClassifierScore", ret_col, n_buckets=5)
        eq = _build_equity_curves(sn, n_buckets=5)
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna()
        n_m = len(rets)
        # Also compute non-neutralized for comparison
        raw = _monthly_bucket_returns(
            total_df,
            "ATCClassifierScore",
            ret_col,
            n_buckets=5,
        )
        raw_eq = _build_equity_curves(raw, n_buckets=5)
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
    result.sector_neutral_summary = pd.DataFrame(sn_rows)
    if output_dir:
        result.sector_neutral_summary.to_parquet(
            output_dir / "robustness_sector_neutral.parquet"
        )

    # ---- 4. Market-cap buckets (optional — requires shares data) ----
    if not skip_mcap:
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
        if mcap_rows:
            result.mcap_bucket_summary = pd.concat(mcap_rows, ignore_index=True)
        if output_dir:
            result.mcap_bucket_summary.to_parquet(
                output_dir / "robustness_mcap_buckets.parquet"
            )

    # ---- 5. Weighting scheme comparison ----
    log.info("=== Weighting scheme ===")
    wt_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        comp = compare_weighting_schemes(
            df[df["SignalType"] == "Total"],
            "ATCClassifierScore",
            ret_col,
        )
        comp["horizon"] = horizon
        wt_rows.append(comp)
    result.weighting_comparison = (
        pd.concat(wt_rows, ignore_index=True) if wt_rows else pd.DataFrame()
    )
    if output_dir:
        result.weighting_comparison.to_parquet(
            output_dir / "robustness_weighting.parquet"
        )

    # ---- 6. OFAT: quantile cutoff ----
    log.info("=== OFAT quantile cutoff ===")
    ofat_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        sub = df[df["SignalType"] == "Total"]
        cutoff_df = ofat_quantile_cutoff(sub, "ATCClassifierScore", ret_col)
        if not cutoff_df.empty:
            cutoff_df["horizon"] = horizon
            ofat_rows.append(cutoff_df)
    result.ofat_quantile = (
        pd.concat(ofat_rows, ignore_index=True) if ofat_rows else pd.DataFrame()
    )
    if output_dir:
        result.ofat_quantile.to_parquet(output_dir / "robustness_ofat_quantile.parquet")

    # ---- 7. OFAT: transaction cost (from portfolio results if available) ----
    log.info("=== OFAT transaction cost ===")
    portfolio_dir = RESULTS_DIR / "portfolio"
    portfolio_candidates = [
        portfolio_dir / f"daily_returns_{universe_name}_lightgbm_weekly_5d.parquet",
        portfolio_dir / f"daily_returns_{universe_name}_weekly_5d.parquet",
        *sorted(portfolio_dir.glob(f"daily_returns_{universe_name}_*_weekly_5d.parquet")),
    ]
    portfolio_path = next((p for p in portfolio_candidates if p.exists()), None)
    if portfolio_path is not None:
        dr = pd.read_parquet(portfolio_path)
        result.ofat_cost = ofat_transaction_cost(dr)
        if output_dir:
            result.ofat_cost.to_parquet(output_dir / "robustness_ofat_cost.parquet")
    else:
        log.info("No weekly 5d portfolio daily returns found in %s — skipping cost OFAT", portfolio_dir)

    # ---- 8. Block bootstrap on decile L/S (ATCClassifierScore, Total) ----
    log.info("=== Block bootstrap ===")
    bootstrap_results: dict[str, Any] = {}
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        sub = df[df["SignalType"] == "Total"]
        monthly = _monthly_bucket_returns(sub, "ATCClassifierScore", ret_col, n_buckets=10)
        eq = _build_equity_curves(monthly, n_buckets=10)
        if eq.empty or "long_short" not in eq.columns:
            continue
        rets = eq["long_short"].dropna().values
        # _build_equity_curves yields *monthly* L/S returns; annualize with 12.
        ci = block_bootstrap(
            rets,
            block_size=3,  # 3 months ≈ 1 quarter block
            n_boot=2000,
            periods_per_year=12,
        )
        ci["horizon"] = horizon
        ci["n_months"] = len(rets)
        bootstrap_results[f"h{horizon}d"] = ci
    result.bootstrap_ci = bootstrap_results
    if output_dir:
        json_path = output_dir / "robustness_bootstrap_ci.json"
        json_path.write_text(json.dumps(bootstrap_results, indent=2, default=str))

    # ---- 9. R8 beta window check — IC sign of pre_event_idio_resid_5d ----
    log.info("=== R8 beta window IC check ===")
    r8_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        ret_col = f"forward_return_{horizon}d"
        for sig_type in SIGNAL_TYPES:
            sub = df[df["SignalType"] == sig_type]
            mask = sub["pre_event_idio_resid_5d"].notna() & sub[ret_col].notna()
            valid = sub[mask]
            if len(valid) < 30:
                continue
            ic, _ = spearmanr(
                valid["pre_event_idio_resid_5d"].values,
                valid[ret_col].values,
            )
            r8_rows.append(
                {
                    "horizon": horizon,
                    "signal_type": sig_type,
                    "ic": ic,
                    "n_events": len(valid),
                    "ic_sign_flips": bool(ic < 0),
                }
            )
    r8_df = pd.DataFrame(r8_rows)
    if output_dir:
        r8_df.to_parquet(output_dir / "robustness_r8_beta_window.parquet")

    result.config = {
        "universe": universe_name,
        "features": available_features,
        "horizons": HORIZONS,
        "signal_types": SIGNAL_TYPES,
        "subperiods": list(SUBPERIODS.keys()),
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
        )
        log.info("Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
