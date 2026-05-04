"""Phase 5.1 — Single-Feature IC Analysis.

Cross-sectional Spearman IC for the 14-column short list, computed by month,
year, and sector, with Newey-West adjusted t-statistics.

Usage::

    python -m backtest.single_feature_ic \
        --features results/features_enhanced.parquet \
        --universe sp500
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backtest.universe import filter_to_universe
from data.cache_utils import build_cache_manifest, write_cache_manifest
from data.config import (
    CACHE_MANIFEST_DIR,
    PRICE_CACHE_DIR,
    RESULTS_DIR,
    UNIVERSE_CACHE_DIR,
    UNIVERSE_NAMES,
)
from data.progress import progress

log = logging.getLogger("backtest.ic")

# ---------------------------------------------------------------------------
# 14-column short list (frozen in Phase 5.1)
# ---------------------------------------------------------------------------
SHORT_LIST_FEATURES: list[str] = [
    "ATCClassifierScore",
    "EventsScore_4_2_1",
    "EventsScore_1_1_1",
    "EventsScore_3_1_0",
    "EventsScore_1_1_0",
    "aspect_Surprise_net_sentiment",
    "theme_FinancialPerformance_net_sentiment",
    "theme_StrategicInitiatives_net_sentiment",
    "qoq_delta_ATCClassifierScore",
    "ATCClassifierScore_sector_pct",
    "pre_event_ret_21d",
    "pre_event_ret_21d_sector_rel",
    "pre_event_idio_resid_5d",
    "qoq_4q_trend_atc",
]

HORIZONS: list[int] = [1, 3, 5, 10, 20]
SIGNAL_TYPES: list[str] = ["Total", "CEO", "CFO", "Analysts", "Executives"]


# ---------------------------------------------------------------------------
# Newey-West (statsmodels not installed — manual Bartlett kernel)
# ---------------------------------------------------------------------------


def _newey_west_se(x: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC standard error of the mean (Bartlett kernel)."""
    n = len(x)
    if n <= max_lag + 1:
        return np.nan
    xm = x - x.mean()
    w = np.dot(xm, xm) / n
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        w += 2.0 * weight * np.dot(xm[: n - lag], xm[lag:]) / n
    return np.sqrt(max(w, 0.0) / n)


def _newey_west_t_stat(x: np.ndarray, max_lag: int) -> float:
    """Newey-West adjusted t-statistic."""
    se = _newey_west_se(x, max_lag)
    if np.isnan(se) or se == 0.0:
        return np.nan
    return float(np.mean(x) / se)


# ---------------------------------------------------------------------------
# Cross-sectional IC
# ---------------------------------------------------------------------------


def _compute_cross_sectional_ic(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    signal_type: str,
    group_col: str = "year_month",
    min_samples: int = 10,
) -> pd.DataFrame:
    """Cross-sectional Spearman IC for each group in *group_col*.

    Returns DataFrame with columns ``[group_col, 'ic', 'n_samples']``.
    """
    sub = df[df["SignalType"] == signal_type]
    if sub.empty:
        return pd.DataFrame(columns=[group_col, "ic", "n_samples"])

    mask = sub[feature_col].notna() & sub[return_col].notna()
    sub = sub[mask]
    if sub.empty:
        return pd.DataFrame(columns=[group_col, "ic", "n_samples"])

    records: list[dict[str, Any]] = []
    for gname, gdf in sub.groupby(group_col, observed=True):
        if len(gdf) < min_samples:
            continue
        gdf = gdf.sort_values("call_entry_date").groupby("BESTTICKER").last()
        if len(gdf) < min_samples:
            continue
        ic, _ = spearmanr(gdf[feature_col].values, gdf[return_col].values)
        records.append(
            {
                group_col: gname,
                "ic": ic if not np.isnan(ic) else np.nan,
                "n_samples": len(gdf),
            }
        )
    return pd.DataFrame(records)


def _compute_sector_monthly_ic(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    signal_type: str,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Monthly Spearman IC within each sector.

    This keeps the sector split cross-sectional at each point in time instead
    of computing one all-history correlation per sector.
    """
    sub = df[df["SignalType"] == signal_type]
    mask = (
        sub[feature_col].notna()
        & sub[return_col].notna()
        & sub["SECTOR"].notna()
        & sub["year_month"].notna()
    )
    sub = sub[mask]
    if sub.empty:
        return pd.DataFrame(columns=["sector", "year_month", "ic", "n_samples"])

    records: list[dict[str, Any]] = []
    for (sector, month), gdf in sub.groupby(["SECTOR", "year_month"], observed=True):
        if len(gdf) < min_samples:
            continue
        gdf = gdf.sort_values("call_entry_date").groupby("BESTTICKER").last()
        if len(gdf) < min_samples:
            continue
        ic, _ = spearmanr(gdf[feature_col].values, gdf[return_col].values)
        records.append(
            {
                "sector": sector,
                "year_month": month,
                "ic": ic if not np.isnan(ic) else np.nan,
                "n_samples": len(gdf),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _ic_summary(ic_series: pd.Series, nw_lag: int) -> dict[str, float]:
    """Compute summary statistics for an IC time series."""
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {
            "n_periods": 0,
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "t_stat": np.nan,
            "nw_t_stat": np.nan,
            "hit_rate": np.nan,
        }
    return {
        "n_periods": n,
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "t_stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(n))
        if ic.std(ddof=1) > 0
        else np.nan,
        "nw_t_stat": _newey_west_t_stat(ic.values, nw_lag),
        "hit_rate": float((ic > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Module-level global set before ProcessPoolExecutor context so forked
# children inherit the filtered DataFrame without pickling it.
_IC_GLOBAL_DF: pd.DataFrame | None = None


def _ic_full_worker(args: tuple) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute monthly, yearly, and sector IC for one (feature, horizon, signal_type) combo.

    Returns (monthly_summary, yearly_rows, sector_rows).
    Yearly IC is derived from monthly IC by grouping by year and taking the mean.
    Sector IC is computed from per-sector-per-month cross-sectional IC.
    """
    feat, horizon, sig_type = args
    df = _IC_GLOBAL_DF
    ret_col = f"forward_return_{horizon}d"

    # ---- Monthly IC (cross-sectional per year_month) ----
    ic_df = _compute_cross_sectional_ic(
        df, feat, ret_col, sig_type, group_col="year_month", min_samples=10
    )
    nw_lag = max(horizon, 1)
    monthly_summary = _ic_summary(ic_df["ic"], nw_lag)
    monthly_summary["feature"] = feat
    monthly_summary["horizon"] = horizon
    monthly_summary["signal_type"] = sig_type
    monthly_summary["total_events"] = int(ic_df["n_samples"].sum()) if len(ic_df) > 0 else 0

    # ---- Yearly IC (derived from monthly IC by grouping by year) ----
    yearly_rows: list[dict[str, Any]] = []
    if not ic_df.empty and "year_month" in ic_df.columns:
        ic_df["year"] = ic_df["year_month"].dt.year
        for year_val, year_group in ic_df.groupby("year"):
            if pd.isna(year_val):
                continue
            yearly_rows.append({
                "feature": feat,
                "horizon": horizon,
                "signal_type": sig_type,
                "year": int(year_val),
                "ic": float(year_group["ic"].mean()),
                "n_samples": int(year_group["n_samples"].sum()),
            })

    # ---- Sector IC (per-sector-per-month, derive summary) ----
    sector_rows: list[dict[str, Any]] = []
    sec_ic_df = _compute_sector_monthly_ic(df, feat, ret_col, sig_type, min_samples=10)
    if not sec_ic_df.empty:
        for sector, sector_ic in sec_ic_df.groupby("sector", observed=True):
            summary = _ic_summary(sector_ic["ic"], nw_lag)
            sector_rows.append({
                "horizon": horizon,
                "signal_type": sig_type,
                "feature": feat,
                "sector": sector,
                "total_events": int(sector_ic["n_samples"].sum()),
                **summary,
            })

    return monthly_summary, yearly_rows, sector_rows


def run_single_feature_ic(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    output_dir: Path | None = None,
    tier: str = "enhanced",
    n_jobs: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run single-feature IC analysis for one universe.

    Returns dict ``SignalType -> summary DataFrame``, plus ``_sector`` key
    for the sector-split table.
    """
    from backtest.splits import ensure_forward_returns, read_feature_columns

    log.info("Loading features from %s", features_path)
    required_cols = [
        "SignalType", "call_entry_date", "availability_date", "BESTTICKER", "SECTOR",
    ]
    df = read_feature_columns(
        features_path,
        [*required_cols, *SHORT_LIST_FEATURES],
        required_columns=required_cols,
    )
    n_total = len(df)

    log.info("Ensuring forward returns")
    df = ensure_forward_returns(
        df, features_path, price_cache_dir,
        entry_date_col="availability_date",
    )

    log.info("Filtering to universe %s", universe_name)
    df = filter_to_universe(df, universe_name)
    in_univ = df["_in_universe"]
    log.info(
        "Universe filter: %d / %d rows in-universe (%.1f%%)",
        in_univ.sum(),
        n_total,
        100 * in_univ.sum() / n_total,
    )
    df = df[in_univ].copy()

    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    missing = [f for f in SHORT_LIST_FEATURES if f not in df.columns]
    if missing:
        log.warning("Missing features (will be skipped): %s", missing)
    features = [f for f in SHORT_LIST_FEATURES if f in df.columns]

    needed_cols = [
        "SignalType", "year_month", "call_entry_date", "BESTTICKER", "SECTOR"
    ] + features + [
        f"forward_return_{h}d" for h in HORIZONS
    ]
    needed_cols = list(dict.fromkeys(c for c in needed_cols if c in df.columns))
    df = df.loc[:, needed_cols].copy()
    gc.collect()

    total_tasks = len(features) * len(HORIZONS) * len(SIGNAL_TYPES)
    tasks = list(itertools.product(features, HORIZONS, SIGNAL_TYPES))

    _n_jobs = n_jobs if n_jobs > 0 else min(os.cpu_count() or 4, 2)

    # ---- Monthly / yearly / sector IC combined (single pass per combo) ----
    global _IC_GLOBAL_DF
    _IC_GLOBAL_DF = df
    try:
        if _n_jobs > 1:
            with ProcessPoolExecutor(max_workers=_n_jobs) as ex:
                all_results = list(ex.map(_ic_full_worker, tasks))
        else:
            all_results = []
            for feat, horizon, sig_type in progress(
                tasks, total=total_tasks, desc="IC analysis"
            ):
                all_results.append(_ic_full_worker((feat, horizon, sig_type)))
    finally:
        _IC_GLOBAL_DF = None

    all_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    for monthly, yearly, sector in all_results:
        all_rows.append(monthly)
        yearly_rows.extend(yearly)
        sector_rows.extend(sector)

    summary_df = pd.DataFrame(all_rows)
    results: dict[str, pd.DataFrame] = {}
    for sig_type in SIGNAL_TYPES:
        results[sig_type] = (
            summary_df[summary_df["signal_type"] == sig_type]
            .drop(columns=["signal_type"])
            .reset_index(drop=True)
        )
    results["_year"] = pd.DataFrame(yearly_rows)
    results["_sector"] = pd.DataFrame(sector_rows)

    # ---- persist ----
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_parquet(output_dir / f"ic_summary_{universe_name}.parquet")

        if not results["_year"].empty:
            results["_year"].to_parquet(
                output_dir / f"ic_yearly_{universe_name}.parquet"
            )

        if not results["_sector"].empty:
            results["_sector"].to_parquet(
                output_dir / f"ic_sector_split_{universe_name}.parquet"
            )

        json_path = output_dir / f"ic_summary_{universe_name}.json"
        json_path.write_text(
            json.dumps(
                {
                    "universe": universe_name,
                    "features_checked": features,
                    "horizons": HORIZONS,
                    "signal_types": SIGNAL_TYPES,
                    "n_total_events": n_total,
                    "n_in_universe": int(in_univ.sum()),
                    "split_outputs": {
                        "monthly_summary": f"ic_summary_{universe_name}.parquet",
                        "yearly": f"ic_yearly_{universe_name}.parquet",
                        "sector": f"ic_sector_split_{universe_name}.parquet",
                    },
                },
                indent=2,
                default=str,
            )
        )

        manifest = build_cache_manifest(
            phase=f"5a_{tier}_{universe_name}",
            parameters={
                "universe_name": universe_name,
                "features": features,
                "horizons": HORIZONS,
                "signal_types": SIGNAL_TYPES,
            },
            input_paths=[
                features_path,
                price_cache_dir / "_manifest.json",
            ],
            source_funcs=[
                run_single_feature_ic,
                _compute_cross_sectional_ic,
                _compute_sector_monthly_ic,
                _ic_summary,
                _ic_full_worker,
                _newey_west_se,
            ],
        )
        write_cache_manifest(
            manifest,
            CACHE_MANIFEST_DIR / f"5a_{tier}_{universe_name}.json",
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Single-feature IC analysis (Phase 5.1)")
    p.add_argument(
        "--features",
        type=Path,
        default=RESULTS_DIR / "features_enhanced.parquet",
        help="Path to enhanced features parquet",
    )
    p.add_argument(
        "--universe",
        choices=["sp500", "sp1500", "ru3k", "all"],
        default="all",
        help="Which universe to run (default: all)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "ic",
        help="Output directory for IC results",
    )
    p.add_argument("--price-cache", type=Path, default=PRICE_CACHE_DIR)
    p.add_argument("--tier", type=str, default="enhanced",
                   help="Feature tier name for cache manifest isolation")
    p.add_argument("--n-jobs", type=int, default=0,
                   help="Parallel workers for IC combos (0=memory-safe auto)")
    args = p.parse_args()

    universes = list(UNIVERSE_NAMES) if args.universe == "all" else [args.universe]
    for univ in universes:
        log.info("==== Running IC analysis for %s ====", univ)
        run_single_feature_ic(
            features_path=args.features,
            universe_name=univ,
            price_cache_dir=args.price_cache,
            output_dir=args.output_dir,
            tier=args.tier,
            n_jobs=args.n_jobs,
        )
        log.info("Results written to %s/ic_summary_%s.parquet", args.output_dir, univ)


if __name__ == "__main__":
    main()
