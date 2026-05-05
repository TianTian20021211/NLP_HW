"""Phase 5.2 — Quintile / Decile Portfolio Analysis.

Required baseline: ATCClassifierScore x universe x horizon x SignalType as
decile spread. Other single features use quintiles.

Equal-weight, dollar-neutral (long $1 / short $1, equal-weight within each leg).
Buckets are formed on each signal availability date so later events in the same
month cannot affect an earlier event's rank.
Reports long-only, short-only, and long-short cumulative returns.

Usage::

    python -m backtest.quintile \
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

from backtest._stats import bucket_returns, build_equity_curves, portfolio_stats

from backtest.universe import filter_to_universe
from data.cache_utils import build_cache_manifest, write_cache_manifest
from data.config import CACHE_MANIFEST_DIR, PRICE_CACHE_DIR, RESULTS_DIR, UNIVERSE_NAMES
from data.progress import progress

log = logging.getLogger("backtest.quintile")

HORIZONS: list[int] = [1, 3, 5, 10, 20]
SIGNAL_TYPES: list[str] = ["Total", "CEO", "CFO", "Analysts", "Executives"]

# Features to run (beyond the mandatory ATCClassifierScore decile baseline)
QUINTILE_FEATURES: list[str] = [
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_QUINTILE_GLOBAL_DF: pd.DataFrame | None = None


def _decile_combo_worker(args: tuple) -> tuple[list[dict[str, Any]], str, pd.DataFrame]:
    """Compute decile stats + equity curve for one (horizon, signal_type) combo."""
    horizon, sig_type = args
    df = _QUINTILE_GLOBAL_DF
    ret_col = f"forward_return_{horizon}d"
    sub = df[df["SignalType"] == sig_type]
    bucket_ret = bucket_returns(
        sub, "ATCClassifierScore", ret_col, n_buckets=10,
        min_samples_per_bucket=1, group_col="bucket_date",
        date_col="availability_date",
    )
    eq = build_equity_curves(bucket_ret, n_buckets=10, group_col="bucket_date")
    results: list[dict[str, Any]] = []
    for leg in ("long_only", "short_only", "long_short"):
        stats = portfolio_stats(eq, leg, ann_factor=252.0 / horizon, count_key="n_event_dates")
        stats["horizon"] = horizon
        stats["signal_type"] = sig_type
        stats["feature"] = "ATCClassifierScore"
        stats["n_buckets"] = 10
        stats["leg"] = leg
        results.append(stats)
    key = f"{horizon}d_{sig_type}"
    return results, key, eq


def _quintile_combo_worker(args: tuple) -> tuple[list[dict[str, Any]], str, pd.DataFrame]:
    """Compute quintile stats + equity curve for one (feature, horizon, signal_type) combo."""
    feat, horizon, sig_type = args
    df = _QUINTILE_GLOBAL_DF
    ret_col = f"forward_return_{horizon}d"
    sub = df[df["SignalType"] == sig_type]
    bucket_ret = bucket_returns(
        sub, feat, ret_col, n_buckets=5,
        min_samples_per_bucket=1, group_col="bucket_date",
        date_col="availability_date",
    )
    eq = build_equity_curves(bucket_ret, n_buckets=5, group_col="bucket_date")
    results: list[dict[str, Any]] = []
    for leg in ("long_only", "short_only", "long_short"):
        stats = portfolio_stats(eq, leg, ann_factor=252.0 / horizon, count_key="n_event_dates")
        stats["horizon"] = horizon
        stats["signal_type"] = sig_type
        stats["feature"] = feat
        stats["n_buckets"] = 5
        stats["leg"] = leg
        results.append(stats)
    key = f"{feat}_{horizon}d_{sig_type}"
    return results, key, eq


def run_quintile_analysis(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    output_dir: Path | None = None,
    tier: str = "enhanced",
    n_jobs: int = 0,
) -> dict[str, Any]:
    """Run quintile/decile portfolio analysis for one universe.

    Returns dict with decile and quintile result DataFrames.
    """
    from backtest.splits import ensure_forward_returns, read_feature_columns

    log.info("Loading features from %s", features_path)
    required_cols = [
        "SignalType", "call_entry_date", "availability_date", "BESTTICKER",
    ]
    df = read_feature_columns(
        features_path,
        [*required_cols, *QUINTILE_FEATURES],
        required_columns=required_cols,
    )

    log.info("Ensuring forward returns")
    df = ensure_forward_returns(
        df, features_path, price_cache_dir,
        entry_date_col="availability_date",
    )

    log.info("Filtering to universe %s", universe_name)
    df = filter_to_universe(df, universe_name)
    in_univ = df["_in_universe"]
    df = df[in_univ].copy()
    log.info("In-universe rows: %d", len(df))

    df["bucket_date"] = pd.to_datetime(df["availability_date"], errors="coerce").dt.normalize()

    available_features = [f for f in QUINTILE_FEATURES if f in df.columns]
    log.info("Features available: %s", available_features)

    needed_cols = [
        "SignalType", "bucket_date", "availability_date", "BESTTICKER"
    ] + available_features + [
        f"forward_return_{h}d" for h in HORIZONS
    ]
    needed_cols = list(dict.fromkeys(c for c in needed_cols if c in df.columns))
    df = df.loc[:, needed_cols].copy()
    gc.collect()

    _n_jobs = n_jobs if n_jobs > 0 else min(os.cpu_count() or 4, 6)
    global _QUINTILE_GLOBAL_DF
    _QUINTILE_GLOBAL_DF = df

    # ---- Decile baseline: ATCClassifierScore ----
    log.info("Running decile baseline (ATCClassifierScore)")
    decile_results: list[dict[str, Any]] = []
    decile_equity: dict[str, pd.DataFrame] = {}
    decile_tasks = list(itertools.product(HORIZONS, SIGNAL_TYPES))

    try:
        if _n_jobs > 1:
            _n_jobs_decile = min(_n_jobs, len(decile_tasks))
            with ProcessPoolExecutor(max_workers=_n_jobs_decile) as ex:
                decile_outputs = list(ex.map(_decile_combo_worker, decile_tasks))
            for stats_list, key, eq in decile_outputs:
                decile_results.extend(stats_list)
                decile_equity[key] = eq
        else:
            for horizon, sig_type in progress(decile_tasks, desc="Decile baseline"):
                stats_list, key, eq = _decile_combo_worker((horizon, sig_type))
                decile_results.extend(stats_list)
                decile_equity[key] = eq

        # ---- Quintile: all features ----
        log.info("Running quintile analysis for %d features", len(available_features))
        quintile_results: list[dict[str, Any]] = []
        quintile_equity: dict[str, pd.DataFrame] = {}
        quintile_tasks = list(itertools.product(available_features, HORIZONS, SIGNAL_TYPES))

        if _n_jobs > 1:
            _n_jobs_quintile = min(_n_jobs, len(quintile_tasks))
            with ProcessPoolExecutor(max_workers=_n_jobs_quintile) as ex:
                quintile_outputs = list(ex.map(_quintile_combo_worker, quintile_tasks))
            for stats_list, key, eq in quintile_outputs:
                quintile_results.extend(stats_list)
                quintile_equity[key] = eq
        else:
            for feat, horizon, sig_type in progress(quintile_tasks, desc="Quintile analysis"):
                stats_list, key, eq = _quintile_combo_worker((feat, horizon, sig_type))
                quintile_results.extend(stats_list)
                quintile_equity[key] = eq
    finally:
        _QUINTILE_GLOBAL_DF = None

    decile_df = pd.DataFrame(decile_results)
    quintile_df = pd.DataFrame(quintile_results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        decile_df.to_parquet(output_dir / f"decile_summary_{universe_name}.parquet")
        quintile_df.to_parquet(
            output_dir / f"quintile_summary_{universe_name}.parquet"
        )

        # Persist equity curves in a compact multi-index format
        all_equity: list[pd.DataFrame] = []
        for key, eq in {**decile_equity, **quintile_equity}.items():
            eq_copy = eq.copy()
            eq_copy["_key"] = key
            all_equity.append(eq_copy)
        if all_equity:
            pd.concat(all_equity).to_parquet(
                output_dir / f"quintile_equity_curves_{universe_name}.parquet"
            )

        json_path = output_dir / f"quintile_meta_{universe_name}.json"
        json_path.write_text(
            json.dumps(
                {
                    "universe": universe_name,
                    "features": available_features,
                    "horizons": HORIZONS,
                    "signal_types": SIGNAL_TYPES,
                    "n_rows_in_universe": len(df),
                },
                indent=2,
                default=str,
            )
        )

        # Write cache manifest for incremental build support
        manifest = build_cache_manifest(
            phase=f"5b_{tier}_{universe_name}",
            parameters={
                "universe_name": universe_name,
                "features": available_features,
                "horizons": HORIZONS,
                "signal_types": SIGNAL_TYPES,
            },
            input_paths=[features_path, price_cache_dir / "_manifest.json"],
            source_funcs=[run_quintile_analysis, _decile_combo_worker, _quintile_combo_worker],
        )
        write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"5b_{tier}_{universe_name}.json")

    return {
        "decile_summary": decile_df,
        "quintile_summary": quintile_df,
        "decile_equity": decile_equity,
        "quintile_equity": quintile_equity,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Quintile/decile portfolios (Phase 5.2)")
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
        default=RESULTS_DIR / "quintile",
    )
    p.add_argument("--price-cache", type=Path, default=PRICE_CACHE_DIR)
    p.add_argument("--tier", type=str, default="enhanced",
                   help="Feature tier name for cache manifest isolation")
    p.add_argument("--n-jobs", type=int, default=0,
                   help="Parallel workers for quintile combos (0=memory-safe auto)")
    args = p.parse_args()

    universes = list(UNIVERSE_NAMES) if args.universe == "all" else [args.universe]
    for univ in universes:
        log.info("==== Quintile analysis for %s ====", univ)
        run_quintile_analysis(
            features_path=args.features,
            universe_name=univ,
            tier=args.tier,
            price_cache_dir=args.price_cache,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
