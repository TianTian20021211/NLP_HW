"""Phase 5.1 — Single-Feature IC Analysis.

Cross-sectional Spearman IC for the 14-column short list, computed by month
and sector, with Newey-West adjusted t-statistics.

Usage::

    python -m backtest.single_feature_ic \
        --features results/features_enhanced.parquet \
        --universe sp500
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.config import (
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
        # Deduplicate: keep latest event per ticker within each group
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
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


# ---------------------------------------------------------------------------
# Universe membership filter
# ---------------------------------------------------------------------------


def _filter_to_universe(
    df: pd.DataFrame,
    universe_name: str,
    tolerance_days: int | None = None,
) -> pd.DataFrame:
    """Add ``_in_universe`` bool column using global PIT snapshots."""
    from backtest.universe import filter_to_universe

    return filter_to_universe(
        df,
        universe_name,
        date_col="call_entry_date",
        ticker_col="BESTTICKER",
        tolerance_days=tolerance_days,
    )


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


def run_single_feature_ic(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run single-feature IC analysis for one universe.

    Returns dict ``SignalType -> summary DataFrame``, plus ``_sector`` key
    for the sector-split table.
    """
    from backtest.splits import get_forward_returns_cached

    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    n_total = len(df)

    log.info("Computing forward returns (cached)")
    fwd = get_forward_returns_cached(features_path, df, price_cache_dir, entry_date_col="availability_date")
    for col in fwd.columns:
        df[col] = fwd[col]

    log.info("Filtering to universe %s", universe_name)
    df = _filter_to_universe(df, universe_name)
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

    total_tasks = len(features) * len(HORIZONS) * len(SIGNAL_TYPES)
    task_iter = itertools.product(features, HORIZONS, SIGNAL_TYPES)

    all_rows: list[dict[str, Any]] = []
    for feat, horizon, sig_type in progress(
        task_iter, total=total_tasks, desc="IC analysis"
    ):
        ret_col = f"forward_return_{horizon}d"
        ic_df = _compute_cross_sectional_ic(
            df, feat, ret_col, sig_type, group_col="year_month", min_samples=10
        )
        nw_lag = max(horizon, 1)
        summary = _ic_summary(ic_df["ic"], nw_lag)
        summary["feature"] = feat
        summary["horizon"] = horizon
        summary["signal_type"] = sig_type
        summary["total_events"] = int(ic_df["n_samples"].sum()) if len(ic_df) > 0 else 0
        all_rows.append(summary)

    summary_df = pd.DataFrame(all_rows)
    results: dict[str, pd.DataFrame] = {}
    for sig_type in SIGNAL_TYPES:
        results[sig_type] = (
            summary_df[summary_df["signal_type"] == sig_type]
            .drop(columns=["signal_type"])
            .reset_index(drop=True)
        )

    # ---- sector-split IC for ATCClassifierScore only ----
    log.info("Computing sector-split IC for ATCClassifierScore")
    sector_rows: list[dict[str, Any]] = []
    for horizon, sig_type in progress(
        itertools.product(HORIZONS, SIGNAL_TYPES),
        total=len(HORIZONS) * len(SIGNAL_TYPES),
        desc="IC sector-split",
    ):
        ret_col = f"forward_return_{horizon}d"
        ic_df = _compute_cross_sectional_ic(
            df,
            "ATCClassifierScore",
            ret_col,
            sig_type,
            group_col="SECTOR",
            min_samples=10,
        )
        for _, row in ic_df.iterrows():
            sector_rows.append(
                {
                    "horizon": horizon,
                    "signal_type": sig_type,
                    "sector": row["SECTOR"],
                    "ic": row["ic"],
                    "n_samples": int(row["n_samples"]),
                }
            )
    results["_sector"] = pd.DataFrame(sector_rows)

    # ---- persist ----
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_parquet(output_dir / f"ic_summary_{universe_name}.parquet")

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
                },
                indent=2,
                default=str,
            )
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
    args = p.parse_args()

    universes = list(UNIVERSE_NAMES) if args.universe == "all" else [args.universe]
    for univ in universes:
        log.info("==== Running IC analysis for %s ====", univ)
        run_single_feature_ic(
            features_path=args.features,
            universe_name=univ,
            price_cache_dir=args.price_cache,
            output_dir=args.output_dir,
        )
        log.info("Results written to %s/ic_summary_%s.parquet", args.output_dir, univ)


if __name__ == "__main__":
    main()
