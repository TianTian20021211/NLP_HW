"""Phase 5.2 — Quintile / Decile Portfolio Analysis.

Required baseline: ATCClassifierScore x universe x horizon x SignalType as
decile spread. Other single features use quintiles.

Equal-weight, dollar-neutral (long $1 / short $1, equal-weight within each leg).
Reports long-only, short-only, and long-short cumulative returns.

Usage::

    python -m backtest.quintile \
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

from data.config import PRICE_CACHE_DIR, RESULTS_DIR, UNIVERSE_NAMES
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
# Universe filter (shared pattern with single_feature_ic.py)
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
# Bucket returns
# ---------------------------------------------------------------------------


def _bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int,
    min_samples_per_bucket: int = 5,
) -> pd.DataFrame:
    """Form equal-size buckets by *feature_col*, compute mean return per bucket.

    Grouping is by ``year_month`` to create a time series of bucket returns.
    Returns DataFrame with ``[year_month, bucket, ret, n]``.
    """
    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=["year_month", "bucket", "ret", "n"])

    records: list[dict[str, Any]] = []
    for month, gdf in sub.groupby("year_month", observed=True):
        # Deduplicate: keep latest event per ticker per month
        gdf = (
            gdf.sort_values("call_entry_date")
            .drop_duplicates(subset=["BESTTICKER"], keep="last")
        )
        if len(gdf) < n_buckets * min_samples_per_bucket:
            continue
        try:
            gdf["_bucket"] = pd.qcut(
                gdf[feature_col], q=n_buckets, labels=False, duplicates="drop"
            )
        except ValueError:
            # Not enough distinct values for qcut
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


def _build_equity_curves(
    bucket_returns: pd.DataFrame,
    n_buckets: int,
) -> pd.DataFrame:
    """From a table of monthly bucket returns, build cumulative equity curves.

    Returns DataFrame indexed by ``year_month`` with columns:
    ``long_only``, ``short_only``, ``long_short``, plus raw per-bucket columns.
    """
    if bucket_returns.empty:
        return pd.DataFrame()

    piv = bucket_returns.pivot_table(
        index="year_month", columns="bucket", values="ret", aggfunc="mean"
    )
    piv.columns = [f"bucket_{int(c)}" for c in piv.columns]
    piv = piv.sort_index()

    cum = (1 + piv).cumprod()
    for col in piv.columns:
        piv[f"cum_{col}"] = cum[col]

    # Long = top bucket, Short = bottom bucket
    top = f"bucket_{n_buckets - 1}"
    bot = "bucket_0"

    if top in piv.columns and bot in piv.columns:
        piv["long_only"] = piv[top]
        piv["short_only"] = -piv[bot]
        piv["long_short"] = piv[top] - piv[bot]

        piv["cum_long_only"] = (1 + piv["long_only"]).cumprod()
        piv["cum_short_only"] = (1 + piv["short_only"]).cumprod()
        piv["cum_long_short"] = (1 + piv["long_short"]).cumprod()

    return piv


def _portfolio_stats(
    equity_curves: pd.DataFrame, leg: str = "long_short"
) -> dict[str, float]:
    """Compute summary statistics for a portfolio leg."""
    col = f"cum_{leg}"
    if col not in equity_curves.columns or len(equity_curves) < 2:
        return {}
    rets = equity_curves[leg].dropna()
    if len(rets) == 0:
        return {}

    def _rolling_sharpe(r: pd.Series, window: int = 12) -> float:
        roll = r.rolling(window).mean() / r.rolling(window).std()
        return float(roll.mean())

    def _max_drawdown(cum: pd.Series) -> float:
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    ann_factor = 12.0  # monthly returns -> annual
    return {
        "ann_return": float(rets.mean() * ann_factor),
        "ann_vol": float(rets.std() * np.sqrt(ann_factor)),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(ann_factor))
        if rets.std() > 0
        else np.nan,
        "rolling_sharpe_12m": _rolling_sharpe(rets, 12),
        "max_drawdown": _max_drawdown(equity_curves[col]),
        "n_months": len(rets),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_quintile_analysis(
    features_path: Path,
    universe_name: str,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run quintile/decile portfolio analysis for one universe.

    Returns dict with decile and quintile result DataFrames.
    """
    from backtest.splits import get_forward_returns_cached

    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    log.info("Computing forward returns (cached)")
    fwd = get_forward_returns_cached(features_path, df, price_cache_dir, entry_date_col="availability_date")
    for col in fwd.columns:
        df[col] = fwd[col]

    log.info("Filtering to universe %s", universe_name)
    df = _filter_to_universe(df, universe_name)
    in_univ = df["_in_universe"]
    df = df[in_univ].copy()
    log.info("In-universe rows: %d", len(df))

    df["year_month"] = df["call_entry_date"].dt.to_period("M")

    available_features = [f for f in QUINTILE_FEATURES if f in df.columns]
    log.info("Features available: %s", available_features)

    # ---- Decile baseline: ATCClassifierScore ----
    log.info("Running decile baseline (ATCClassifierScore)")
    decile_results: list[dict[str, Any]] = []
    decile_equity: dict[str, pd.DataFrame] = {}

    for horizon, sig_type in progress(
        list(itertools.product(HORIZONS, SIGNAL_TYPES)),
        desc="Decile baseline",
    ):
        ret_col = f"forward_return_{horizon}d"
        sub = df[df["SignalType"] == sig_type]
        bucket_ret = _bucket_returns(sub, "ATCClassifierScore", ret_col, n_buckets=10)
        eq = _build_equity_curves(bucket_ret, n_buckets=10)
        stats = _portfolio_stats(eq, "long_short")
        stats["horizon"] = horizon
        stats["signal_type"] = sig_type
        stats["feature"] = "ATCClassifierScore"
        stats["n_buckets"] = 10
        decile_results.append(stats)
        decile_equity[f"{horizon}d_{sig_type}"] = eq

    # ---- Quintile: all features ----
    log.info("Running quintile analysis for %d features", len(available_features))
    quintile_results: list[dict[str, Any]] = []
    quintile_equity: dict[str, pd.DataFrame] = {}

    task_iter = list(
        itertools.product(available_features, HORIZONS, SIGNAL_TYPES)
    )
    for feat, horizon, sig_type in progress(task_iter, desc="Quintile analysis"):
        ret_col = f"forward_return_{horizon}d"
        sub = df[df["SignalType"] == sig_type]
        bucket_ret = _bucket_returns(sub, feat, ret_col, n_buckets=5)
        eq = _build_equity_curves(bucket_ret, n_buckets=5)

        # long-only / short-only / long-short stats
        for leg in ("long_only", "short_only", "long_short"):
            stats = _portfolio_stats(eq, leg)
            stats["horizon"] = horizon
            stats["signal_type"] = sig_type
            stats["feature"] = feat
            stats["n_buckets"] = 5
            stats["leg"] = leg
            quintile_results.append(stats)

        quintile_equity[f"{feat}_{horizon}d_{sig_type}"] = eq

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
    args = p.parse_args()

    universes = list(UNIVERSE_NAMES) if args.universe == "all" else [args.universe]
    for univ in universes:
        log.info("==== Quintile analysis for %s ====", univ)
        run_quintile_analysis(
            features_path=args.features,
            universe_name=univ,
            price_cache_dir=args.price_cache,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
