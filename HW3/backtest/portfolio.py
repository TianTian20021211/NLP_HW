"""Phase 5.4 — Rebalanced Portfolio Simulation.

Cadences: daily, weekly, monthly.
Capital convention: independent stock selection by cohort -> net by stock ->
scale to fixed gross=200% / net=0%.

Metrics: turnover, gross/net exposure, 5 bps post-cost Sharpe, capacity proxy
(20d ADDV, holding count, top-10 concentration, %ADV consumed).

Usage::

    python -m backtest.portfolio \\
        --signals results/predictions_enhanced.parquet \\
        --universe sp500 --cadence weekly
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from data.config import (
    AUDIT_DIR,
    CACHE_MANIFEST_DIR,
    PRICE_CACHE_DIR,
    RESULTS_DIR,
    UNIVERSE_CACHE_DIR,
)
from data.progress import progress
from backtest.universe import filter_to_universe
from features.audit import validate_trade_log

log = logging.getLogger("backtest.portfolio")

MAX_QUOTE_FORWARD_DAYS = 5
"""Maximum trading-day roll-forward for portfolio entry/exit quote lookup."""


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------


def _make_trading_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_dir: Path,
    tickers: set[str],
) -> pd.DatetimeIndex:
    """Build a trading calendar from price data of *tickers*.

    Takes the union of all dates across ticker price files. Falls back to
    ``pd.bdate_range`` when no price data is available.
    """
    all_dates: set[pd.Timestamp] = set()
    for tkr in tickers:
        path = price_dir / f"{tkr}.parquet"
        if not path.exists():
            continue
        px = pd.read_parquet(path, columns=["date"])
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        all_dates.update(d for d in px["date"].dropna() if start <= d <= end)
    if not all_dates:
        log.warning("No price dates found; falling back to business-day range")
        return pd.bdate_range(start, end)
    return pd.DatetimeIndex(sorted(all_dates))


def _rebalance_dates(
    calendar: pd.DatetimeIndex,
    cadence: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    weekly_day: str = "monday",
) -> pd.DatetimeIndex:
    """Return rebalance dates within [start, end].

    - daily: every trading day
    - weekly: Monday close by default, or Friday close for timing robustness
    - monthly: first trading day of each month
    """
    cal = calendar[(calendar >= start) & (calendar <= end)]
    if cadence == "daily":
        return cal
    if cadence == "weekly":
        # Monday close, or the next trading day when Monday is a holiday.
        # For Friday timing robustness, use the last trading day in each ISO week.
        # Group by ISO year as well as week so late-December / early-January
        # weeks do not collide across calendar years.
        if weekly_day not in {"monday", "friday"}:
            raise ValueError(f"Unknown weekly_day: {weekly_day}")
        iso = cal.isocalendar()
        week_groups = pd.Series(cal, index=cal).groupby(
            [iso["year"].to_numpy(), iso["week"].to_numpy()],
            sort=True,
        )
        if weekly_day == "friday":
            return pd.DatetimeIndex([g.iloc[-1] for _, g in week_groups])
        return pd.DatetimeIndex([g.iloc[0] for _, g in week_groups])
    if cadence == "monthly":
        month_groups = cal.to_series().groupby(cal.to_period("M"))
        return pd.DatetimeIndex(
            [g.iloc[0] for _, g in month_groups]
        )
    raise ValueError(f"Unknown cadence: {cadence}")


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def _load_price_table(
    tickers: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_dir: Path,
) -> pd.DataFrame:
    """Load price data for *tickers* into a tall DataFrame.

    Columns: ``[date, ticker, adj_close, volume]``.
    """
    frames: list[pd.DataFrame] = []
    for tkr in progress(sorted(tickers), desc="Loading prices", unit="tkr"):
        path = price_dir / f"{tkr}.parquet"
        if not path.exists():
            continue
        px = pd.read_parquet(path, columns=["date", "adj_close", "volume"])
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px[
            px["date"].notna()
            & px["adj_close"].notna()
            & (px["adj_close"] > 0)
            & px["volume"].notna()
            & (px["volume"] > 0)
        ]
        px = px[(px["date"] >= start) & (px["date"] <= end)]
        if px.empty:
            continue
        px["ticker"] = tkr
        frames.append(px[["date", "ticker", "adj_close", "volume"]])
    if not frames:
        raise ValueError(f"No price data found for any ticker in {price_dir}")
    return pd.concat(frames, ignore_index=True)


def _build_close_matrix(price_table: pd.DataFrame) -> pd.DataFrame:
    """Wide close matrix indexed by date with one column per ticker."""
    close = price_table.pivot_table(
        index="date",
        columns="ticker",
        values="adj_close",
        aggfunc="last",
    )
    return close.sort_index()


def _load_universe_snapshots(universe_name: str) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, set[str]]]:
    """Load PIT universe snapshots as date index plus membership sets."""
    path = UNIVERSE_CACHE_DIR / f"{universe_name}_pit.parquet"
    if not path.exists():
        return pd.DatetimeIndex([]), {}
    univ = pd.read_parquet(path, columns=["date", "ticker"])
    if univ.empty:
        return pd.DatetimeIndex([]), {}
    univ["date"] = pd.to_datetime(univ["date"], errors="coerce").astype("datetime64[ns]")
    univ = univ.dropna(subset=["date", "ticker"])
    snapshots = {
        pd.Timestamp(d): set(g["ticker"].astype(str))
        for d, g in univ.groupby("date", sort=True)
    }
    return pd.DatetimeIndex(sorted(snapshots)), snapshots


def _coverage_by_rebalance_date(
    rebalance_dates: pd.DatetimeIndex,
    universe_name: str,
    close_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Compute PIT member quote coverage for each rebalance date."""
    snapshot_dates, snapshots = _load_universe_snapshots(universe_name)
    rows: list[dict[str, Any]] = []
    if len(snapshot_dates) == 0:
        return pd.DataFrame(
            columns=[
                "date", "universe", "snapshot_date", "pit_members",
                "tradeable_members", "coverage_ratio", "missing_quote_ratio",
            ]
        )

    for rb_date in rebalance_dates:
        pos = snapshot_dates.searchsorted(rb_date, side="right") - 1
        if pos < 0:
            members: set[str] = set()
            snap_date = pd.NaT
        else:
            snap_date = pd.Timestamp(snapshot_dates[pos])
            members = snapshots.get(snap_date, set())

        if rb_date in close_matrix.index and members:
            quote_row = close_matrix.loc[rb_date]
            tradeable = int(quote_row.reindex(sorted(members)).notna().sum())
        else:
            tradeable = 0

        n_members = len(members)
        coverage = tradeable / n_members if n_members else np.nan
        rows.append({
            "date": rb_date,
            "universe": universe_name,
            "snapshot_date": snap_date,
            "pit_members": n_members,
            "tradeable_members": tradeable,
            "coverage_ratio": coverage,
            "missing_quote_ratio": 1.0 - coverage if np.isfinite(coverage) else np.nan,
        })
    return pd.DataFrame(rows)


def _slug(value: str) -> str:
    """Filesystem-safe tag for model/scenario output names."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return safe or "untagged"


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------


def _build_cohort_weights(
    scores: pd.DataFrame,
    long_frac: float = 0.2,
    n_min_positions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert scores to cohort-aggregated portfolio weights.

    For each cohort (unique signal date in *scores*):
      1. Drop NaN scores.
      2. Deduplicate to one signal per ticker (keep the one with highest
         absolute score).
      3. Rank by score descending.
      4. Select top/bottom *long_frac* (with *n_min_positions* floor) for
         equal-weight long/short raw weights.

    Then:
      5. Concatenate all cohort raw weights.
      6. Net by ticker (sum weights across cohorts).
      7. Re-center to net zero (subtract mean weight).
      8. Scale to gross exposure = 2.0 (sum of absolute weights).

    Parameters
    ----------
    scores:
        DataFrame with ``[date, ticker, score]`` for eligible signals
        in the lookback window.
    long_frac:
        Fraction of stocks to long/short in each cohort.
    n_min_positions:
        Minimum stocks required in each leg of a single cohort.

    Returns
    -------
    Tuple of (cohort_weights, aggregated_weights):
        cohort_weights: DataFrame ``[cohort_date, ticker, raw_weight]``
            Raw per-cohort weights before cross-cohort aggregation.
        aggregated_weights: DataFrame ``[ticker, raw_weight]``
            Final weights after netting by ticker, re-centering, and
            scaling to gross = 2.0 / net = 0.
    """
    required_cols = {"date", "ticker", "score"}
    if not required_cols.issubset(scores.columns):
        raise ValueError(
            f"scores must contain {required_cols}, got {set(scores.columns)}"
        )

    cohort_frames: list[pd.DataFrame] = []

    for cohort_date, group in scores.groupby("date", sort=True):
        valid = group[group["score"].notna()].copy()
        n = len(valid)
        if n < 2 * n_min_positions:
            continue

        # Deduplicate per ticker: keep row with highest absolute score.
        valid["_abs_score"] = valid["score"].abs()
        best_idx = valid.groupby("ticker")["_abs_score"].idxmax()
        valid = valid.loc[best_idx]
        valid = valid.drop(columns=["_abs_score"])
        n_after = len(valid)
        if n_after < 2 * n_min_positions:
            continue

        # Rank by score descending.
        valid = valid.sort_values("score", ascending=False)
        cutoff = max(n_min_positions, int(np.ceil(n_after * long_frac)))

        long_cohort = valid.head(cutoff)
        short_cohort = valid.tail(cutoff)

        n_long = len(long_cohort)
        n_short = len(short_cohort)
        if n_long == 0 or n_short == 0:
            continue

        long_w = 1.0 / n_long
        short_w = -1.0 / n_short

        rows: list[dict[str, Any]] = []
        for _, row in long_cohort.iterrows():
            rows.append({
                "cohort_date": cohort_date,
                "ticker": row["ticker"],
                "raw_weight": long_w,
            })
        for _, row in short_cohort.iterrows():
            rows.append({
                "cohort_date": cohort_date,
                "ticker": row["ticker"],
                "raw_weight": short_w,
            })
        cohort_frames.append(pd.DataFrame(rows))

    if not cohort_frames:
        return (
            pd.DataFrame(columns=["cohort_date", "ticker", "raw_weight"]),
            pd.DataFrame(columns=["ticker", "raw_weight"]),
        )

    cohort_weights = pd.concat(cohort_frames, ignore_index=True)

    # --- Aggregate by ticker across cohorts ---
    agg_weights = cohort_weights.groupby("ticker", as_index=False)["raw_weight"].sum()

    # Re-center to net zero (subtract mean from each ticker).
    mean_w = agg_weights["raw_weight"].mean()
    agg_weights["raw_weight"] = agg_weights["raw_weight"] - mean_w

    # Scale to gross exposure = 2.0.
    gross = agg_weights["raw_weight"].abs().sum()
    if gross > 1e-10:
        agg_weights["raw_weight"] = agg_weights["raw_weight"] * (2.0 / gross)

    return cohort_weights, agg_weights


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    """Output of a portfolio simulation run."""

    daily_returns: pd.DataFrame  # date x [pnl, gross_exposure, net_exposure, turnover]
    weights_history: dict[
        pd.Timestamp, pd.DataFrame
    ]  # rebalance_date -> aggregated ticker weights
    cohort_weights_history: dict[
        pd.Timestamp, pd.DataFrame
    ] = field(default_factory=dict)  # rebalance_date -> raw cohort weights
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    universe_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    gap_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, Any] = field(default_factory=dict)

    def cumulative_returns(self) -> pd.Series:
        return (1 + self.daily_returns["pnl"]).cumprod()

    def summary(self, cost_bps: float = 5.0) -> dict[str, float]:
        r = self.daily_returns
        n_days = len(r)
        if n_days < 2:
            return {}

        # Pre-cost
        pnl = r["pnl"]
        ann_ret = float(pnl.mean() * 252)
        ann_vol = float(pnl.std() * np.sqrt(252))
        pre_cost_sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan

        # Post-cost: deduct transaction cost from daily returns
        cost_per_day = cost_bps / 10_000.0 * r["turnover"]
        post_pnl = pnl - cost_per_day
        post_ann_ret = float(post_pnl.mean() * 252)
        post_sharpe = (
            float(post_ann_ret / (post_pnl.std() * np.sqrt(252)))
            if post_pnl.std() > 0
            else np.nan
        )

        cum = self.cumulative_returns()
        peak = cum.cummax()
        dd = (cum - peak) / peak

        total_pos_days = float(r["n_positions"].sum()) if "n_positions" in r.columns else 0.0

        gap_stats: dict[str, float] = {}
        if total_pos_days > 0:
            gap_stats["ffill_1d_position_day_share"] = float(
                r["n_ffill_1d_positions"].sum() / total_pos_days
            )
            gap_stats["ffill_2d_position_day_share"] = float(
                r["n_ffill_2d_positions"].sum() / total_pos_days
            )
            if "n_long_gap_recovered_positions" in r.columns:
                gap_stats["long_gap_recovered_position_day_share"] = float(
                    r["n_long_gap_recovered_positions"].sum() / total_pos_days
                )
                gap_stats["possible_delisting_or_unavailable_position_day_share"] = float(
                    r["n_possible_delisting_or_unavailable_positions"].sum() / total_pos_days
                )
        else:
            gap_stats["ffill_1d_position_day_share"] = 0.0
            gap_stats["ffill_2d_position_day_share"] = 0.0
            gap_stats["long_gap_recovered_position_day_share"] = 0.0
            gap_stats["possible_delisting_or_unavailable_position_day_share"] = 0.0

        entry_delay: dict[str, float] = {}
        if (
            not self.trade_log.empty
            and "actual_entry_date" in self.trade_log.columns
            and "planned_entry_date" in self.trade_log.columns
        ):
            log_df = self.trade_log
            has_entry = (
                log_df["actual_entry_date"].notna()
                & log_df["planned_entry_date"].notna()
            )
            if has_entry.any():
                delay = (
                    pd.to_datetime(log_df.loc[has_entry, "actual_entry_date"])
                    - pd.to_datetime(log_df.loc[has_entry, "planned_entry_date"])
                )
                entry_delay["avg_entry_delay_trading_days"] = float(
                    delay.dt.days.mean()
                )
                entry_delay["median_entry_delay_trading_days"] = float(
                    delay.dt.days.median()
                )

        return {
            "n_days": n_days,
            "ann_return_pre_cost": ann_ret,
            "ann_vol": ann_vol,
            "sharpe_pre_cost": pre_cost_sharpe,
            "sharpe_post_cost_5bps": post_sharpe,
            "max_drawdown": float(dd.min()),
            "ann_turnover": float(r["turnover"].mean() * 252),
            "avg_gross_exposure": float(r["gross_exposure"].mean()),
            "avg_net_exposure": float(r["net_exposure"].mean()),
            "n_rebalances": int(r["_rebalance"].sum()) if "_rebalance" in r.columns else 0,
            **entry_delay,
            **gap_stats,
        }


@dataclass
class SimulationState:
    """Mutable state accumulated through a portfolio simulation loop."""

    current_weights: dict[str, float] = field(default_factory=dict)
    prev_weights: dict[str, float] = field(default_factory=dict)
    current_entry_dates: dict[str, pd.Timestamp] = field(default_factory=dict)
    current_entry_prices: dict[str, float] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    missing_streaks: dict[str, int] = field(default_factory=dict)
    censored_tickers: set[str] = field(default_factory=set)
    today_turnover: float = 0.0
    daily_records: list[dict[str, Any]] = field(default_factory=list)
    trade_records: list[dict[str, Any]] = field(default_factory=list)
    long_gap_records: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    weights_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    cohort_weights_history: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)

    # Numpy parallel arrays (populated after ticker_to_idx is built)
    last_prices_arr: np.ndarray | None = None
    missing_streaks_arr: np.ndarray | None = None
    is_censored_arr: np.ndarray | None = None
    weights_arr: np.ndarray | None = None
    entry_dates_arr: np.ndarray | None = None


_EMPTY_COLUMNS = [
    "date", "pnl", "gross_exposure", "net_exposure", "turnover", "_rebalance",
    "n_positions", "ffill_1d_weight", "ffill_2d_weight",
    "long_gap_recovered_weight", "possible_delisting_or_unavailable_weight",
    "n_ffill_1d_positions", "n_ffill_2d_positions",
    "n_long_gap_recovered_positions", "n_possible_delisting_or_unavailable_positions",
]


def _advance_daily_state(
    tickers: list[str],
    weights: dict[str, float],
    last_prices: dict[str, float],
    missing_streaks: dict[str, int],
    censored_tickers: set[str],
    p_today: dict[str, float],
    p_next: dict[str, float],
) -> dict:
    """Advance one day of the gap-state machine. Returns a daily record dict.

    Pure function over plain dicts — no pandas, no SimulationState, no
    close_matrix slicing. Mutates *last_prices*, *missing_streaks*, and
    *censored_tickers* in place.
    """
    import math

    if not tickers:
        return {
            "pnl": 0.0, "gross_exposure": 0.0, "net_exposure": 0.0,
            "n_positions": 0, "ffill_1d_weight": 0.0, "ffill_2d_weight": 0.0,
            "n_ffill_1d": 0, "n_ffill_2d": 0, "long_gap_tickers": {},
        }

    gross = sum(abs(w) for w in weights.values())
    net = sum(weights.values())
    n_positions = len(tickers)

    pnl = 0.0
    ffill_1d_weight = 0.0
    ffill_2d_weight = 0.0
    n_ffill_1d = 0
    n_ffill_2d = 0
    long_gap_tickers: dict[str, float] = {}

    for tkr in tickers:
        w = weights.get(tkr, 0.0)
        today_px = p_today.get(tkr, float("nan"))
        next_px = p_next.get(tkr, float("nan"))

        if not math.isnan(today_px) and float(today_px) > 0:
            last_prices.setdefault(tkr, float(today_px))

        if tkr in censored_tickers:
            if math.isnan(next_px):
                long_gap_tickers[tkr] = float(w)
            continue

        if not math.isnan(next_px) and float(next_px) > 0:
            base_px = last_prices.get(tkr)
            if base_px is not None and base_px > 0:
                pnl += float(w) * (float(next_px) / base_px - 1.0)
            last_prices[tkr] = float(next_px)
            missing_streaks[tkr] = 0
            continue

        if tkr not in last_prices:
            continue
        streak = missing_streaks.get(tkr, 0) + 1
        missing_streaks[tkr] = streak
        if streak == 1:
            ffill_1d_weight += abs(float(w))
            n_ffill_1d += 1
        elif streak == 2:
            ffill_2d_weight += abs(float(w))
            n_ffill_2d += 1
        else:
            long_gap_tickers[tkr] = float(w)
            censored_tickers.add(tkr)

    return {
        "pnl": pnl,
        "gross_exposure": gross,
        "net_exposure": net,
        "n_positions": n_positions,
        "ffill_1d_weight": ffill_1d_weight,
        "ffill_2d_weight": ffill_2d_weight,
        "n_ffill_1d": n_ffill_1d,
        "n_ffill_2d": n_ffill_2d,
        "long_gap_tickers": long_gap_tickers,
    }


def _advance_daily_state_numpy(
    weights_arr: np.ndarray,
    last_prices_arr: np.ndarray,
    missing_streaks_arr: np.ndarray,
    is_censored_arr: np.ndarray,
    entry_dates_arr: np.ndarray,
    prices_2d: np.ndarray,
    d: int,
    d_next: int,
    idx_to_ticker: list[str],
) -> dict:
    """Vectorized daily PnL — 4 boolean-mask passes, no per-ticker loop.

    Mutates *last_prices_arr*, *missing_streaks_arr*, *is_censored_arr*
    in place. Returns the same dict shape as ``_advance_daily_state``.
    """
    has_position = (
        (weights_arr != 0.0)
        & (entry_dates_arr >= 0)
        & (entry_dates_arr <= d)
    )

    if not has_position.any():
        return {
            "pnl": 0.0, "gross_exposure": 0.0, "net_exposure": 0.0,
            "n_positions": 0, "ffill_1d_weight": 0.0, "ffill_2d_weight": 0.0,
            "n_ffill_1d": 0, "n_ffill_2d": 0, "long_gap_tickers": {},
        }

    pos_idx = np.flatnonzero(has_position)
    weights_pos = weights_arr[pos_idx]
    p_today = prices_2d[pos_idx, d]
    p_next = prices_2d[pos_idx, d_next]

    gross = float(np.abs(weights_pos).sum())
    net = float(weights_pos.sum())
    n_positions = len(pos_idx)

    # --- Pass 1: setdefault for newly-seen valid today prices (all positions) ---
    need_init = (
        np.isnan(last_prices_arr[pos_idx])
        & np.isfinite(p_today) & (p_today > 0)
    )
    if need_init.any():
        last_prices_arr[pos_idx[need_init]] = p_today[need_init]

    # --- Separate censored from active (non-censored) ---
    is_cens = is_censored_arr[pos_idx]
    non_cens = ~is_cens

    long_gap_tickers: dict[str, float] = {}

    # Censored positions: only flag long_gap when next quote is missing
    if is_cens.any():
        cens_next = p_next[is_cens]
        cens_nan = ~np.isfinite(cens_next)
        if cens_nan.any():
            cens_abs_idx = pos_idx[is_cens][cens_nan]
            for i in cens_abs_idx:
                long_gap_tickers[idx_to_ticker[i]] = float(weights_arr[i])

    if not non_cens.any():
        return {
            "pnl": 0.0,
            "gross_exposure": gross,
            "net_exposure": net,
            "n_positions": n_positions,
            "ffill_1d_weight": 0.0,
            "ffill_2d_weight": 0.0,
            "n_ffill_1d": 0,
            "n_ffill_2d": 0,
            "long_gap_tickers": long_gap_tickers,
        }

    active_idx = pos_idx[non_cens]
    weights_a = weights_pos[non_cens]
    p_next_a = p_next[non_cens]

    # --- Pass 2: compute PnL for tickers with valid next price ---
    can_compute = (
        np.isfinite(p_next_a) & (p_next_a > 0)
        & np.isfinite(last_prices_arr[active_idx])
        & (last_prices_arr[active_idx] > 0)
    )
    pnl = 0.0
    if can_compute.any():
        ret = p_next_a[can_compute] / last_prices_arr[active_idx[can_compute]] - 1.0
        pnl = float(np.dot(weights_a[can_compute], ret))
        last_prices_arr[active_idx[can_compute]] = p_next_a[can_compute]
        missing_streaks_arr[active_idx[can_compute]] = 0

    # --- Pass 3: missing quotes → increment streaks ---
    has_last = np.isfinite(last_prices_arr[active_idx])
    missing = ~np.isfinite(p_next_a) & has_last

    ffill_1d_weight = 0.0
    ffill_2d_weight = 0.0
    n_ffill_1d = 0
    n_ffill_2d = 0

    if missing.any():
        missing_streaks_arr[active_idx[missing]] += 1
        streaks = missing_streaks_arr[active_idx]
        is_1d = missing & (streaks == 1)
        is_2d = missing & (streaks == 2)
        long_gap_mask = missing & (streaks > 2)

        if is_1d.any():
            ffill_1d_weight = float(np.abs(weights_a[is_1d]).sum())
            n_ffill_1d = int(is_1d.sum())
        if is_2d.any():
            ffill_2d_weight = float(np.abs(weights_a[is_2d]).sum())
            n_ffill_2d = int(is_2d.sum())
        if long_gap_mask.any():
            is_censored_arr[active_idx[long_gap_mask]] = True
            for i in active_idx[long_gap_mask]:
                long_gap_tickers[idx_to_ticker[i]] = float(weights_arr[i])

    return {
        "pnl": pnl,
        "gross_exposure": gross,
        "net_exposure": net,
        "n_positions": n_positions,
        "ffill_1d_weight": ffill_1d_weight,
        "ffill_2d_weight": ffill_2d_weight,
        "n_ffill_1d": n_ffill_1d,
        "n_ffill_2d": n_ffill_2d,
        "long_gap_tickers": long_gap_tickers,
    }


if _NUMBA_AVAILABLE:

    @_njit(cache=True)
    def _advance_daily_state_numba(
        weights_arr,
        last_prices_arr,
        missing_streaks_arr,
        is_censored_arr,
        entry_dates_arr,
        prices_2d,
        d,
        d_next,
        long_gap_out,
    ):
        """Single fused loop over all tickers — numba-compiled.

        Returns a tuple of scalar results. *long_gap_out* is filled with
        ticker indices that hit a long gap. Mutates the state arrays in place.
        """
        n_tickers = weights_arr.shape[0]
        pnl = 0.0
        gross = 0.0
        net = 0.0
        n_positions = 0
        ffill_1d_weight = 0.0
        ffill_2d_weight = 0.0
        n_ffill_1d = 0
        n_ffill_2d = 0
        n_long_gap = 0

        for i in range(n_tickers):
            w = weights_arr[i]
            if w == 0.0:
                continue
            ed = entry_dates_arr[i]
            if ed < 0 or ed > d:
                continue

            n_positions += 1
            gross += abs(w)
            net += w

            today_px = prices_2d[i, d]
            next_px = prices_2d[i, d_next]

            # Pass 1: setdefault
            if not np.isnan(today_px) and today_px > 0.0:
                if np.isnan(last_prices_arr[i]):
                    last_prices_arr[i] = today_px

            # Censored check
            if is_censored_arr[i]:
                if np.isnan(next_px):
                    long_gap_out[n_long_gap] = i
                    n_long_gap += 1
                continue

            # Pass 2: valid next price
            if not np.isnan(next_px) and next_px > 0.0:
                base_px = last_prices_arr[i]
                if not np.isnan(base_px) and base_px > 0.0:
                    pnl += w * (next_px / base_px - 1.0)
                last_prices_arr[i] = next_px
                missing_streaks_arr[i] = 0
                continue

            # Pass 3: missing next price
            if np.isnan(last_prices_arr[i]):
                continue
            streak = missing_streaks_arr[i] + 1
            missing_streaks_arr[i] = streak
            if streak == 1:
                ffill_1d_weight += abs(w)
                n_ffill_1d += 1
            elif streak == 2:
                ffill_2d_weight += abs(w)
                n_ffill_2d += 1
            else:
                long_gap_out[n_long_gap] = i
                n_long_gap += 1
                is_censored_arr[i] = True

        return (pnl, gross, net, n_positions,
                ffill_1d_weight, ffill_2d_weight,
                n_ffill_1d, n_ffill_2d, n_long_gap)


def _advance_daily_state_numba_wrapper(
    weights_arr: np.ndarray,
    last_prices_arr: np.ndarray,
    missing_streaks_arr: np.ndarray,
    is_censored_arr: np.ndarray,
    entry_dates_arr: np.ndarray,
    prices_2d: np.ndarray,
    d: int,
    d_next: int,
    idx_to_ticker: list[str],
) -> dict:
    """Python wrapper that calls the numba kernel and builds the result dict."""
    n_tickers = weights_arr.shape[0]
    long_gap_out = np.zeros(n_tickers, dtype=np.int64)

    (pnl, gross, net, n_positions,
     ffill_1d_weight, ffill_2d_weight,
     n_ffill_1d, n_ffill_2d, n_long_gap) = _advance_daily_state_numba(
        weights_arr, last_prices_arr, missing_streaks_arr, is_censored_arr,
        entry_dates_arr, prices_2d, d, d_next, long_gap_out,
    )

    long_gap_tickers: dict[str, float] = {}
    for j in range(n_long_gap):
        i = long_gap_out[j]
        long_gap_tickers[idx_to_ticker[i]] = float(weights_arr[i])

    return {
        "pnl": float(pnl),
        "gross_exposure": float(gross),
        "net_exposure": float(net),
        "n_positions": int(n_positions),
        "ffill_1d_weight": float(ffill_1d_weight),
        "ffill_2d_weight": float(ffill_2d_weight),
        "n_ffill_1d": int(n_ffill_1d),
        "n_ffill_2d": int(n_ffill_2d),
        "long_gap_tickers": long_gap_tickers,
    }


def _empty_result(
    universe_coverage: pd.DataFrame | None = None,
    trade_records: list[dict[str, Any]] | None = None,
    trade_log_violations: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Build a standard empty PortfolioResult with stable schemas."""
    return PortfolioResult(
        daily_returns=pd.DataFrame(columns=_EMPTY_COLUMNS),
        weights_history={},
        cohort_weights_history={},
        trade_log=pd.DataFrame(trade_records) if trade_records else pd.DataFrame(),
        trade_log_violations=(
            trade_log_violations
            if trade_log_violations is not None
            else pd.DataFrame()
        ),
        universe_coverage=(
            universe_coverage
            if universe_coverage is not None
            else pd.DataFrame()
        ),
    )


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------


def _next_valid_quote(
    close_matrix: pd.DataFrame,
    ticker: str,
    planned_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    cal_pos: dict[pd.Timestamp, int],
    max_forward_days: int = MAX_QUOTE_FORWARD_DAYS,
) -> tuple[pd.Timestamp, float]:
    """Find the next valid close on or after *planned_date*, up to
    *max_forward_days* consecutive trading days.

    Returns ``(actual_date, price)``.  If no valid quote is found within the
    tolerance window, returns ``(pd.NaT, np.nan)``.
    """
    if planned_date not in cal_pos:
        return pd.NaT, np.nan
    start_pos = cal_pos[planned_date]
    for offset in range(max_forward_days + 1):
        pos = start_pos + offset
        if pos >= len(calendar):
            break
        try_date = calendar[pos]
        if (
            try_date in close_matrix.index
            and ticker in close_matrix.columns
            and pd.notna(close_matrix.at[try_date, ticker])
        ):
            return try_date, float(close_matrix.at[try_date, ticker])
    return pd.NaT, np.nan


def _audit_long_gap_recovery(
    close_matrix: pd.DataFrame,
    cal_list: list[pd.Timestamp],
    cal_pos: dict[pd.Timestamp, int],
    long_gap_records: dict[pd.Timestamp, dict[str, float]],
    max_recovery_days: int = 30,
) -> dict[tuple[pd.Timestamp, str], bool]:
    """Check whether long-gap tickers resumed trading within *max_recovery_days*.

    For each ``(gap_date, ticker)`` in *long_gap_records*, scan forward up to
    *max_recovery_days* trading days in the calendar.  If a valid adjusted
    close is found anywhere inside that window the gap is considered
    *recovered*, otherwise it stays *unresolved*.

    Returns a dict keyed by ``(gap_date, ticker)`` with ``True`` for
    recovered, ``False`` otherwise.
    """
    result: dict[tuple[pd.Timestamp, str], bool] = {}
    n_total = sum(len(v) for v in long_gap_records.values())
    if n_total == 0:
        return result

    for gap_date, gap_dict in long_gap_records.items():
        gap_pos = cal_pos.get(gap_date)
        if gap_pos is None:
            for tkr in gap_dict:
                result[(gap_date, tkr)] = False
            continue
        end_pos = min(gap_pos + max_recovery_days, len(cal_list) - 1)
        for tkr in gap_dict:
            found = False
            for j in range(gap_pos + 1, end_pos + 1):
                check_date = cal_list[j]
                if (
                    check_date in close_matrix.index
                    and tkr in close_matrix.columns
                    and pd.notna(close_matrix.at[check_date, tkr])
                ):
                    found = True
                    break
            result[(gap_date, tkr)] = found
    return result


def _apply_long_gap_recovery(
    daily_df: pd.DataFrame,
    state: SimulationState,
    close_matrix: pd.DataFrame,
    calendar: list[pd.Timestamp],
    cal_pos_by_date: dict[pd.Timestamp, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 30-trading-day long-gap recovery audit and update *daily_df*.

    Returns (updated daily_df, gap_records_df).
    """
    recovery_result: dict[tuple[pd.Timestamp, str], bool] = {}
    if not daily_df.empty and state.long_gap_records:
        log.info(
            "Long-gap recovery audit: %d dates with gap events",
            len(state.long_gap_records),
        )
        recovery_result = _audit_long_gap_recovery(
            close_matrix, calendar, cal_pos_by_date, state.long_gap_records,
            max_recovery_days=30,
        )

        # Map recovery results back to daily_df rows
        for idx in daily_df.index:
            date = daily_df.at[idx, "date"]
            gap_dict = state.long_gap_records.get(date, {})
            if not gap_dict:
                continue
            recovered_w = 0.0
            unrecovered_w = 0.0
            recovered_n = 0
            unrecovered_n = 0
            for tkr, w in gap_dict.items():
                if recovery_result.get((date, tkr), False):
                    recovered_w += abs(w)
                    recovered_n += 1
                else:
                    unrecovered_w += abs(w)
                    unrecovered_n += 1
            daily_df.at[idx, "long_gap_recovered_weight"] = recovered_w
            daily_df.at[idx, "possible_delisting_or_unavailable_weight"] = unrecovered_w
            daily_df.at[idx, "n_long_gap_recovered_positions"] = recovered_n
            daily_df.at[idx, "n_possible_delisting_or_unavailable_positions"] = unrecovered_n

        n_recovered = sum(1 for v in recovery_result.values() if v)
        n_unrecovered = sum(1 for v in recovery_result.values() if not v)
        log.info(
            "Long-gap recovery audit: %d recovered, %d unrecovered (possible delisting)",
            n_recovered, n_unrecovered,
        )

    # Build gap_records_df for persistence
    gap_records_frames: list[dict[str, Any]] = []
    if state.long_gap_records:
        for gap_date, gap_dict in state.long_gap_records.items():
            for tkr, w in gap_dict.items():
                recovered = recovery_result.get((gap_date, tkr), False) if recovery_result else False
                gap_records_frames.append({
                    "gap_date": gap_date,
                    "ticker": tkr,
                    "weight": w,
                    "recovered": recovered,
                })
    gap_records_df = pd.DataFrame(gap_records_frames) if gap_records_frames else pd.DataFrame()

    return daily_df, gap_records_df


class PortfolioSimulator:
    """Rebalanced portfolio simulation.

    Parameters
    ----------
    price_dir:
        Path to ``{ticker}.parquet`` price files.
    universe_name:
        Which universe to use (for initial ticker list).
    """

    def __init__(
        self,
        price_dir: Path = PRICE_CACHE_DIR,
        universe_name: str = "sp500",
    ) -> None:
        self.price_dir = Path(price_dir)
        self.universe_name = universe_name
        self._price_table: pd.DataFrame | None = None
        self._calendar: pd.DatetimeIndex | None = None
        self._signals: pd.DataFrame | None = None
        self._market_cache: dict[
            tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp, int, str, str],
            tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, int], pd.DatetimeIndex, pd.DataFrame],
        ] = {}

    # -------------------------------------------------------------------
    # C1.3 — Validate signals, normalize dates, apply PIT universe filter
    # -------------------------------------------------------------------

    def _validate_and_filter_signals(
        self,
        signals: pd.DataFrame,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
        """Normalize signal columns, date bounds, and PIT universe membership."""
        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
        signals = signals[signals["date"].notna()]
        signals["ticker"] = signals["ticker"].astype(str)

        if start is None:
            start = signals["date"].min()
        if end is None:
            end = signals["date"].max()

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        n_before = len(signals)
        signals = filter_to_universe(
            signals,
            self.universe_name,
            date_col="date",
            ticker_col="ticker",
        )
        signals = signals[signals["_in_universe"]].copy()
        log.info(
            "Universe filter (%s): %d / %d signals kept",
            self.universe_name,
            len(signals),
            n_before,
        )
        return signals, start, end, n_before

    # -------------------------------------------------------------------
    # C1.4 — Collect traded tickers, coverage tickers, and union
    # -------------------------------------------------------------------

    def _collect_price_tickers(
        self,
        signals: pd.DataFrame,
    ) -> tuple[set[str], set[str], set[str]]:
        """Collect traded tickers, coverage tickers, and union price tickers."""
        tickers = set(signals["ticker"].unique())
        _, snapshots = _load_universe_snapshots(self.universe_name)
        coverage_tickers: set[str] = set()
        for members in snapshots.values():
            coverage_tickers.update(members)
        price_tickers = tickers | coverage_tickers
        return tickers, coverage_tickers, price_tickers

    # -------------------------------------------------------------------
    # C1.5 — Load prices, build calendar, rebalance dates, coverage
    # -------------------------------------------------------------------

    def _load_calendar_prices_and_coverage(
        self,
        price_tickers: set[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        lookback: int,
        cadence: str,
        weekly_day: str = "monday",
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, int], pd.DatetimeIndex, pd.DataFrame]:
        """Load prices, close matrix, trading calendar, rebalance dates, and coverage."""
        cache_key = (
            tuple(sorted(price_tickers)),
            pd.Timestamp(start).normalize(),
            pd.Timestamp(end).normalize(),
            int(lookback),
            cadence,
            weekly_day,
        )
        cached = self._market_cache.get(cache_key)
        if cached is not None:
            close_matrix, calendar, cal_pos_by_date, rebal_dates, universe_coverage = cached
            self._price_table = None
            self._calendar = calendar
            log.info(
                "Simulation market cache hit: %s cadence, %d lookback, %s weekly timing",
                cadence,
                lookback,
                weekly_day,
            )
            return close_matrix, calendar, cal_pos_by_date, rebal_dates, universe_coverage

        calendar_end = end + pd.Timedelta(days=30)
        self._price_table = _load_price_table(
            price_tickers,
            start - pd.Timedelta(days=lookback + 30),
            calendar_end,
            self.price_dir,
        )
        close_matrix = _build_close_matrix(self._price_table)
        self._calendar = pd.DatetimeIndex(
            close_matrix.index[(close_matrix.index >= start) & (close_matrix.index <= calendar_end)]
        )
        rebal_dates = _rebalance_dates(self._calendar, cadence, start, end, weekly_day=weekly_day)

        log.info(
            "Simulation: %s cadence, %d lookback, %s weekly timing, %d rebalance dates",
            cadence,
            lookback,
            weekly_day,
            len(rebal_dates),
        )

        universe_coverage = _coverage_by_rebalance_date(
            rebal_dates,
            self.universe_name,
            close_matrix,
        )

        cal_pos_by_date = {d: i for i, d in enumerate(self._calendar)}

        result = (close_matrix, self._calendar, cal_pos_by_date, rebal_dates, universe_coverage)
        self._market_cache[cache_key] = result
        return result

    # -------------------------------------------------------------------
    # C1.6 — Compute lookback start date (static helper)
    # -------------------------------------------------------------------

    @staticmethod
    def _rebalance_lookback_start(
        calendar: pd.DatetimeIndex,
        cal_pos_by_date: dict[pd.Timestamp, int],
        date: pd.Timestamp,
        lookback: int,
    ) -> pd.Timestamp:
        """Return the first eligible signal date for a rebalance date."""
        rb_pos = cal_pos_by_date[date]
        return (
            calendar[max(0, rb_pos - lookback + 1)]
            if rb_pos >= lookback - 1
            else calendar[0]
        )

    # -------------------------------------------------------------------
    # C1.7 — Look up entry quotes and rescale weights
    # -------------------------------------------------------------------

    def _lookup_tradeable_entries(
        self,
        agg_weights_df: pd.DataFrame,
        close_matrix: pd.DataFrame,
        calendar: list[pd.Timestamp],
        cal_pos_by_date: dict[pd.Timestamp, int],
        planned_entry_date: pd.Timestamp,
        lookback: int,
    ) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], dict[str, float], list[dict[str, Any]]]:
        """Find entry quotes, remove untradeable names, and rescale gross to 2.0."""
        entry_dates: dict[str, pd.Timestamp] = {}
        entry_prices: dict[str, float] = {}
        skip_records: list[dict[str, Any]] = []

        if not agg_weights_df.empty:
            for _, row in agg_weights_df.iterrows():
                tkr = str(row["ticker"])
                actual_d, actual_px = _next_valid_quote(
                    close_matrix, tkr, planned_entry_date, calendar, cal_pos_by_date,
                )
                if pd.isna(actual_d):
                    skip_records.append({
                        "trade_id": f"{planned_entry_date.date()}:{tkr}:skip",
                        "ticker": tkr,
                        "signal_date": planned_entry_date,
                        "planned_entry_date": planned_entry_date,
                        "actual_entry_date": pd.NaT,
                        "planned_exit_date": pd.NaT,
                        "actual_exit_date": pd.NaT,
                        "skip_reason": "skip_no_entry_quote",
                        "entry_price": np.nan,
                        "exit_price": np.nan,
                        "horizon_days": lookback,
                        "universe": self.universe_name,
                    })
                else:
                    entry_dates[tkr] = actual_d
                    entry_prices[tkr] = actual_px

            # Remove tickers with no entry quote and re-scale to gross=2.0.
            tradeable = list(entry_dates)
            agg_weights_df = agg_weights_df[
                agg_weights_df["ticker"].isin(tradeable)
            ].copy()
            if not agg_weights_df.empty:
                gross_after = agg_weights_df["raw_weight"].abs().sum()
                if gross_after > 1e-10:
                    agg_weights_df["raw_weight"] *= 2.0 / gross_after

        return agg_weights_df, entry_dates, entry_prices, skip_records

    # -------------------------------------------------------------------
    # C1.8 — Append trade records for entered positions
    # -------------------------------------------------------------------

    def _append_entered_trade_records(
        self,
        state: SimulationState,
        close_matrix: pd.DataFrame,
        calendar: list[pd.Timestamp],
        cal_pos_by_date: dict[pd.Timestamp, int],
        rebal_dates: pd.DatetimeIndex,
        date: pd.Timestamp,
        lookback: int,
        entry_dates: dict[str, pd.Timestamp],
        entry_prices: dict[str, float],
    ) -> None:
        """Append trade records for successfully entered positions."""
        next_reb_pos = rebal_dates.searchsorted(date, side="right")
        planned_exit = (
            pd.Timestamp(rebal_dates[next_reb_pos])
            if next_reb_pos < len(rebal_dates)
            else pd.NaT
        )
        for tkr, w in state.current_weights.items():
            entry_price = entry_prices.get(tkr, np.nan)
            actual_entry = entry_dates.get(tkr, pd.NaT)
            exit_price = np.nan
            actual_exit = pd.NaT
            skip_reason = None
            if pd.notna(planned_exit):
                actual_exit, exit_price = _next_valid_quote(
                    close_matrix, tkr, planned_exit,
                    calendar, cal_pos_by_date,
                )
                if pd.isna(actual_exit):
                    skip_reason = "right_censored_no_exit_quote"
            else:
                skip_reason = "right_censored_no_exit_quote"

            state.trade_records.append({
                "trade_id": f"{date.date()}:{tkr}",
                "ticker": tkr,
                "signal_date": date,
                "planned_entry_date": date,
                "actual_entry_date": actual_entry,
                "planned_exit_date": planned_exit,
                "actual_exit_date": actual_exit,
                "skip_reason": skip_reason,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "horizon_days": lookback,
                "universe": self.universe_name,
                "weight": w,
            })

    # -------------------------------------------------------------------
    # C1.9 — Process one rebalance date
    # -------------------------------------------------------------------

    def _process_rebalance(
        self,
        state: SimulationState,
        signals: pd.DataFrame,
        close_matrix: pd.DataFrame,
        calendar: list[pd.Timestamp],
        cal_pos_by_date: dict[pd.Timestamp, int],
        rebal_dates: pd.DatetimeIndex,
        date: pd.Timestamp,
        lookback: int,
        long_frac: float,
    ) -> None:
        """Build weights for one rebalance date and update state."""
        lookback_start = self._rebalance_lookback_start(
            self._calendar, cal_pos_by_date, date, lookback,
        )
        eligible = self._signals.loc[lookback_start:date].reset_index()

        # ---- Build cohort-aggregated weights ----
        cohort_weights_df, agg_weights_df = _build_cohort_weights(
            eligible, long_frac=long_frac, n_min_positions=5,
        )

        # Store raw cohort weights (before cross-cohort aggregation).
        if not cohort_weights_df.empty:
            state.cohort_weights_history[date] = cohort_weights_df

        # ---- Look up entry prices and rescale ----
        agg_weights_df, entry_dates, entry_prices, skip_records = self._lookup_tradeable_entries(
            agg_weights_df, close_matrix, calendar, cal_pos_by_date, date, lookback,
        )

        # No eligible signals — keep existing positions, skip this rebalance.
        if agg_weights_df.empty:
            return

        # Set current weights and entry dates from tradeable names.
        state.current_weights = dict(
            zip(agg_weights_df["ticker"], agg_weights_df["raw_weight"])
        )
        state.current_entry_dates = {
            t: entry_dates[t] for t in state.current_weights if t in entry_dates
        }
        state.current_entry_prices = {
            t: entry_prices[t] for t in state.current_weights if t in entry_prices
        }

        # A rebalance creates a fresh executable portfolio. Reset quote-state
        # tracking so any delayed fills or stale gaps from the prior portfolio
        # cannot bleed into the new holdings.
        state.last_prices = dict(state.current_entry_prices)
        state.missing_streaks = {t: 0 for t in state.current_weights}
        state.censored_tickers = set()

        # Populate parallel numpy arrays (Step 2).
        state.weights_arr[:] = 0.0
        state.last_prices_arr[:] = np.nan
        state.missing_streaks_arr[:] = 0
        state.is_censored_arr[:] = False
        for tkr, w in state.current_weights.items():
            idx = self._ticker_to_idx.get(tkr)
            if idx is not None:
                state.weights_arr[idx] = w
                if tkr in state.current_entry_prices:
                    state.last_prices_arr[idx] = float(state.current_entry_prices[tkr])
                if tkr in state.current_entry_dates:
                    ed = state.current_entry_dates[tkr]
                    state.entry_dates_arr[idx] = self._day_index.get(ed, -1)

        # ---- Turnover ----
        all_tkrs = set(state.current_weights) | set(state.prev_weights)
        state.today_turnover = float(
            sum(abs(state.current_weights.get(t, 0.0) - state.prev_weights.get(t, 0.0)) for t in all_tkrs)
        )

        # Store weights for this rebalance date.
        state.weights_history[date] = agg_weights_df

        # ---- Trade log: skip records first, then entered positions ----
        state.trade_records.extend(skip_records)
        self._append_entered_trade_records(
            state, close_matrix, calendar, cal_pos_by_date, rebal_dates,
            date, lookback, entry_dates, entry_prices,
        )

        state.prev_weights = state.current_weights.copy()

    # -------------------------------------------------------------------
    # C1.10 — Compute one daily PnL row
    # -------------------------------------------------------------------

    def _compute_daily_pnl(
        self,
        state: SimulationState,
        close_matrix: pd.DataFrame,
        calendar: list[pd.Timestamp],
        cal_pos_by_date: dict[pd.Timestamp, int],
        date: pd.Timestamp,
        next_date: pd.Timestamp,
        did_rebalance: bool,
    ) -> None:
        """Compute one daily PnL row with delayed-entry and gap accounting.

        Uses the numpy vectorized path when arrays are populated (production),
        falling back to the dict-based kernel (tests / bootstrap).
        """
        if state.last_prices_arr is not None:
            self._compute_daily_pnl_vectorized(
                state, date, next_date, did_rebalance,
            )
            return

        # --- Dict-based fallback path ---
        active_weights = {
            t: w
            for t, w in state.current_weights.items()
            if pd.Timestamp(state.current_entry_dates.get(t, date)) <= date
        }

        if not active_weights:
            state.daily_records.append({
                "date": date,
                "pnl": 0.0,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "turnover": state.today_turnover,
                "_rebalance": int(did_rebalance),
                "n_positions": 0,
                "ffill_1d_weight": 0.0,
                "ffill_2d_weight": 0.0,
                "long_gap_recovered_weight": 0.0,
                "possible_delisting_or_unavailable_weight": 0.0,
                "n_ffill_1d_positions": 0,
                "n_ffill_2d_positions": 0,
                "n_long_gap_recovered_positions": 0,
                "n_possible_delisting_or_unavailable_positions": 0,
            })
            return

        tickers_list = list(active_weights.keys())

        if date not in close_matrix.index or next_date not in close_matrix.index:
            gross = float(sum(abs(w) for w in active_weights.values()))
            net = float(sum(active_weights.values()))
            state.daily_records.append({
                "date": date,
                "pnl": 0.0,
                "gross_exposure": gross,
                "net_exposure": net,
                "turnover": state.today_turnover,
                "_rebalance": int(did_rebalance),
                "n_positions": len(tickers_list),
                "ffill_1d_weight": 0.0,
                "ffill_2d_weight": 0.0,
                "long_gap_recovered_weight": 0.0,
                "possible_delisting_or_unavailable_weight": 0.0,
                "n_ffill_1d_positions": 0,
                "n_ffill_2d_positions": 0,
                "n_long_gap_recovered_positions": 0,
                "n_possible_delisting_or_unavailable_positions": 0,
            })
            return

        p_today_s = close_matrix.loc[date]
        p_next_s = close_matrix.loc[next_date]
        p_today = {t: float(p_today_s[t]) if t in p_today_s.index and pd.notna(p_today_s[t]) else float("nan") for t in tickers_list}
        p_next = {t: float(p_next_s[t]) if t in p_next_s.index and pd.notna(p_next_s[t]) else float("nan") for t in tickers_list}

        result = _advance_daily_state(
            tickers=tickers_list,
            weights=active_weights,
            last_prices=state.last_prices,
            missing_streaks=state.missing_streaks,
            censored_tickers=state.censored_tickers,
            p_today=p_today,
            p_next=p_next,
        )

        if result["long_gap_tickers"]:
            state.long_gap_records[date] = result["long_gap_tickers"]

        state.daily_records.append({
            "date": date,
            "pnl": result["pnl"],
            "gross_exposure": result["gross_exposure"],
            "net_exposure": result["net_exposure"],
            "turnover": state.today_turnover,
            "_rebalance": int(did_rebalance),
            "n_positions": result["n_positions"],
            "ffill_1d_weight": result["ffill_1d_weight"],
            "ffill_2d_weight": result["ffill_2d_weight"],
            "long_gap_recovered_weight": 0.0,
            "possible_delisting_or_unavailable_weight": 0.0,
            "n_ffill_1d_positions": result["n_ffill_1d"],
            "n_ffill_2d_positions": result["n_ffill_2d"],
            "n_long_gap_recovered_positions": 0,
            "n_possible_delisting_or_unavailable_positions": 0,
        })

    def _compute_daily_pnl_vectorized(
        self,
        state: SimulationState,
        date: pd.Timestamp,
        next_date: pd.Timestamp,
        did_rebalance: bool,
    ) -> None:
        """Vectorized daily PnL — numba when available, pure numpy fallback."""
        d = self._day_index[date]
        d_next = self._day_index[next_date]

        if _NUMBA_AVAILABLE:
            result = _advance_daily_state_numba_wrapper(
                weights_arr=state.weights_arr,
                last_prices_arr=state.last_prices_arr,
                missing_streaks_arr=state.missing_streaks_arr,
                is_censored_arr=state.is_censored_arr,
                entry_dates_arr=state.entry_dates_arr,
                prices_2d=self._prices_2d,
                d=d,
                d_next=d_next,
                idx_to_ticker=self._idx_to_ticker,
            )
        else:
            result = _advance_daily_state_numpy(
                weights_arr=state.weights_arr,
                last_prices_arr=state.last_prices_arr,
                missing_streaks_arr=state.missing_streaks_arr,
                is_censored_arr=state.is_censored_arr,
                entry_dates_arr=state.entry_dates_arr,
                prices_2d=self._prices_2d,
                d=d,
                d_next=d_next,
                idx_to_ticker=self._idx_to_ticker,
            )

        if result["long_gap_tickers"]:
            state.long_gap_records[date] = result["long_gap_tickers"]

        state.daily_records.append({
            "date": date,
            "pnl": result["pnl"],
            "gross_exposure": result["gross_exposure"],
            "net_exposure": result["net_exposure"],
            "turnover": state.today_turnover,
            "_rebalance": int(did_rebalance),
            "n_positions": result["n_positions"],
            "ffill_1d_weight": result["ffill_1d_weight"],
            "ffill_2d_weight": result["ffill_2d_weight"],
            "long_gap_recovered_weight": 0.0,
            "possible_delisting_or_unavailable_weight": 0.0,
            "n_ffill_1d_positions": result["n_ffill_1d"],
            "n_ffill_2d_positions": result["n_ffill_2d"],
            "n_long_gap_recovered_positions": 0,
            "n_possible_delisting_or_unavailable_positions": 0,
        })

    # -------------------------------------------------------------------
    # C1.12 — Post-process: validate trade log, apply gap recovery, build result
    # -------------------------------------------------------------------

    def _post_process_result(
        self,
        state: SimulationState,
        close_matrix: pd.DataFrame,
        calendar: list[pd.Timestamp],
        cal_pos_by_date: dict[pd.Timestamp, int],
        universe_coverage: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cadence: str,
        lookback: int,
        long_frac: float,
        transaction_cost_bps: float,
        weekly_day: str = "monday",
    ) -> PortfolioResult:
        """Validate trade log, apply gap recovery, and return PortfolioResult."""
        daily_df = pd.DataFrame(state.daily_records)

        # ---- Long-gap recovery audit ----
        daily_df, gap_records_df = _apply_long_gap_recovery(
            daily_df, state, close_matrix, calendar, cal_pos_by_date,
        )

        # ---- Post-process trade log: mark unrecovered gaps as delisting exits ----
        if not gap_records_df.empty:
            unrecovered_tickers = set(
                gap_records_df.loc[~gap_records_df["recovered"], "ticker"].unique()
            )
            if unrecovered_tickers:
                for record in state.trade_records:
                    if (record.get("ticker") in unrecovered_tickers
                            and record.get("skip_reason") == "right_censored_no_exit_quote"):
                        record["skip_reason"] = "delisting_exit_used"

        # ---- Validate trade log ----
        trade_log_df = pd.DataFrame(state.trade_records)
        trade_log_violations_df = pd.DataFrame()
        if not trade_log_df.empty:
            trade_log_violations_df = validate_trade_log(trade_log_df)
            if not trade_log_violations_df.empty:
                vc = trade_log_violations_df["check"].value_counts()
                summary_lines = [f"    {chk}: {cnt}" for chk, cnt in vc.items()]
                log.warning(
                    "Trade log validation: %d violations found (e.g. planned<=actual date order)\n%s",
                    len(trade_log_violations_df),
                    "\n".join(summary_lines),
                )

        if daily_df.empty:
            log.warning("No daily returns generated")
            return PortfolioResult(
                daily_returns=pd.DataFrame(
                    columns=["date", "pnl", "gross_exposure", "net_exposure", "turnover",
                             "n_positions", "ffill_1d_weight", "ffill_2d_weight",
                             "long_gap_recovered_weight", "possible_delisting_or_unavailable_weight",
                             "n_ffill_1d_positions", "n_ffill_2d_positions",
                             "n_long_gap_recovered_positions", "n_possible_delisting_or_unavailable_positions"]
                ),
                weights_history={},
                cohort_weights_history={},
                trade_log=pd.DataFrame(state.trade_records),
                trade_log_violations=trade_log_violations_df,
                universe_coverage=universe_coverage,
            )

        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        return PortfolioResult(
            daily_returns=daily_df,
            weights_history=state.weights_history,
            cohort_weights_history=state.cohort_weights_history,
            trade_log=pd.DataFrame(state.trade_records),
            trade_log_violations=trade_log_violations_df,
            universe_coverage=universe_coverage,
            gap_records=gap_records_df,
            config={
                "cadence": cadence,
                "lookback": lookback,
                "weekly_day": weekly_day,
                "long_frac": long_frac,
                "transaction_cost_bps": transaction_cost_bps,
                "start": str(start.date()),
                "end": str(end.date()),
                "universe": self.universe_name,
            },
        )

    # -------------------------------------------------------------------
    # C1.13 — Clean orchestration run() method
    # -------------------------------------------------------------------

    def run(
        self,
        signals: pd.DataFrame,
        cadence: str = "weekly",
        lookback: int = 5,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        long_frac: float = 0.2,
        transaction_cost_bps: float = 5.0,
        weekly_day: str = "monday",
    ) -> PortfolioResult:
        """Run portfolio simulation.

        Parameters
        ----------
        signals:
            DataFrame with ``[date, ticker, score]``.  *date* is the
            availability date of each signal.
        cadence:
            ``daily``, ``weekly``, or ``monthly``.
        weekly_day:
            Weekly rebalance timing, ``monday`` or ``friday``. Ignored for
            daily/monthly cadences.
        lookback:
            Number of trading days to look back for eligible signals at each
            rebalance date.
        start, end:
            Date bounds. Default: min/max of *signals* date.
        long_frac:
            Fraction of stocks in each leg (0.2 = quintile).
        transaction_cost_bps:
            One-way transaction cost in basis points.

        Returns
        -------
        PortfolioResult with daily returns, weights history, and config.
        """
        # C1.3 — Validate and filter signals
        signals, start, end, n_before = self._validate_and_filter_signals(
            signals, start, end,
        )
        if signals.empty:
            log.warning("No signals left after PIT universe filtering")
            return _empty_result()

        # Pre-index signals by date for O(log N) lookups at each rebalance.
        self._signals = signals.set_index("date").sort_index()

        # C1.4 — Collect ticker sets
        tickers, coverage_tickers, price_tickers = self._collect_price_tickers(signals)

        # C1.5 — Load prices, calendar, rebalance dates, coverage
        close_matrix, calendar, cal_pos_by_date, rebal_dates, universe_coverage = (
            self._load_calendar_prices_and_coverage(
                price_tickers, start, end, lookback, cadence, weekly_day=weekly_day,
            )
        )

        # Build flat price matrix and index mappings once (Step 1).
        self._ticker_to_idx = {t: i for i, t in enumerate(close_matrix.columns)}
        self._idx_to_ticker = list(close_matrix.columns)
        self._day_index = {d: i for i, d in enumerate(close_matrix.index)}
        self._prices_2d = close_matrix.values.T.astype(np.float64)  # (n_tickers, n_days)

        # ---- Simulation loop ----
        state = SimulationState()
        n_tickers = len(self._idx_to_ticker)
        state.last_prices_arr = np.full(n_tickers, np.nan, dtype=np.float64)
        state.missing_streaks_arr = np.zeros(n_tickers, dtype=np.int32)
        state.is_censored_arr = np.zeros(n_tickers, dtype=bool)
        state.weights_arr = np.zeros(n_tickers, dtype=np.float64)
        state.entry_dates_arr = np.full(n_tickers, -1, dtype=np.int64)
        cal_list = list(calendar)
        rebal_set = set(rebal_dates)
        for i in progress(
            range(len(cal_list) - 1),
            desc=f"Simulating ({cadence})",
            unit="day",
        ):
            date = cal_list[i]
            next_date = cal_list[i + 1]
            if date < start:
                continue
            if date > end:
                break

            state.today_turnover = 0.0
            did_rebalance = date in rebal_set

            if did_rebalance:
                self._process_rebalance(
                    state, signals, close_matrix, cal_list, cal_pos_by_date,
                    rebal_dates, date, lookback, long_frac,
                )

            self._compute_daily_pnl(
                state, close_matrix, cal_list, cal_pos_by_date,
                date, next_date, did_rebalance,
            )

        return self._post_process_result(
            state, close_matrix, cal_list, cal_pos_by_date, universe_coverage,
            start, end, cadence, lookback, long_frac, transaction_cost_bps, weekly_day,
        )


# ---------------------------------------------------------------------------
# Capacity proxy
# ---------------------------------------------------------------------------


def compute_capacity_metrics(
    weights_history: dict[pd.Timestamp, pd.DataFrame],
    price_dir: Path = PRICE_CACHE_DIR,
    aum_grid: tuple[float, ...] = (10_000_000, 50_000_000, 100_000_000),
    addv_window: int = 20,
    price_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute capacity metrics from a weights history.

    Parameters
    ----------
    weights_history:
        Map from rebalance_date to ticker weights DataFrame.
    price_dir:
        Path to ticker price parquet files (used when *price_table* is None).
    aum_grid:
        Asset levels for %ADV consumption.
    addv_window:
        Trading days for average daily dollar volume computation.
    price_table:
        Optional pre-loaded price DataFrame with columns ``[date, ticker, adj_close, volume]``.
        When provided, it is used instead of loading individual parquet files from *price_dir*.

    Returns
    -------
    Dict with holding count, top-10 concentration, and %ADV at each AUM level.
    """
    # Collect all tickers
    all_tickers: set[str] = set()
    for wdf in weights_history.values():
        all_tickers.update(wdf["ticker"].tolist())

    # Load price/volume for ADV calc
    vol_data: dict[str, pd.Series] = {}
    if price_table is not None:
        pt = price_table.copy()
        pt["dollar_volume"] = pt["adj_close"].astype(float) * pt["volume"].astype(float)
        for tkr in all_tickers:
            mask = pt["ticker"] == tkr
            if mask.any():
                vol_data[tkr] = pt.loc[mask, ["date", "dollar_volume"]].set_index("date")["dollar_volume"]
    else:
        for tkr in all_tickers:
            path = price_dir / f"{tkr}.parquet"
            if not path.exists():
                continue
            px = pd.read_parquet(path, columns=["date", "adj_close", "volume"])
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px = px.dropna(subset=["adj_close", "volume"])
            px = px[px["adj_close"] > 0]
            if px.empty:
                continue
            px["dollar_volume"] = px["adj_close"].astype(float) * px["volume"].astype(
                float
            )
            vol_data[tkr] = px.set_index("date")["dollar_volume"]

    n_holdings: list[int] = []
    top10_concentration: list[float] = []
    adv_consumption: dict[float, list[float]] = {a: [] for a in aum_grid}

    for date, wdf in sorted(weights_history.items()):
        if wdf.empty:
            continue
        wdf = wdf[wdf["raw_weight"] != 0]
        weights = wdf.set_index("ticker")["raw_weight"].abs()

        n_holdings.append(len(weights))
        top10 = weights.nlargest(10).sum() / weights.sum() if len(weights) > 0 else 1.0
        top10_concentration.append(float(top10))

        for aum in aum_grid:
            consumed_pct: list[float] = []
            for tkr, w in weights.items():
                if tkr not in vol_data:
                    continue
                series = vol_data[tkr]
                lookback_vals = series[series.index < date]
                if len(lookback_vals) < addv_window:
                    continue
                adv = float(lookback_vals.iloc[-addv_window:].mean())
                if adv > 0:
                    consumed_pct.append(float(aum * w / adv))
            if consumed_pct:
                adv_consumption[aum].append(float(np.max(consumed_pct)))

    result: dict[str, Any] = {
        "avg_n_holdings": float(np.mean(n_holdings)) if n_holdings else 0.0,
        "avg_top10_concentration": float(np.mean(top10_concentration))
        if top10_concentration
        else 0.0,
    }
    for aum, vals in adv_consumption.items():
        if vals:
            result[f"pct_adv_consumed_{int(aum/1e6)}M_avg"] = float(np.mean(vals))
            result[f"pct_adv_consumed_{int(aum/1e6)}M_p95"] = float(
                np.percentile(vals, 95)
            )
        else:
            result[f"pct_adv_consumed_{int(aum/1e6)}M_avg"] = np.nan
            result[f"pct_adv_consumed_{int(aum/1e6)}M_p95"] = np.nan
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _normalize_signals(
    signals_path: Path,
    features_path: Path | None,
    signal_type: str,
    date_col: str,
) -> pd.DataFrame:
    """Load signals or OOS predictions and return standard [date, ticker, score] columns."""
    signals = pd.read_parquet(signals_path)
    log.info("Loaded %d signal rows from %s", len(signals), signals_path)

    # ---- Convert OOS prediction format [df_index, y_pred] -> [date, ticker, score] ----
    if "date" not in signals.columns and "df_index" in signals.columns:
        if features_path is None:
            log.error(
                "Signals file has 'df_index' column but no 'date'/'ticker' -- "
                "this is Phase 4 OOS prediction format. Pass --features "
                "<features.parquet> to join with feature metadata."
            )
            raise SystemExit(1)
        log.info("Joining OOS predictions with features from %s", features_path)
        from backtest.splits import read_feature_columns

        meta_cols = ["BESTTICKER", date_col]
        features = read_feature_columns(
            features_path,
            [*meta_cols, "SignalType"],
            required_columns=meta_cols,
        )
        if "SignalType" in features.columns:
            meta_cols.append("SignalType")
        signals = signals.merge(
            features[meta_cols],
            left_on="df_index",
            right_index=True,
            how="left",
        )
        if signal_type.lower() != "all" and "SignalType" in signals.columns:
            before = len(signals)
            signals = signals[signals["SignalType"] == signal_type].copy()
            log.info(
                "SignalType filter %s: %d / %d rows kept",
                signal_type,
                len(signals),
                before,
            )
        signals = signals.rename(
            columns={
                date_col: "date",
                "BESTTICKER": "ticker",
                "y_pred": "score",
            }
        )
        signals = signals[signals["date"].notna() & signals["ticker"].notna()]
        log.info("After join: %d rows with date+ticker", len(signals))
    else:
        if signal_type.lower() != "all" and "SignalType" in signals.columns:
            before = len(signals)
            signals = signals[signals["SignalType"] == signal_type].copy()
            log.info(
                "SignalType filter %s: %d / %d rows kept",
                signal_type,
                len(signals),
                before,
            )
        rename_cols: dict[str, str] = {}
        if "ticker" not in signals.columns and "BESTTICKER" in signals.columns:
            rename_cols["BESTTICKER"] = "ticker"
        if "score" not in signals.columns and "y_pred" in signals.columns:
            rename_cols["y_pred"] = "score"
        if "date" not in signals.columns and date_col in signals.columns:
            rename_cols[date_col] = "date"
        if rename_cols:
            signals = signals.rename(columns=rename_cols)

    required_cols = {"date", "ticker", "score"}
    missing_cols = sorted(required_cols - set(signals.columns))
    if missing_cols:
        raise KeyError(f"signals missing required columns after normalization: {missing_cols}")

    return signals


def _portfolio_output_suffix(
    universe: str,
    cadence: str,
    lookback: int,
    tag: str | None,
    signals: pd.DataFrame,
    weekly_day: str = "monday",
) -> str:
    """Build the existing output filename suffix."""
    suffix_parts = [universe]
    if tag:
        suffix_parts.append(_slug(tag))
    suffix_parts.extend([cadence, f"{lookback}d"])
    if cadence == "weekly" and weekly_day != "monday":
        suffix_parts.append(_slug(weekly_day))
    return "_".join(suffix_parts)


def _weights_history_to_frame(
    weights_history: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Convert rebalance-date keyed weights history to tall DataFrame."""
    weight_frames: list[pd.DataFrame] = []
    for date, wdf in weights_history.items():
        if not wdf.empty:
            wdf_copy = wdf.copy()
            wdf_copy["rebalance_date"] = date
            weight_frames.append(wdf_copy)
    if weight_frames:
        return pd.concat(weight_frames, ignore_index=True)
    return pd.DataFrame(columns=["ticker", "raw_weight", "rebalance_date"])


def _cohort_weights_history_to_frame(
    cohort_weights_history: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Convert raw cohort weights history to tall DataFrame."""
    cohort_weight_frames: list[pd.DataFrame] = []
    for date, cwdf in cohort_weights_history.items():
        if not cwdf.empty:
            cwdf_copy = cwdf.copy()
            cwdf_copy["rebalance_date"] = date
            cohort_weight_frames.append(cwdf_copy)
    if cohort_weight_frames:
        return pd.concat(cohort_weight_frames, ignore_index=True)
    return pd.DataFrame(
        columns=["cohort_date", "ticker", "raw_weight", "rebalance_date"]
    )


def _persist_portfolio_results(
    result: PortfolioResult,
    output_dir: Path,
    audit_dir: Path,
    suffix: str,
    cost_bps: float,
    price_cache_dir: Path,
    tag: str | None,
    price_table: pd.DataFrame | None = None,
    # Cache manifest parameters (optional for backward compat).
    signals_path: Path | None = None,
    features_path: Path | None = None,
    signal_type: str | None = None,
    date_col: str | None = None,
) -> Path:
    """Write portfolio, audit, capacity, and summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    summary = result.summary(cost_bps=cost_bps)
    cap = compute_capacity_metrics(
        result.weights_history, price_cache_dir, price_table=price_table,
    )

    result.daily_returns.to_parquet(
        output_dir / f"daily_returns_{suffix}.parquet"
    )

    # Persist weights history as a tall DataFrame
    weights_out = _weights_history_to_frame(result.weights_history)
    weights_out.to_parquet(output_dir / f"weights_{suffix}.parquet")

    # Persist raw cohort weights (before cross-cohort aggregation)
    cohort_weights_out = _cohort_weights_history_to_frame(result.cohort_weights_history)
    cohort_weights_out.to_parquet(
        output_dir / f"cohort_weights_{suffix}.parquet"
    )

    trade_log = result.trade_log.copy()
    if trade_log.empty:
        trade_log = pd.DataFrame(
            columns=[
                "trade_id", "ticker", "signal_date", "planned_entry_date",
                "actual_entry_date", "planned_exit_date", "actual_exit_date",
                "skip_reason", "entry_price", "exit_price", "horizon_days",
                "universe", "weight",
            ]
        )
    trade_log.to_parquet(audit_dir / f"trade_execution_log_{suffix}.parquet")
    trade_log.to_parquet(audit_dir / "trade_execution_log.parquet")

    # Persist trade log violations — always write both generic and scenario-
    # specific files so downstream consumers can distinguish "no violations"
    # from "scenario not run."
    violations = result.trade_log_violations
    if not violations.empty:
        violations.to_parquet(
            audit_dir / f"trade_execution_violations_{suffix}.parquet"
        )
        log.warning("Trade log violations: %d rows written", len(violations))
    else:
        # Write empty sentinel so the file exists
        pd.DataFrame().to_parquet(audit_dir / f"trade_execution_violations_{suffix}.parquet")
    # Always write a generic file (even if empty, so manifest can reference it)
    violations.to_parquet(audit_dir / "trade_execution_violations.parquet")

    coverage = result.universe_coverage.copy()
    coverage.to_csv(audit_dir / f"universe_coverage_by_date_{suffix}.csv", index=False)
    coverage.to_csv(audit_dir / "universe_coverage_by_date.csv", index=False)

    # Persist gap accounting details
    gap_records = result.gap_records
    if not gap_records.empty:
        gap_records.to_parquet(output_dir / f"gap_accounting_{suffix}.parquet")
        log.info("Gap accounting: %d gap event rows written", len(gap_records))

    combined = {**summary, **cap, "config": {**result.config, "tag": tag}}
    json_path = output_dir / f"summary_{suffix}.json"
    json_path.write_text(json.dumps(combined, indent=2, default=str))

    log.info("Portfolio simulation complete.")
    log.info("Sharpe (post-cost 5bps): %.3f", summary.get("sharpe_post_cost_5bps", np.nan))
    log.info("Max drawdown: %.2f%%", summary.get("max_drawdown", np.nan) * 100)
    log.info("Annual turnover: %.2f", summary.get("ann_turnover", np.nan))

    # --- Write cache manifest for this simulation run ---
    try:
        from data.cache_utils import build_cache_manifest, write_cache_manifest

        cfg = result.config
        phase_key = f"5c_{suffix}"

        manifest_params = {
            "universe": cfg.get("universe"),
            "cadence": cfg.get("cadence"),
            "lookback": cfg.get("lookback"),
            "long_frac": cfg.get("long_frac"),
            "cost_bps": cost_bps,
            "weekly_day": cfg.get("weekly_day", "monday"),
            "tag": tag,
            "signal_type": signal_type,
            "date_col": date_col,
        }

        manifest_inputs: list[Path] = []
        if signals_path is not None:
            manifest_inputs.append(signals_path)
        if features_path is not None:
            manifest_inputs.append(features_path)
        manifest_inputs.append(price_cache_dir / "_manifest.json")
        universe_name = cfg.get("universe", "sp500")
        manifest_inputs.append(UNIVERSE_CACHE_DIR / f"{universe_name}_pit.parquet")

        manifest_sources = [
            PortfolioSimulator.run,
            PortfolioSimulator._process_rebalance,
            PortfolioSimulator._compute_daily_pnl,
            PortfolioSimulator._post_process_result,
            _build_cohort_weights,
            _persist_portfolio_results,
        ]

        manifest = build_cache_manifest(
            phase=phase_key,
            parameters=manifest_params,
            input_paths=manifest_inputs,
            source_funcs=manifest_sources,
        )
        write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"{phase_key}.json")
        log.info("Cache manifest written: %s.json", phase_key)
    except Exception:
        log.warning("Failed to write cache manifest for %s", suffix, exc_info=True)

    return json_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Rebalanced portfolio simulation (Phase 5.4)"
    )
    p.add_argument(
        "--signals",
        type=Path,
        required=True,
        help="Path to signals/predictions parquet [date, ticker, score]",
    )
    p.add_argument(
        "--universe",
        choices=["sp500", "sp1500", "ru3k"],
        default="sp500",
    )
    p.add_argument(
        "--cadence",
        choices=["daily", "weekly", "monthly"],
        default="weekly",
    )
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument(
        "--weekly-day",
        choices=["monday", "friday"],
        default="monday",
        help="Weekly rebalance timing for weekly cadence robustness",
    )
    p.add_argument("--long-frac", type=float, default=0.2)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "portfolio")
    p.add_argument("--price-cache", type=Path, default=PRICE_CACHE_DIR)
    p.add_argument(
        "--tag",
        default=None,
        help="Optional model/scenario tag included in output filenames",
    )
    p.add_argument(
        "--signal-type",
        default="Total",
        help="SignalType to trade when metadata is available; use 'all' to disable",
    )
    p.add_argument(
        "--date-col",
        choices=["call_entry_date", "availability_date"],
        default="availability_date",
        help="Feature metadata date to use as signal availability for OOS predictions",
    )
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to features parquet (needed when --signals is OOS predictions "
             "with df_index instead of date/ticker columns)",
    )
    args = p.parse_args()

    signals = _normalize_signals(
        signals_path=args.signals,
        features_path=args.features,
        signal_type=args.signal_type,
        date_col=args.date_col,
    )

    sim = PortfolioSimulator(
        price_dir=args.price_cache,
        universe_name=args.universe,
    )
    result = sim.run(
        signals=signals,
        cadence=args.cadence,
        lookback=args.lookback,
        long_frac=args.long_frac,
        transaction_cost_bps=args.cost_bps,
        weekly_day=args.weekly_day,
    )

    # Resolve tag for output filenames
    tag = args.tag
    if tag is None and "model" in signals.columns:
        models = signals["model"].dropna().astype(str).unique()
        if len(models) == 1:
            tag = models[0]

    suffix = _portfolio_output_suffix(
        universe=args.universe,
        cadence=args.cadence,
        lookback=args.lookback,
        tag=tag,
        signals=signals,
        weekly_day=args.weekly_day,
    )

    _persist_portfolio_results(
        result=result,
        output_dir=Path(args.output_dir),
        audit_dir=AUDIT_DIR,
        suffix=suffix,
        cost_bps=args.cost_bps,
        price_cache_dir=args.price_cache,
        tag=tag,
        price_table=sim._price_table,
        signals_path=args.signals,
        features_path=args.features,
        signal_type=args.signal_type,
        date_col=args.date_col,
    )


if __name__ == "__main__":
    main()
