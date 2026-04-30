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

from data.config import (
    AUDIT_DIR,
    PRICE_CACHE_DIR,
    RESULTS_DIR,
    UNIVERSE_CACHE_DIR,
)
from data.progress import progress
from backtest.universe import filter_to_universe
from features.audit import validate_trade_log

log = logging.getLogger("backtest.portfolio")


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
) -> pd.DatetimeIndex:
    """Return rebalance dates within [start, end].

    - daily: every trading day
    - weekly: every Monday (or next trading day if Monday is not in calendar)
    - monthly: first trading day of each month
    """
    cal = calendar[(calendar >= start) & (calendar <= end)]
    if cadence == "daily":
        return cal
    if cadence == "weekly":
        # Monday close, or the next trading day when Monday is a holiday.
        # Group by ISO year as well as week so late-December / early-January
        # weeks do not collide across calendar years.
        iso = cal.isocalendar()
        week_groups = pd.Series(cal, index=cal).groupby(
            [iso["year"].to_numpy(), iso["week"].to_numpy()],
            sort=True,
        )
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
            **gap_stats,
        }


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------


def _next_valid_quote(
    close_matrix: pd.DataFrame,
    ticker: str,
    planned_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    cal_pos: dict[pd.Timestamp, int],
    max_forward_days: int = 3,
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

    def run(
        self,
        signals: pd.DataFrame,
        cadence: str = "weekly",
        lookback: int = 5,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        long_frac: float = 0.2,
        transaction_cost_bps: float = 5.0,
    ) -> PortfolioResult:
        """Run portfolio simulation.

        Parameters
        ----------
        signals:
            DataFrame with ``[date, ticker, score]``.  *date* is the
            availability date of each signal.
        cadence:
            ``daily``, ``weekly``, or ``monthly``.
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

        # Enforce the requested PIT universe before any portfolio construction.
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
        if signals.empty:
            log.warning("No signals left after PIT universe filtering")
            return PortfolioResult(
                daily_returns=pd.DataFrame(
                    columns=["date", "pnl", "gross_exposure", "net_exposure", "turnover", "_rebalance",
                             "n_positions", "ffill_1d_weight", "ffill_2d_weight",
                             "long_gap_recovered_weight", "possible_delisting_or_unavailable_weight",
                             "n_ffill_1d_positions", "n_ffill_2d_positions",
                             "n_long_gap_recovered_positions", "n_possible_delisting_or_unavailable_positions"]
                ),
                weights_history={},
                cohort_weights_history={},
                trade_log=pd.DataFrame(),
                universe_coverage=pd.DataFrame(),
            )

        # Build calendar and price table. Include all historical PIT members in
        # the price load so the coverage audit is not limited to traded names.
        tickers = set(signals["ticker"].unique())
        _, snapshots = _load_universe_snapshots(self.universe_name)
        coverage_tickers: set[str] = set()
        for members in snapshots.values():
            coverage_tickers.update(members)
        price_tickers = tickers | coverage_tickers

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
        rebal_dates = _rebalance_dates(self._calendar, cadence, start, end)

        log.info(
            "Simulation: %s cadence, %d lookback, %d rebalance dates",
            cadence,
            lookback,
            len(rebal_dates),
        )

        universe_coverage = _coverage_by_rebalance_date(
            rebal_dates,
            self.universe_name,
            close_matrix,
        )

        # ---- Run simulation loop ----
        weights_history: dict[pd.Timestamp, pd.DataFrame] = {}
        cohort_weights_history: dict[pd.Timestamp, pd.DataFrame] = {}
        daily_records: list[dict[str, Any]] = []
        trade_records: list[dict[str, Any]] = []

        prev_weights: dict[str, float] = {}
        current_weights: dict[str, float] = {}
        current_entry_dates: dict[str, pd.Timestamp] = {}
        cal_list = list(self._calendar)
        cal_pos = {d: i for i, d in enumerate(cal_list)}
        rebal_set = set(rebal_dates)

        # Gap accounting: date -> {ticker: weight} for gaps > 2 trading days
        long_gap_records: dict[pd.Timestamp, dict[str, float]] = {}

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

            did_rebalance = date in rebal_set
            turnover = 0.0

            if did_rebalance:
                rb_pos = cal_pos[date]
                lookback_start = (
                    self._calendar[max(0, rb_pos - lookback + 1)]
                    if rb_pos >= lookback - 1
                    else self._calendar[0]
                )
                eligible = signals[
                    (signals["date"] >= lookback_start) & (signals["date"] <= date)
                ]

                # ---- Build cohort-aggregated weights ----
                cohort_weights_df, agg_weights_df = _build_cohort_weights(
                    eligible, long_frac=long_frac, n_min_positions=5,
                )

                # Store raw cohort weights (before cross-cohort aggregation).
                if not cohort_weights_df.empty:
                    cohort_weights_history[date] = cohort_weights_df

                # ---- Look up entry prices for weighted tickers ----
                entry_dates: dict[str, pd.Timestamp] = {}
                entry_prices: dict[str, float] = {}
                if not agg_weights_df.empty:
                    for _, row in agg_weights_df.iterrows():
                        tkr = str(row["ticker"])
                        actual_d, actual_px = _next_valid_quote(
                            close_matrix, tkr, date, cal_list, cal_pos, max_forward_days=3,
                        )
                        if pd.isna(actual_d):
                            trade_records.append({
                                "trade_id": f"{date.date()}:{tkr}:skip",
                                "ticker": tkr,
                                "signal_date": date,
                                "planned_entry_date": date,
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

                    current_weights = dict(
                        zip(agg_weights_df["ticker"], agg_weights_df["raw_weight"])
                    )
                    current_entry_dates = {
                        t: entry_dates[t] for t in current_weights if t in entry_dates
                    }
                else:
                    current_weights = {}
                    current_entry_dates = {}

                # ---- Turnover ----
                all_tkrs = set(current_weights) | set(prev_weights)
                turnover = float(
                    sum(abs(current_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tkrs)
                )

                weights_history[date] = agg_weights_df

                # ---- Trade log for entered positions ----
                next_reb_pos = rebal_dates.searchsorted(date, side="right")
                planned_exit = (
                    pd.Timestamp(rebal_dates[next_reb_pos])
                    if next_reb_pos < len(rebal_dates)
                    else pd.NaT
                )
                for tkr, w in current_weights.items():
                    entry_price = entry_prices.get(tkr, np.nan)
                    actual_entry = entry_dates.get(tkr, pd.NaT)
                    exit_price = np.nan
                    actual_exit = pd.NaT
                    skip_reason = None
                    if pd.notna(planned_exit):
                        actual_exit, exit_price = _next_valid_quote(
                            close_matrix, tkr, planned_exit,
                            cal_list, cal_pos, max_forward_days=3,
                        )
                        if pd.isna(actual_exit):
                            skip_reason = "right_censored_no_exit_quote"
                    else:
                        skip_reason = "right_censored_no_exit_quote"

                    trade_records.append({
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

                prev_weights = current_weights.copy()

            active_weights = {
                t: w
                for t, w in current_weights.items()
                if pd.Timestamp(current_entry_dates.get(t, date)) <= date
            }

            if active_weights:
                weights = pd.Series(active_weights, dtype="float64")
                tickers_list = list(weights.index)
                n_positions_today = len(tickers_list)

                if date in close_matrix.index and next_date in close_matrix.index:
                    p_today_s = close_matrix.loc[date]
                    p_next_s = close_matrix.loc[next_date]
                    p_t = p_today_s.reindex(tickers_list)
                    p_n = p_next_s.reindex(tickers_list)

                    has_today = p_t.notna()
                    has_next = p_n.notna()
                    has_both = has_today & has_next

                    ret = pd.Series(0.0, index=tickers_list)
                    ret[has_both] = p_n[has_both] / p_t[has_both] - 1.0

                    pnl = float((weights[has_both] * ret[has_both]).sum())

                    no_next = ~has_next & has_today

                    ffill_1d = pd.Series(False, index=tickers_list)
                    ffill_2d = pd.Series(False, index=tickers_list)
                    long_gap = pd.Series(False, index=tickers_list)

                    if no_next.any():
                        i_next_cal = cal_pos.get(next_date)
                        if i_next_cal is not None and i_next_cal + 1 < len(cal_list):
                            d2 = cal_list[i_next_cal + 1]
                            if d2 in close_matrix.index:
                                p_d2 = close_matrix.loc[d2].reindex(tickers_list)
                                ffill_1d = no_next & p_d2.notna()

                                still_no_next = no_next & ~ffill_1d
                                if still_no_next.any() and i_next_cal + 2 < len(cal_list):
                                    d3 = cal_list[i_next_cal + 2]
                                    if d3 in close_matrix.index:
                                        p_d3 = close_matrix.loc[d3].reindex(tickers_list)
                                        ffill_2d = still_no_next & p_d3.notna()
                                        long_gap = still_no_next & ~ffill_2d
                                    else:
                                        long_gap = still_no_next
                                else:
                                    long_gap = still_no_next
                            else:
                                long_gap = no_next
                        else:
                            long_gap = no_next

                        if long_gap.any():
                            long_gap_records[date] = {
                                tkr: float(weights[tkr]) for tkr in tickers_list[long_gap]
                            }

                    ffill_1d_weight = float(weights[ffill_1d].abs().sum()) if ffill_1d.any() else 0.0
                    ffill_2d_weight = float(weights[ffill_2d].abs().sum()) if ffill_2d.any() else 0.0
                    n_ffill_1d = int(ffill_1d.sum())
                    n_ffill_2d = int(ffill_2d.sum())
                else:
                    pnl = 0.0
                    ffill_1d_weight = 0.0
                    ffill_2d_weight = 0.0
                    n_ffill_1d = 0
                    n_ffill_2d = 0

                gross = float(weights.abs().sum())
                net = float(weights.sum())
            else:
                pnl = 0.0
                gross = 0.0
                net = 0.0
                n_positions_today = 0
                ffill_1d_weight = 0.0
                ffill_2d_weight = 0.0
                n_ffill_1d = 0
                n_ffill_2d = 0

            daily_records.append({
                "date": date,
                "pnl": pnl,
                "gross_exposure": gross,
                "net_exposure": net,
                "turnover": turnover,
                "_rebalance": int(did_rebalance),
                "n_positions": n_positions_today,
                "ffill_1d_weight": ffill_1d_weight,
                "ffill_2d_weight": ffill_2d_weight,
                "long_gap_recovered_weight": 0.0,
                "possible_delisting_or_unavailable_weight": 0.0,
                "n_ffill_1d_positions": n_ffill_1d,
                "n_ffill_2d_positions": n_ffill_2d,
                "n_long_gap_recovered_positions": 0,
                "n_possible_delisting_or_unavailable_positions": 0,
            })

        daily_df = pd.DataFrame(daily_records)

        # ---- Long-gap recovery audit ----
        recovery_result: dict[tuple[pd.Timestamp, str], bool] = {}
        if not daily_df.empty and long_gap_records:
            log.info(
                "Long-gap recovery audit: %d dates with gap events",
                len(long_gap_records),
            )
            recovery_result = _audit_long_gap_recovery(
                close_matrix, cal_list, cal_pos, long_gap_records, max_recovery_days=30,
            )

            # Map recovery results back to daily_df rows
            for idx in daily_df.index:
                date = daily_df.at[idx, "date"]
                gap_dict = long_gap_records.get(date, {})
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

        # Validate trade log
        trade_log_df = pd.DataFrame(trade_records)
        trade_log_violations_df = pd.DataFrame()
        if not trade_log_df.empty:
            trade_log_violations_df = validate_trade_log(trade_log_df)
            if not trade_log_violations_df.empty:
                # Count violations by type
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
                trade_log=pd.DataFrame(trade_records),
                trade_log_violations=trade_log_violations_df,
                universe_coverage=universe_coverage,
            )

        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        # Build gap_records DataFrame for persistence
        gap_records_frames: list[pd.DataFrame] = []
        if long_gap_records:
            for gap_date, gap_dict in long_gap_records.items():
                for tkr, w in gap_dict.items():
                    recovered = recovery_result.get((gap_date, tkr), False) if recovery_result else False
                    gap_records_frames.append({
                        "gap_date": gap_date,
                        "ticker": tkr,
                        "weight": w,
                        "recovered": recovered,
                    })
        gap_records_df = pd.DataFrame(gap_records_frames) if gap_records_frames else pd.DataFrame()

        return PortfolioResult(
            daily_returns=daily_df,
            weights_history=weights_history,
            cohort_weights_history=cohort_weights_history,
            trade_log=pd.DataFrame(trade_records),
            trade_log_violations=trade_log_violations_df,
            universe_coverage=universe_coverage,
            gap_records=gap_records_df,
            config={
                "cadence": cadence,
                "lookback": lookback,
                "long_frac": long_frac,
                "transaction_cost_bps": transaction_cost_bps,
                "start": str(start.date()),
                "end": str(end.date()),
                "universe": self.universe_name,
            },
        )


# ---------------------------------------------------------------------------
# Capacity proxy
# ---------------------------------------------------------------------------


def compute_capacity_metrics(
    weights_history: dict[pd.Timestamp, pd.DataFrame],
    price_dir: Path = PRICE_CACHE_DIR,
    aum_grid: tuple[float, ...] = (10_000_000, 50_000_000, 100_000_000),
    addv_window: int = 20,
) -> dict[str, Any]:
    """Compute capacity metrics from a weights history.

    Parameters
    ----------
    weights_history:
        Map from rebalance_date to ticker weights DataFrame.
    price_dir:
        Path to ticker price parquet files.
    aum_grid:
        Asset levels for %ADV consumption.
    addv_window:
        Trading days for average daily dollar volume computation.

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

    signals = pd.read_parquet(args.signals)
    log.info("Loaded %d signal rows from %s", len(signals), args.signals)

    # ---- Convert OOS prediction format [df_index, y_pred] → [date, ticker, score] ----
    if "date" not in signals.columns and "df_index" in signals.columns:
        if args.features is None:
            log.error(
                "Signals file has 'df_index' column but no 'date'/'ticker' — "
                "this is Phase 4 OOS prediction format. Pass --features "
                "<features.parquet> to join with feature metadata."
            )
            raise SystemExit(1)
        log.info("Joining OOS predictions with features from %s", args.features)
        features = pd.read_parquet(args.features)
        meta_cols = ["BESTTICKER", args.date_col]
        if "SignalType" in features.columns:
            meta_cols.append("SignalType")
        missing_meta = [c for c in meta_cols if c not in features.columns]
        if missing_meta:
            raise KeyError(f"feature metadata missing columns: {missing_meta}")
        signals = signals.merge(
            features[meta_cols],
            left_on="df_index",
            right_index=True,
            how="left",
        )
        if args.signal_type.lower() != "all" and "SignalType" in signals.columns:
            before = len(signals)
            signals = signals[signals["SignalType"] == args.signal_type].copy()
            log.info(
                "SignalType filter %s: %d / %d rows kept",
                args.signal_type,
                len(signals),
                before,
            )
        signals = signals.rename(
            columns={
                args.date_col: "date",
                "BESTTICKER": "ticker",
                "y_pred": "score",
            }
        )
        signals = signals[signals["date"].notna() & signals["ticker"].notna()]
        log.info("After join: %d rows with date+ticker", len(signals))
    else:
        if args.signal_type.lower() != "all" and "SignalType" in signals.columns:
            before = len(signals)
            signals = signals[signals["SignalType"] == args.signal_type].copy()
            log.info(
                "SignalType filter %s: %d / %d rows kept",
                args.signal_type,
                len(signals),
                before,
            )
        rename_cols: dict[str, str] = {}
        if "ticker" not in signals.columns and "BESTTICKER" in signals.columns:
            rename_cols["BESTTICKER"] = "ticker"
        if "score" not in signals.columns and "y_pred" in signals.columns:
            rename_cols["y_pred"] = "score"
        if "date" not in signals.columns and args.date_col in signals.columns:
            rename_cols[args.date_col] = "date"
        if rename_cols:
            signals = signals.rename(columns=rename_cols)

    required_cols = {"date", "ticker", "score"}
    missing_cols = sorted(required_cols - set(signals.columns))
    if missing_cols:
        raise KeyError(f"signals missing required columns after normalization: {missing_cols}")

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
    )

    summary = result.summary(cost_bps=args.cost_bps)
    cap = compute_capacity_metrics(result.weights_history, args.price_cache)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    if tag is None and "model" in signals.columns:
        models = signals["model"].dropna().astype(str).unique()
        if len(models) == 1:
            tag = models[0]
    suffix_parts = [args.universe]
    if tag:
        suffix_parts.append(_slug(tag))
    suffix_parts.extend([args.cadence, f"{args.lookback}d"])
    suffix = "_".join(suffix_parts)

    result.daily_returns.to_parquet(
        output_dir / f"daily_returns_{suffix}.parquet"
    )

    # Persist weights history as a tall DataFrame
    weight_frames: list[pd.DataFrame] = []
    for date, wdf in result.weights_history.items():
        if not wdf.empty:
            wdf_copy = wdf.copy()
            wdf_copy["rebalance_date"] = date
            weight_frames.append(wdf_copy)
    if weight_frames:
        weights_out = pd.concat(weight_frames, ignore_index=True)
    else:
        weights_out = pd.DataFrame(columns=["ticker", "raw_weight", "rebalance_date"])
    weights_out.to_parquet(output_dir / f"weights_{suffix}.parquet")

    # Persist raw cohort weights (before cross-cohort aggregation)
    cohort_weight_frames: list[pd.DataFrame] = []
    for date, cwdf in result.cohort_weights_history.items():
        if not cwdf.empty:
            cwdf_copy = cwdf.copy()
            cwdf_copy["rebalance_date"] = date
            cohort_weight_frames.append(cwdf_copy)
    if cohort_weight_frames:
        cohort_weights_out = pd.concat(cohort_weight_frames, ignore_index=True)
    else:
        cohort_weights_out = pd.DataFrame(
            columns=["cohort_date", "ticker", "raw_weight", "rebalance_date"]
        )
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
    trade_log.to_parquet(AUDIT_DIR / f"trade_execution_log_{suffix}.parquet")
    trade_log.to_parquet(AUDIT_DIR / "trade_execution_log.parquet")

    # Persist trade log violations
    violations = result.trade_log_violations
    if not violations.empty:
        violations.to_parquet(
            AUDIT_DIR / f"trade_execution_violations_{suffix}.parquet"
        )
        log.warning("Trade log violations: %d rows written", len(violations))
    # Always write a generic file (even if empty, so manifest can reference it)
    violations.to_parquet(AUDIT_DIR / "trade_execution_violations.parquet")

    coverage = result.universe_coverage.copy()
    coverage.to_csv(AUDIT_DIR / f"universe_coverage_by_date_{suffix}.csv", index=False)
    coverage.to_csv(AUDIT_DIR / "universe_coverage_by_date.csv", index=False)

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


if __name__ == "__main__":
    main()
