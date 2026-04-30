"""Point-in-time universe filtering helpers for backtests."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.config import UNIVERSE_CACHE_DIR

log = logging.getLogger("backtest.universe")


def _default_tolerance_days(universe: pd.DataFrame) -> int:
    """Infer a stale-snapshot tolerance from PIT snapshot frequency."""
    dates = pd.DatetimeIndex(pd.to_datetime(universe["date"], errors="coerce").dropna().unique())
    if len(dates) < 2:
        return 65
    gaps = pd.Series(dates.sort_values()).diff().dt.days.dropna()
    if gaps.empty:
        return 65
    # Daily SP500 snapshots should only need holiday/weekend slack. Monthly ETF
    # snapshots need wider slack so mid-month events use the latest month-end.
    return 7 if float(gaps.median()) <= 7 else 65


def filter_to_universe(
    df: pd.DataFrame,
    universe_name: str,
    *,
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
    universe_dir: Path = UNIVERSE_CACHE_DIR,
    tolerance_days: int | None = None,
    output_col: str = "_in_universe",
) -> pd.DataFrame:
    """Add a PIT membership boolean column.

    For every event, this uses the latest global universe snapshot on or before
    ``date_col`` and then checks whether the event ticker is in that snapshot.
    This avoids the common bug where a per-ticker backward asof join keeps a
    removed member alive until the asof tolerance expires.
    """
    out = df.copy()
    out[output_col] = False

    if out.empty:
        return out
    if date_col not in out.columns:
        raise KeyError(f"date column missing for universe filter: {date_col}")
    if ticker_col not in out.columns:
        raise KeyError(f"ticker column missing for universe filter: {ticker_col}")

    universe_path = universe_dir / f"{universe_name}_pit.parquet"
    if not universe_path.exists():
        log.warning("Universe file not found: %s; marking all rows out-of-universe", universe_path)
        return out

    universe = pd.read_parquet(universe_path, columns=["date", "ticker"])
    if universe.empty:
        log.warning("Universe %s is empty; marking all rows out-of-universe", universe_name)
        return out

    universe["date"] = pd.to_datetime(universe["date"], errors="coerce").astype("datetime64[ns]")
    universe["ticker"] = universe["ticker"].astype(str)
    universe = universe.dropna(subset=["date", "ticker"]).drop_duplicates()
    if universe.empty:
        return out

    tolerance = _default_tolerance_days(universe) if tolerance_days is None else tolerance_days
    snapshot_dates = pd.DatetimeIndex(universe["date"].drop_duplicates().sort_values())

    event_dates = pd.to_datetime(out[date_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    event_tickers = out[ticker_col]
    pos = snapshot_dates.searchsorted(event_dates, side="right") - 1
    valid = (pos >= 0) & ~pd.isna(event_dates) & event_tickers.notna().to_numpy()

    snapshot_for_event = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    snapshot_for_event[valid] = snapshot_dates.values[pos[valid]]

    if tolerance is not None:
        event_day = event_dates.astype("datetime64[D]")
        snap_day = snapshot_for_event.astype("datetime64[D]")
        stale = valid & ((event_day - snap_day).astype("timedelta64[D]").astype(float) > tolerance)
        valid &= ~stale

    event_keys = pd.DataFrame({
        "_row_pos": np.arange(len(out), dtype=np.int64),
        "_snapshot_date": snapshot_for_event,
        "_ticker": event_tickers.astype(str).to_numpy(),
    })
    event_keys = event_keys[valid].copy()
    if event_keys.empty:
        return out

    universe_keys = universe.rename(columns={"date": "_snapshot_date", "ticker": "_ticker"})
    matched = event_keys.merge(
        universe_keys.assign(_member=True),
        on=["_snapshot_date", "_ticker"],
        how="left",
    )
    member_pos = matched.loc[matched["_member"].fillna(False), "_row_pos"].to_numpy(dtype=np.int64)
    out.iloc[member_pos, out.columns.get_loc(output_col)] = True
    return out
