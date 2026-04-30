"""Phase 2 - Feature Engineering.

Timestamp fields (2.1) -> Enhanced features 85 cols (2.2) -> Stretch features (2.3).

Every operation that depends on historical ordering respects PIT constraints:
- QoQ joins use ``(BESTTICKER, availability_date)`` with strictly less than.
- PIT percentiles are expanding and sector-relative.
- Pre-event momentum anchors on T = date(MOSTIMPORTANTDATEUTC), not entry_date.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.config import PRICE_CACHE_DIR, SIGNALS_PARQUET
from data.progress import progress

log = logging.getLogger("engineer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASPECTS = [
    "CurrentState", "Forecast", "Other",
    "StrategicPosition", "Surprise",
]
THEMES = [
    "CapitalAllocation", "ESG", "FinancialPerformance",
    "MacroeconomicFactors", "MarketAndCompetitivePosition",
    "OperationalPerformance", "Other", "RegulatoryAndLegalIssues",
    "StrategicInitiatives",
]
GICS_SECTORS = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services", "Utilities",
    "Real Estate",
]
MAGNITUDE_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

# Columns the plan explicitly forbids from entering features.
EXCLUDED_FEATURE_COLS = {"QTR_YEAR", "INGESTDATEUTC"}
EXCLUDED_FEATURE_PATTERN = r"^Return_\d+d$"

# EventScore columns used in enhanced features.
EVENT_SCORE_VARIANTS = [
    "EventsScore_1_1_1", "EventsScore_4_2_1",
    "EventsScore_3_1_0", "EventsScore_1_1_0",
]

IDENTIFIER_COLS = [
    "SignalType", "BESTTICKER", "SECTOR", "MOSTIMPORTANTDATEUTC",
    "INGESTDATEUTC", "call_hour_utc", "QTR_YEAR",
]


# ---------------------------------------------------------------------------
# 2.1  Timestamp fields
# ---------------------------------------------------------------------------

def _parse_utc(series: pd.Series) -> pd.Series:
    """Parse a string column as UTC datetime; failed parses become NaT."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def entry_rule(ts: pd.Series) -> pd.Series:
    """Compute entry date from a UTC timestamp series.

    hour < 13 UTC  -> same business-day close entry  (BMO)
    hour >= 13 UTC -> next business-day close entry   (AMC / gray zone)

    Weekends are rolled forward to Monday with ``numpy.busday_offset``. Market
    holidays are still handled later by price-aware execution logic, because
    this timestamp stage intentionally does not depend on a price calendar.
    """
    dt = _parse_utc(ts).dt.tz_localize(None)
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")

    valid = dt.notna()
    if not valid.any():
        return out

    valid_dt = dt.loc[valid]
    days = valid_dt.dt.normalize().to_numpy(dtype="datetime64[D]").copy()
    after_cutoff = valid_dt.dt.hour.to_numpy() >= 13

    # Apply the BMO/AMC day rule first, then roll non-business calendar dates
    # forward. This makes Friday AMC -> Monday, while Saturday BMO/AMC -> Monday.
    days[after_cutoff] = days[after_cutoff] + np.timedelta64(1, "D")
    rolled = np.busday_offset(days, 0, roll="forward")
    out.loc[valid] = pd.to_datetime(rolled)
    return out


def compute_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add *call_entry_date*, *ingest_entry_date*, and *availability_date*.

    Returns a new DataFrame with the three date columns appended.
    """
    df = df.copy()
    df["call_entry_date"] = entry_rule(df["MOSTIMPORTANTDATEUTC"])
    df["ingest_entry_date"] = entry_rule(df["INGESTDATEUTC"])
    df["availability_date"] = (
        df[["call_entry_date", "ingest_entry_date"]]
        .max(axis=1)
        .astype("datetime64[ns]")
    )
    return df


# ---------------------------------------------------------------------------
# Aspect / Theme column parsing
# ---------------------------------------------------------------------------

def parse_aspect_theme_column(col_name: str) -> dict[str, str] | None:
    """Parse ``AspectTheme_{Aspect}_{Theme} - {Magnitude} - {Sentiment}``.

    Returns None for non-AspectTheme columns.
    """
    if not col_name.startswith("AspectTheme_"):
        return None
    rest = col_name[len("AspectTheme_"):]
    parts = rest.split(" - ")
    if len(parts) != 3:
        return None
    at_part, magnitude, sentiment = parts
    # First token is the Aspect; remainder is the Theme.
    tokens = at_part.split("_", 1)
    if len(tokens) != 2:
        return None
    return {
        "aspect": tokens[0],
        "theme": tokens[1],
        "magnitude": magnitude,
        "sentiment": sentiment,
    }


def _group_aspect_theme_cols(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Return ``{aspect: {theme: [col_names]}}`` for every AspectTheme column."""
    groups: dict[str, dict[str, list[str]]] = {
        a: {t: [] for t in THEMES} for a in ASPECTS
    }
    for col in df.columns:
        parsed = parse_aspect_theme_column(col)
        if parsed is None:
            continue
        groups[parsed["aspect"]][parsed["theme"]].append(col)
    return groups


# ---------------------------------------------------------------------------
# 2.2  Row-level enhanced features  (60 columns)
# ---------------------------------------------------------------------------

def _aspect_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-Aspect totals / net sentiment / magnitude-weighted  (5 x 3 = 15 cols)."""
    out = pd.DataFrame(index=df.index)
    at_groups = _group_aspect_theme_cols(df)
    for aspect in ASPECTS:
        cols: list[str] = []
        for theme in THEMES:
            cols.extend(at_groups[aspect][theme])

        if not cols:
            # Aspect not present — fill with NaN/0
            out[f"aspect_{aspect}_total"] = 0.0
            out[f"aspect_{aspect}_net_sentiment"] = 0.0
            out[f"aspect_{aspect}_mag_weighted"] = 0.0
            continue

        sub = df[cols].fillna(0)
        out[f"aspect_{aspect}_total"] = sub.sum(axis=1)

        # net sentiment = sum(positive) - sum(negative)
        pos_cols = [c for c in cols if "Positive" in c]
        neg_cols = [c for c in cols if "Negative" in c]
        out[f"aspect_{aspect}_net_sentiment"] = (
            sub[pos_cols].sum(axis=1) - sub[neg_cols].sum(axis=1)
        )

        # magnitude-weighted: High x3 + Med x2 + Low x1
        high_cols = [c for c in cols if "High" in c]
        med_cols = [c for c in cols if "Medium" in c]
        low_cols = [c for c in cols if "Low" in c]
        out[f"aspect_{aspect}_mag_weighted"] = (
            3 * sub[high_cols].sum(axis=1)
            + 2 * sub[med_cols].sum(axis=1)
            + 1 * sub[low_cols].sum(axis=1)
        )
    return out


def _theme_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-Theme totals / net sentiment / magnitude-weighted  (9 x 3 = 27 cols)."""
    out = pd.DataFrame(index=df.index)
    at_groups = _group_aspect_theme_cols(df)
    for theme in THEMES:
        cols: list[str] = []
        for aspect in ASPECTS:
            cols.extend(at_groups[aspect][theme])

        if not cols:
            out[f"theme_{theme}_total"] = 0.0
            out[f"theme_{theme}_net_sentiment"] = 0.0
            out[f"theme_{theme}_mag_weighted"] = 0.0
            continue

        sub = df[cols].fillna(0)
        out[f"theme_{theme}_total"] = sub.sum(axis=1)

        pos_cols = [c for c in cols if "Positive" in c]
        neg_cols = [c for c in cols if "Negative" in c]
        out[f"theme_{theme}_net_sentiment"] = (
            sub[pos_cols].sum(axis=1) - sub[neg_cols].sum(axis=1)
        )

        high_cols = [c for c in cols if "High" in c]
        med_cols = [c for c in cols if "Medium" in c]
        low_cols = [c for c in cols if "Low" in c]
        out[f"theme_{theme}_mag_weighted"] = (
            3 * sub[high_cols].sum(axis=1)
            + 2 * sub[med_cols].sum(axis=1)
            + 1 * sub[low_cols].sum(axis=1)
        )
    return out


def _sector_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode GICS 11 sectors. Missing/unknown sectors -> all zeros."""
    out = pd.DataFrame(index=df.index)
    for sector in GICS_SECTORS:
        col = f"sector_{sector.replace(' ', '_')}"
        out[col] = (df["SECTOR"] == sector).astype("int8")
    return out


def compute_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """60 row-level columns — zero look-ahead risk.

    ================================ ====
    Subgroup                         Cols
    ================================ ====
    Headline                          1
    EventScore variants               4
    Per-Aspect totals / net / mw      15
    Per-Theme totals / net / mw       27
    Call-length controls              2
    Sector one-hot                    11
    ================================ ====
    """
    out = pd.DataFrame(index=df.index)

    # Headline  (1)
    out["ATCClassifierScore"] = df["ATCClassifierScore"].astype("float64")

    # EventScore variants  (4)
    for ev_col in EVENT_SCORE_VARIANTS:
        if ev_col in df.columns:
            out[ev_col] = df[ev_col].astype("float64")
        else:
            out[ev_col] = np.nan

    # Per-Aspect features  (15)
    af = _aspect_features(df)
    out[af.columns] = af

    # Per-Theme features  (27)
    tf = _theme_features(df)
    out[tf.columns] = tf

    # Call-length controls  (2)
    for col in ["DOCSENTENCECOUNT", "Sentences"]:
        out[col] = df[col].astype("float64") if col in df.columns else np.nan

    # Sector one-hot  (11)
    so = _sector_onehot(df)
    out[so.columns] = so

    return out


# ---------------------------------------------------------------------------
# 2.2  Time-series features  (16 columns)
# ---------------------------------------------------------------------------

def compute_qoq_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """QoQ deltas: within each ticker, diff vs most-recent prior event.

    Uses ``(BESTTICKER, SignalType, availability_date)`` with a strict prior
    availability date. Rows that become available on the same date never chain
    into each other.
    """
    out = pd.DataFrame(index=df.index)
    qoq_cols = (
        ["ATCClassifierScore"]
        + [f"aspect_{a}_total" for a in ASPECTS]
        + [f"theme_{t}_total" for t in THEMES]
    )
    # Only compute for columns that exist in df (row features must be joined first)
    available = [c for c in qoq_cols if c in df.columns]
    if not available:
        return out

    keys = ["BESTTICKER"]
    if "SignalType" in df.columns:
        keys.append("SignalType")
    block_cols = keys + ["availability_date"]

    work_cols = block_cols + available
    if "call_entry_date" in df.columns:
        work_cols.append("call_entry_date")

    work = df[work_cols].copy()
    work["_orig_idx"] = df.index
    sort_cols = keys + ["availability_date"]
    if "call_entry_date" in work.columns:
        sort_cols.append("call_entry_date")
    work = work.sort_values(sort_cols)

    # Collapse simultaneous rows to one date-level value per ticker/slice, then
    # shift across distinct availability dates. Joining the shifted block back
    # gives every same-day row the same strictly-prior baseline.
    last_by_date = work.groupby(block_cols, sort=False)[available].last()
    prev_by_date = (
        last_by_date
        .groupby(level=keys, sort=False)
        .shift(1)
        .rename(columns=lambda c: f"_prev_{c}")
        .reset_index()
    )
    work = work.merge(prev_by_date, on=block_cols, how="left")

    for col in available:
        out.loc[work["_orig_idx"], f"qoq_delta_{col}"] = (
            work[col].to_numpy(dtype="float64")
            - work[f"_prev_{col}"].to_numpy(dtype="float64")
        )
    return out


def compute_4q_trend(df: pd.DataFrame) -> pd.Series:
    """4Q rolling trend slope of ATC — within each ticker, sorted by
    availability_date.

    The OLS slope for a 4-point window x=[0,1,2,3] has the closed form
    ``(-3*y_{t-3} - y_{t-2} + y_{t-1} + 3*y_t) / 10``, which avoids a
    per-row Python loop and ``np.polyfit`` call.
    """
    keys = ["BESTTICKER"]
    if "SignalType" in df.columns:
        keys.append("SignalType")
    block_cols = keys + ["availability_date"]

    work_cols = block_cols + ["ATCClassifierScore"]
    if "call_entry_date" in df.columns:
        work_cols.append("call_entry_date")

    work = df[work_cols].copy()
    work["_orig_idx"] = df.index
    sort_cols = keys + ["availability_date"]
    if "call_entry_date" in work.columns:
        sort_cols.append("call_entry_date")
    work = work.sort_values(sort_cols)

    atc_by_date = work.groupby(block_cols, sort=False)["ATCClassifierScore"].last()
    lagged = pd.concat(
        {
            "_lag1": atc_by_date.groupby(level=keys, sort=False).shift(1),
            "_lag2": atc_by_date.groupby(level=keys, sort=False).shift(2),
            "_lag3": atc_by_date.groupby(level=keys, sort=False).shift(3),
        },
        axis=1,
    ).reset_index()
    work = work.merge(lagged, on=block_cols, how="left")

    slope_values = (
        -3 * work["_lag3"].to_numpy(dtype="float64")
        - work["_lag2"].to_numpy(dtype="float64")
        + work["_lag1"].to_numpy(dtype="float64")
        + 3 * work["ATCClassifierScore"].to_numpy(dtype="float64")
    ) / 10.0
    slope = pd.Series(np.nan, index=df.index, dtype="float64")
    slope.loc[work["_orig_idx"]] = slope_values
    return slope.rename("qoq_4q_trend_atc")


def compute_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """16 time-series columns: QoQ deltas (15) + 4Q rolling trend (1).

    Assumes *df* already has row-feature columns joined and is sorted by
    ``availability_date`` ascending.
    """
    out = compute_qoq_deltas(df)
    out["qoq_4q_trend_atc"] = compute_4q_trend(df)
    return out


# ---------------------------------------------------------------------------
# 2.2  Cross-sectional PIT percentiles  (6 columns)
# ---------------------------------------------------------------------------

def compute_pit_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-relative expanding percentiles of ATC + per-Aspect totals.

    Implementation: compare each row against historical events in the same
    sector and SignalType with ``availability_date < call_entry_date``. This is
    intentionally stricter than self-inclusive expanding ranks: rows that
    become available on the same date never rank each other, and the current
    row never ranks itself.

    Returns 6 columns (1 ATC + 5 Aspects).
    """
    out = pd.DataFrame(index=df.index)
    percentile_cols = ["ATCClassifierScore"] + [f"aspect_{a}_total" for a in ASPECTS]
    available = [c for c in percentile_cols if c in df.columns]
    if not available:
        return out

    group_keys = ["SECTOR"]
    if "SignalType" in df.columns:
        group_keys.append("SignalType")

    work = df[group_keys + ["availability_date", "call_entry_date"] + available].copy()
    work["_orig_idx"] = df.index

    for col in available:
        result = pd.Series(np.nan, index=df.index, dtype="float64")
        for _, grp in work.groupby(group_keys, dropna=False, sort=False):
            pct = _strict_historical_percentile(
                values=grp[col],
                history_dates=grp["availability_date"],
                cutoff_dates=grp["call_entry_date"],
            )
            result.loc[grp["_orig_idx"]] = pct
        out[f"{col}_sector_pct"] = result

    return out


def _strict_historical_percentile(
    values: pd.Series,
    history_dates: pd.Series,
    cutoff_dates: pd.Series,
) -> np.ndarray:
    """Empirical percentile using only rows with ``history_date < cutoff``.

    A Fenwick tree over discretized values keeps the implementation exact while
    avoiding an O(n^2) scan inside each sector/signal group.
    """
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    hist = pd.to_datetime(history_dates).to_numpy(dtype="datetime64[ns]")
    cutoff = pd.to_datetime(cutoff_dates).to_numpy(dtype="datetime64[ns]")
    pct = np.full(len(vals), np.nan, dtype="float64")

    valid_hist = (~np.isnan(vals)) & (~np.isnat(hist))
    valid_query = (~np.isnan(vals)) & (~np.isnat(cutoff))
    if not valid_hist.any() or not valid_query.any():
        return pct

    unique_vals = np.sort(np.unique(vals[valid_hist]))
    if len(unique_vals) == 0:
        return pct

    hist_idx = np.flatnonzero(valid_hist)
    hist_order = hist_idx[np.argsort(hist[hist_idx], kind="mergesort")]
    hist_dates = hist[hist_order]
    hist_codes = np.searchsorted(unique_vals, vals[hist_order], side="left") + 1

    query_idx = np.flatnonzero(valid_query)
    query_order = query_idx[np.argsort(cutoff[query_idx], kind="mergesort")]

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
        q_cutoff = cutoff[qi]
        while add_pos < len(hist_order) and hist_dates[add_pos] < q_cutoff:
            add(int(hist_codes[add_pos]))
            total += 1
            add_pos += 1
        if total == 0:
            continue
        q_code = int(np.searchsorted(unique_vals, vals[qi], side="right"))
        pct[qi] = prefix_sum(q_code) / total

    return pct


# ---------------------------------------------------------------------------
# 2.2  Pre-event price momentum  (3 columns)
# ---------------------------------------------------------------------------

def _load_price_series(
    ticker: str,
    price_cache_dir: Path,
) -> pd.DataFrame | None:
    """Load a ticker's price history, returning a DataFrame with a
    DatetimeIndex and columns ``adj_close, volume``."""
    path = price_cache_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Keep only trading days with valid close
    df = df[df["adj_close"].notna() & (df["adj_close"] > 0)]
    return df


def _trading_day_before(
    target: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the last trading day strictly before *target*."""
    mask = calendar < target
    if not mask.any():
        return None
    return calendar[mask][-1]


def _daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Simple daily returns: ``adj_close.pct_change()``."""
    return prices["adj_close"].pct_change().dropna()


def compute_momentum_features(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Pre-event price momentum (3 columns) — fully vectorized.

    * T = date part of ``MOSTIMPORTANTDATEUTC`` (not entry_date).
    * All price bars are strictly ``<= T-1``.

    Strategy:
      1. Pre-compute n-day returns and rolling-beta for every (ticker, date).
      2. Compute anchor_date for every event (trading day before T).
      3. ``merge_asof`` to join each event with its pre-computed metrics.
      4. Compute residual features from the merged frame.
    """
    out = pd.DataFrame(index=df.index, dtype="float64")
    out["pre_event_ret_21d"] = np.nan
    out["pre_event_ret_21d_sector_rel"] = np.nan
    out["pre_event_idio_resid_5d"] = np.nan

    if not price_cache_dir.exists():
        log.warning("price cache dir %s missing — skipping momentum features", price_cache_dir)
        return out

    has_sector = "SECTOR" in df.columns

    # -- 1. Load all prices and compute daily returns --
    unique_tickers = df[ticker_col].dropna().unique()
    price_cache: dict[str, pd.DataFrame] = {}
    daily_ret_cache: dict[str, pd.Series] = {}
    loaded = 0
    log.info("loading prices for %d unique tickers ...", len(unique_tickers))
    for tkr in progress(unique_tickers, desc="loading prices", unit="tkr"):
        px = _load_price_series(str(tkr), price_cache_dir)
        if px is not None and not px.empty:
            price_cache[str(tkr)] = px
            daily_ret_cache[str(tkr)] = _daily_returns(px)
            loaded += 1
    log.info("loaded prices for %d / %d tickers", loaded, len(unique_tickers))
    if not price_cache:
        return out

    # Build global trading calendar
    all_dates = pd.DatetimeIndex(
        np.unique(np.concatenate([p.index.values for p in price_cache.values()]))
    ).sort_values()

    # -- 2. Map ticker -> sector (vectorized: O(n_rows), not O(n_tickers × n_rows)) --
    ticker_sector_map: dict[str, str] = {}
    if has_sector:
        ticker_sector_map = (
            df.groupby(ticker_col)["SECTOR"]
            .first()
            .astype(str)
            .to_dict()
        )

    # -- 3. Sector median daily returns --
    sector_daily_rets: dict[str, pd.Series] = {}
    # Also pre-compute cumprod + shifted ratios for fast n-day return lookup
    sector_metrics: dict[str, pd.DataFrame] = {}  # date index, cols: ret_5d, ret_21d
    all_sectors = sorted(set(ticker_sector_map.values()))
    for sector in progress(all_sectors, desc="sector returns", unit="sec"):
        sector_tkrs = [t for t, s in ticker_sector_map.items()
                       if s == sector and t in daily_ret_cache]
        if not sector_tkrs:
            continue
        sector_ret = pd.DataFrame({t: daily_ret_cache[t] for t in sector_tkrs}).median(axis=1)
        sector_ret = sector_ret.dropna().sort_index()
        sector_daily_rets[sector] = sector_ret
        # Pre-compute n-day returns using log-returns (avoids cumprod overflow)
        log_rets = np.log(1.0 + sector_ret)
        cum_log = log_rets.cumsum()
        sec_metrics = pd.DataFrame(index=sector_ret.index)
        sec_metrics["sector_ret_5d"] = np.exp(cum_log - cum_log.shift(5)) - 1.0
        sec_metrics["sector_ret_21d"] = np.exp(cum_log - cum_log.shift(21)) - 1.0
        sector_metrics[sector] = sec_metrics

    # -- 4. Build per-ticker metrics, merge_asof with events --
    # Parse anchor dates: trading day strictly before T, via searchsorted
    # (vectorized O(n log m), not per-row O(n×m))
    t_dt = _parse_utc(df["MOSTIMPORTANTDATEUTC"]).dt.tz_localize(None)
    events = pd.DataFrame({
        "_orig_idx": df.index,
        ticker_col: df[ticker_col].astype(str).values,
        "_t": t_dt.values,
    })
    anchor_idx = all_dates.searchsorted(events["_t"].values) - 1
    valid_anchor = (anchor_idx >= 0) & events["_t"].notna().values
    events["anchor_date"] = pd.NaT
    events.loc[valid_anchor, "anchor_date"] = all_dates[anchor_idx[valid_anchor]]
    events = events[events["anchor_date"].notna()].copy()

    # Build ticker metrics in batches to control memory
    batch_size = 500
    ticker_list = sorted(price_cache.keys())
    all_metrics_parts: list[pd.DataFrame] = []
    total_batches = (len(ticker_list) + batch_size - 1) // batch_size

    for batch_idx in progress(range(0, len(ticker_list), batch_size),
                              desc="pre-computing metrics", total=total_batches,
                              unit="batch"):
        batch_tkrs = ticker_list[batch_idx:batch_idx + batch_size]
        batch_parts = []
        for tkr in batch_tkrs:
            px = price_cache[tkr]
            stock_rets = daily_ret_cache[tkr]
            sector = ticker_sector_map.get(tkr, "")

            tkr_metrics = pd.DataFrame(index=stock_rets.index)
            # Stock n-day returns from price ratios (fast, vectorized)
            tkr_metrics["stock_ret_5d"] = (
                px["adj_close"] / px["adj_close"].shift(5) - 1.0
            )
            tkr_metrics["stock_ret_21d"] = (
                px["adj_close"] / px["adj_close"].shift(21) - 1.0
            )
            tkr_metrics[ticker_col] = tkr
            tkr_metrics["date"] = tkr_metrics.index

            if sector and sector in sector_metrics:
                sec_m = sector_metrics[sector]
                # Align: join sector metrics onto the stock's date index
                tkr_metrics = tkr_metrics.join(
                    sec_m[["sector_ret_5d", "sector_ret_21d"]], how="left"
                )
                # Rolling beta: 60-trading-day OLS of stock on sector
                sec_rets = sector_daily_rets[sector]
                common_idx = stock_rets.index.intersection(sec_rets.index)
                if len(common_idx) >= 30:
                    s_aligned = stock_rets.reindex(common_idx)
                    m_aligned = sec_rets.reindex(common_idx)
                    rolling_cov = s_aligned.rolling(60, min_periods=30).cov(m_aligned)
                    rolling_var = m_aligned.rolling(60, min_periods=30).var()
                    beta_series = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)
                    # For an event anchored on T-1, beta must be estimated on
                    # data ending at T-5. Shifting by 4 trading rows keeps the
                    # T-4..T-1 interval out of the beta estimate.
                    beta_df = beta_series.reindex(stock_rets.index).shift(4)
                    tkr_metrics["beta"] = beta_df.values
                else:
                    tkr_metrics["beta"] = np.nan
                    tkr_metrics["sector_ret_5d"] = np.nan
                    tkr_metrics["sector_ret_21d"] = np.nan

            # Keep only rows with at least stock returns
            tkr_metrics = tkr_metrics.dropna(subset=["stock_ret_21d", "stock_ret_5d"])
            if len(tkr_metrics) > 0:
                batch_parts.append(tkr_metrics)

        if batch_parts:
            all_metrics_parts.append(pd.concat(batch_parts, ignore_index=True))

    if not all_metrics_parts:
        return out

    all_metrics = pd.concat(all_metrics_parts, ignore_index=True)

    # -- 5. merge_asof per ticker (pandas 3.x merge_asof by= is broken) --
    # Pre-group by ticker to avoid O(n_tickers × n_rows) boolean masks.
    merged_parts: list[pd.DataFrame] = []
    events_by_tkr = dict(list(events.groupby(ticker_col)))
    metrics_by_tkr = dict(list(all_metrics.groupby(ticker_col)))
    common_tickers = sorted(set(events_by_tkr) & set(metrics_by_tkr))
    for tkr in progress(common_tickers, desc="merging metrics", unit="tkr",
                         mininterval=0.2):
        ev = events_by_tkr[tkr].sort_values("anchor_date")
        mt = metrics_by_tkr[tkr].sort_values("date")
        # Ensure matching datetime resolution (parquet round-trips can flip ns<->s)
        ev["anchor_date"] = ev["anchor_date"].astype("datetime64[ns]")
        mt["date"] = mt["date"].astype("datetime64[ns]")
        mg = pd.merge_asof(
            ev, mt,
            left_on="anchor_date", right_on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=10),
        )
        merged_parts.append(mg)

    if not merged_parts:
        return out
    merged = pd.concat(merged_parts, ignore_index=True)

    # -- 6. Compute final features --
    valid_mask = merged["stock_ret_21d"].notna()
    out.loc[merged.loc[valid_mask, "_orig_idx"], "pre_event_ret_21d"] = (
        merged.loc[valid_mask, "stock_ret_21d"].values
    )

    sec_valid = valid_mask & merged["sector_ret_21d"].notna()
    idx_sec = merged.loc[sec_valid, "_orig_idx"]
    out.loc[idx_sec, "pre_event_ret_21d_sector_rel"] = (
        merged.loc[sec_valid, "stock_ret_21d"].values
        - merged.loc[sec_valid, "sector_ret_21d"].values
    )

    resid_valid = sec_valid & merged["beta"].notna() & merged["sector_ret_5d"].notna()
    idx_resid = merged.loc[resid_valid, "_orig_idx"]
    out.loc[idx_resid, "pre_event_idio_resid_5d"] = (
        merged.loc[resid_valid, "stock_ret_5d"].values
        - merged.loc[resid_valid, "beta"].values
        * merged.loc[resid_valid, "sector_ret_5d"].values
    )

    return out


# ---------------------------------------------------------------------------
# 2.3  Stretch features
# ---------------------------------------------------------------------------

def stretch_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of AspectTheme column names (non-Fluff/Filler).

    These are the candidate pool for LassoCV selection inside each
    walk-forward fold (Phase 4). No full-sample pre-screening.
    """
    cols: list[str] = []
    for col in df.columns:
        parsed = parse_aspect_theme_column(col)
        if parsed is None:
            continue
        if parsed["aspect"] not in ASPECTS:
            continue
        cols.append(col)
    return cols


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    include_momentum: bool = True,
) -> pd.DataFrame:
    """Compute all features for a signal DataFrame.

    Parameters
    ----------
    df:
        Signal DataFrame (must contain the columns loaded by Phase 1, i.e.
        identifiers + EventScores + ATCClassifierScore + non-Fluff/Filler
        AspectTheme_* columns + call_hour_utc).
    price_cache_dir:
        Path to ``data/cache/prices/`` for pre-event momentum features.
    tier:
        ``"enhanced"`` -> 85 engineered columns.
        ``"stretch"``  -> enhanced + all AspectTheme_* columns.
    include_momentum:
        Set False to skip momentum (useful when price data is unavailable).

    Returns
    -------
    DataFrame with feature columns. Identifier columns (BESTTICKER, SECTOR,
    availability_date, call_entry_date, MOSTIMPORTANTDATEUTC) are preserved
    for downstream use.
    """
    t_total = time.time()

    # --- 2.1 Timestamps ---
    t0 = time.time()
    df = compute_timestamps(df)
    log.info("2.1 timestamps: %.1fs", time.time() - t0)

    # Track original row position for 1:1 joins after sorting.
    df["_row_id"] = np.arange(len(df), dtype="int64")

    # --- 2.2 Row-level features (60 cols) ---
    t0 = time.time()
    row_feat = compute_row_features(df)
    log.info("2.2 row features (%d cols): %.1fs", len(row_feat.columns), time.time() - t0)

    # Combine identifiers + timestamps + row features into one frame for
    # time-series / PIT steps which need SECTOR / BESTTICKER / availability_date
    id_cols = [
        "BESTTICKER", "SECTOR", "availability_date",
        "call_entry_date", "MOSTIMPORTANTDATEUTC", "SignalType",
    ]
    available_ids = [c for c in id_cols if c in df.columns]
    base = pd.concat([
        df[available_ids + ["_row_id"]].reset_index(drop=True),
        row_feat.reset_index(drop=True),
    ], axis=1)

    # Sort by availability_date for PIT-safe time-series operations
    base = base.sort_values("availability_date").reset_index(drop=True)

    # --- 2.2 Time-series features (16 cols) ---
    t0 = time.time()
    ts_feat = compute_timeseries_features(base)
    log.info("2.2 time-series features (%d cols): %.1fs", len(ts_feat.columns), time.time() - t0)
    base = pd.concat([base, ts_feat], axis=1)

    # --- 2.2 PIT percentiles (6 cols) ---
    t0 = time.time()
    pit_feat = compute_pit_percentiles(base)
    log.info("2.2 PIT percentiles (%d cols): %.1fs", len(pit_feat.columns), time.time() - t0)
    base = pd.concat([base, pit_feat], axis=1)

    # --- 2.2 Pre-event momentum (3 cols) ---
    if include_momentum:
        t0 = time.time()
        mom_feat = compute_momentum_features(
            base, price_cache_dir=price_cache_dir
        )
        log.info("2.2 momentum features (%d cols): %.1fs", len(mom_feat.columns), time.time() - t0)
        base = pd.concat([base, mom_feat], axis=1)

    # --- 2.3 Stretch columns ---
    if tier == "stretch":
        t0 = time.time()
        stretch_cols = stretch_feature_cols(df)
        stretch_df = df[["_row_id"] + stretch_cols].copy()
        base = base.merge(stretch_df, on="_row_id", how="left")
        log.info("2.3 stretch columns (%d cols): %.1fs", len(stretch_cols), time.time() - t0)

    # --- Clean up ---
    base = base.sort_values("_row_id").reset_index(drop=True)
    base = base.drop(columns=["_row_id"], errors="ignore")

    # --- Drop excluded columns ---
    cols_to_drop = [c for c in EXCLUDED_FEATURE_COLS if c in base.columns]
    if cols_to_drop:
        base = base.drop(columns=cols_to_drop)

    log.info("total build_features: %.1fs  (%d rows x %d cols)",
             time.time() - t_total, len(base), len(base.columns))
    return base


def load_signals_df(path: Path = SIGNALS_PARQUET) -> pd.DataFrame:
    """Load the signal parquet cache into a DataFrame."""
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from data.config import set_global_seed

    parser = argparse.ArgumentParser(description="Phase 2 — Feature Engineering")
    parser.add_argument("--signals", type=Path, default=SIGNALS_PARQUET)
    parser.add_argument("--prices", type=Path, default=PRICE_CACHE_DIR)
    parser.add_argument("--tier", choices=["enhanced", "stretch"], default="enhanced")
    parser.add_argument("--no-momentum", action="store_true",
                        help="Skip pre-event momentum features")
    parser.add_argument("--output", type=Path, default=None,
                        help="Parquet output path (optional)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Limit to N rows for quick testing")
    args = parser.parse_args()

    set_global_seed()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("loading signals from %s", args.signals)
    df = load_signals_df(args.signals)
    if args.sample > 0:
        df = df.head(args.sample)
    log.info("signals: %d rows x %d cols", len(df), len(df.columns))

    features = build_features(
        df,
        price_cache_dir=args.prices,
        tier=args.tier,
        include_momentum=not args.no_momentum,
    )
    log.info("features: %d rows x %d cols", len(features), len(features.columns))

    if args.output:
        features.to_parquet(args.output, compression="zstd")
        log.info("wrote %s", args.output)
    else:
        print(features.head())
        print(f"\nColumns ({len(features.columns)}):")
        for c in features.columns:
            print(f"  {c}")


if __name__ == "__main__":
    main()
