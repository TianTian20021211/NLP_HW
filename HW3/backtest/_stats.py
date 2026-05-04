"""Leaf utility module for backtest statistics.

This module provides shared statistics functions used across the backtest
pipeline. It is a leaf module and must NOT import from other backtest.* modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr as _spearmanr


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-d arrays."""
    return float(_spearmanr(a, b, nan_policy="omit").statistic)


def make_median_imputer() -> Any:
    """SimpleImputer(strategy='median') with keep_empty_features fallback."""
    from sklearn.impute import SimpleImputer

    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def dedup_latest_per_ticker(
    df: pd.DataFrame,
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Sort by date_col ascending and keep the last row per ticker_col."""
    return (
        df.sort_values(date_col)
        .drop_duplicates(subset=[ticker_col], keep="last")
    )


def bucket_returns(
    df: pd.DataFrame,
    feature_col: str,
    return_col: str,
    n_buckets: int,
    min_samples_per_bucket: int = 5,
    group_col: str = "year_month",
    date_col: str = "call_entry_date",
    ticker_col: str = "BESTTICKER",
) -> pd.DataFrame:
    """Equal-size buckets by feature_col; mean return_col per bucket and group.

    Returns DataFrame with columns [group_col, "bucket", "ret", "n"].
    Returns empty standard-schema DataFrame if no valid rows.
    """
    mask = df[feature_col].notna() & df[return_col].notna()
    sub = df[mask]
    if sub.empty:
        return pd.DataFrame(columns=[group_col, "bucket", "ret", "n"])

    records: list[dict[str, Any]] = []
    for gname, gdf in sub.groupby(group_col, observed=True):
        gdf = dedup_latest_per_ticker(
            gdf, date_col=date_col, ticker_col=ticker_col,
        )
        if len(gdf) < n_buckets * min_samples_per_bucket:
            continue
        gdf = gdf.copy()
        try:
            gdf["_bucket"] = pd.qcut(
                gdf[feature_col], q=n_buckets, labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        buckets = gdf.groupby("_bucket")[return_col]
        for b_idx, b_ret in buckets.mean().items():
            records.append({
                group_col: gname,
                "bucket": int(b_idx),
                "ret": float(b_ret),
                "n": int(buckets.size()[b_idx]),
            })
    return pd.DataFrame(records)


def build_equity_curves(
    bucket_returns_df: pd.DataFrame,
    n_buckets: int,
    group_col: str = "year_month",
) -> pd.DataFrame:
    """Build monthly bucket, cumulative bucket, and long/short equity curves.

    Only observed bucket columns are created (when qcut drops buckets).
    L/S = long_only + short_only.
    """
    if bucket_returns_df.empty:
        return pd.DataFrame()

    piv = bucket_returns_df.pivot_table(
        index=group_col, columns="bucket", values="ret", aggfunc="mean",
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


def max_drawdown_from_equity(cum: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a cumulative equity series.

    Returns NaN for empty input.
    """
    if len(cum) == 0:
        return float("nan")
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def portfolio_stats(
    equity_curves: pd.DataFrame,
    leg: str = "long_short",
    ann_factor: float = 12.0,
    count_key: str = "n_months",
) -> dict[str, float]:
    """Compute annualized return, vol, Sharpe, rolling Sharpe, drawdown, count.

    Returns {} for missing leg, no observations, or one-row equity.
    Returns NaN Sharpe for zero volatility.
    """
    cum_col = f"cum_{leg}"
    if cum_col not in equity_curves.columns or len(equity_curves) < 2:
        return {}
    rets = equity_curves[leg].dropna()
    if len(rets) == 0:
        return {}

    std = float(rets.std())
    mean_ret = float(rets.mean())

    def _rolling_sharpe(r: pd.Series, window: int = 12) -> float:
        roll = r.rolling(window).mean() / r.rolling(window).std()
        return float(roll.mean())

    result: dict[str, float] = {
        "ann_return": mean_ret * ann_factor,
        "ann_vol": std * np.sqrt(ann_factor),
        "sharpe": (
            mean_ret / std * np.sqrt(ann_factor) if std > 0
            else float("nan")
        ),
        "rolling_sharpe_12m": _rolling_sharpe(rets, 252),
        "max_drawdown": max_drawdown_from_equity(equity_curves[cum_col]),
        count_key: len(rets),
    }

    return result
