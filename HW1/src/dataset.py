"""Feature+label join, variant canonicalization, and X/y construction.

This module is the bridge between the wide feature_table (which keeps
both FinBERT and LM sentiment families as independent column clusters)
and the model layer (which expects a single canonical ``sentiment_*``
family). Per Part II section 2.3, the on-disk schema and the dataset canonical
schema must remain separate.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .features import event_columns
from .io_paths import error, feature_table_path, labels_path


Extraction = Literal["ext_finbert_llm", "ext_lm_llm", "ext_finbert_only"]
SplitMode = Literal["primary", "secondary"]
Target = Literal["binary", "regression"]

PRIMARY_TRAIN_PER_TICKER = 5
SECONDARY_CUTOFF = "2025-04-01"

_KEY_COLS = ("ticker", "call_date")
_META_COLS = ("ticker", "call_date", "entry_date", "quarter")

_FINBERT_COLS = (
    "sentiment_call",
    "sentiment_qa",
    "sentiment_dispersion",
    "role_sent_CEO",
    "role_sent_CEO_missing",
    "role_sent_CFO",
    "role_sent_CFO_missing",
    "sentiment_delta",
)

_LM_COLS = (
    "sentiment_lm_call",
    "sentiment_lm_qa",
    "sentiment_lm_dispersion",
    "role_sent_lm_CEO",
    "role_sent_lm_CEO_missing",
    "role_sent_lm_CFO",
    "role_sent_lm_CFO_missing",
    "sentiment_lm_delta",
)

_LM_TO_CANONICAL = {
    "sentiment_lm_call": "sentiment_call",
    "sentiment_lm_qa": "sentiment_qa",
    "sentiment_lm_dispersion": "sentiment_dispersion",
    "role_sent_lm_CEO": "role_sent_CEO",
    "role_sent_lm_CEO_missing": "role_sent_CEO_missing",
    "role_sent_lm_CFO": "role_sent_CFO",
    "role_sent_lm_CFO_missing": "role_sent_CFO_missing",
    "sentiment_lm_delta": "sentiment_delta",
}


def load_feature_table() -> pd.DataFrame:
    """Read the feature_table parquet; fail loud if missing."""
    p = feature_table_path()
    if not p.is_file():
        error(f"feature_table parquet missing: {p}")
    return pd.read_parquet(p)


def load_labels() -> pd.DataFrame:
    """Read the labels parquet; fail loud if missing."""
    p = labels_path()
    if not p.is_file():
        error(f"labels parquet missing: {p}")
    return pd.read_parquet(p)


def join_features_labels(
    feature_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inner-join feature rows and labels on ``(ticker, call_date)``.

    Uses an inner join: a call without a usable ``call_date`` never enters
    the labels frame, and we don't want to propagate such rows into the
    modeling layer with fabricated labels.
    """
    fx = feature_df if feature_df is not None else load_feature_table()
    ly = labels_df if labels_df is not None else load_labels()
    merged = fx.merge(ly, on=list(_KEY_COLS), how="inner", validate="one_to_one")
    merged = merged.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    return merged


def split_primary(df: pd.DataFrame, k: int = PRIMARY_TRAIN_PER_TICKER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker first ``k`` calls → train, rest → test.

    Default ``k=5`` matches Part II §2.6. E0 smoke tests override with
    ``k=1`` to guarantee a non-empty test set on the tiny 2-call subset.
    Preserves input ordering inside each split; ``call_date`` is the sort
    key (not ``quarter``, which is fiscal-calendar dependent for NKE / FDX).
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.sort_values(["ticker", "call_date"]).groupby("ticker", sort=False):
        train_parts.append(group.iloc[:k])
        test_parts.append(group.iloc[k:])
    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else df.iloc[:0]
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else df.iloc[:0]
    return train, test


def split_secondary(df: pd.DataFrame, cutoff: str = SECONDARY_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global date cutoff: ``call_date < cutoff`` → train (sanity split)."""
    cutoff_ts = pd.to_datetime(cutoff)
    dates = pd.to_datetime(df["call_date"])
    train = df[dates < cutoff_ts].reset_index(drop=True)
    test = df[dates >= cutoff_ts].reset_index(drop=True)
    return train, test


def canonicalize_variant(df: pd.DataFrame, extraction: Extraction) -> pd.DataFrame:
    """Collapse the two sentiment families down to the variant selected.

    - ``ext_finbert_llm``: keep FinBERT sentiment columns, drop LM family.
    - ``ext_lm_llm``: rename LM columns to the canonical ``sentiment_*``
      names so downstream code (rule / logreg / xgb / ridge) can be
      extraction-agnostic; the original FinBERT columns are dropped.
    - ``ext_finbert_only``: keep FinBERT sentiment columns, drop the LM
      family **and** every event-derived column (see
      :func:`features.event_columns`).

    The input ``df`` is not mutated.
    """
    out = df.copy()
    if extraction == "ext_finbert_llm":
        out = out.drop(columns=[c for c in _LM_COLS if c in out.columns])
    elif extraction == "ext_lm_llm":
        missing = [c for c in _LM_COLS if c not in out.columns]
        if missing:
            error(
                "canonicalize_variant(ext_lm_llm) needs LM columns in feature_table; "
                f"missing: {missing}. Run features.build_feature_table(include_lm=True)."
            )
        out = out.drop(columns=[c for c in _FINBERT_COLS if c in out.columns])
        out = out.rename(columns=_LM_TO_CANONICAL)
    elif extraction == "ext_finbert_only":
        out = out.drop(columns=[c for c in _LM_COLS if c in out.columns])
        out = out.drop(columns=[c for c in event_columns() if c in out.columns])
    else:
        error(f"unknown extraction variant: {extraction}")
    return out


def build_xy(
    df: pd.DataFrame,
    horizon: str,
    target: Target = "binary",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a canonicalized frame into ``(X, y, meta)``.

    ``horizon`` is ``"5d"`` / ``"21d"`` / etc. — the excess-return column
    ``excess_{horizon}`` drives ``y``. Rows with NaN in that column are
    dropped (per-horizon dropping per Part II §2.4). ``meta`` keeps
    bookkeeping columns (ticker, call_date, entry_date, quarter, raw_h,
    excess_h) that the backtest needs but the model must not see.
    """
    excess_col = f"excess_{horizon}"
    raw_col = f"raw_{horizon}"
    if excess_col not in df.columns or raw_col not in df.columns:
        error(f"build_xy: missing horizon columns {excess_col!r}/{raw_col!r}")

    keep = df[df[excess_col].notna()].reset_index(drop=True)

    meta_cols = [c for c in _META_COLS if c in keep.columns]
    meta_cols = meta_cols + [raw_col, excess_col]
    meta = keep[meta_cols].copy()

    label_related = {c for c in keep.columns if c.startswith(("raw_", "excess_")) or c == "entry_date"}
    feature_cols = [c for c in keep.columns if c not in label_related and c not in _META_COLS]
    X = keep[feature_cols].copy()

    if target == "binary":
        y = (keep[excess_col] > 0).astype(int)
    elif target == "regression":
        y = keep[excess_col].astype(float)
    else:
        error(f"unknown target: {target}")

    y.index = X.index
    return X, y, meta
