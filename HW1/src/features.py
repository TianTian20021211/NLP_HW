"""Call-level JSON -> one feature row per ``(ticker, call_date)``.

Consumes records produced by :mod:`aggregate` (optionally both the
``variant="finbert"`` and ``variant="lm"`` lines) and emits a flat wide
table keyed on ``(ticker, call_date)``. See Part II §2.3 for the full
schema and missingness convention.

Key design notes:

- **Fail loud on structure**: anything but the semantic-NaN cases enumerated
  in §2.3 (delta first row, ``ret_21d_prior`` below price-history window,
  overlap with < 2 non-empty role sets) is treated as a structural error
  and raises. ``_skipped`` call records (``_empty_record`` path) raise
  immediately — we never emit a row for them.
- **QoQ deltas**: computed per-ticker on ``call_date`` ascending order, so
  they stay correct even for tickers with fiscal-calendar quirks
  (``quarter`` is not used as the ordering key).
- **Two-family canonicalization**: when ``include_lm=True`` both the
  FinBERT and LM sentiment columns are written to disk; the dataset layer
  is responsible for renaming the active family to the canonical
  ``sentiment_*`` names at model time.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from .aggregate import load_call_record
from .cache_keys import stage_key, write_sidecar
from .io_paths import ensure_dirs, error, feature_table_path
from .prices import load_prices


_GUIDANCE_ORDINAL = {"lowered": -1.0, "none": 0.0, "maintained": 0.5, "raised": 1.0}
_ROLES_TRACKED = ("CEO", "CFO")

_EVENT_COLUMNS = (
    "is_guidance_raised",
    "is_guidance_lowered",
    "is_guidance_maintained",
    "guidance_ordinal",
    "guidance_disagree",
    "llm_agree_guidance",
    "llm_agree_guidance_missing",
    "n_qa_minus_pres_risks",
    "wins_overlap",
    "risks_overlap",
    "wins_overlap_missing",
    "risks_overlap_missing",
    "guidance_delta",
    "risks_persistence",
    "wins_persistence",
    "theme_drift",
)


def event_columns() -> tuple[str, ...]:
    """Return the canonical list of event-derived feature columns.

    Exposed so :mod:`dataset` can drop them wholesale for the
    ``ext_finbert_only`` extraction variant without re-enumerating the
    list in two places.
    """
    return _EVENT_COLUMNS


def build_feature_table(
    calls: Sequence[tuple[str, str]],
    include_lm: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """Materialize the full ``(ticker, call_date)`` feature table.

    ``calls`` is ``[(ticker, quarter), ...]``. FinBERT call records are
    always required; LM call records are additionally required when
    ``include_lm=True``. The parquet is rewritten from scratch (no
    incremental add-column path) so Part II §2.3's E1→E2 schema bump is
    just a re-run with ``include_lm=True``.
    """
    per_ticker: dict[str, list[dict]] = {}
    for ticker, quarter in calls:
        rec_fb = load_call_record(ticker, quarter, variant="finbert")
        _reject_skipped(rec_fb, ticker, quarter)
        rec_lm = None
        if include_lm:
            rec_lm = load_call_record(ticker, quarter, variant="lm")
            _reject_skipped(rec_lm, ticker, quarter)
        row = _base_row(rec_fb, rec_lm, include_lm=include_lm)
        per_ticker.setdefault(ticker, []).append(row)

    rows: list[dict] = []
    for ticker, ticker_rows in per_ticker.items():
        ticker_rows.sort(key=lambda r: r["call_date"])
        prices_df = _load_prices_or_none(ticker)
        prev: dict | None = None
        for row in ticker_rows:
            row.update(_delta_row(row, prev, include_lm=include_lm))
            row["ret_21d_prior"] = _ret_21d_prior(prices_df, row["call_date"])
            rows.append(row)
            prev = row

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    if df.duplicated(["ticker", "call_date"]).any():
        error("feature_table has duplicate (ticker, call_date) keys; aggregate produced dupes")
    df = drop_internal_columns(df)
    if write:
        ensure_dirs()
        path = feature_table_path()
        df.to_parquet(path, index=False)
        write_sidecar(path, "features")
    return df


def _reject_skipped(record: dict, ticker: str, quarter: str) -> None:
    if record.get("_skipped"):
        error(
            f"skipped call record reached features stage: {ticker}_{quarter} "
            f"reason={record.get('_skipped')!r}"
        )


def _base_row(rec_fb: dict, rec_lm: dict | None, include_lm: bool) -> dict:
    """Extract the non-delta, non-momentum half of the feature row.

    Pulls sentiment / dispersion / event fields from the FinBERT record and
    (when ``include_lm=True``) the matching sentiment family from the LM
    record. Structural misses (missing sub-dict, missing scalar) raise.
    """
    consensus = _require_block(rec_fb, "consensus")
    dispersion = _require_block(rec_fb, "dispersion")
    n_units = _require_block(rec_fb, "n_units")

    row: dict = {
        "ticker": rec_fb["ticker"],
        "call_date": rec_fb["call_date"],
        "quarter": rec_fb["quarter"],
        "n_units_total": int(n_units["total"]),
        "n_units_presenter": int(n_units["presenter"]),
        "n_units_qa": int(n_units["qa"]),
        "sentiment_call": _require_scalar_allowing_none(consensus, "sentiment_call"),
        "sentiment_qa": _require_scalar_allowing_none(consensus, "sentiment_qa"),
        "sentiment_dispersion": _require_scalar_allowing_none(dispersion, "sentiment_dispersion"),
    }

    role_means = dispersion.get("role_sent_means") or {}
    for role in _ROLES_TRACKED:
        v = role_means.get(role)
        row[f"role_sent_{role}"] = float(v) if _is_number(v) else math.nan
        row[f"role_sent_{role}_missing"] = 0 if _is_number(v) else 1

    row["wins_overlap"] = _require_scalar_allowing_none(dispersion, "wins_overlap")
    row["risks_overlap"] = _require_scalar_allowing_none(dispersion, "risks_overlap")
    row["wins_overlap_missing"] = 0 if _is_number(row["wins_overlap"]) else 1
    row["risks_overlap_missing"] = 0 if _is_number(row["risks_overlap"]) else 1

    row["n_qa_minus_pres_risks"] = int(dispersion.get("qa_minus_pres_risks_count", 0))

    guidance_call = consensus.get("guidance_call")
    if guidance_call not in _GUIDANCE_ORDINAL:
        error(
            f"unexpected guidance_call={guidance_call!r} in "
            f"{rec_fb['ticker']}_{rec_fb['quarter']}"
        )
    row["is_guidance_raised"] = int(guidance_call == "raised")
    row["is_guidance_lowered"] = int(guidance_call == "lowered")
    row["is_guidance_maintained"] = int(guidance_call == "maintained")
    row["guidance_ordinal"] = _GUIDANCE_ORDINAL[guidance_call]
    row["guidance_disagree"] = int(dispersion.get("guidance_disagree", 0))

    agree, agree_missing = _llm_agree_guidance(rec_fb)
    row["llm_agree_guidance"] = agree
    row["llm_agree_guidance_missing"] = agree_missing

    if include_lm:
        if rec_lm is None:
            error("include_lm=True but LM record not supplied")
        consensus_lm = _require_block(rec_lm, "consensus")
        dispersion_lm = _require_block(rec_lm, "dispersion")
        row["sentiment_lm_call"] = _require_scalar_allowing_none(consensus_lm, "sentiment_call")
        row["sentiment_lm_qa"] = _require_scalar_allowing_none(consensus_lm, "sentiment_qa")
        row["sentiment_lm_dispersion"] = _require_scalar_allowing_none(
            dispersion_lm, "sentiment_dispersion"
        )
        role_means_lm = dispersion_lm.get("role_sent_means") or {}
        for role in _ROLES_TRACKED:
            v = role_means_lm.get(role)
            row[f"role_sent_lm_{role}"] = float(v) if _is_number(v) else math.nan
            row[f"role_sent_lm_{role}_missing"] = 0 if _is_number(v) else 1

    # Top-5 phrase sets are stored for QoQ persistence / drift deltas but
    # not emitted to disk as features; we drop them after the delta pass.
    row["_wins_top5"] = _phrase_set(consensus.get("wins_top5", []))
    row["_risks_top5"] = _phrase_set(consensus.get("risks_top5", []))

    return row


def _delta_row(row: dict, prev: dict | None, include_lm: bool) -> dict:
    """Compute QoQ delta columns using ``prev`` (same ticker, previous call)."""
    out: dict[str, float] = {}
    out["sentiment_delta"] = _maybe_delta(row.get("sentiment_call"), prev and prev.get("sentiment_call"))
    out["guidance_delta"] = _maybe_delta(row.get("guidance_ordinal"), prev and prev.get("guidance_ordinal"))
    if include_lm:
        out["sentiment_lm_delta"] = _maybe_delta(
            row.get("sentiment_lm_call"), prev and prev.get("sentiment_lm_call")
        )
    if prev is None:
        out["risks_persistence"] = math.nan
        out["wins_persistence"] = math.nan
        out["theme_drift"] = math.nan
        return out
    out["risks_persistence"] = _persistence(row["_risks_top5"], prev["_risks_top5"])
    out["wins_persistence"] = _persistence(row["_wins_top5"], prev["_wins_top5"])
    out["theme_drift"] = _theme_drift(
        row["_wins_top5"] | row["_risks_top5"],
        prev["_wins_top5"] | prev["_risks_top5"],
    )
    return out


def _llm_agree_guidance(rec_fb: dict) -> tuple[float, int]:
    """Return ``(agreement, missing_flag)`` per Part II §2.3 semantics.

    ``agreement``:
      - ``1.0`` if every model listed in ``rec_fb["models"]`` appears in
        ``consensus.guidance_by_model`` and they all share the same label;
      - ``0.0`` if ≥ 2 present models disagree;
      - ``NaN`` if any expected model is missing (equivalently: only
        produced ``_empty`` / ``_failed`` presenter records).

    ``missing_flag`` is ``1`` iff any expected model is missing, ``0``
    otherwise (independent of agree/disagree outcome).
    """
    expected = rec_fb.get("models")
    if not expected:
        error(f"call record has no 'models' field: {rec_fb['ticker']}_{rec_fb['quarter']}")
    by_model = rec_fb.get("consensus", {}).get("guidance_by_model") or {}
    missing = [m for m in expected if m not in by_model]
    missing_flag = 1 if missing else 0
    if missing:
        return math.nan, missing_flag
    labels = {by_model[m] for m in expected}
    return (1.0 if len(labels) == 1 else 0.0), missing_flag


def _require_block(record: dict, name: str) -> dict:
    block = record.get(name)
    if not isinstance(block, dict):
        error(
            f"call record {record.get('ticker')}_{record.get('quarter')} "
            f"missing required block: {name}"
        )
    return block


def _require_scalar_allowing_none(block: dict, key: str) -> float:
    """Fetch a float scalar; allow ``None`` -> ``NaN`` but not KeyError."""
    if key not in block:
        error(f"missing scalar field {key!r} in block: {list(block.keys())}")
    v = block[key]
    if v is None:
        return math.nan
    return float(v)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def _phrase_set(top5: list) -> set[str]:
    """Re-normalize aggregate ``wins_top5`` / ``risks_top5`` entries to a set.

    ``aggregate`` already lowercases + regex-cleans phrases when counting,
    but it writes the *display* form back out, so we lowercase again here
    for the Jaccard / persistence calculations.
    """
    out: set[str] = set()
    for entry in top5 or []:
        if not entry:
            continue
        phrase = entry[0] if isinstance(entry, (list, tuple)) else entry
        if isinstance(phrase, str) and phrase.strip():
            out.add(phrase.strip().lower())
    return out


def _maybe_delta(cur, prev) -> float:
    if cur is None or prev is None:
        return math.nan
    try:
        cur_f = float(cur)
        prev_f = float(prev)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(cur_f) or math.isnan(prev_f):
        return math.nan
    return cur_f - prev_f


def _persistence(cur: set[str], prev: set[str]) -> float:
    if not prev:
        return math.nan
    return len(cur & prev) / len(prev)


def _theme_drift(cur_union: set[str], prev_union: set[str]) -> float:
    union = cur_union | prev_union
    if not union:
        return math.nan
    return 1.0 - len(cur_union & prev_union) / len(union)


def _load_prices_or_none(ticker: str):
    """Return the prices DataFrame or ``None`` if cache is cold.

    ``features`` tolerates a missing price cache (ret_21d_prior becomes
    NaN for every call of that ticker) so the notebook can iterate on
    features without always refreshing prices. The backtest layer has its
    own fail-loud check on price availability.
    """
    try:
        return load_prices(ticker)
    except RuntimeError:
        return None


def _ret_21d_prior(prices_df, call_date: str) -> float:
    """21-trading-day return ending one trading day before ``call_date``.

    Follows the label convention: ``T`` is the first trading day ≥
    ``call_date``; we use ``Close_{T-1}`` as the numerator and
    ``Close_{T-22}`` (22 trading days earlier) as the denominator. When
    any lookup falls outside the prices window the feature is NaN (by
    §2.3 it does *not* get a ``_missing`` indicator column, to avoid
    encoding an early-sample flag).
    """
    if prices_df is None or call_date is None:
        return math.nan
    call_ts = pd.to_datetime(call_date)
    idx = prices_df["Date"].searchsorted(call_ts, side="left")
    if idx == 0:
        return math.nan
    t_minus_1 = idx - 1
    t_minus_22 = t_minus_1 - 21
    if t_minus_22 < 0:
        return math.nan
    try:
        num = float(prices_df["Close"].iloc[t_minus_1])
        den = float(prices_df["Close"].iloc[t_minus_22])
    except (KeyError, IndexError):
        return math.nan
    if not den:
        return math.nan
    return num / den - 1.0


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the transient ``_wins_top5 / _risks_top5`` columns before persisting.

    Kept out of :func:`build_feature_table` so callers that want to
    introspect top-5 sets (for reporting) still have the raw sets.
    """
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
