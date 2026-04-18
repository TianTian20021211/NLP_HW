"""Quality gates that gate progress from P0 -> P3 (see Plan 1.4 / 1.8).

Functions here are pure: they take a list of unit dicts (and optionally
extraction records) and compute simple fractions we can compare against
the targets in the plan. Used by the notebook to decide whether to
proceed to the next phase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .extract_llm import DEFAULT_MODELS, load_extraction
from .io_paths import QC_DIR, ensure_dirs, list_transcripts, parse_stem
from .parser import parse_file, read_units


PASS_TARGETS = {
    "executive_unknown_rate": 0.05,
    "qa_orphan_rate": 0.02,
    "parse_failures": 0,
    "llm_json_failure_rate": 0.02,
    "sentiment_nan_rate": 0.05,
    # Part 2 additions (Part II §2.8 / §2.9)
    "sentiment_lm_nan_rate": 0.25,
    "ret_21d_prior_nan_count": 2,
    "run_cells_nan_rate": 0.1,
}


def parse_quality(units: Sequence[dict]) -> dict:
    """Stage-A QC: executive-unknown + QA-orphan rates + dropped-date count."""
    presenters = [u for u in units if u["kind"] == "presenter"]
    qa = [u for u in units if u["kind"] == "qa"]
    exec_unk = sum(1 for u in presenters if u["speaker_role"] == "Unknown")
    orphans = sum(1 for u in qa if any("orphan" in w for w in (u.get("warnings") or [])))
    parse_failures = sum(1 for u in units if u.get("call_date") is None)
    return {
        "n_presenter": len(presenters),
        "n_qa": len(qa),
        "executive_unknown_rate": exec_unk / max(1, len(presenters)),
        "qa_orphan_rate": orphans / max(1, len(qa)),
        "parse_failures": parse_failures,
    }


def extraction_quality(units: Sequence[dict], models: Sequence[str] = DEFAULT_MODELS) -> dict:
    """Stage-C QC: LLM JSON-failure rate per model (cache-file driven)."""
    per_model: dict[str, dict] = {}
    for m in models:
        seen = 0
        failed = 0
        for u in units:
            rec = load_extraction(m, u["unit_id"])
            if rec is None:
                continue
            seen += 1
            if rec.get("_failed"):
                failed += 1
        per_model[m] = {
            "n_seen": seen,
            "n_failed": failed,
            "failure_rate": (failed / seen) if seen else None,
        }
    return per_model


def sentiment_quality(units: Sequence[dict]) -> dict:
    """Stage-B QC: fraction of non-empty units without a valid sentiment score."""
    from .sentiment_finbert import load_sentiment

    seen = 0
    nan = 0
    for u in units:
        if not (u.get("text") or "").strip():
            continue
        try:
            rec = load_sentiment(u["unit_id"])
        except Exception:
            continue
        seen += 1
        if rec.get("sentiment") is None:
            nan += 1
    return {
        "n_seen": seen,
        "n_nan": nan,
        "nan_rate": (nan / seen) if seen else None,
    }


def evaluate_gates(metrics: dict) -> list[tuple[str, bool, float | int, float | int]]:
    """Compare each known metric to its pass target.

    Returns `[(metric_name, passed, actual, target), ...]`, skipping
    metrics not present in `metrics`. Useful for the notebook summary cell.
    """
    out: list[tuple[str, bool, float | int, float | int]] = []
    for k, tgt in PASS_TARGETS.items():
        if k not in metrics:
            continue
        actual = metrics[k]
        if actual is None:
            continue
        passed = actual <= tgt
        out.append((k, passed, actual, tgt))
    return out


def corpus_parse_quality() -> dict:
    """Parse every transcript once (no cache writes) and aggregate QC.

    Used as a pre-flight check before triggering expensive stages.
    """
    agg = {"n_presenter": 0, "n_qa": 0, "exec_unknown": 0, "orphans": 0, "parse_failures": 0, "files": 0}
    per_file: list[dict] = []
    for p in list_transcripts():
        tic, q = parse_stem(p)
        units = parse_file(p, tic, q)
        pq = parse_quality(units)
        per_file.append({"ticker": tic, "quarter": q, **pq})
        agg["n_presenter"] += pq["n_presenter"]
        agg["n_qa"] += pq["n_qa"]
        agg["exec_unknown"] += int(pq["executive_unknown_rate"] * pq["n_presenter"])
        agg["orphans"] += int(pq["qa_orphan_rate"] * pq["n_qa"])
        agg["parse_failures"] += pq["parse_failures"]
        agg["files"] += 1
    overall = {
        "files": agg["files"],
        "n_presenter": agg["n_presenter"],
        "n_qa": agg["n_qa"],
        "executive_unknown_rate": agg["exec_unknown"] / max(1, agg["n_presenter"]),
        "qa_orphan_rate": agg["orphans"] / max(1, agg["n_qa"]),
        "parse_failures": agg["parse_failures"],
    }
    return {"overall": overall, "per_file": per_file}


def save_qc_report(name: str, payload: dict) -> Path:
    """Dump a QC payload to `data/cache/qc/<name>.json` (for the writeup)."""
    ensure_dirs()
    out = QC_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return out


def _jsonable(obj):
    """Best-effort fallback for numpy / pandas scalar types inside payloads."""
    try:
        return obj.item()
    except Exception:
        return str(obj)


def sentiment_lm_quality(units: Sequence[dict]) -> dict:
    """Part 2 QC: fraction of non-empty units with no LM dictionary hit.

    Mirrors :func:`sentiment_quality`: only counts units that carry actual
    text — short "thanks" style QA answers with zero pos/neg hits are
    expected and not a failure. Threshold in ``PASS_TARGETS`` is 25%
    because LM is a sparse-dictionary method and routinely leaves short
    QA turns unscored.
    """
    from .lexicon_lm import load_sentiment_lm

    seen = 0
    nan = 0
    for u in units:
        if not (u.get("text") or "").strip():
            continue
        try:
            rec = load_sentiment_lm(u["unit_id"])
        except Exception:
            continue
        seen += 1
        if rec.get("sentiment") is None:
            nan += 1
    return {
        "n_seen": seen,
        "n_nan": nan,
        "nan_rate": (nan / seen) if seen else None,
    }


def label_quality(labels_df: pd.DataFrame) -> dict:
    """Part 2 QC: per-horizon NaN rate on excess labels + boundary markers.

    Threshold is applied per-horizon (not aggregated) because the +63d
    horizon naturally carries higher NaN rates than +1d. Returns the per-
    horizon rates plus ``n_rows`` so the caller can sanity-check that the
    label generator ran.
    """
    from .labels import HORIZONS, label_nan_rate

    rates = label_nan_rate(labels_df)
    return {
        "n_rows": int(len(labels_df)),
        "horizons": list(rates.keys()),
        "nan_rate_by_horizon": rates,
    }


def feature_momentum_quality(feature_df: pd.DataFrame) -> dict:
    """Part 2 QC: how many ``ret_21d_prior`` NaNs slipped through."""
    col = "ret_21d_prior"
    if col not in feature_df.columns:
        return {col: None}
    n_nan = int(feature_df[col].isna().sum())
    return {"ret_21d_prior_nan_count": n_nan, "n_rows": int(len(feature_df))}


def run_cells_quality(
    preds_df: pd.DataFrame,
    score_col: str = "score",
    score_kind: str = "binary",
) -> dict:
    """Part 2 QC: pathologies in a ``preds_*.parquet`` frame.

    Flags:

    * ``nan_rate``: fraction of rows whose model score is NaN.
    * ``std_zero``: scores have ``std==0`` — model collapsed to a
      constant prediction (single-class sentinel or malformed train).
    * ``prob_out_of_range``: only populated for ``score_kind="binary"``;
      classifier probabilities outside ``[0, 1]`` indicate a numerical
      bug upstream. For ``score_kind="regression"`` or ``"rule"`` the
      score range is unbounded and this check is ``None``.
    """
    if score_col not in preds_df.columns:
        return {"nan_rate": None, "std_zero": None, "prob_out_of_range": None}
    scores = preds_df[score_col]
    n = len(scores)
    nan_rate = float(scores.isna().mean()) if n else 0.0
    finite = scores.dropna()
    std_zero = bool(finite.std() == 0) if len(finite) > 1 else True
    if score_kind == "binary" and not finite.empty:
        lo, hi = float(finite.min()), float(finite.max())
        oor: bool | None = bool(lo < -1e-9 or hi > 1.0 + 1e-9)
    else:
        oor = None
    return {"nan_rate": nan_rate, "std_zero": std_zero, "prob_out_of_range": oor}
