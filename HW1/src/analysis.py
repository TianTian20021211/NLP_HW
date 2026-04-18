"""Qualitative + extra-credit analyses on top of the call-level JSONs.

Covers the four PDF deliverable items not yet implemented as separate
modules:

1. **Per-ticker cross-quarter narrative** (PDF §6 — "for 2-3 companies, a
   paragraph showing the story your pipeline tells across quarters").
2. **Reactive-vs-proactive officer analysis** (PDF §6 stretch / extra
   credit — surfaces ``qa_minus_pres_risks`` already captured in the
   ``calls`` JSONs).
3. **LLM head-to-head comparison** (PDF §6 extra credit — uses
   ``consensus.guidance_by_model`` to compute per-call gemma vs llama
   agreement on the guidance label).
4. **Cross-sectional long-short signal** (PDF §6 extra credit — for each
   quarter rank the 14 tickers by ``score = sentiment_call + 0.5 *
   guidance_ordinal``, long top-3 / short bottom-3, on +5d excess return).

All functions are pure consumers of the on-disk caches written by
``aggregate.py`` (``data/cache/calls/``) and the price cache; they never
re-invoke FinBERT or the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .aggregate import load_call_record
from .io_paths import (
    ANALYSIS,
    analysis_path,
    ensure_dirs,
    error,
    list_transcripts,
    parse_stem,
)


_GUIDANCE_ORDINAL = {"lowered": -1.0, "none": 0.0, "maintained": 0.5, "raised": 1.0}


def _all_calls() -> list[tuple[str, str]]:
    return [parse_stem(p) for p in list_transcripts()]


def _per_ticker_call_records(ticker: str) -> list[dict]:
    """Load every call JSON for ``ticker`` sorted chronologically by call_date."""
    recs: list[dict] = []
    for tic, q in _all_calls():
        if tic != ticker:
            continue
        rec = load_call_record(tic, q, variant="finbert")
        recs.append(rec)
    recs.sort(key=lambda r: r.get("call_date") or "")
    return recs


def _short_quarter(rec: dict) -> str:
    q = (rec.get("quarter") or "").replace("Q", "").replace("-", " ")
    return q.strip()


def _topk_phrases(top: list, k: int = 3) -> list[str]:
    out: list[str] = []
    for entry in (top or [])[:k]:
        if not entry:
            continue
        if isinstance(entry, (list, tuple)):
            out.append(str(entry[0]))
        else:
            out.append(str(entry))
    return out


def cross_quarter_narrative(ticker: str) -> dict:
    """Build a narrative payload for one ticker across every quarter we have.

    Returns ``{ticker, quarters: [{quarter, call_date, sentiment_call,
    sentiment_qa, guidance_call, top_wins, top_risks, qa_only_risks_n}],
    summary: {n_quarters, avg_sentiment, n_guidance_raised, ...}}``. The
    notebook composes the actual report paragraph from this structured
    output.
    """
    recs = _per_ticker_call_records(ticker)
    if not recs:
        error(f"no call records for ticker {ticker!r}")
    quarters: list[dict] = []
    for r in recs:
        cons = r.get("consensus") or {}
        disp = r.get("dispersion") or {}
        quarters.append({
            "quarter": r.get("quarter"),
            "call_date": r.get("call_date"),
            "sentiment_call": cons.get("sentiment_call"),
            "sentiment_qa": cons.get("sentiment_qa"),
            "guidance_call": cons.get("guidance_call"),
            "guidance_by_model": cons.get("guidance_by_model") or {},
            "top_wins": _topk_phrases(cons.get("wins_top5", []), k=3),
            "top_risks": _topk_phrases(cons.get("risks_top5", []), k=3),
            "qa_only_risks": disp.get("qa_minus_pres_risks", [])[:5],
            "qa_only_risks_n": int(disp.get("qa_minus_pres_risks_count", 0)),
            "role_sent_means": disp.get("role_sent_means", {}),
            "guidance_disagree": int(disp.get("guidance_disagree", 0)),
        })

    sents = [q["sentiment_call"] for q in quarters if q["sentiment_call"] is not None]
    summary = {
        "n_quarters": len(quarters),
        "avg_sentiment": float(np.mean(sents)) if sents else None,
        "min_sentiment": (
            min((q for q in quarters if q["sentiment_call"] is not None),
                key=lambda q: q["sentiment_call"])
            if sents else None
        ),
        "max_sentiment": (
            max((q for q in quarters if q["sentiment_call"] is not None),
                key=lambda q: q["sentiment_call"])
            if sents else None
        ),
        "n_guidance_raised": sum(1 for q in quarters if q["guidance_call"] == "raised"),
        "n_guidance_lowered": sum(1 for q in quarters if q["guidance_call"] == "lowered"),
        "n_guidance_maintained": sum(1 for q in quarters if q["guidance_call"] == "maintained"),
        "qa_only_risks_total": sum(q["qa_only_risks_n"] for q in quarters),
    }
    return {"ticker": ticker, "quarters": quarters, "summary": summary}


def write_narratives(tickers: Sequence[str]) -> Path:
    """Persist a JSON payload covering each ``ticker`` in ``tickers``.

    Saved to ``data/cache/analysis/narratives.json`` so the notebook
    output cell and the report writer can both read it without
    re-traversing every call record.
    """
    ensure_dirs()
    payload = {t: cross_quarter_narrative(t) for t in tickers}
    p = analysis_path("narratives.json")
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def reactive_proactive_table(calls: Sequence[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Aggregate ``qa_minus_pres_risks`` across every call into a tidy table.

    Returns one row per call with ``[ticker, quarter, call_date,
    n_qa_only_risks, n_pres_risks, top_qa_only_risks]``. Higher
    ``n_qa_only_risks`` flags calls where analysts surfaced concerns the
    executives had not raised in prepared remarks — the S&P "reactive"
    pattern in the PDF.
    """
    if calls is None:
        calls = _all_calls()
    rows: list[dict] = []
    for ticker, quarter in calls:
        rec = load_call_record(ticker, quarter, variant="finbert")
        cons = rec.get("consensus") or {}
        disp = rec.get("dispersion") or {}
        pres_risks = (rec.get("by_section") or {}).get("presenter", {}).get("risks_top5", [])
        rows.append({
            "ticker": ticker,
            "quarter": quarter,
            "call_date": rec.get("call_date"),
            "sentiment_call": cons.get("sentiment_call"),
            "guidance_call": cons.get("guidance_call"),
            "n_pres_risks": len(pres_risks),
            "n_qa_only_risks": int(disp.get("qa_minus_pres_risks_count", 0)),
            "top_qa_only_risks": "; ".join(disp.get("qa_minus_pres_risks", [])[:5]),
        })
    df = pd.DataFrame(rows).sort_values(["ticker", "call_date"]).reset_index(drop=True)
    p = analysis_path("reactive_proactive.csv")
    df.to_csv(p, index=False)
    return df


def reactive_proactive_summary(df: pd.DataFrame) -> dict:
    """Summarize the reactive-vs-proactive table across the corpus.

    Returns ``{corpus_avg_qa_only, corpus_avg_pres_risks,
    top_reactive_calls: [(ticker, quarter, n)], by_ticker_avg: dict}``.
    """
    by_tic = df.groupby("ticker").agg(
        n_calls=("quarter", "count"),
        avg_qa_only=("n_qa_only_risks", "mean"),
        avg_pres=("n_pres_risks", "mean"),
    ).reset_index()
    top10 = df.nlargest(10, "n_qa_only_risks")[
        ["ticker", "quarter", "call_date", "n_qa_only_risks", "top_qa_only_risks"]
    ].to_dict(orient="records")
    return {
        "corpus_avg_qa_only": float(df["n_qa_only_risks"].mean()),
        "corpus_avg_pres_risks": float(df["n_pres_risks"].mean()),
        "top_reactive_calls": top10,
        "by_ticker_avg": by_tic.to_dict(orient="records"),
    }


def llm_agreement_per_call(calls: Sequence[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Per-call gemma vs llama agreement on the call-level guidance label.

    Reads ``consensus.guidance_by_model`` from each call JSON. ``agree=1``
    when both models emit the same label; ``agree=0`` when they
    disagree or one is missing.
    """
    if calls is None:
        calls = _all_calls()
    rows: list[dict] = []
    for ticker, quarter in calls:
        rec = load_call_record(ticker, quarter, variant="finbert")
        cons = rec.get("consensus") or {}
        per_model: dict[str, str] = cons.get("guidance_by_model") or {}
        gemma = per_model.get("gemma3:4b")
        llama = per_model.get("llama3.1:8b")
        agree = int(bool(gemma) and bool(llama) and gemma == llama)
        rows.append({
            "ticker": ticker,
            "quarter": quarter,
            "call_date": rec.get("call_date"),
            "guidance_consensus": cons.get("guidance_call"),
            "gemma3_4b": gemma,
            "llama3_1_8b": llama,
            "agree": agree,
        })
    df = pd.DataFrame(rows).sort_values(["ticker", "call_date"]).reset_index(drop=True)
    p = analysis_path("llm_agreement_per_call.csv")
    df.to_csv(p, index=False)
    return df


def llm_agreement_per_ticker(per_call: pd.DataFrame) -> pd.DataFrame:
    """Roll the per-call agreement table up to per-ticker (for the figure)."""
    grp = per_call.groupby("ticker").agg(
        n_calls=("quarter", "count"),
        n_agree=("agree", "sum"),
    ).reset_index()
    grp["agree_rate"] = grp["n_agree"] / grp["n_calls"].clip(lower=1)
    grp["n_disagree"] = grp["n_calls"] - grp["n_agree"]
    p = analysis_path("llm_agreement_per_ticker.csv")
    grp.to_csv(p, index=False)
    return grp


def cross_sectional_backtest(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    horizon: str = "5d",
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Quarterly cross-sectional long-short on the 14-name universe.

    For each quarter (calendar quarter of ``call_date``):

    1. Compute ``score = sentiment_call + 0.5 · guidance_ordinal`` per
       ticker (skip rows with NaN sentiment).
    2. Long the top ``top_k`` by score, short the bottom ``top_k``.
    3. PnL = mean(excess_long) - mean(excess_short).
    4. Compound across quarters as the cross-sectional equity series.

    Returns ``(per_quarter_df, summary)`` and writes
    ``analysis/cross_sectional_quarterly.csv``.
    """
    excess_col = f"excess_{horizon}"
    if excess_col not in labels_df.columns:
        error(f"labels frame missing {excess_col!r}")
    merged = feature_df.merge(labels_df[["ticker", "call_date", excess_col]],
                              on=["ticker", "call_date"], how="inner")
    if "guidance_ordinal" not in merged.columns:
        error("feature_df missing 'guidance_ordinal'")
    merged["score"] = merged["sentiment_call"].astype(float) + 0.5 * merged["guidance_ordinal"].astype(float)
    merged["call_dt"] = pd.to_datetime(merged["call_date"])
    merged["cs_quarter"] = merged["call_dt"].dt.to_period("Q").astype(str)

    rows: list[dict] = []
    for cs_q, group in merged.groupby("cs_quarter", sort=True):
        usable = group.dropna(subset=["score", excess_col])
        if len(usable) < 2 * top_k:
            continue
        ranked = usable.sort_values("score", ascending=False).reset_index(drop=True)
        longs = ranked.head(top_k)
        shorts = ranked.tail(top_k)
        ret_long = float(longs[excess_col].mean())
        ret_short = float(shorts[excess_col].mean())
        ret_ls = ret_long - ret_short
        rows.append({
            "quarter": cs_q,
            "n_long": int(len(longs)),
            "n_short": int(len(shorts)),
            "n_universe": int(len(usable)),
            "ret_long_avg": ret_long,
            "ret_short_avg": ret_short,
            "ret_ls": ret_ls,
            "long_tickers": ",".join(longs["ticker"].tolist()),
            "short_tickers": ",".join(shorts["ticker"].tolist()),
        })
    df = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    if df.empty:
        error("cross_sectional_backtest produced no quarters with >= 2*top_k usable rows")
    df["equity"] = (1.0 + df["ret_ls"]).cumprod()
    p = analysis_path("cross_sectional_quarterly.csv")
    df.to_csv(p, index=False)

    bets = df["ret_ls"].to_numpy()
    sharpe = (
        float(bets.mean() / bets.std(ddof=1) * np.sqrt(len(bets)))
        if len(bets) > 2 and bets.std(ddof=1) > 0 else None
    )
    summary = {
        "horizon": horizon,
        "top_k": top_k,
        "n_quarters": int(len(df)),
        "mean_ret_ls": float(bets.mean()),
        "std_ret_ls": float(bets.std(ddof=1)) if len(bets) > 1 else 0.0,
        "hit_rate_quarters": float((bets > 0).mean()),
        "final_equity": float(df["equity"].iloc[-1]),
        "sharpe_quarterly": sharpe,
    }
    (analysis_path("cross_sectional_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return df, summary


def momentum_only_baseline(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    horizon: str = "5d",
) -> dict:
    """Standalone signal: ``ret_21d_prior > 0 → long``, on the test split.

    PDF §6 extra credit: "comparison of an external signal (price
    momentum) alone vs combined". This function isolates the momentum
    column to give a clean alone-vs-combined number.
    """
    excess_col = f"excess_{horizon}"
    if "ret_21d_prior" not in feature_df.columns:
        error("feature_df missing 'ret_21d_prior'")
    if excess_col not in labels_df.columns:
        error(f"labels frame missing {excess_col!r}")
    merged = feature_df.merge(
        labels_df[["ticker", "call_date", excess_col]],
        on=["ticker", "call_date"], how="inner",
    )
    merged = merged.dropna(subset=["ret_21d_prior", excess_col])
    longs = merged[merged["ret_21d_prior"] > 0]
    excess = longs[excess_col].to_numpy()
    if len(excess) == 0:
        return {"horizon": horizon, "n_trades": 0, "pnl_sum": 0.0, "hit_rate": None,
                "sharpe": None, "final_equity": 1.0}
    pnl_sum = float(excess.sum())
    hit_rate = float((excess > 0).mean())
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(len(excess)))
        if len(excess) > 2 and excess.std(ddof=1) > 0 else None
    )
    final_equity = float(np.prod(1.0 + excess))
    return {
        "horizon": horizon,
        "n_trades": int(len(excess)),
        "pnl_sum": pnl_sum,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "final_equity": final_equity,
    }


def write_combined_summary(
    narratives_path: Path,
    rp_summary: dict,
    llm_per_ticker: pd.DataFrame,
    cs_summary: dict,
    momentum_summary: dict,
) -> Path:
    """Bundle every analysis artifact into one ``analysis_summary.json``.

    Lets the notebook surface a single block to the user (and the
    report writer pick keys without juggling four separate files).
    """
    out = {
        "narratives_path": str(narratives_path),
        "reactive_proactive": rp_summary,
        "llm_agreement_overall": {
            "n_calls_total": int(llm_per_ticker["n_calls"].sum()),
            "n_agree_total": int(llm_per_ticker["n_agree"].sum()),
            "agree_rate_overall": float(
                llm_per_ticker["n_agree"].sum()
                / max(1, llm_per_ticker["n_calls"].sum())
            ),
            "by_ticker": llm_per_ticker.to_dict(orient="records"),
        },
        "cross_sectional": cs_summary,
        "momentum_only_baseline": momentum_summary,
    }
    p = analysis_path("analysis_summary.json")
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return p
