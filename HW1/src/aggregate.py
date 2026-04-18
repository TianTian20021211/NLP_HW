"""Unit-level caches -> call-level consensus + dispersion.

Inputs (read-only):
  - `data/cache/units/<call>.jsonl`
  - `data/cache/sentiment/<unit_id>.json`           (variant="finbert")
  - `data/cache/sentiment_lm/<unit_id>.json`        (variant="lm")
  - `data/cache/extractions/<model>/<unit_id>.json`

Output (overwritten on every call):
  - `data/cache/calls/<call>.json`                  (variant="finbert")
  - `data/cache/calls_lm/<call>.json`               (variant="lm")

Re-running this stage never touches the LLMs, FinBERT, or the LM
dictionary scorer; it is a pure function of the on-disk caches. See Plan
1.7 and Part II §2.2.

Variant handling: the event half of the record (wins / risks / guidance,
everything derived from the LLM extraction caches) is identical across
variants. Only the sentiment scalar plumbed into consensus / dispersion /
by-role / by-section changes. Both variants share the same weighting and
dedup logic downstream, so aggregate output for variant="finbert" is
bit-for-bit identical to Part 1 except for the new `_cache_key` field.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from typing import Iterable, Literal, Sequence

from .cache_keys import KEY_FIELD, stage_key
from .extract_llm import DEFAULT_MODELS, load_extraction
from .extract_qa_group import load_qa_group
from .io_paths import call_lm_path, call_path, ensure_dirs, error
from .parser import read_units


Variant = Literal["finbert", "lm"]


def _variant_stage(variant: Variant) -> str:
    if variant == "finbert":
        return "calls"
    if variant == "lm":
        return "calls_lm"
    error(f"unknown variant: {variant}")
    return ""


def _variant_call_path(ticker: str, quarter: str, variant: Variant):
    if variant == "finbert":
        return call_path(ticker, quarter)
    if variant == "lm":
        return call_lm_path(ticker, quarter)
    error(f"unknown variant: {variant}")


def _load_unit_sentiment(uid: str, variant: Variant) -> float | None:
    """Read the per-unit sentiment scalar for ``variant``, swallowing failures.

    Mirrors the existing Part 1 behavior of ``_enrich``: a missing or stale
    cache file degrades to ``None`` so ``_weighted_mean`` can skip the unit.
    Upstream QC (Stage B for FinBERT, Part 2 QC for LM) is responsible for
    catching cache-missing cases before aggregation runs.
    """
    if variant == "finbert":
        from .sentiment_finbert import load_sentiment

        rec = load_sentiment(uid)
        return rec.get("sentiment")
    if variant == "lm":
        from .lexicon_lm import load_sentiment_lm

        rec = load_sentiment_lm(uid)
        return rec.get("sentiment")
    error(f"unknown variant: {variant}")
    return None


_PRESENTER_ROLES = ("CEO", "CFO", "CTO", "IR", "Other_Exec")
_DISPERSION_ROLES = ("CEO", "CFO", "CTO", "Other_Exec")
_GUIDANCE_TIE_ORDER = ("lowered", "raised", "maintained", "none")

_GUIDANCE_WEIGHT_PRESENTER_SENIOR = 2
_GUIDANCE_WEIGHT_PRESENTER_OTHER = 1
_GUIDANCE_WEIGHT_GROUP_SENIOR = 5
_GUIDANCE_WEIGHT_GROUP_OTHER = 1


def _call_id(unit: dict) -> str:
    return f"{unit['ticker']}_{unit['quarter']}"


def _dedup_gid(unit: dict) -> tuple:
    """Counting-bucket id: per-unit for presenter, per (call, role) for QA.

    Used by :func:`_top_phrases` and :func:`_guidance_vote` to avoid
    counting an officer's Q&A extraction once per answer unit — under the
    batched-per-officer path all QA units of the same ``(call, role)``
    share the same extraction record, and we want it to contribute a
    single vote.
    """
    if unit.get("kind") == "qa":
        return ("qa_group", _call_id(unit), unit.get("speaker_role") or "Unknown")
    return ("unit", unit["unit_id"])


def build_call_record(
    ticker: str,
    quarter: str,
    models: Sequence[str] = DEFAULT_MODELS,
    variant: Variant = "finbert",
) -> dict:
    """Assemble the call-level JSON from on-disk unit caches.

    ``variant`` selects which per-unit sentiment family feeds the sentiment
    tracks (consensus / dispersion / by_role / by_section). Event fields
    (wins / risks / guidance) come from the shared LLM extraction caches
    and are variant-invariant. No model invocations happen here.
    """
    units = read_units(ticker, quarter)
    if not units:
        return _empty_record(ticker, quarter, None, reason="no units")

    enriched = _enrich(units, models, variant=variant)
    presenters = [u for u in enriched if u["kind"] == "presenter"]
    qa = [u for u in enriched if u["kind"] == "qa"]

    consensus = _consensus(presenters, qa)
    dispersion = _dispersion(presenters, qa)
    by_role = _by_role(presenters, qa)
    by_section = {"presenter": _section_metrics(presenters), "qa": _section_metrics(qa)}

    record = {
        "ticker": ticker,
        "quarter": quarter,
        "call_date": units[0].get("call_date"),
        "variant": variant,
        "n_units": {
            "total": len(units),
            "presenter": len(presenters),
            "qa": len(qa),
        },
        "models": list(models),
        "consensus": consensus,
        "dispersion": dispersion,
        "by_role": by_role,
        "by_section": by_section,
    }
    return record


def write_call_record(record: dict, variant: Variant = "finbert") -> None:
    """Persist a call record JSON, stamping the current stage cache key.

    ``_cache_key`` embeds ``stage_key(_variant_stage(variant))`` so
    :func:`load_call_record` and :func:`cache_keys.prune_stale` can detect
    stale records after any edit to ``aggregate.py`` (or ``lexicon_lm.py``
    for the LM variant).
    """
    ensure_dirs()
    stage = _variant_stage(variant)
    record[KEY_FIELD] = stage_key(stage)
    record["variant"] = variant
    p = _variant_call_path(record["ticker"], record["quarter"], variant)
    p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def build_and_write(
    ticker: str,
    quarter: str,
    models: Sequence[str] = DEFAULT_MODELS,
    variant: Variant = "finbert",
) -> dict:
    rec = build_call_record(ticker, quarter, models=models, variant=variant)
    write_call_record(rec, variant=variant)
    return rec


def load_call_record(ticker: str, quarter: str, variant: Variant = "finbert") -> dict:
    """Read a call record with freshness validation.

    Treats any of {missing file, unreadable JSON, missing ``_cache_key``,
    stale ``_cache_key``} as fatal so downstream feature generation can
    trust that the record was produced by the current source tree.
    """
    p = _variant_call_path(ticker, quarter, variant)
    if not p.is_file():
        error(f"call record missing ({variant}): {p}")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        error(f"call record unreadable ({variant}): {p} ({e})")
    if not isinstance(rec, dict) or rec.get(KEY_FIELD) != stage_key(_variant_stage(variant)):
        error(f"call record stale ({variant}): {p}")
    return rec


def _enrich(
    units: Sequence[dict],
    models: Sequence[str],
    variant: Variant = "finbert",
) -> list[dict]:
    """Attach sentiment + model extractions to each unit in-memory.

    Presenter units read their own per-unit extraction cache. Q&A units
    read the batched ``(call, role)`` group cache — so every CEO Q&A
    unit in a given call shares the same ``_ext`` payload. Missing
    caches are tolerated: missing sentiment -> ``None``, missing
    extraction -> skipped in that model's contribution. Sentiment loader
    is chosen by ``variant``.
    """
    group_cache: dict[tuple[str, str, str], dict | None] = {}

    def _ext_for(unit: dict, model: str) -> dict | None:
        if unit.get("kind") == "qa":
            ckey = (_call_id(unit), unit.get("speaker_role") or "Unknown", model)
            if ckey not in group_cache:
                group_cache[ckey] = load_qa_group(model, ckey[0], ckey[1])
            return group_cache[ckey]
        return load_extraction(model, unit["unit_id"])

    enriched: list[dict] = []
    for u in units:
        uid = u["unit_id"]
        try:
            sent = _load_unit_sentiment(uid, variant)
        except Exception:
            sent = None
        ext_by_model: dict[str, dict] = {}
        for m in models:
            rec = _ext_for(u, m)
            if rec is not None:
                ext_by_model[m] = rec
        enriched.append({**u, "_sentiment": sent, "_ext": ext_by_model})
    return enriched


def _weighted_mean(values: Iterable[tuple[float | None, int]]) -> float | None:
    """Length-weighted mean, skipping `None` values. Returns `None` if empty."""
    total = 0.0
    w_sum = 0
    for v, w in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if w <= 0:
            continue
        total += v * w
        w_sum += w
    if w_sum == 0:
        return None
    return total / w_sum


def _consensus(presenters: list[dict], qa: list[dict]) -> dict:
    """Official 'what did this call say' track: sentiment + top items + guidance.

    ``guidance_by_model`` exposes each LLM's call-level guidance separately
    (same weighting scheme as ``guidance_call`` but restricted to one model
    at a time). Part 2's ``llm_agree_guidance`` feature needs this: the
    majority-vote collapse inside :func:`_unit_consensus_ext` otherwise
    hides per-model disagreement.
    """
    sent_call = _weighted_mean((u["_sentiment"], u["n_chars"]) for u in presenters)
    sent_qa = _weighted_mean((u["_sentiment"], u["n_chars"]) for u in qa)

    wins_top5 = _top_phrases(presenters, field="wins", k=5)
    risks_top5 = _top_phrases(presenters + qa, field="risks", k=5)

    guidance_call = _guidance_vote(presenters)
    guidance_by_model = _guidance_by_model(presenters)

    return {
        "sentiment_call": sent_call,
        "sentiment_qa": sent_qa,
        "wins_top5": wins_top5,
        "risks_top5": risks_top5,
        "guidance_call": guidance_call,
        "guidance_by_model": guidance_by_model,
    }


def _dispersion(presenters: list[dict], qa: list[dict]) -> dict:
    """Cross-officer disagreement: sentiment std, guidance disagreement, overlap."""
    by_role_sent: dict[str, list[tuple[float, int]]] = {}
    by_role_wins: dict[str, set[str]] = {}
    by_role_risks: dict[str, set[str]] = {}
    by_role_guidance: dict[str, list[str]] = {}

    for u in presenters:
        r = u["speaker_role"]
        if r not in _DISPERSION_ROLES:
            continue
        if u["_sentiment"] is not None:
            by_role_sent.setdefault(r, []).append((u["_sentiment"], u["n_chars"]))
        wins, risks, gv = _unit_consensus_ext(u)
        by_role_wins.setdefault(r, set()).update(_norm_phrase(w) for w in wins)
        by_role_risks.setdefault(r, set()).update(_norm_phrase(w) for w in risks)
        if gv:
            by_role_guidance.setdefault(r, []).append(gv)

    role_sent_means: dict[str, float] = {}
    for r, pairs in by_role_sent.items():
        m = _weighted_mean((v, w) for v, w in pairs)
        if m is not None:
            role_sent_means[r] = m
    sent_disp = _std(list(role_sent_means.values())) if len(role_sent_means) >= 2 else None

    guidance_labels: set[str] = set()
    for labels in by_role_guidance.values():
        for g in labels:
            if g and g != "none":
                guidance_labels.add(g)
    guidance_disagree = 1 if len(guidance_labels) >= 2 else 0

    wins_overlap = _mean_pairwise_jaccard(by_role_wins)
    risks_overlap = _mean_pairwise_jaccard(by_role_risks)

    qa_risks: set[str] = set()
    for u in qa:
        _, risks, _ = _unit_consensus_ext(u)
        qa_risks.update(_norm_phrase(w) for w in risks)
    pres_risks: set[str] = set()
    for u in presenters:
        _, risks, _ = _unit_consensus_ext(u)
        pres_risks.update(_norm_phrase(w) for w in risks)
    qa_minus_pres = sorted(qa_risks - pres_risks)

    return {
        "sentiment_dispersion": sent_disp,
        "guidance_disagree": guidance_disagree,
        "wins_overlap": wins_overlap,
        "risks_overlap": risks_overlap,
        "qa_minus_pres_risks": qa_minus_pres,
        "qa_minus_pres_risks_count": len(qa_minus_pres),
        "role_sent_means": role_sent_means,
    }


def _by_role(presenters: list[dict], qa: list[dict]) -> dict:
    """Per-role slice of the same metrics; presenter + qa units both attributed to speaker role."""
    groups: dict[str, list[dict]] = {r: [] for r in _PRESENTER_ROLES + ("Analyst", "Unknown")}
    for u in presenters + qa:
        groups.setdefault(u["speaker_role"], []).append(u)
    out: dict[str, dict] = {}
    for role, ulist in groups.items():
        if not ulist:
            continue
        out[role] = _section_metrics(ulist)
    return out


def _section_metrics(units: list[dict]) -> dict:
    sent = _weighted_mean((u["_sentiment"], u["n_chars"]) for u in units)
    wins = _top_phrases(units, field="wins", k=5)
    risks = _top_phrases(units, field="risks", k=5)
    return {
        "n_units": len(units),
        "sentiment": sent,
        "wins_top5": wins,
        "risks_top5": risks,
        "guidance_vote": _guidance_vote(units),
    }


def _unit_consensus_ext(unit: dict) -> tuple[list[str], list[str], str | None]:
    """Collapse the multi-model extraction into one `(wins, risks, guidance)`.

    Wins/risks are the union across models (each phrase counted once per
    unit, same as in `_top_phrases`). Guidance is the majority across
    models; `none` loses to any non-`none` (tie -> most conservative).
    Both ``_failed`` and ``_empty`` records are treated as "no evidence"
    from that model on that unit and contribute no wins / risks / guidance
    vote — so ``guidance == None`` genuinely means "no model committed a
    direction here" rather than "models agreed on none".
    """
    wins_all: list[str] = []
    risks_all: list[str] = []
    guidance_votes: list[str] = []
    for m, rec in unit.get("_ext", {}).items():
        if _is_no_evidence(rec):
            continue
        wins_all.extend(rec.get("wins", []) or [])
        risks_all.extend(rec.get("risks", []) or [])
        g = rec.get("guidance")
        if g:
            guidance_votes.append(g)
    wins = _dedupe(wins_all)
    risks = _dedupe(risks_all)
    guidance = _guidance_pick(guidance_votes) if guidance_votes else None
    return wins, risks, guidance


def _is_no_evidence(rec: dict | None) -> bool:
    """A model-per-unit record carries no evidence if missing or flagged.

    ``_failed`` records never reached the LLM at all; ``_empty`` records
    hit the LLM on empty text and fell back to default ``guidance="none"``
    without any signal. Both get skipped everywhere so that the absence of
    a signal propagates cleanly (see Part II §2.3 "llm_agree_guidance
    missing semantics").
    """
    if rec is None:
        return True
    return bool(rec.get("_failed") or rec.get("_empty"))


def _top_phrases(units: list[dict], field: str, k: int) -> list[tuple[str, int]]:
    """Count normalized phrases across units/groups (each bucket contributes each phrase once per model).

    A "bucket" is one presenter unit, or one ``(call, role)`` Q&A group.
    Under the batched-per-officer path every CEO Q&A answer points to the
    same cached extraction; without dedup the CEO's top 3 wins would be
    counted once per answer and dominate the top-5 rankings.

    Returns `[(phrase, count), ...]` sorted by descending count, len `≤ k`.
    """
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    seen_buckets: set[tuple] = set()
    for u in units:
        gid = _dedup_gid(u)
        if gid in seen_buckets:
            continue
        seen_buckets.add(gid)
        for _, rec in u.get("_ext", {}).items():
            if _is_no_evidence(rec):
                continue
            seen: set[str] = set()
            for raw in rec.get(field, []) or []:
                key = _norm_phrase(raw)
                if not key or key in seen:
                    continue
                seen.add(key)
                counts[key] += 1
                display.setdefault(key, raw.strip())
    top = counts.most_common(k)
    return [[display[k_], v] for k_, v in top]


def _guidance_vote(units: list[dict]) -> str:
    """Majority-vote guidance across presenter units and Q&A officer groups.

    Presenter units vote per-unit with CEO/CFO ``×2``. Q&A units collapse
    to one vote per ``(call, speaker_role)`` group, with CEO/CFO groups
    weighted ``×5`` to compensate for the votes-per-officer collapse
    from the batched extraction (a CEO who used to contribute ~18 Q&A
    unit votes now contributes one group vote).

    Ties resolved conservatively: ``lowered > raised > maintained > none``.
    Empty input returns ``"none"``.
    """
    weighted: Counter[str] = Counter()
    seen_buckets: set[tuple] = set()
    for u in units:
        gid = _dedup_gid(u)
        if gid in seen_buckets:
            continue
        seen_buckets.add(gid)
        _, _, g = _unit_consensus_ext(u)
        if not g:
            continue
        senior = u.get("speaker_role") in ("CEO", "CFO")
        if u.get("kind") == "qa":
            w = _GUIDANCE_WEIGHT_GROUP_SENIOR if senior else _GUIDANCE_WEIGHT_GROUP_OTHER
        else:
            w = _GUIDANCE_WEIGHT_PRESENTER_SENIOR if senior else _GUIDANCE_WEIGHT_PRESENTER_OTHER
        weighted[g] += w
    if not weighted:
        return "none"
    best = max(weighted.items(), key=lambda kv: (kv[1], -_GUIDANCE_TIE_ORDER.index(kv[0])))
    return best[0]


def _guidance_pick(votes: list[str]) -> str:
    c = Counter(votes)
    return max(c.items(), key=lambda kv: (kv[1], -_GUIDANCE_TIE_ORDER.index(kv[0])))[0]


def _guidance_by_model(units: list[dict]) -> dict[str, str]:
    """Per-model call-level guidance vote over the given units.

    Same weighting + tie-break as :func:`_guidance_vote`, but each model's
    votes are tallied independently (no cross-model majority). Models with
    no non-failed extraction contributing a guidance label are omitted.
    Call sites pass the units they consider authoritative for guidance
    (currently presenter units only, matching ``_consensus``).
    """
    all_models: set[str] = set()
    for u in units:
        all_models.update((u.get("_ext") or {}).keys())
    out: dict[str, str] = {}
    for model in sorted(all_models):
        weighted: Counter[str] = Counter()
        seen_buckets: set[tuple] = set()
        for u in units:
            gid = _dedup_gid(u)
            if gid in seen_buckets:
                continue
            seen_buckets.add(gid)
            rec = (u.get("_ext") or {}).get(model)
            if _is_no_evidence(rec):
                continue
            g = rec.get("guidance")
            if not g:
                continue
            senior = u.get("speaker_role") in ("CEO", "CFO")
            if u.get("kind") == "qa":
                w = _GUIDANCE_WEIGHT_GROUP_SENIOR if senior else _GUIDANCE_WEIGHT_GROUP_OTHER
            else:
                w = _GUIDANCE_WEIGHT_PRESENTER_SENIOR if senior else _GUIDANCE_WEIGHT_PRESENTER_OTHER
            weighted[g] += w
        if not weighted:
            continue
        best = max(weighted.items(), key=lambda kv: (kv[1], -_GUIDANCE_TIE_ORDER.index(kv[0])))
        out[model] = best[0]
    return out


_PHRASE_CLEAN = re.compile(r"[^a-z0-9 ]+")


def _norm_phrase(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PHRASE_CLEAN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = _norm_phrase(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it.strip())
    return out


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _mean_pairwise_jaccard(sets_by_key: dict[str, set[str]]) -> float | None:
    keys = [k for k, v in sets_by_key.items() if v]
    if len(keys) < 2:
        return None
    sims: list[float] = []
    for a, b in combinations(keys, 2):
        sa, sb = sets_by_key[a], sets_by_key[b]
        union = sa | sb
        if not union:
            continue
        sims.append(len(sa & sb) / len(union))
    return sum(sims) / len(sims) if sims else None


def _empty_record(ticker: str, quarter: str, call_date: str | None, reason: str) -> dict:
    return {
        "ticker": ticker,
        "quarter": quarter,
        "call_date": call_date,
        "n_units": {"total": 0, "presenter": 0, "qa": 0},
        "consensus": {},
        "dispersion": {},
        "by_role": {},
        "by_section": {},
        "_skipped": reason,
    }
