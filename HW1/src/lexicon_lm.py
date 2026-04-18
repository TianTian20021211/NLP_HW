"""Loughran-McDonald dictionary sentiment per unit (Part II §2.2 baseline).

Produces a scalar in ``[-1, 1]`` per unit:

    sentiment = (n_pos - n_neg) / (n_pos + n_neg)

with ``sentiment = None`` when the unit contains no dictionary hits (so
the weighted mean in :mod:`aggregate` skips it rather than diluting with
a fake neutral 0). Cache lives at ``data/cache/sentiment_lm/<unit_id>.json``.

Dictionary loading: the Notre Dame "Master Dictionary" CSV encodes
``Positive`` / ``Negative`` as "first year classified" integers — any
non-zero value means the word belongs to that class, ``0`` means it does
not. We filter on ``col > 0`` to build the ``POS`` / ``NEG`` lowercase
word sets.

Tokenization: ``re.findall(r"[a-zA-Z]+", text.lower())``. Deliberately
naive — no lemmatization, hyphenated words are split (``"year-over-year"``
-> ``["year", "over", "year"]``). This matches the LM convention (the
dictionary itself is a flat word list) and makes results trivially
reproducible.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .cache_keys import KEY_FIELD, stage_key
from .io_paths import EXTERNAL, ensure_dirs, error, sentiment_lm_path


DEFAULT_DICT_PATH = EXTERNAL / "LM_dict.csv"
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


@lru_cache(maxsize=2)
def load_lm_dict(path: str | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(POS, NEG)`` lowercase word sets from the Master Dictionary CSV.

    Cached so the 86k-row CSV is parsed at most once per process. ``path``
    defaults to ``data/external/LM_dict.csv``; override it in tests only.
    """
    import csv

    p = Path(path) if path else DEFAULT_DICT_PATH
    if not p.is_file():
        error(
            f"Loughran-McDonald dict missing: {p}. Download Master Dictionary "
            "CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ "
            "and place it there."
        )
    pos: set[str] = set()
    neg: set[str] = set()
    with p.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if "Word" not in reader.fieldnames or "Positive" not in reader.fieldnames or "Negative" not in reader.fieldnames:
            error(f"LM_dict.csv missing required columns; got {reader.fieldnames}")
        for row in reader:
            w = (row.get("Word") or "").strip().lower()
            if not w:
                continue
            try:
                pos_year = int(row.get("Positive") or 0)
                neg_year = int(row.get("Negative") or 0)
            except ValueError:
                continue
            if pos_year > 0:
                pos.add(w)
            if neg_year > 0:
                neg.add(w)
    if not pos or not neg:
        error(f"LM_dict.csv parse produced empty POS/NEG sets: |pos|={len(pos)} |neg|={len(neg)}")
    return frozenset(pos), frozenset(neg)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def score_unit_lm(text: str, pos: Iterable[str] | None = None, neg: Iterable[str] | None = None) -> dict:
    """Score a single text blob against the LM dictionary.

    Returns ``{sentiment, n_pos, n_neg, n_words}``. When the text has no
    dictionary hits (``n_pos + n_neg == 0``) the ``sentiment`` field is
    ``None`` so downstream weighted means can skip this unit.
    """
    if pos is None or neg is None:
        pos_set, neg_set = load_lm_dict()
    else:
        pos_set = frozenset(pos)
        neg_set = frozenset(neg)
    toks = _tokenize(text or "")
    n_pos = 0
    n_neg = 0
    for t in toks:
        if t in pos_set:
            n_pos += 1
        elif t in neg_set:
            n_neg += 1
    total = n_pos + n_neg
    sentiment = (n_pos - n_neg) / total if total > 0 else None
    return {
        "sentiment": sentiment,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_words": len(toks),
    }


def score_units_lm(units: Sequence[dict], overwrite: bool = False) -> list[dict]:
    """Batch-score every unit and cache to ``data/cache/sentiment_lm/<uid>.json``.

    Cache behavior mirrors :func:`sentiment_finbert.score_units`: an existing
    file whose ``_cache_key`` still matches the current stage fingerprint is
    reused as-is; stale or missing files trigger a recompute. Returns the
    cached dicts in the same order as ``units``.
    """
    ensure_dirs()
    pos_set, neg_set = load_lm_dict()
    key = stage_key("sentiment_lm")

    results: list[dict | None] = [None] * len(units)
    pending_idx: list[int] = []
    pending_texts: list[str] = []
    pending_uids: list[str] = []

    for i, u in enumerate(units):
        uid = u["unit_id"]
        p = sentiment_lm_path(uid)
        if p.is_file() and not overwrite:
            try:
                cached = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict) and cached.get(KEY_FIELD) == key:
                results[i] = cached
                continue
        pending_idx.append(i)
        pending_texts.append(u.get("text", "") or "")
        pending_uids.append(uid)

    for i, uid, text in zip(pending_idx, pending_uids, pending_texts):
        rec = {"unit_id": uid, **score_unit_lm(text, pos_set, neg_set), KEY_FIELD: key}
        sentiment_lm_path(uid).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        results[i] = rec

    out: list[dict] = []
    for r in results:
        if r is None:
            error("score_units_lm: internal bookkeeping error (unit left unscored)")
        out.append(r)  # type: ignore[arg-type]
    return out


def load_sentiment_lm(unit_id: str) -> dict:
    """Load a cached LM record. Errors if missing or stale."""
    p = sentiment_lm_path(unit_id)
    if not p.is_file():
        error(f"sentiment_lm cache missing: {p}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get(KEY_FIELD) != stage_key("sentiment_lm"):
        error(f"sentiment_lm cache stale: {p}")
    return rec
