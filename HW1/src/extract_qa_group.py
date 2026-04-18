"""Stage C batched extraction: one Q&A officer -> one LLM call.

Groups every Q&A answer by the same ``speaker_role`` inside a call and
sends the whole block as a single ``/api/chat`` request. The LLM is
asked to summarize ACROSS the officer's answers (top wins / risks /
guidance) rather than per-answer.

Why a separate module from :mod:`extract_llm`:

``cache_keys.stage_key('extraction', <model>)`` hashes ``extract_llm.py``
so any edit to that module invalidates every per-unit extraction cache.
The batched path has its own stage (``extraction_qa_group``) whose key
tracks only this file — that way iterating on the group prompt does
not wipe the presenter caches that still live under the old stage.
"""

from __future__ import annotations

import json
import time
from typing import Sequence

from .cache_keys import KEY_FIELD, stage_key
from .extract_llm import _MAX_ITEMS, _MAX_STR_LEN, _normalize
from .io_paths import (
    ensure_dirs,
    error,
    qa_group_extraction_path,
    qa_group_failure_path,
)


_SYSTEM_PROMPT = (
    "You extract structured information from ONE officer's entire Q&A "
    "contribution during an earnings call. Reply with ONE JSON object "
    "and nothing else."
)


def _group_prompt(units: Sequence[dict]) -> str:
    """Build the batched prompt listing every (question, answer) for one officer."""
    if not units:
        error("_group_prompt: empty unit list")
    role = units[0].get("speaker_role", "Unknown")
    title = units[0].get("speaker_title") or ""
    header_bits = [f"Officer role: {role}"]
    if title:
        header_bits.append(f"Officer title: {title}")
    header_bits.append(
        "Below are every (analyst question, officer answer) pair this "
        "officer gave during the Q&A portion of this earnings call, in "
        "the order they occurred."
    )
    lines: list[str] = header_bits + [""]
    for i, u in enumerate(units, 1):
        q = (u.get("question_text") or "").strip()
        a = (u.get("text") or "").strip()
        if q:
            lines.append(f"[Q{i}] {q[:600]}")
        lines.append(f"[A{i}] {a}")
        lines.append("")
    lines.append("Summarize ACROSS ALL of this officer's answers above:")
    lines.append(
        f'- "wins": up to {_MAX_ITEMS} short noun phrases describing the '
        "MOST important positive achievements this officer explicitly "
        "emphasized. Empty list if none. Do not invent."
    )
    lines.append(
        f'- "risks": up to {_MAX_ITEMS} short noun phrases describing the '
        "MOST important risks, concerns, or weaknesses this officer "
        "explicitly mentioned. Empty list if none. Do not invent."
    )
    lines.append(
        '- "guidance": one of "raised", "maintained", "lowered", "none". '
        'Use "none" unless this officer gave forward guidance somewhere '
        "in their answers."
    )
    lines.append("")
    lines.append(
        'Return JSON exactly: {"wins":[...], "risks":[...], "guidance":"..."}'
    )
    return "\n".join(lines)


def _ollama_chat_qa(model: str, user: str, temperature: float) -> str:
    """Thin wrapper around :func:`extract_llm._ollama_chat` using the QA system prompt.

    We override the system prompt to steer the model to cross-answer
    summarization semantics; everything else (timeout, ``think=false``,
    ``format=json``) is inherited.
    """
    import requests

    from .extract_llm import OLLAMA_URL

    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    try:
        r = requests.post(url, json=payload, timeout=180.0)
    except requests.RequestException as e:
        error(f"ollama request failed: {e}")
    if r.status_code != 200:
        error(f"ollama {r.status_code}: {r.text[:500]}")
    data = r.json()
    return (data.get("message") or {}).get("content", "")


def _call_id(unit: dict) -> str:
    return f"{unit['ticker']}_{unit['quarter']}"


def _normalize_group(raw: dict) -> dict:
    """Same shape as :func:`extract_llm._normalize` but enforces group caps.

    The per-unit normalizer already truncates to ``_MAX_ITEMS`` phrases of
    ``_MAX_STR_LEN`` chars each; we rely on that and simply delegate.
    """
    return _normalize(raw)


def extract_qa_group(
    units: Sequence[dict],
    model: str,
    overwrite: bool = False,
) -> dict:
    """Extract one ``{wins, risks, guidance}`` record for an officer's QA contribution.

    ``units`` must all share ``(ticker, quarter, speaker_role)`` and must
    all be ``kind="qa"``. The resulting record is cached at
    ``data/cache/extractions/<model_slug>/qa_groups/<call>__<role>.json``.

    Empty-text groups short-circuit to a default record. On three JSON /
    HTTP failures we write a sentinel ``_failed=True`` record and dump
    the offending prompt under ``qa_groups/_failures/``.
    """
    ensure_dirs()
    if not units:
        error("extract_qa_group: empty unit list")
    first = units[0]
    call_id = _call_id(first)
    role = first.get("speaker_role") or "Unknown"
    p = qa_group_extraction_path(model, call_id, role)
    key = stage_key("extraction_qa_group", model)

    if p.is_file() and not overwrite:
        cached = json.loads(p.read_text(encoding="utf-8"))
        if cached.get(KEY_FIELD) == key:
            return cached

    non_empty = [u for u in units if (u.get("text") or "").strip()]
    if not non_empty:
        rec = {
            "call_id": call_id,
            "role": role,
            "model": model,
            "n_answers": len(units),
            "unit_ids": [u["unit_id"] for u in units],
            "wins": [],
            "risks": [],
            "guidance": "none",
            "_empty": True,
            KEY_FIELD: key,
        }
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec

    user = _group_prompt(non_empty)
    last_err: str | None = None
    for attempt, temp in enumerate((0.0, 0.1, 0.1)):
        try:
            raw = _ollama_chat_qa(model=model, user=user, temperature=temp)
            parsed = json.loads(raw)
            norm = _normalize_group(parsed)
            rec = {
                "call_id": call_id,
                "role": role,
                "model": model,
                "n_answers": len(units),
                "unit_ids": [u["unit_id"] for u in units],
                **norm,
                KEY_FIELD: key,
            }
            p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            return rec
        except (json.JSONDecodeError, RuntimeError) as e:
            last_err = f"attempt {attempt} ({temp}): {e}"
            time.sleep(0.3)

    fpath = qa_group_failure_path(model, call_id, role)
    fpath.write_text(
        f"{last_err}\n----\n{user[:_MAX_STR_LEN * 20]}\n", encoding="utf-8"
    )
    rec = {
        "call_id": call_id,
        "role": role,
        "model": model,
        "n_answers": len(units),
        "unit_ids": [u["unit_id"] for u in units],
        "wins": [],
        "risks": [],
        "guidance": "none",
        "_failed": True,
        KEY_FIELD: key,
    }
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def load_qa_group(model: str, call_id: str, role: str) -> dict | None:
    """Return cached batched-QA extraction for ``(model, call, role)`` or None.

    A mismatch between the stored ``_cache_key`` and the current
    ``stage_key('extraction_qa_group', model)`` is treated as a miss, so
    downstream aggregators never mix results from different prompt revs.
    """
    p = qa_group_extraction_path(model, call_id, role)
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get(KEY_FIELD) != stage_key("extraction_qa_group", model):
        return None
    return rec
