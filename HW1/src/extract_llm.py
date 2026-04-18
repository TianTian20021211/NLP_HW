"""Ollama LLM zero-shot extraction for wins / risks / guidance.

Each unit is sent to a local Ollama endpoint (`http://localhost:11434`)
with `format=json` and `think=false`. The JSON response is validated,
truncated, and cached per model per unit at
`data/cache/extractions/<model_slug>/<unit_id>.json`.

Failures are retried up to twice with temperature 0.1; if still broken,
a sentinel `_failed=True` record is stored so the pipeline keeps moving
(Plan 1.6).
"""

from __future__ import annotations

import json
import os
import time
from typing import Sequence

import requests

from .cache_keys import KEY_FIELD, stage_key
from .io_paths import (
    ensure_dirs,
    error,
    extraction_failure_path,
    extraction_path,
)


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = ("gemma3:4b", "llama3.1:8b")
_GUIDANCE_VALUES = ("raised", "maintained", "lowered", "none")
_MAX_ITEMS = 3
_MAX_STR_LEN = 160

_SYSTEM_PROMPT = (
    "You extract structured information from a single segment of an earnings call. "
    "Reply with ONE JSON object and nothing else."
)


def _user_prompt(unit: dict) -> str:
    """Build the zero-shot user prompt per Plan 1.6 spec."""
    lines: list[str] = [
        f"Speaker role: {unit.get('speaker_role', 'Unknown')}",
        f"Segment kind: {unit.get('kind', 'presenter')}",
    ]
    if unit.get("kind") == "qa" and unit.get("question_text"):
        lines.append(f"Analyst question: {unit['question_text'][:800]}")
    lines.append("Segment text:")
    lines.append('"""')
    lines.append(unit.get("text", ""))
    lines.append('"""')
    lines.append("")
    lines.append("Extract:")
    lines.append(
        '- "wins": up to 3 short noun phrases describing positive achievements '
        "explicitly stated in this segment. Empty list if none. Do not invent."
    )
    lines.append(
        '- "risks": up to 3 short noun phrases describing risks, concerns, or '
        "weaknesses explicitly stated. Empty list if none. Do not invent."
    )
    lines.append(
        '- "guidance": one of "raised", "maintained", "lowered", "none". '
        'Use "none" if this segment does not contain forward guidance.'
    )
    lines.append("")
    lines.append('Return JSON exactly: {"wins":[...], "risks":[...], "guidance":"..."}')
    return "\n".join(lines)


def _ollama_chat(model: str, user: str, temperature: float, timeout: float = 120.0) -> str:
    """POST to `/api/chat`; return the raw `message.content` string.

    Raises `RuntimeError` on HTTP failure so the retry wrapper can catch it.
    """
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
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        error(f"ollama request failed: {e}")
    if r.status_code != 200:
        error(f"ollama {r.status_code}: {r.text[:500]}")
    data = r.json()
    msg = data.get("message") or {}
    return msg.get("content", "")


def _normalize(raw: dict) -> dict:
    """Coerce a raw JSON dict into the strict `{wins,risks,guidance}` shape.

    Accepts common near-miss shapes (`guidance=None`, `wins` as a string, etc.)
    and falls back to safe defaults (`[]` / `"none"`) with no `_failed` flag.
    """
    wins = raw.get("wins", [])
    risks = raw.get("risks", [])
    guidance = raw.get("guidance", "none")

    wins = _coerce_list(wins)
    risks = _coerce_list(risks)
    if not isinstance(guidance, str):
        guidance = "none"
    guidance = guidance.strip().lower()
    if guidance not in _GUIDANCE_VALUES:
        guidance = "none"
    return {"wins": wins, "risks": risks, "guidance": guidance}


def _coerce_list(val: object) -> list[str]:
    """Turn `val` into a clean `list[str]` of ≤ 3 short phrases."""
    if val is None:
        return []
    if isinstance(val, str):
        items = [val]
    elif isinstance(val, list):
        items = [str(x) for x in val if isinstance(x, (str, int, float)) and str(x).strip()]
    else:
        items = []
    items = [i.strip().strip('"').strip("'") for i in items if i and i.strip()]
    items = [i[:_MAX_STR_LEN] for i in items]
    return items[:_MAX_ITEMS]


def extract_unit(unit: dict, model: str, overwrite: bool = False) -> dict:
    """Extract (and cache) `wins/risks/guidance` for one unit with one model.

    Returns the cached dict. Empty-text units short-circuit to the default
    record without hitting the LLM. Cached records whose ``_cache_key`` no
    longer matches the current ``extract_llm.py`` fingerprint are treated
    as stale and recomputed.
    """
    ensure_dirs()
    uid = unit["unit_id"]
    p = extraction_path(model, uid)
    key = stage_key("extraction", model)
    if p.is_file() and not overwrite:
        cached = json.loads(p.read_text(encoding="utf-8"))
        if cached.get(KEY_FIELD) == key:
            return cached

    if not unit.get("text", "").strip():
        rec = {
            "unit_id": uid,
            "model": model,
            "wins": [],
            "risks": [],
            "guidance": "none",
            "_empty": True,
            KEY_FIELD: key,
        }
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec

    user = _user_prompt(unit)
    last_err: str | None = None
    for attempt, temp in enumerate((0.0, 0.1, 0.1)):
        try:
            raw = _ollama_chat(model=model, user=user, temperature=temp)
            parsed = json.loads(raw)
            norm = _normalize(parsed)
            rec = {"unit_id": uid, "model": model, **norm, KEY_FIELD: key}
            p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            return rec
        except (json.JSONDecodeError, RuntimeError) as e:
            last_err = f"attempt {attempt} ({temp}): {e}"
            time.sleep(0.3)

    fpath = extraction_failure_path(model, uid)
    fpath.write_text(f"{last_err}\n----\n{user[:2000]}\n", encoding="utf-8")
    rec = {
        "unit_id": uid,
        "model": model,
        "wins": [],
        "risks": [],
        "guidance": "none",
        "_failed": True,
        KEY_FIELD: key,
    }
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def extract_units(
    units: Sequence[dict],
    models: Sequence[str] = DEFAULT_MODELS,
    overwrite: bool = False,
) -> dict[str, list[dict]]:
    """Run every `model` over every `unit`, returning `{model -> [records]}`."""
    results: dict[str, list[dict]] = {m: [] for m in models}
    for m in models:
        for u in units:
            results[m].append(extract_unit(u, model=m, overwrite=overwrite))
    return results


def load_extraction(model: str, unit_id: str) -> dict | None:
    """Return the cached extraction for `(model, unit_id)` or None.

    Stale records (``_cache_key`` mismatches the current source fingerprint)
    are also reported as missing, so downstream aggregators do not mix
    results from different code revisions.
    """
    p = extraction_path(model, unit_id)
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get(KEY_FIELD) != stage_key("extraction", model):
        return None
    return rec


def ollama_available(models: Sequence[str] = DEFAULT_MODELS, timeout: float = 3.0) -> tuple[bool, str]:
    """Return `(ok, detail)` pinging `/api/tags` and checking models exist."""
    url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        return False, f"no ollama at {OLLAMA_URL}: {e}"
    if r.status_code != 200:
        return False, f"ollama {r.status_code}"
    names = {m.get("name", "") for m in r.json().get("models", [])}
    missing = [m for m in models if m not in names and not any(n.startswith(m) for n in names)]
    if missing:
        return False, f"missing models: {missing}; have: {sorted(names)}"
    return True, f"ok; {len(names)} models"
