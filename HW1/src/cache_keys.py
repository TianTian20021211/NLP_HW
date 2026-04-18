"""Source-fingerprint cache invalidation.

Every cache-writing stage embeds a ``_cache_key`` field (or a sidecar
``<file>.key`` for the jsonl units cache) that hashes the source files it
depends on. Readers compare the cached key with the current ``stage_key``
and treat any mismatch as a miss, which triggers a recompute-and-overwrite.

Use :func:`prune_stale` from a notebook cell to proactively sweep stale
files off disk; the read-path check already guarantees correctness, so
pruning is purely for tidiness.

Stage -> source-file dependency table (declared in ``_STAGE_SOURCES``):

* ``units``      — ``parser.py``, ``roles.py``
* ``sentiment``  — ``sentiment_finbert.py``
* ``extraction`` — ``extract_llm.py`` (plus the model slug as an ``extra``)
* ``calls``      — ``aggregate.py``
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .io_paths import (
    CALLS,
    CALLS_LM,
    EXTRACTIONS,
    FEATURES,
    LABELS,
    SENTIMENT,
    SENTIMENT_LM,
    UNITS,
    error,
)


_SRC = Path(__file__).resolve().parent

_STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "units": ("parser.py", "roles.py"),
    "sentiment": ("sentiment_finbert.py",),
    "sentiment_lm": ("lexicon_lm.py",),
    "extraction": ("extract_llm.py",),
    "extraction_qa_group": ("extract_qa_group.py",),
    "calls": ("aggregate.py",),
    "calls_lm": ("aggregate.py", "lexicon_lm.py"),
    "features": ("features.py", "aggregate.py"),
    "labels": ("labels.py",),
}

KEY_FIELD = "_cache_key"
_KEY_LEN = 16


def _ast_fingerprint(path: Path) -> bytes:
    """SHA256 over the AST dump of `path`.

    Using the AST rather than raw bytes means comment-only or whitespace-only
    edits do not invalidate caches. If the file is mid-edit and fails to
    parse we fall back to raw bytes so the pipeline still makes progress.
    """
    data = path.read_bytes()
    try:
        tree = ast.parse(data, filename=str(path))
    except SyntaxError:
        return hashlib.sha256(data).digest()
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).digest()


def stage_key(stage: str, *extras: str) -> str:
    """Return a short fingerprint for ``stage`` given the current source tree.

    ``extras`` get hashed in alongside the source files — use it for
    parameters that change cache semantics without living in a source file
    (e.g. the model slug for an extraction record).
    """
    if stage not in _STAGE_SOURCES:
        error(f"unknown cache stage: {stage}")
    h = hashlib.sha256()
    h.update(stage.encode("utf-8"))
    for name in _STAGE_SOURCES[stage]:
        h.update(b"\x00")
        h.update(_ast_fingerprint(_SRC / name))
    for e in extras:
        h.update(b"\x00")
        h.update(e.encode("utf-8"))
    return h.hexdigest()[:_KEY_LEN]


def is_fresh(rec: dict, stage: str, *extras: str) -> bool:
    """True iff ``rec[KEY_FIELD]`` matches the current stage key."""
    return isinstance(rec, dict) and rec.get(KEY_FIELD) == stage_key(stage, *extras)


def _sidecar_path(primary: Path) -> Path:
    return primary.parent / (primary.name + ".key")


def write_sidecar(primary: Path, stage: str, *extras: str) -> None:
    """Write the current stage key next to ``primary`` as ``<primary>.key``.

    Used for the units jsonl where embedding a key inside every line would
    be noisy; one sidecar per call is enough.
    """
    _sidecar_path(primary).write_text(stage_key(stage, *extras), encoding="utf-8")


def read_sidecar(primary: Path) -> str | None:
    """Return the sidecar key for ``primary`` or ``None`` if missing/empty."""
    p = _sidecar_path(primary)
    if not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8").strip()
    return txt or None


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    return root.glob("*.json")


def _prune_units(key: str) -> int:
    if not UNITS.is_dir():
        return 0
    deleted = 0
    for jl in UNITS.glob("*.jsonl"):
        if read_sidecar(jl) == key:
            continue
        jl.unlink(missing_ok=True)
        _sidecar_path(jl).unlink(missing_ok=True)
        deleted += 1
    return deleted


def _prune_json_dir(root: Path, key: str) -> int:
    deleted = 0
    for p in _iter_json_files(root):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            cached = rec.get(KEY_FIELD) if isinstance(rec, dict) else None
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached == key:
            continue
        p.unlink(missing_ok=True)
        deleted += 1
    return deleted


def _prune_extraction(model: str, key: str) -> int:
    slug = model.replace("/", "_").replace(":", "_")
    root = EXTRACTIONS / slug
    if not root.is_dir():
        return 0
    deleted = _prune_json_dir(root, key)
    failures = root / "_failures"
    if failures.is_dir():
        for f in failures.glob("*.txt"):
            f.unlink(missing_ok=True)
            deleted += 1
    return deleted


def _prune_extraction_qa_group(model: str, key: str) -> int:
    slug = model.replace("/", "_").replace(":", "_")
    root = EXTRACTIONS / slug / "qa_groups"
    if not root.is_dir():
        return 0
    deleted = _prune_json_dir(root, key)
    failures = root / "_failures"
    if failures.is_dir():
        for f in failures.glob("*.txt"):
            f.unlink(missing_ok=True)
            deleted += 1
    return deleted


def prune_stale(stage: str, *extras: str) -> int:
    """Delete every cache file for ``stage`` whose key does not match.

    Missing-key files (legacy caches written before this module existed)
    are treated as stale and removed. Returns the number of files deleted.

    Usage::

        prune_stale("units")
        prune_stale("sentiment")
        for m in MODELS:
            prune_stale("extraction", m)
        prune_stale("calls")
    """
    key = stage_key(stage, *extras)
    if stage == "units":
        return _prune_units(key)
    if stage == "sentiment":
        return _prune_json_dir(SENTIMENT, key)
    if stage == "sentiment_lm":
        return _prune_json_dir(SENTIMENT_LM, key)
    if stage == "extraction":
        if not extras:
            error("prune_stale('extraction') requires a model slug, e.g. prune_stale('extraction', 'gemma3:4b')")
        return _prune_extraction(extras[0], key)
    if stage == "extraction_qa_group":
        if not extras:
            error("prune_stale('extraction_qa_group') requires a model slug")
        return _prune_extraction_qa_group(extras[0], key)
    if stage == "calls":
        return _prune_json_dir(CALLS, key)
    if stage == "calls_lm":
        return _prune_json_dir(CALLS_LM, key)
    if stage == "features":
        return _prune_parquet_dir(FEATURES, key)
    if stage == "labels":
        return _prune_parquet_dir(LABELS, key)
    error(f"prune_stale: unsupported stage {stage}")
    return 0


def _prune_parquet_dir(root: Path, key: str) -> int:
    """Wipe every parquet under ``root`` that carries a stale sidecar key.

    Features / labels parquets keep the current stage key in a sibling
    ``<file>.key`` sidecar (same convention as units jsonl) because the
    parquet schema itself has no convenient slot for metadata.
    """
    if not root.is_dir():
        return 0
    deleted = 0
    for p in root.glob("*.parquet"):
        if read_sidecar(p) == key:
            continue
        p.unlink(missing_ok=True)
        _sidecar_path(p).unlink(missing_ok=True)
        deleted += 1
    return deleted
