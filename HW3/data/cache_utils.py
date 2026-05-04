"""Shared cache-manifest utilities for pipeline-phase result caching.

Each pipeline phase writes a lightweight JSON manifest alongside its output
artifacts.  ``run_all.py`` reads the manifest before launching the phase
subprocess and skips the phase when all input hashes still match — turning
the pipeline into a content-addressed incremental build.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

CACHE_MANIFEST_VERSION = 1


@dataclass
class CacheManifest:
    """Lightweight content-addressed manifest for one pipeline artifact.

    Every field that could change the output is captured: file hashes,
    function source-code hashes, parameter values, and a schema version.
    """

    cache_version: int
    computed_at: str
    phase: str
    parameters: dict[str, Any]
    input_hashes: dict[str, str]  # str(path) → sha256 hex
    source_hash: str


def build_cache_manifest(
    phase: str,
    parameters: dict[str, Any],
    input_paths: Sequence[Path],
    source_funcs: Sequence[Callable],
) -> CacheManifest:
    """Hash every input file and every source function into a manifest."""
    input_hashes: dict[str, str] = {}
    for p in input_paths:
        if p.exists():
            input_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            input_hashes[str(p)] = ""

    try:
        source_blob = "\n\n".join(inspect.getsource(f) for f in source_funcs)
        source_hash = hashlib.sha256(source_blob.encode()).hexdigest()[:16]
    except (OSError, TypeError):
        source_hash = "0" * 16

    return CacheManifest(
        cache_version=CACHE_MANIFEST_VERSION,
        computed_at=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        parameters=parameters,
        input_hashes=input_hashes,
        source_hash=source_hash,
    )


def write_cache_manifest(manifest: CacheManifest, path: Path) -> None:
    """Write *manifest* as JSON to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2))


_hash_cache: dict[str, str] = {}


def validate_cache_manifest(manifest_path: Path) -> bool:
    """Return True if every input file hash in the manifest still matches."""
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    if data.get("cache_version") != CACHE_MANIFEST_VERSION:
        return False

    input_hashes: dict[str, str] = data.get("input_hashes", {})
    if not input_hashes:
        return False

    for path_str, expected_hash in input_hashes.items():
        cached = _hash_cache.get(path_str)
        if cached is not None:
            if cached != expected_hash:
                return False
            continue
        p = Path(path_str)
        if not p.exists():
            return False
        current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        _hash_cache[path_str] = current_hash
        if current_hash != expected_hash:
            return False

    return True
