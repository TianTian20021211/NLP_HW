"""Batched Stage-C driver for single-GPU hosts.

Presenter units still go through :func:`src.extract_llm.extract_unit`
(one LLM call per unit). Q&A units are grouped by ``(call, speaker_role)``
and each group is sent through :func:`src.extract_qa_group.extract_qa_group`
as a single batched LLM call — so one CEO contributes one extraction
instead of one per answer. This typically cuts Stage-C requests by
~4x on our corpus (most calls have 15-30 Q&A answers but only 2-3
officers) without losing per-unit FinBERT sentiment.

Model iteration order is OUTERMOST so a single model stays resident in
VRAM for its whole-corpus pass — critical on hosts where the two
models (gemma3:4b + llama3.1:8b) don't fit together on one GPU.
Within a model, up to ``concurrency`` requests are in flight at once;
Ollama batches them on the GPU via ``OLLAMA_NUM_PARALLEL`` while we
overlap tokenization, JSON decode, and disk I/O.

Deliberately kept out of :mod:`extract_llm` so that file's AST
fingerprint (which keys the per-unit extraction cache) stays stable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Sequence

from tqdm.auto import tqdm

from .extract_llm import DEFAULT_MODELS, extract_unit
from .extract_qa_group import extract_qa_group


def _group_qa(units: Iterable[dict]) -> list[list[dict]]:
    """Group QA units by ``(ticker, quarter, speaker_role)``; preserve order."""
    buckets: dict[tuple[str, str, str], list[dict]] = {}
    order: list[tuple[str, str, str]] = []
    for u in units:
        if u.get("kind") != "qa":
            continue
        key = (
            u.get("ticker", ""),
            u.get("quarter", ""),
            u.get("speaker_role") or "Unknown",
        )
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(u)
    return [buckets[k] for k in order]


def extract_many(
    units: Iterable[dict],
    models: Sequence[str] = DEFAULT_MODELS,
    concurrency: int = 4,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    """Run every model across the corpus with presenter/QA dispatch.

    Returns ``{model -> {"presenter": [...], "qa_group": [...]}}``. The
    actual consumer of these extractions is :func:`src.aggregate`, which
    reads the per-unit and per-group caches on disk; the return value is
    diagnostic.
    """
    unit_list = list(units)
    presenter_units = [u for u in unit_list if u.get("kind") != "qa"]
    qa_groups = _group_qa(unit_list)

    results: dict[str, dict[str, list[dict]]] = {}
    workers = max(1, concurrency)
    for m in models:
        pres_out: list[dict | None] = [None] * len(presenter_units)
        grp_out: list[dict | None] = [None] * len(qa_groups)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict = {}
            for i, u in enumerate(presenter_units):
                futures[pool.submit(extract_unit, u, m, overwrite)] = ("p", i)
            for i, g in enumerate(qa_groups):
                futures[pool.submit(extract_qa_group, g, m, overwrite)] = ("g", i)

            iterator = as_completed(futures)
            if progress:
                desc = f"LLM {m}  ({len(presenter_units)} pres + {len(qa_groups)} qa-groups)"
                iterator = tqdm(iterator, total=len(futures), desc=desc)
            for fut in iterator:
                kind, i = futures[fut]
                if kind == "p":
                    pres_out[i] = fut.result()
                else:
                    grp_out[i] = fut.result()

        results[m] = {
            "presenter": [r for r in pres_out if r is not None],
            "qa_group": [r for r in grp_out if r is not None],
        }
    return results
