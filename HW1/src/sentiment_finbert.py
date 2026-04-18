"""FinBERT sentiment per unit.

Produces a scalar in ``[-1, 1]`` per unit: length-weighted mean of
``P(positive) - P(negative)`` over sentence-level (or 512-token-window)
chunks. Cache lives at ``data/cache/sentiment/<unit_id>.json``.

Implementation notes (Stage B hot loop):

* Tokenises every unit up-front, flattens all windows across the whole
  input list, then runs FinBERT in a single length-sorted batch loop.
  This keeps the GPU saturated for short QA turns that would otherwise
  each produce a tiny one- or two-window batch.
* Defaults to ``cuda`` + ``bfloat16`` when available (Ada/Ampere have
  native BF16 Tensor Cores). Falls back to ``cpu`` + ``float32``.
* No tokenizer decode/re-encode round-trip: window token ids are padded
  directly into the model's expected ``input_ids`` layout.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Iterable, Sequence

import numpy as np

from .cache_keys import KEY_FIELD, stage_key
from .io_paths import SENTIMENT, error, ensure_dirs, sentiment_path


MODEL_NAME = "ProsusAI/finbert"
_MAX_TOKENS = 512
_STRIDE = 256
_DEFAULT_BATCH = 32

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` on sentence terminators followed by a capitalized token.

    Good enough for FinBERT windowing; we are not doing syntactic parsing,
    just preventing one run-on 5000-char unit from dominating the average.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(device: str, explicit):  # -> torch.dtype
    import torch

    if explicit is not None:
        return explicit
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class FinBertScorer:
    """Lazily-loaded FinBERT wrapper with global-batch inference.

    Construct once, then feed a whole unit list to :meth:`score_many` (or a
    single text to :meth:`score_unit`). The model and tokenizer are loaded
    on first use and reused across calls.

    Parameters
    ----------
    model_name : str
        HF repo id. Defaults to ProsusAI/finbert.
    device : str | None
        Torch device string. Defaults to ``"cuda"`` if available else ``"cpu"``.
    dtype : torch.dtype | None
        Weight dtype. Defaults to bfloat16 on Ampere/Ada, float16 on older
        CUDA GPUs, and float32 on CPU.
    batch_size : int
        GPU micro-batch size across flattened windows. 32 keeps peak VRAM
        well under 2 GB for BERT-base at 512 tokens; bump to 64 if you have
        headroom.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        dtype=None,
        batch_size: int = _DEFAULT_BATCH,
    ):
        self.model_name = model_name
        self.device = _pick_device(device)
        self.dtype = _pick_dtype(self.device, dtype)
        self.batch_size = int(batch_size)
        self._tok = None
        self._model = None
        self._label_order: list[str] | None = None
        self._pos_idx: int = -1
        self._neg_idx: int = -1
        self._cls_id: int = 0
        self._sep_id: int = 0
        self._pad_id: int = 0

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, dtype=self.dtype
        )
        model.eval()
        model.to(self.device)
        self._model = model

        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self._label_order = [id2label[i] for i in range(len(id2label))]
        self._pos_idx = self._label_order.index("positive")
        self._neg_idx = self._label_order.index("negative")

        self._cls_id = int(self._tok.cls_token_id)
        self._sep_id = int(self._tok.sep_token_id)
        self._pad_id = int(self._tok.pad_token_id or 0)

    def _windows_for_sentence(self, sentence: str) -> Iterable[list[int]]:
        """Yield 510-token windows covering ``sentence`` (no special tokens).

        Long sentences get a sliding window of stride 256 so the tail still
        contributes to the unit's score. Empty sentences yield nothing.
        """
        assert self._tok is not None
        ids = self._tok.encode(sentence, add_special_tokens=False)
        if not ids:
            return
        budget = _MAX_TOKENS - 2
        if len(ids) <= budget:
            yield ids
            return
        for start in range(0, len(ids), _STRIDE):
            chunk = ids[start : start + budget]
            if not chunk:
                break
            yield chunk
            if start + budget >= len(ids):
                break

    def _windows_for_text(self, text: str) -> list[list[int]]:
        """Flatten sentence splitting + windowing into one list of token-id windows."""
        if not text or not text.strip():
            return []
        self._lazy_load()
        out: list[list[int]] = []
        for s in _split_sentences(text):
            for w in self._windows_for_sentence(s):
                out.append(w)
        return out

    def score_unit(self, text: str) -> dict:
        """Return ``{"sentiment": float|None, "n_windows": int, "model": str}``.

        Empty or whitespace text returns ``sentiment=None``. Scores are clipped
        to ``[-1, 1]`` to guard against floating-point drift in softmax.
        """
        return self.score_many([text])[0]

    def score_many(self, texts: Sequence[str]) -> list[dict]:
        """Batch-score a list of texts in a single GPU sweep.

        Windows across all texts are flattened and sorted by length so that
        each micro-batch pads to its bucket's max length rather than the
        corpus max. Per-text results preserve input order.
        """
        self._lazy_load()
        n = len(texts)
        if n == 0:
            return []

        per_text_windows: list[list[list[int]]] = [self._windows_for_text(t) for t in texts]

        flat_ids: list[list[int]] = []
        flat_text_idx: list[int] = []
        flat_weights: list[int] = []
        for ti, windows in enumerate(per_text_windows):
            for w in windows:
                flat_ids.append(w)
                flat_text_idx.append(ti)
                flat_weights.append(len(w))

        totals = np.zeros(n, dtype=np.float64)
        total_w = np.zeros(n, dtype=np.int64)

        if flat_ids:
            self._run_inference(flat_ids, flat_text_idx, flat_weights, totals, total_w)

        results: list[dict] = []
        for ti, windows in enumerate(per_text_windows):
            model_name = self.model_name
            if not windows:
                results.append({"sentiment": None, "n_windows": 0, "model": model_name})
                continue
            if total_w[ti] == 0:
                results.append({"sentiment": None, "n_windows": len(windows), "model": model_name})
                continue
            score = float(totals[ti] / total_w[ti])
            score = max(-1.0, min(1.0, score))
            results.append({"sentiment": score, "n_windows": len(windows), "model": model_name})
        return results

    def _run_inference(
        self,
        flat_ids: list[list[int]],
        flat_text_idx: list[int],
        flat_weights: list[int],
        totals: np.ndarray,
        total_w: np.ndarray,
    ) -> None:
        """Push all flattened windows through FinBERT and accumulate into `totals`.

        Windows are sorted by length descending before being sliced into
        micro-batches of ``self.batch_size``, which keeps padding close to
        the true window length for each batch.
        """
        import torch

        assert self._model is not None
        order = sorted(range(len(flat_ids)), key=lambda i: len(flat_ids[i]), reverse=True)
        bs = self.batch_size
        pad_id = self._pad_id
        cls_id = self._cls_id
        sep_id = self._sep_id

        for start in range(0, len(order), bs):
            chunk = order[start : start + bs]
            batch_ids = [flat_ids[i] for i in chunk]
            max_core = max(len(b) for b in batch_ids)
            max_len = max_core + 2

            ids_np = np.full((len(batch_ids), max_len), pad_id, dtype=np.int64)
            attn_np = np.zeros((len(batch_ids), max_len), dtype=np.int64)
            for bi, ids in enumerate(batch_ids):
                L = len(ids)
                ids_np[bi, 0] = cls_id
                ids_np[bi, 1 : 1 + L] = ids
                ids_np[bi, 1 + L] = sep_id
                attn_np[bi, : L + 2] = 1

            input_ids = torch.from_numpy(ids_np).to(self.device, non_blocking=True)
            attention_mask = torch.from_numpy(attn_np).to(self.device, non_blocking=True)

            with torch.inference_mode():
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits.float(), dim=-1)
            signed = (probs[:, self._pos_idx] - probs[:, self._neg_idx]).detach().cpu().numpy()

            for bi, orig in enumerate(chunk):
                s = float(signed[bi])
                if math.isnan(s):
                    continue
                ti = flat_text_idx[orig]
                w = flat_weights[orig]
                totals[ti] += s * w
                total_w[ti] += w


def score_units(
    units: Sequence[dict],
    scorer: FinBertScorer | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Run FinBERT on every unit, writing a cache file per unit.

    Returns a list of the cache dicts in the same order as ``units``. If
    ``overwrite`` is False, existing cache files are reused when their
    ``_cache_key`` still matches the current source fingerprint; stale or
    missing files trigger a recompute.

    Scoring is done in a single global batch sweep, so this function is
    efficient even when called once per transcript with ~20 units.
    """
    ensure_dirs()
    scorer = scorer or FinBertScorer()
    key = stage_key("sentiment")

    results: list[dict | None] = [None] * len(units)
    pending_idx: list[int] = []
    pending_texts: list[str] = []
    pending_uids: list[str] = []

    for i, u in enumerate(units):
        uid = u["unit_id"]
        p = sentiment_path(uid)
        if p.is_file() and not overwrite:
            try:
                cached = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict) and cached.get(KEY_FIELD) == key:
                results[i] = cached
                continue
        pending_idx.append(i)
        pending_texts.append(u.get("text", ""))
        pending_uids.append(uid)

    if pending_texts:
        if progress:
            try:
                from tqdm.auto import tqdm

                tqdm.write(
                    f"FinBERT scoring {len(pending_texts)} unit(s) on {scorer.device}/{scorer.dtype}"
                )
            except ImportError:
                pass
        scored = scorer.score_many(pending_texts)
        for i, uid, rec in zip(pending_idx, pending_uids, scored):
            rec = {"unit_id": uid, **rec, KEY_FIELD: key}
            sentiment_path(uid).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            results[i] = rec

    out: list[dict] = []
    for r in results:
        if r is None:
            error("score_units: internal bookkeeping error (unit left unscored)")
        out.append(r)  # type: ignore[arg-type]
    return out


def load_sentiment(unit_id: str) -> dict:
    """Load a cached sentiment record. Errors if the cache is missing or stale."""
    p = sentiment_path(unit_id)
    if not p.is_file():
        error(f"sentiment cache missing: {p}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get(KEY_FIELD) != stage_key("sentiment"):
        error(f"sentiment cache stale: {p}")
    return rec


def test():
    pass
