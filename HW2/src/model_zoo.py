from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.errors import error
from src.handcrafted_features import handcrafted_features_matrix
from src.model_config import (
    FAMILY_TIE_BREAK_ORDER,
    NEG_LABEL,
    NON_TRANSFORMER_ENSEMBLE_POOL,
    OOF_N_SPLITS,
    POS_LABEL,
    RECALL_FLOOR_DEFAULT,
    RULE_BOILERPLATE_FEATURE_INDICES,
    RULE_SUBSTANTIVE_FEATURE_INDICES,
    ZooRunConfig,
)
from src.paths import CACHE_DIR
from src.train_split import stratified_supervision_split


def _hf_trainer(model: Any, args: Any, train_dataset: Any, tokenizer: Any) -> Any:
    from transformers import Trainer

    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        return Trainer(
            model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer
        )
    return Trainer(model=model, args=args, train_dataset=train_dataset, tokenizer=tokenizer)


def _ensure_transformers_default_logdir_shim() -> None:
    import transformers.training_args as ta

    if hasattr(ta, "default_logdir"):
        return

    def default_logdir() -> str:
        return "runs"

    ta.default_logdir = default_logdir  # type: ignore[assignment, misc]


def _setfit_model_for_sentence_transformer_backbone(model_id: str) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer
    from setfit import SetFitModel

    return SetFitModel(
        model_body=SentenceTransformer(model_id),
        model_head=LogisticRegression(),
    )


def _gold_final_to_binary(series: pd.Series) -> np.ndarray:
    return (series.astype(str).str.lower() == "substantive").astype(np.int64).to_numpy()


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)[:120]


def rule_pseudo_probability_from_handcrafted(h: np.ndarray) -> np.ndarray:
    """
    Weighted rule score from handcrafted columns, mapped to (0,1) via logistic squashing.
    Substantive-positive weights on finance / number signals; negative on housekeeping patterns.
    """
    if h.ndim != 2 or h.shape[1] != 30:
        error(f"Expected handcrafted matrix (n, 30), got {h.shape}")
    bp_idx = np.array(RULE_BOILERPLATE_FEATURE_INDICES, dtype=int)
    su_idx = np.array(RULE_SUBSTANTIVE_FEATURE_INDICES, dtype=int)
    bp = h[:, bp_idx].sum(axis=1)
    su = h[:, su_idx].sum(axis=1)
    raw = su - bp
    return 1.0 / (1.0 + np.exp(-raw))


def search_threshold_max_macro_f1(
    y_true: np.ndarray,
    p_substantive: np.ndarray,
    *,
    recall_floor: float = RECALL_FLOOR_DEFAULT,
    n_grid: int = 1001,
) -> tuple[float | None, float, bool]:
    """
    Among thresholds with substantive recall >= recall_floor on ``y_true`` / scores,
    return (best_threshold, best_macro_f1, eligible). If none eligible, threshold is None
    and eligible is False (macro_f1 set to -1.0).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    p_substantive = np.asarray(p_substantive, dtype=np.float64)
    best_t: float | None = None
    best_f1 = -1.0
    eligible = False
    for t in np.linspace(0.0, 1.0, int(n_grid)):
        pred = (p_substantive >= t).astype(np.int64)
        rec_sub = recall_score(y_true, pred, pos_label=POS_LABEL, zero_division=0)
        if rec_sub < recall_floor - 1e-12:
            continue
        eligible = True
        f1m = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1m > best_f1 + 1e-15 or (math.isclose(f1m, best_f1) and (best_t is None or t < best_t)):
            best_f1 = f1m
            best_t = float(t)
    if not eligible:
        return None, -1.0, False
    return best_t, float(best_f1), True


def per_fold_threshold_stats(
    y: np.ndarray,
    p: np.ndarray,
    fold_of_index: np.ndarray,
    *,
    n_folds: int,
    recall_floor: float = RECALL_FLOOR_DEFAULT,
) -> tuple[float, float]:
    """Mean and std of best eligible thresholds computed on each fold's OOF slice only."""
    ts: list[float] = []
    for k in range(int(n_folds)):
        m = fold_of_index == k
        if not np.any(m):
            continue
        t_star, _f1m, ok = search_threshold_max_macro_f1(y[m], p[m], recall_floor=recall_floor)
        if ok and t_star is not None:
            ts.append(t_star)
    if not ts:
        return float("nan"), float("nan")
    arr = np.asarray(ts, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def rank_average_scores(matrix: np.ndarray) -> np.ndarray:
    """
    ``matrix`` shape (n_samples, n_models) with higher = more substantive preference.
    Per model, rank samples (average ranks for ties), map ranks to [0,1], then
    average ranks across models.
    """
    x = np.asarray(matrix, dtype=np.float64)
    n, m = x.shape
    if m == 0:
        error("rank_average_scores: empty model list")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    ranked = np.zeros_like(x)
    for model_idx in range(m):
        col = x[:, model_idx]
        order = np.argsort(col, kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            k = start
            while k + 1 < n and col[order[k + 1]] == col[order[start]]:
                k += 1
            mean_rank = float((start + k) / 2.0)
            r[order[start : k + 1]] = mean_rank
            start = k + 1
        ranked[:, model_idx] = r / max(n - 1, 1)
    return ranked.mean(axis=1)


def classification_report_row(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    bp_f1 = f1_score(y_true, y_pred, pos_label=NEG_LABEL, average="binary", zero_division=0)
    su_f1 = f1_score(y_true, y_pred, pos_label=POS_LABEL, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_boilerplate": float(
            precision_score(y_true, y_pred, pos_label=NEG_LABEL, average="binary", zero_division=0)
        ),
        "recall_boilerplate": float(
            recall_score(y_true, y_pred, pos_label=NEG_LABEL, average="binary", zero_division=0)
        ),
        "f1_boilerplate": float(bp_f1),
        "precision_substantive": float(
            precision_score(y_true, y_pred, pos_label=POS_LABEL, average="binary", zero_division=0)
        ),
        "recall_substantive": float(
            recall_score(y_true, y_pred, pos_label=POS_LABEL, average="binary", zero_division=0)
        ),
        "f1_substantive": float(su_f1),
    }


def _sort_leaderboard_key(family_id: str) -> tuple[int, str]:
    is_ens = 1 if family_id.startswith("G_") or family_id.startswith("H_") else 0
    return (-is_ens, family_id)


def winner_sort_key(
    r: dict[str, Any],
    *,
    prefer_ensemble_on_tie: bool = True,
) -> tuple[float, int, str]:
    tf1 = float(r.get("test_macro_f1", -1.0))
    fid = str(r["family_id"])
    ens = 0 if (prefer_ensemble_on_tie and (fid.startswith("G_") or fid.startswith("H_"))) else 1
    return (-tf1, ens, fid)


def _jsonable_config(cfg: ZooRunConfig) -> dict[str, Any]:
    out = asdict(cfg)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


@dataclass
class _FamilyRun:
    family_id: str
    oof_p: np.ndarray | None = None
    test_p: np.ndarray | None = None
    train_seconds: float = 0.0
    threshold: float | None = None
    threshold_std_across_folds: float = float("nan")
    oof_macro_f1_at_threshold: float = -1.0
    eligible: bool = False
    test_metrics: dict[str, Any] = field(default_factory=dict)
    infer_sents_per_sec: float = 0.0
    artifact_path: str | None = None
    notes: str = ""


def _load_sentence_transformers(model_id: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing sentence-transformers (needed for B/C embeddings). "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_id)


def _encode_st(model: Any, texts: list[str], batch_size: int, *, show_progress_bar: bool) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float64)


def _embedding_cache_path(model_id: str, sentence_ids: np.ndarray | None = None) -> Path:
    stem = f"st_embeddings_{_sanitize_model_id(model_id)}"
    if sentence_ids is None:
        return CACHE_DIR / f"{stem}.npz"
    joined = "\0".join(sentence_ids.astype(str).tolist()).encode("utf-8")
    digest = hashlib.sha1(joined).hexdigest()[:12]
    return CACHE_DIR / f"{stem}_{digest}.npz"


def load_or_compute_embeddings(
    sentence_ids: np.ndarray,
    texts: list[str],
    model_id: str,
    *,
    batch_size: int,
    use_cache: bool,
    show_progress: bool = False,
) -> np.ndarray:
    path = _embedding_cache_path(model_id, sentence_ids)
    legacy_path = _embedding_cache_path(model_id)
    if use_cache and path.exists():
        data = np.load(path, allow_pickle=True)
        if np.array_equal(data["sentence_id"].astype(str), sentence_ids.astype(str)):
            return np.asarray(data["embeddings"], dtype=np.float64)
    if use_cache and legacy_path.exists() and legacy_path != path:
        data = np.load(legacy_path, allow_pickle=True)
        if np.array_equal(data["sentence_id"].astype(str), sentence_ids.astype(str)):
            mat = np.asarray(data["embeddings"], dtype=np.float64)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, sentence_id=sentence_ids.astype(str), embeddings=mat.astype(np.float32))
            return mat
    model = _load_sentence_transformers(model_id)
    mat = _encode_st(model, texts, batch_size, show_progress_bar=show_progress)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sentence_id=sentence_ids.astype(str), embeddings=mat.astype(np.float32))
    return mat


def _make_group_kfold(seed: int) -> GroupKFold:
    params = inspect.signature(GroupKFold).parameters
    if "shuffle" in params:
        return GroupKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=OOF_N_SPLITS)


def _group_kfold_splits(y: np.ndarray, groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = _make_group_kfold(seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _oof_group_indices(
    groups: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return fold id per row (for variance) and list of (train_idx, val_idx) for each fold."""
    fold_of = np.full(len(groups), -1, dtype=np.int32)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (train_idx, val_idx) in enumerate(_group_kfold_splits(y, groups, seed)):
        fold_of[val_idx] = k
        splits.append((train_idx, val_idx))
    if (fold_of < 0).any():
        error("GroupKFold failed to assign some rows to a fold")
    return fold_of, splits


def _fit_lr_oof(
    x_tv: np.ndarray,
    y_tv: np.ndarray,
    groups: np.ndarray,
    cfg: ZooRunConfig,
) -> tuple[np.ndarray, float]:
    oof = np.zeros(len(y_tv), dtype=np.float64)
    t0 = time.perf_counter()
    splits = _group_kfold_splits(y_tv, groups, cfg.seed)
    for tr, va in tqdm(
        splits,
        desc="B LogReg OOF",
        unit="fold",
        disable=not cfg.show_progress,
        leave=False,
    ):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=cfg.sklearn_max_iter,
                        class_weight=cfg.logreg_class_weight,
                        random_state=cfg.seed,
                        solver=cfg.logreg_solver,
                    ),
                ),
            ]
        )
        pipe.fit(x_tv[tr], y_tv[tr])
        proba = pipe.predict_proba(x_tv[va])[:, 1]
        oof[va] = proba
    return oof, time.perf_counter() - t0


def _fit_hgb_oof(
    x_tv: np.ndarray,
    y_tv: np.ndarray,
    groups: np.ndarray,
    cfg: ZooRunConfig,
) -> tuple[np.ndarray, float]:
    oof = np.zeros(len(y_tv), dtype=np.float64)
    t0 = time.perf_counter()
    splits = _group_kfold_splits(y_tv, groups, cfg.seed)
    for tr, va in tqdm(
        splits,
        desc="C HistGB OOF",
        unit="fold",
        disable=not cfg.show_progress,
        leave=False,
    ):
        clf = HistGradientBoostingClassifier(
            random_state=cfg.seed,
            max_depth=cfg.histgb_max_depth,
            learning_rate=cfg.histgb_learning_rate,
            max_iter=cfg.histgb_max_iter,
        )
        clf.fit(x_tv[tr], y_tv[tr])
        oof[va] = clf.predict_proba(x_tv[va])[:, 1]
    return oof, time.perf_counter() - t0


def _final_fit_lr(x_train: np.ndarray, y_train: np.ndarray, cfg: ZooRunConfig) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg.sklearn_max_iter,
                    class_weight=cfg.logreg_class_weight,
                    random_state=cfg.seed,
                    solver=cfg.logreg_solver,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def _final_fit_hgb(x_train: np.ndarray, y_train: np.ndarray, cfg: ZooRunConfig) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        random_state=cfg.seed,
        max_depth=cfg.histgb_max_depth,
        learning_rate=cfg.histgb_learning_rate,
        max_iter=cfg.histgb_max_iter,
    )
    clf.fit(x_train, y_train)
    return clf


def _ensure_fasttext_numpy2_compat() -> None:
    """Patch fasttext's predict once: vendored wheels use np.array(..., copy=False), which raises on NumPy 2.x."""
    import fasttext.FastText as ftm

    cls = ftm._FastText
    if getattr(cls, "_numpy2_predict_patched", False):
        return
    cls.predict = _fasttext_predict_numpy2_compat  # type: ignore[method-assign]
    cls._numpy2_predict_patched = True


def _fasttext_predict_numpy2_compat(
    self: Any,
    text: Any,
    k: int = 1,
    threshold: float = 0.0,
    on_unicode_error: str = "strict",
) -> Any:
    """
    Mirrors fasttext.FastText._FastText.predict; only the return path uses np.asarray for probabilities
    so NumPy 2.x accepts the buffer view from the fastText C++ binding.
    """

    def check(entry: str) -> str:
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        entry += "\n"
        return entry

    if type(text) == list:
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, all_probs
    text_single = check(text)
    predictions = self.f.predict(text_single, k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())
    return labels, np.asarray(probs)


def _fasttext_train_predict_oof(
    texts: list[str],
    y_tv: np.ndarray,
    groups: np.ndarray,
    cfg: ZooRunConfig,
) -> tuple[np.ndarray, float]:
    try:
        import fasttext
    except ImportError as e:
        raise ImportError(
            "Missing fasttext (needed for D_fasttext). Install with: pip install fasttext"
        ) from e
    _ensure_fasttext_numpy2_compat()
    oof = np.zeros(len(y_tv), dtype=np.float64)
    t0 = time.perf_counter()
    tmpdir = Path(tempfile.mkdtemp(prefix="ft_zoo_"))
    try:
        splits = _group_kfold_splits(y_tv, groups, cfg.seed)
        for tr, va in tqdm(
            splits,
            desc="D FastText OOF",
            unit="fold",
            disable=not cfg.show_progress,
            leave=False,
        ):
            train_path = tmpdir / "train.txt"
            lines: list[str] = []
            for i in tr:
                lab = "__label__substantive" if y_tv[i] == POS_LABEL else "__label__boilerplate"
                t = texts[i].replace("\n", " ").replace("\r", " ")
                lines.append(f"{lab} {t}")
            train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            model = fasttext.train_supervised(
                str(train_path),
                lr=cfg.fasttext_lr,
                epoch=cfg.fasttext_epoch,
                dim=cfg.fasttext_dim,
                wordNgrams=cfg.fasttext_word_ngrams,
                minCount=cfg.fasttext_min_count,
                verbose=0,
            )
            oof[va] = _fasttext_p_sub_batch(model, [texts[j] for j in va], show_progress=False)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return oof, time.perf_counter() - t0


def _fasttext_final_train_predict(
    train_texts: list[str],
    train_y: np.ndarray,
    predict_texts: list[str],
    cfg: ZooRunConfig,
) -> tuple[Any, np.ndarray, str]:
    import fasttext

    _ensure_fasttext_numpy2_compat()
    tmpdir = Path(tempfile.mkdtemp(prefix="ft_final_"))
    train_path = tmpdir / "train.txt"
    lines = []
    for t, yi in zip(train_texts, train_y.tolist()):
        lab = "__label__substantive" if yi == POS_LABEL else "__label__boilerplate"
        lines.append(f"{lab} {t.replace(chr(10), ' ').replace(chr(13), ' ')}")
    train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = fasttext.train_supervised(
        str(train_path),
        lr=cfg.fasttext_lr,
        epoch=cfg.fasttext_epoch,
        dim=cfg.fasttext_dim,
        wordNgrams=cfg.fasttext_word_ngrams,
        minCount=cfg.fasttext_min_count,
        verbose=0,
    )
    out = _fasttext_p_sub_batch(model, predict_texts, show_progress=cfg.show_progress)
    model_path = CACHE_DIR / "zoo_D_fasttext.bin"
    model.save_model(str(model_path))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return model, out, str(model_path)


def _fasttext_one_p_sub(labels: Any, probs: Any) -> float:
    p_sub = 0.0
    for lab, pr in zip(labels, probs):
        if lab == "__label__substantive":
            return float(pr)
    return p_sub


def _fasttext_p_sub_batch(model: Any, texts: list[str], *, show_progress: bool) -> np.ndarray:
    clean = [raw.replace("\n", " ").replace("\r", " ") for raw in texts]
    if not clean:
        return np.zeros(0, dtype=np.float64)
    if show_progress and len(clean) > 32:
        batches: list[list[str]] = []
        step = 4096
        for start in range(0, len(clean), step):
            batches.append(clean[start : start + step])
        out_parts: list[np.ndarray] = []
        for batch in tqdm(batches, desc="FastText predict", unit="batch", leave=False):
            out_parts.append(_fasttext_p_sub_batch(model, batch, show_progress=False))
        return np.concatenate(out_parts)
    labels, probs = model.predict(clean, k=2)
    return np.asarray([_fasttext_one_p_sub(lab, pr) for lab, pr in zip(labels, probs)], dtype=np.float64)


def _try_import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _torch_cuda_available() -> bool:
    torch = _try_import_torch()
    return bool(torch is not None and torch.cuda.is_available())


def _datasets_map(ds: Any, fn: Any, *, show_progress: bool, **kwargs: Any) -> Any:
    """Call ``Dataset.map`` with progress disabled when possible (API differs by ``datasets`` version)."""
    for extra in (
        {"disable": not show_progress},
        {"disable_progress_bar": not show_progress},
    ):
        try:
            return ds.map(fn, **extra, **kwargs)
        except TypeError:
            continue
    return ds.map(fn, **kwargs)


def _setfit_training_arguments(cls: Any, *, show_progress: bool, **kwargs: Any) -> Any:
    try:
        return cls(**kwargs, disable_tqdm=not show_progress)
    except TypeError:
        return cls(**kwargs)


def _transformer_p_sub_batches(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    show_progress: bool,
    desc: str,
) -> np.ndarray:
    torch = _try_import_torch()
    if torch is None:
        error("PyTorch is required for transformer inference.")
    out = np.zeros(len(texts), dtype=np.float64)
    if not texts:
        return out
    model.eval()
    device = next(model.parameters()).device
    n_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=desc,
            unit="batch",
            total=n_batches,
            disable=not show_progress,
            leave=False,
        ):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            logits = model(**enc).logits
            out[start : start + len(chunk)] = torch.softmax(logits, dim=-1)[:, POS_LABEL].cpu().numpy()
    return out


def _finbert_oof_and_final(
    texts_tv: list[str],
    y_tv: np.ndarray,
    groups: np.ndarray,
    texts_test: list[str],
    cfg: ZooRunConfig,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    torch = _try_import_torch()
    if torch is None:
        error("PyTorch + transformers required for FinBERT (E_finbert).")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(cfg.finbert_model_id)
    oof = np.zeros(len(y_tv), dtype=np.float64)
    t_train = 0.0
    fold_splits = _group_kfold_splits(y_tv, groups, cfg.seed)
    use_fp16 = bool(cfg.finbert_use_fp16_if_cuda and torch.cuda.is_available())
    for fold_i, (tr, va) in enumerate(
        tqdm(
            fold_splits,
            desc="E FinBERT OOF",
            unit="fold",
            disable=not cfg.show_progress,
            leave=False,
        )
    ):
        ds_tr = Dataset.from_dict({"text": [texts_tv[i] for i in tr], "labels": y_tv[tr].tolist()})

        def tok(batch: dict[str, Any]) -> dict[str, Any]:
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=cfg.finbert_max_length,
                padding=False,
            )
            enc["labels"] = batch["labels"]
            return enc

        ds_tr_t = _datasets_map(
            ds_tr,
            tok,
            show_progress=cfg.show_progress,
            batched=True,
            remove_columns=["text"],
            desc="tokenize",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.finbert_model_id,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        t0 = time.perf_counter()
        args = TrainingArguments(
            output_dir=str(CACHE_DIR / f"finbert_fold_{fold_i}"),
            learning_rate=cfg.finbert_learning_rate,
            per_device_train_batch_size=cfg.finbert_batch_size,
            per_device_eval_batch_size=cfg.finbert_eval_batch_size,
            gradient_accumulation_steps=cfg.finbert_gradient_accumulation_steps,
            num_train_epochs=cfg.finbert_epochs,
            seed=cfg.seed + cfg.finbert_seed_offset + fold_i,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=not cfg.show_progress,
            fp16=use_fp16,
            dataloader_num_workers=cfg.finbert_dataloader_num_workers,
        )

        trainer = _hf_trainer(model, args, ds_tr_t, tokenizer)
        trainer.train()
        t_train += time.perf_counter() - t0
        oof[va] = _transformer_p_sub_batches(
            model,
            tokenizer,
            [texts_tv[j] for j in va],
            batch_size=cfg.finbert_eval_batch_size,
            max_length=cfg.finbert_max_length,
            show_progress=cfg.show_progress,
            desc="  FinBERT val",
        )

    full_texts_tv = texts_tv
    full_y_tv = y_tv
    ds_full = Dataset.from_dict({"text": full_texts_tv, "labels": full_y_tv.tolist()})

    def tok2(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(batch["text"], truncation=True, max_length=cfg.finbert_max_length, padding=False)
        enc["labels"] = batch["labels"]
        return enc

    ds_full_t = _datasets_map(
        ds_full,
        tok2,
        show_progress=cfg.show_progress,
        batched=True,
        remove_columns=["text"],
        desc="tokenize (full TV)",
    )
    final_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.finbert_model_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    t0 = time.perf_counter()
    fargs = TrainingArguments(
        output_dir=str(CACHE_DIR / "finbert_final_tv"),
        learning_rate=cfg.finbert_learning_rate,
        per_device_train_batch_size=cfg.finbert_batch_size,
        per_device_eval_batch_size=cfg.finbert_eval_batch_size,
        gradient_accumulation_steps=cfg.finbert_gradient_accumulation_steps,
        num_train_epochs=cfg.finbert_epochs,
        seed=cfg.seed + cfg.finbert_seed_offset + 99,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=not cfg.show_progress,
        fp16=use_fp16,
        dataloader_num_workers=cfg.finbert_dataloader_num_workers,
    )
    ftrainer = _hf_trainer(final_model, fargs, ds_full_t, tokenizer)
    ftrainer.train()
    t_train += time.perf_counter() - t0
    test_p = _transformer_p_sub_batches(
        final_model,
        tokenizer,
        texts_test,
        batch_size=cfg.finbert_eval_batch_size,
        max_length=cfg.finbert_max_length,
        show_progress=cfg.show_progress,
        desc="FinBERT test",
    )
    artifact_dir = CACHE_DIR / "zoo_E_finbert_final"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    return oof, test_p, t_train, str(artifact_dir)


def _to_float64_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _setfit_positive_class_probability(value: Any, *, context: str) -> np.ndarray:
    arr = _to_float64_numpy(value)
    if arr.ndim != 2 or arr.shape[1] <= POS_LABEL:
        error(f"Unexpected SetFit {context} predict_proba shape: {arr.shape}")
    return arr[:, POS_LABEL]


def _setfit_oof_and_final(
    texts_tv: list[str],
    y_tv: np.ndarray,
    groups: np.ndarray,
    texts_test: list[str],
    cfg: ZooRunConfig,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    _ensure_transformers_default_logdir_shim()
    try:
        from datasets import Dataset
        from setfit import Trainer, TrainingArguments as SFTrainingArguments
    except ImportError as e:
        error(f"setfit + datasets required for F_setfit: {e}")

    oof = np.zeros(len(y_tv), dtype=np.float64)
    t_train = 0.0
    sf_splits = _group_kfold_splits(y_tv, groups, cfg.seed)
    use_amp = bool(cfg.setfit_use_amp_if_cuda and _torch_cuda_available())
    for fold_i, (tr, va) in enumerate(
        tqdm(
            sf_splits,
            desc="F SetFit OOF",
            unit="fold",
            disable=not cfg.show_progress,
            leave=False,
        )
    ):
        texts_tr = [texts_tv[i] for i in tr]
        y_tr = y_tv[tr]
        ds = Dataset.from_dict({"text": texts_tr, "label": y_tr.tolist()})
        model = _setfit_model_for_sentence_transformer_backbone(cfg.sentence_transformers_model_id)
        out_dir = str(CACHE_DIR / f"setfit_fold_{fold_i}")
        args = _setfit_training_arguments(
            SFTrainingArguments,
            show_progress=cfg.show_progress,
            output_dir=out_dir,
            batch_size=cfg.setfit_batch_size,
            num_epochs=cfg.setfit_num_epochs,
            num_iterations=cfg.setfit_num_iterations,
            samples_per_label=cfg.setfit_samples_per_label,
            max_length=cfg.setfit_max_length,
            use_amp=use_amp,
            seed=cfg.seed + cfg.setfit_seed_offset + fold_i,
            logging_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            show_progress_bar=cfg.show_progress,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        t0 = time.perf_counter()
        trainer.train()
        t_train += time.perf_counter() - t0
        p_va = model.predict_proba([texts_tv[i] for i in va])
        oof[va] = _setfit_positive_class_probability(p_va, context="OOF")

    model_f = _setfit_model_for_sentence_transformer_backbone(cfg.sentence_transformers_model_id)
    ds_f = Dataset.from_dict({"text": texts_tv, "label": y_tv.tolist()})
    args_f = _setfit_training_arguments(
        SFTrainingArguments,
        show_progress=cfg.show_progress,
        output_dir=str(CACHE_DIR / "setfit_final_tv"),
        batch_size=cfg.setfit_batch_size,
        num_epochs=cfg.setfit_num_epochs,
        num_iterations=cfg.setfit_num_iterations,
        samples_per_label=cfg.setfit_samples_per_label,
        max_length=cfg.setfit_max_length,
        use_amp=use_amp,
        seed=cfg.seed + cfg.setfit_seed_offset + 99,
        logging_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        show_progress_bar=cfg.show_progress,
        report_to="none",
    )
    trainer_f = Trainer(model=model_f, args=args_f, train_dataset=ds_f)
    t0 = time.perf_counter()
    trainer_f.train()
    t_train += time.perf_counter() - t0
    pt = model_f.predict_proba(texts_test)
    test_p = _setfit_positive_class_probability(pt, context="test")
    artifact_dir = CACHE_DIR / "zoo_F_setfit_final"
    model_f.save_pretrained(str(artifact_dir))
    return oof, test_p, t_train, str(artifact_dir)


def _measure_throughput_sklearn(
    predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, *, n_cap: int = 2048
) -> float:
    xb = x[: min(len(x), n_cap)]
    t0 = time.perf_counter()
    predict_fn(xb)
    dt = time.perf_counter() - t0
    if dt <= 0:
        return float("inf")
    return float(len(xb) / dt)


def _measure_throughput_text(
    predict_fn: Callable[[list[str]], np.ndarray], texts: list[str], *, n_cap: int = 512
) -> float:
    texts_b = texts[: min(len(texts), n_cap)]
    t0 = time.perf_counter()
    predict_fn(texts_b)
    dt = time.perf_counter() - t0
    if dt <= 0:
        return float("inf")
    return float(len(texts_b) / dt)


def run_classifier_zoo(
    split_df: pd.DataFrame,
    cfg: ZooRunConfig | None = None,
) -> dict[str, Any]:
    """
    Train and evaluate classifier zoo (plan/model.md): OOF thresholding on train∪val,
    test metrics once per family, mean/rank ensembles, winner selection and optional artifact save.
    """
    cfg = cfg or ZooRunConfig()
    need_cols = {"text", "gold_final", "split", "source_file", "sentence_id"}
    if not need_cols.issubset(split_df.columns):
        error(f"split_df missing columns: {sorted(need_cols - set(split_df.columns))}")

    if cfg.show_progress:
        print("[zoo] start (OOF + test)", flush=True)

    texts_all = split_df["text"].astype(str).tolist()
    sid = split_df["sentence_id"].astype(str).to_numpy()
    groups_all = split_df["source_file"].astype(str).to_numpy()
    y_all = _gold_final_to_binary(split_df["gold_final"])
    is_tv = split_df["split"].isin(["train", "val"]).to_numpy()
    is_test = (split_df["split"] == "test").to_numpy()
    idx_tv = np.flatnonzero(is_tv)
    idx_test = np.flatnonzero(is_test)

    texts_tv = [texts_all[i] for i in idx_tv]
    y_tv = y_all[idx_tv]
    g_tv = groups_all[idx_tv]
    texts_test = [texts_all[i] for i in idx_test]
    y_test = y_all[idx_test]

    h_all = handcrafted_features_matrix(split_df["text"])
    h_tv = h_all[idx_tv]
    h_test = h_all[idx_test]

    if cfg.show_progress:
        print("[zoo] sentence embeddings: all supervised rows …", flush=True)
    emb_all = load_or_compute_embeddings(
        sid,
        texts_all,
        cfg.sentence_transformers_model_id,
        batch_size=cfg.embedding_batch_size,
        use_cache=cfg.cache_embeddings,
        show_progress=cfg.show_progress,
    )
    emb_tv = emb_all[idx_tv]
    emb_test = emb_all[idx_test]
    x_tv = np.hstack([emb_tv, h_tv])
    x_test = np.hstack([emb_test, h_test])

    fold_of_tv, _ = _oof_group_indices(g_tv, y_tv, cfg.seed)

    families: dict[str, _FamilyRun] = {}
    fasttext_final_model: Any | None = None
    final_lr_model: Pipeline | None = None
    final_hgb_model: HistGradientBoostingClassifier | None = None

    pr_tv = rule_pseudo_probability_from_handcrafted(h_tv)
    t_rules = 0.0
    fr = _FamilyRun("A_rules", oof_p=pr_tv, train_seconds=t_rules)
    families[fr.family_id] = fr

    if cfg.show_progress:
        print("[zoo] B/C/D OOF models …", flush=True)
    oof_b, t_b = _fit_lr_oof(x_tv, y_tv, g_tv, cfg)
    families["B_logreg_enriched"] = _FamilyRun("B_logreg_enriched", oof_p=oof_b, train_seconds=t_b)

    oof_c, t_c = _fit_hgb_oof(x_tv, y_tv, g_tv, cfg)
    families["C_histgb_enriched"] = _FamilyRun("C_histgb_enriched", oof_p=oof_c, train_seconds=t_c)

    oof_d, t_d = _fasttext_train_predict_oof(texts_tv, y_tv, g_tv, cfg)
    families["D_fasttext"] = _FamilyRun("D_fasttext", oof_p=oof_d, train_seconds=t_d)

    if cfg.skip_finbert:
        families["E_finbert"] = _FamilyRun("E_finbert", notes="skipped_skip_finbert")
    else:
        oof_e, test_e, t_e, art_e = _finbert_oof_and_final(texts_tv, y_tv, g_tv, texts_test, cfg)
        families["E_finbert"] = _FamilyRun(
            "E_finbert",
            oof_p=oof_e,
            test_p=test_e,
            train_seconds=t_e,
            artifact_path=art_e,
        )

    if cfg.skip_setfit:
        families["F_setfit"] = _FamilyRun("F_setfit", notes="skipped_skip_setfit")
    else:
        oof_f, test_f, t_f, art_f = _setfit_oof_and_final(texts_tv, y_tv, g_tv, texts_test, cfg)
        families["F_setfit"] = _FamilyRun(
            "F_setfit",
            oof_p=oof_f,
            test_p=test_f,
            train_seconds=t_f,
            artifact_path=art_f,
        )

    for fid, fr in families.items():
        if fr.oof_p is None:
            fr.eligible = False
            continue
        t_star, f1m, ok = search_threshold_max_macro_f1(y_tv, fr.oof_p, recall_floor=cfg.recall_floor)
        _, s = per_fold_threshold_stats(
            y_tv, fr.oof_p, fold_of_tv, n_folds=OOF_N_SPLITS, recall_floor=cfg.recall_floor
        )
        fr.threshold = t_star
        fr.oof_macro_f1_at_threshold = f1m if ok else -1.0
        fr.eligible = ok
        fr.threshold_std_across_folds = s if not math.isnan(s) else float("nan")
        if fr.test_p is None:
            if fid == "A_rules":
                fr.test_p = rule_pseudo_probability_from_handcrafted(h_test)
            elif fid == "B_logreg_enriched":
                final_lr_model = _final_fit_lr(x_tv, y_tv, cfg)
                fr.test_p = final_lr_model.predict_proba(x_test)[:, 1]
            elif fid == "C_histgb_enriched":
                final_hgb_model = _final_fit_hgb(x_tv, y_tv, cfg)
                fr.test_p = final_hgb_model.predict_proba(x_test)[:, 1]
            elif fid == "D_fasttext":
                fasttext_final_model, ptest, art_d = _fasttext_final_train_predict(texts_tv, y_tv, texts_test, cfg)
                fr.test_p = ptest
                fr.artifact_path = art_d
            elif fid in ("E_finbert", "F_setfit") and not fr.notes:
                pass
        if fr.test_p is not None and fr.threshold is not None:
            pred = (fr.test_p >= fr.threshold).astype(np.int64)
            fr.test_metrics = classification_report_row(y_test, pred) | {
                "confusion_matrix": confusion_matrix(y_test, pred, labels=[NEG_LABEL, POS_LABEL]).tolist(),
            }
        elif fr.test_p is not None and not fr.eligible:
            fr.test_metrics = {"note": "ineligible_no_threshold; test probs computed for diagnostics only"}

    def finalize_sklearn_artifacts() -> None:
        nonlocal final_lr_model, final_hgb_model
        if final_lr_model is None:
            final_lr_model = _final_fit_lr(x_tv, y_tv, cfg)
        path_b = CACHE_DIR / "zoo_B_logreg_pipeline.pkl"
        with path_b.open("wb") as f:
            pickle.dump({"pipeline": final_lr_model, "feature_dim": x_tv.shape[1]}, f)
        families["B_logreg_enriched"].artifact_path = str(path_b)
        if final_hgb_model is None:
            final_hgb_model = _final_fit_hgb(x_tv, y_tv, cfg)
        path_c = CACHE_DIR / "zoo_C_histgb.pkl"
        with path_c.open("wb") as f:
            pickle.dump({"model": final_hgb_model, "feature_dim": x_tv.shape[1]}, f)
        families["C_histgb_enriched"].artifact_path = str(path_c)

    finalize_sklearn_artifacts()

    pool_ids = list(NON_TRANSFORMER_ENSEMBLE_POOL)
    ranked: list[tuple[str, float]] = []
    for pid in pool_ids:
        fr = families[pid]
        if fr.eligible:
            ranked.append((pid, fr.oof_macro_f1_at_threshold))
    ranked.sort(key=lambda z: (-z[1], z[0]))
    top5 = [fid for fid, _ in ranked[:5]]

    def build_prob_matrix(ids: list[str], *, oof: bool) -> np.ndarray | None:
        cols: list[np.ndarray] = []
        for pid in ids:
            frp = families[pid]
            p = frp.oof_p if oof else frp.test_p
            if p is None:
                return None
            cols.append(p)
        return np.stack(cols, axis=1)

    mat_top_oof = build_prob_matrix(top5, oof=True)
    if mat_top_oof is not None and mat_top_oof.shape[1] > 0:
        ens_mean_p = mat_top_oof.mean(axis=1)
        ens_rank_p = rank_average_scores(mat_top_oof)
        mat_top_test = build_prob_matrix(top5, oof=False)
        for eid, p_ens in [("G_ensemble_mean", ens_mean_p), ("H_ensemble_rankmean", ens_rank_p)]:
            t_star, f1m, ok = search_threshold_max_macro_f1(y_tv, p_ens, recall_floor=cfg.recall_floor)
            _, s = per_fold_threshold_stats(y_tv, p_ens, fold_of_tv, n_folds=OOF_N_SPLITS, recall_floor=cfg.recall_floor)
            fe = _FamilyRun(eid, oof_p=p_ens, train_seconds=0.0)
            fe.threshold = t_star
            fe.oof_macro_f1_at_threshold = f1m if ok else -1.0
            fe.eligible = ok
            fe.threshold_std_across_folds = s
            if mat_top_test is not None:
                if eid == "G_ensemble_mean":
                    fe.test_p = mat_top_test.mean(axis=1)
                else:
                    fe.test_p = rank_average_scores(mat_top_test)
                if fe.threshold is not None and fe.test_p is not None:
                    pred = (fe.test_p >= fe.threshold).astype(np.int64)
                    fe.test_metrics = classification_report_row(y_test, pred) | {
                        "confusion_matrix": confusion_matrix(y_test, pred, labels=[NEG_LABEL, POS_LABEL]).tolist(),
                    }
            families[eid] = fe

    rows: list[dict[str, Any]] = []
    for fid in FAMILY_TIE_BREAK_ORDER:
        if fid not in families:
            continue
        fr = families[fid]
        if fr.oof_p is None:
            rows.append(
                {
                    "family_id": fid,
                    "eligible_oof": False,
                    "threshold": None,
                    "oof_macro_f1": -1.0,
                    "threshold_std_across_folds": float("nan"),
                    "train_seconds": fr.train_seconds,
                    "notes": fr.notes or "skipped_or_no_oof",
                }
            )
            continue
        tm = fr.test_metrics
        row = {
            "family_id": fid,
            "eligible_oof": fr.eligible,
            "threshold": fr.threshold,
            "oof_macro_f1": fr.oof_macro_f1_at_threshold,
            "threshold_std_across_folds": fr.threshold_std_across_folds,
            "train_seconds": fr.train_seconds,
            "notes": fr.notes,
        }
        if isinstance(tm, dict) and "accuracy" in tm:
            row.update(
                {
                    "test_accuracy": tm["accuracy"],
                    "test_macro_f1": tm["macro_f1"],
                    "test_precision_boilerplate": tm["precision_boilerplate"],
                    "test_recall_boilerplate": tm["recall_boilerplate"],
                    "test_f1_boilerplate": tm["f1_boilerplate"],
                    "test_precision_substantive": tm["precision_substantive"],
                    "test_recall_substantive": tm["recall_substantive"],
                    "test_f1_substantive": tm["f1_substantive"],
                    "test_confusion_matrix": tm.get("confusion_matrix"),
                }
            )
        rows.append(row)

    speed_by_family: dict[str, float] = {}
    for row in rows:
        fid = str(row["family_id"])
        spd = float("nan")
        if fid == "A_rules":
            spd = _measure_throughput_sklearn(lambda xb: rule_pseudo_probability_from_handcrafted(xb), h_test)
        elif fid == "B_logreg_enriched" and families[fid].artifact_path:
            blob = pickle.loads(Path(families[fid].artifact_path).read_bytes())
            pipe = blob["pipeline"] if isinstance(blob, dict) else blob
            spd = _measure_throughput_sklearn(lambda xb: pipe.predict_proba(xb)[:, 1], x_test)
        elif fid == "C_histgb_enriched" and families[fid].artifact_path:
            blob = pickle.loads(Path(families[fid].artifact_path).read_bytes())
            clf = blob["model"]
            spd = _measure_throughput_sklearn(lambda xb: clf.predict_proba(xb)[:, 1], x_test)
        elif fid == "D_fasttext" and fasttext_final_model is not None:
            spd = _measure_throughput_text(
                lambda tx: _fasttext_p_sub_batch(fasttext_final_model, tx, show_progress=False),
                texts_test,
            )
        elif fid == "E_finbert" and families[fid].artifact_path:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(families[fid].artifact_path)
            model = AutoModelForSequenceClassification.from_pretrained(families[fid].artifact_path)
            torch = _try_import_torch()
            if torch is not None and torch.cuda.is_available():
                model.to("cuda")
            spd = _measure_throughput_text(
                lambda tx: _transformer_p_sub_batches(
                    model,
                    tokenizer,
                    tx,
                    batch_size=cfg.finbert_eval_batch_size,
                    max_length=cfg.finbert_max_length,
                    show_progress=False,
                    desc="FinBERT throughput",
                ),
                texts_test,
            )
        elif fid == "F_setfit" and families[fid].artifact_path:
            from setfit import SetFitModel

            model = SetFitModel.from_pretrained(families[fid].artifact_path)
            spd = _measure_throughput_text(
                lambda tx: _setfit_positive_class_probability(
                    model.predict_proba(tx),
                    context="throughput",
                ),
                texts_test,
            )
        elif fid in ("G_ensemble_mean", "H_ensemble_rankmean"):
            component_speeds = [
                speed_by_family[pid]
                for pid in top5
                if pid in speed_by_family and math.isfinite(speed_by_family[pid]) and speed_by_family[pid] > 0
            ]
            if component_speeds:
                spd = 1.0 / sum(1.0 / s for s in component_speeds)
        row["approx_infer_sents_per_sec"] = spd
        if math.isfinite(spd) and spd > 0:
            speed_by_family[fid] = spd

    leaderboard = [r for r in rows if "test_macro_f1" in r]
    leaderboard.sort(key=lambda r: (-float(r["test_macro_f1"]), *_sort_leaderboard_key(str(r["family_id"]))))

    candidates = [r for r in leaderboard if r.get("eligible_oof")]
    if cfg.winner_requires_test_recall:
        candidates = [r for r in candidates if float(r.get("test_recall_substantive", -1.0)) >= cfg.recall_floor - 1e-9]
    winner: dict[str, Any] | None = None
    if candidates:
        winner = sorted(candidates, key=lambda r: winner_sort_key(r, prefer_ensemble_on_tie=cfg.prefer_ensemble_on_tie))[0]

    out_dir = cfg.winner_artifact_dir or (CACHE_DIR / "zoo_winner")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": cfg.seed,
        "show_progress": cfg.show_progress,
        "recall_floor": cfg.recall_floor,
        "model_hyperparameters": _jsonable_config(cfg),
        "sentence_transformers_model_id": cfg.sentence_transformers_model_id,
        "finbert": {
            "model_id": cfg.finbert_model_id,
            "max_length": cfg.finbert_max_length,
            "epochs": cfg.finbert_epochs,
            "batch_size": cfg.finbert_batch_size,
            "eval_batch_size": cfg.finbert_eval_batch_size,
            "use_fp16_if_cuda": cfg.finbert_use_fp16_if_cuda,
        },
        "setfit": {
            "batch_size": cfg.setfit_batch_size,
            "num_epochs": cfg.setfit_num_epochs,
            "num_iterations": cfg.setfit_num_iterations,
            "use_amp_if_cuda": cfg.setfit_use_amp_if_cuda,
        },
        "oof_splits": OOF_N_SPLITS,
        "ensemble_top5_family_ids": top5,
        "winner_family_id": winner["family_id"] if winner else None,
        "winner_threshold": float(winner["threshold"]) if winner and winner.get("threshold") is not None else None,
        "winner_artifact_dir": str(out_dir),
        "winner_requires_test_recall": cfg.winner_requires_test_recall,
    }
    (out_dir / "zoo_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if winner:
        for stale_file in ("winner_sklearn.pkl", "winner_fasttext.bin"):
            (out_dir / stale_file).unlink(missing_ok=True)
        for stale_dir in ("winner_finbert", "winner_setfit"):
            shutil.rmtree(out_dir / stale_dir, ignore_errors=True)
        wt = winner.get("threshold")
        if wt is not None:
            (out_dir / "winner_threshold.txt").write_text(str(wt), encoding="utf-8")
        wf = str(winner["family_id"])
        winner_manifest: dict[str, Any] = {
            "family_id": wf,
            "threshold": float(wt) if wt is not None else None,
            "source_artifact": families.get(wf).artifact_path if wf in families else None,
        }
        if wf == "B_logreg_enriched" and families["B_logreg_enriched"].artifact_path:
            dest = out_dir / "winner_sklearn.pkl"
            shutil.copy2(families["B_logreg_enriched"].artifact_path, dest)
            winner_manifest["artifact"] = str(dest)
        elif wf == "C_histgb_enriched" and families["C_histgb_enriched"].artifact_path:
            dest = out_dir / "winner_sklearn.pkl"
            shutil.copy2(families["C_histgb_enriched"].artifact_path, dest)
            winner_manifest["artifact"] = str(dest)
        elif wf == "D_fasttext" and families["D_fasttext"].artifact_path:
            dest = out_dir / "winner_fasttext.bin"
            shutil.copy2(families["D_fasttext"].artifact_path, dest)
            winner_manifest["artifact"] = str(dest)
        elif wf == "E_finbert" and families["E_finbert"].artifact_path:
            dest = out_dir / "winner_finbert"
            shutil.copytree(families["E_finbert"].artifact_path, dest, dirs_exist_ok=True)
            winner_manifest["artifact"] = str(dest)
        elif wf == "F_setfit" and families["F_setfit"].artifact_path:
            dest = out_dir / "winner_setfit"
            shutil.copytree(families["F_setfit"].artifact_path, dest, dirs_exist_ok=True)
            winner_manifest["artifact"] = str(dest)
        elif wf in ("G_ensemble_mean", "H_ensemble_rankmean"):
            winner_manifest.update(
                {
                    "ensemble_method": "mean_probability" if wf == "G_ensemble_mean" else "rank_average",
                    "component_family_ids": top5,
                    "component_artifacts": {
                        fid: families[fid].artifact_path or ("code:rule_pseudo_probability_from_handcrafted" if fid == "A_rules" else None)
                        for fid in top5
                    },
                }
            )
        elif wf == "A_rules":
            winner_manifest["artifact"] = "code:rule_pseudo_probability_from_handcrafted"
        (out_dir / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2), encoding="utf-8")

    if cfg.show_progress:
        print("[zoo] done.", flush=True)

    return {
        "config": meta,
        "leaderboard": leaderboard,
        "winner": winner,
        "families_artifact": {k: v.artifact_path for k, v in families.items()},
    }


def run_classifier_zoo_from_gold(
    *,
    gold_labeled_path: Path | None = None,
    cfg: ZooRunConfig | None = None,
) -> dict[str, Any]:
    """Load gold parquet, stratified split (``feature.md``), then ``run_classifier_zoo``."""
    cfg = cfg or ZooRunConfig()
    split_df, _meta = stratified_supervision_split(gold_labeled_path=gold_labeled_path, seed=cfg.seed)
    return run_classifier_zoo(split_df, cfg)
