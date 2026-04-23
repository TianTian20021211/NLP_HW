from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.handcrafted_features import handcrafted_features_matrix
from src.model_config import POS_LABEL, ZooRunConfig
from src.model_zoo import (
    _ensure_fasttext_numpy2_compat,
    _fasttext_p_sub_batch,
    _setfit_positive_class_probability,
    _transformer_p_sub_batches,
    rank_average_scores,
    rule_pseudo_probability_from_handcrafted,
)
from src.paths import CACHE_DIR, PROJECT_ROOT
from src.sentence_extraction import extract_sentence_rows_from_text


@dataclass(frozen=True)
class SentencePrediction:
    sentence_id: str
    text: str
    p_substantive: float
    label: str
    component_scores: dict[str, float]


@dataclass(frozen=True)
class DocumentPrediction:
    source_name: str
    family_id: str
    threshold: float
    component_family_ids: tuple[str, ...]
    extraction_meta: dict[str, int | str]
    sentences: list[SentencePrediction]

    @property
    def boilerplate_count(self) -> int:
        return sum(1 for sent in self.sentences if sent.label == "Boilerplate")

    @property
    def substantive_count(self) -> int:
        return sum(1 for sent in self.sentences if sent.label == "Substantive")

    @property
    def classified_count(self) -> int:
        return len(self.sentences)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(raw: str | None, *, winner_dir: Path) -> Path | None:
    if not raw or raw.startswith("code:"):
        return None
    path = Path(raw).expanduser()
    if path.exists():
        return path
    if not path.is_absolute():
        rel = PROJECT_ROOT / path
        if rel.exists():
            return rel
        rel_winner = winner_dir / path
        if rel_winner.exists():
            return rel_winner

    by_name_candidates = [
        CACHE_DIR / path.name,
        winner_dir / path.name,
        PROJECT_ROOT / path.name,
    ]
    for candidate in by_name_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Model artifact not found: {raw}")


def _hf_cache_snapshot(model_id: str) -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub / f"models--{model_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        snapshot = repo_dir / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            return dirs[0]
    return None


def _resolve_sentence_transformer_model(model_id: str) -> str:
    env_path = os.environ.get("HW2_SENTENCE_TRANSFORMER_MODEL")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    model_tail = model_id.split("/")[-1]
    candidates.extend(
        [
            PROJECT_ROOT / "models" / model_id,
            PROJECT_ROOT / "models" / model_tail,
            CACHE_DIR / "models" / model_id,
            CACHE_DIR / "models" / model_tail,
            CACHE_DIR / model_id,
            CACHE_DIR / model_tail,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    cached = _hf_cache_snapshot(model_id)
    if cached is not None:
        return str(cached)
    return model_id


class WinnerPredictor:
    def __init__(
        self,
        *,
        family_id: str,
        threshold: float,
        component_family_ids: tuple[str, ...],
        ensemble_method: str | None,
        sentence_transformer_model_id: str,
        embedding_batch_size: int,
        components: dict[str, Any],
    ) -> None:
        self.family_id = family_id
        self.threshold = threshold
        self.component_family_ids = component_family_ids
        self.ensemble_method = ensemble_method
        self.sentence_transformer_model_id = sentence_transformer_model_id
        self.embedding_batch_size = embedding_batch_size
        self.components = components
        self._sentence_transformer: Any | None = None

    @classmethod
    def load(cls, winner_dir: Path | None = None) -> "WinnerPredictor":
        winner_dir = winner_dir or (CACHE_DIR / "zoo_winner")
        manifest = _read_json(winner_dir / "winner_manifest.json")
        run_meta_path = winner_dir / "zoo_run_meta.json"
        run_meta = _read_json(run_meta_path) if run_meta_path.exists() else {}

        family_id = str(manifest["family_id"])
        threshold_raw = manifest.get("threshold")
        if threshold_raw is None:
            threshold_raw = (winner_dir / "winner_threshold.txt").read_text(encoding="utf-8").strip()
        threshold = float(threshold_raw)

        component_family_ids = tuple(
            manifest.get("component_family_ids") or ([family_id] if family_id else [])
        )
        ensemble_method = manifest.get("ensemble_method")
        component_artifacts = dict(manifest.get("component_artifacts") or {})
        if not component_artifacts and manifest.get("artifact"):
            component_artifacts[family_id] = manifest["artifact"]
        if family_id == "A_rules":
            component_artifacts[family_id] = "code:rule_pseudo_probability_from_handcrafted"

        components: dict[str, Any] = {}
        for component_id in component_family_ids:
            components[component_id] = _load_component(
                component_id,
                component_artifacts.get(component_id),
                winner_dir=winner_dir,
            )

        hyper = dict(run_meta.get("model_hyperparameters") or {})
        default_cfg = ZooRunConfig()
        st_model_id = str(
            run_meta.get("sentence_transformers_model_id")
            or hyper.get("sentence_transformers_model_id")
            or default_cfg.sentence_transformers_model_id
        )
        batch_size = int(hyper.get("embedding_batch_size") or default_cfg.embedding_batch_size)
        return cls(
            family_id=family_id,
            threshold=threshold,
            component_family_ids=component_family_ids,
            ensemble_method=str(ensemble_method) if ensemble_method is not None else None,
            sentence_transformer_model_id=st_model_id,
            embedding_batch_size=batch_size,
            components=components,
        )

    def predict_document(self, text: str, source_name: str = "uploaded.txt") -> DocumentPrediction:
        rows, meta = extract_sentence_rows_from_text(text, source_name)
        texts = [str(row["text"]) for row in rows]
        probabilities, component_scores = self.predict_texts(texts)
        predictions: list[SentencePrediction] = []
        for row, prob, scores in zip(rows, probabilities.tolist(), component_scores):
            label = "Substantive" if prob >= self.threshold else "Boilerplate"
            predictions.append(
                SentencePrediction(
                    sentence_id=str(row["sentence_id"]),
                    text=str(row["text"]),
                    p_substantive=float(prob),
                    label=label,
                    component_scores=scores,
                )
            )
        return DocumentPrediction(
            source_name=str(meta["source_file"]),
            family_id=self.family_id,
            threshold=self.threshold,
            component_family_ids=self.component_family_ids,
            extraction_meta=meta,
            sentences=predictions,
        )

    def predict_texts(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, float]]]:
        if not texts:
            return np.zeros(0, dtype=np.float64), []
        h = handcrafted_features_matrix(texts)
        x: np.ndarray | None = None
        component_arrays: dict[str, np.ndarray] = {}

        for component_id in self.component_family_ids:
            component = self.components[component_id]
            if component_id == "A_rules":
                p = rule_pseudo_probability_from_handcrafted(h)
            elif component_id in {"B_logreg_enriched", "C_histgb_enriched"}:
                if x is None:
                    emb = self._encode_embeddings(texts)
                    x = np.hstack([emb, h])
                expected_dim = int(component.get("feature_dim") or x.shape[1])
                if x.shape[1] != expected_dim:
                    raise ValueError(f"{component_id} expected feature_dim={expected_dim}, got {x.shape[1]}")
                model = component["pipeline"] if component_id == "B_logreg_enriched" else component["model"]
                p = np.asarray(model.predict_proba(x)[:, POS_LABEL], dtype=np.float64)
            elif component_id == "D_fasttext":
                p = _fasttext_p_sub_batch(component, texts, show_progress=False)
            elif component_id == "E_finbert":
                model, tokenizer, max_length, batch_size = component
                p = _transformer_p_sub_batches(
                    model,
                    tokenizer,
                    texts,
                    batch_size=batch_size,
                    max_length=max_length,
                    show_progress=False,
                    desc="FinBERT GUI",
                )
            elif component_id == "F_setfit":
                p = _setfit_positive_class_probability(component.predict_proba(texts), context="GUI")
            else:
                raise ValueError(f"Unsupported winner component: {component_id}")
            component_arrays[component_id] = np.asarray(p, dtype=np.float64)

        matrix = np.stack([component_arrays[cid] for cid in self.component_family_ids], axis=1)
        if self.ensemble_method == "mean_probability" or self.family_id == "G_ensemble_mean":
            final_p = matrix.mean(axis=1)
        elif self.ensemble_method == "rank_average" or self.family_id == "H_ensemble_rankmean":
            final_p = rank_average_scores(matrix)
        elif len(self.component_family_ids) == 1:
            final_p = matrix[:, 0]
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

        score_rows = [
            {component_id: float(component_arrays[component_id][i]) for component_id in self.component_family_ids}
            for i in range(len(texts))
        ]
        return np.asarray(final_p, dtype=np.float64), score_rows

    def _encode_embeddings(self, texts: list[str]) -> np.ndarray:
        if self._sentence_transformer is None:
            from sentence_transformers import SentenceTransformer

            resolved_model = _resolve_sentence_transformer_model(self.sentence_transformer_model_id)
            self._sentence_transformer = SentenceTransformer(resolved_model)
        emb = self._sentence_transformer.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float64)


def _load_component(component_id: str, artifact: str | None, *, winner_dir: Path) -> Any:
    if component_id == "A_rules":
        return "code:rule_pseudo_probability_from_handcrafted"
    path = _resolve_artifact_path(artifact, winner_dir=winner_dir)
    if component_id == "B_logreg_enriched":
        if path is None:
            path = _resolve_artifact_path("zoo_B_logreg_pipeline.pkl", winner_dir=winner_dir)
        return pickle.loads(path.read_bytes())
    if component_id == "C_histgb_enriched":
        if path is None:
            path = _resolve_artifact_path("zoo_C_histgb.pkl", winner_dir=winner_dir)
        return pickle.loads(path.read_bytes())
    if component_id == "D_fasttext":
        if path is None:
            path = _resolve_artifact_path("zoo_D_fasttext.bin", winner_dir=winner_dir)
        import fasttext

        _ensure_fasttext_numpy2_compat()
        return fasttext.load_model(str(path))
    if component_id == "E_finbert":
        if path is None:
            path = winner_dir / "winner_finbert"
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        meta = _read_json(winner_dir / "zoo_run_meta.json") if (winner_dir / "zoo_run_meta.json").exists() else {}
        finbert_meta = dict(meta.get("finbert") or {})
        return model, tokenizer, int(finbert_meta.get("max_length") or 256), int(finbert_meta.get("eval_batch_size") or 32)
    if component_id == "F_setfit":
        if path is None:
            path = winner_dir / "winner_setfit"
        from setfit import SetFitModel

        return SetFitModel.from_pretrained(str(path))
    raise ValueError(f"Unsupported winner component: {component_id}")


def load_winner_predictor() -> WinnerPredictor:
    return WinnerPredictor.load()
