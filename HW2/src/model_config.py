from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

RECALL_FLOOR_DEFAULT = 0.96
POS_LABEL = 1
NEG_LABEL = 0
OOF_N_SPLITS = 5

FAMILY_TIE_BREAK_ORDER: tuple[str, ...] = (
    "A_rules",
    "B_logreg_enriched",
    "C_histgb_enriched",
    "D_fasttext",
    "E_finbert",
    "F_setfit",
    "G_ensemble_mean",
    "H_ensemble_rankmean",
)

NON_TRANSFORMER_ENSEMBLE_POOL: tuple[str, ...] = (
    "A_rules",
    "B_logreg_enriched",
    "C_histgb_enriched",
    "D_fasttext",
)

RULE_BOILERPLATE_FEATURE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 9, 16, 18, 19)
RULE_SUBSTANTIVE_FEATURE_INDICES: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 28, 29)


@dataclass
class ZooRunConfig:
    """
    All classifier-zoo hyperparameters in one place.

    Defaults target the full requirements pipeline on an 8 GB NVIDIA GPU. The training
    code automatically falls back to CPU-safe behavior when CUDA is unavailable.
    Substantive is the positive class (``y=1``).
    """

    seed: int = 42
    recall_floor: float = RECALL_FLOOR_DEFAULT

    sentence_transformers_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 256

    sklearn_max_iter: int = 2000
    logreg_solver: str = "lbfgs"
    logreg_class_weight: str | dict[int, float] | None = "balanced"

    histgb_max_depth: int = 8
    histgb_learning_rate: float = 0.06
    histgb_max_iter: int = 200

    fasttext_lr: float = 0.5
    fasttext_epoch: int = 25
    fasttext_dim: int = 50
    fasttext_word_ngrams: int = 2
    fasttext_min_count: int = 1

    finbert_model_id: str = "ProsusAI/finbert"
    finbert_max_length: int = 256
    finbert_epochs: int = 2
    finbert_learning_rate: float = 2e-5
    finbert_batch_size: int = 16
    finbert_eval_batch_size: int = 32
    finbert_gradient_accumulation_steps: int = 1
    finbert_dataloader_num_workers: int = 2
    finbert_use_fp16_if_cuda: bool = True
    finbert_seed_offset: int = 11_000

    setfit_batch_size: int | tuple[int, int] = 128
    setfit_num_epochs: tuple[int, int] = (1, 2)
    setfit_num_iterations: int | None = 10
    setfit_samples_per_label: int = 2
    setfit_max_length: int | None = 256
    setfit_use_amp_if_cuda: bool = True
    setfit_seed_offset: int = 12_000

    winner_requires_test_recall: bool = True
    prefer_ensemble_on_tie: bool = True
    cache_embeddings: bool = True
    winner_artifact_dir: Path | None = None

    skip_finbert: bool = False
    skip_setfit: bool = False
    show_progress: bool = True

    leaderboard_csv_name: str = "zoo_leaderboard.csv"
    leaderboard_json_name: str = "zoo_leaderboard.json"

    _GPU_BATCH_NOTE: ClassVar[str] = "Defaults are sized for an 8 GB CUDA GPU."

    @classmethod
    def for_8gb_gpu(cls, **overrides: object) -> "ZooRunConfig":
        """Explicit constructor for the target machine profile."""
        return cls(**overrides)

    @classmethod
    def quick_non_transformer(cls, **overrides: object) -> "ZooRunConfig":
        """Fast local smoke profile; does not satisfy the full model-family requirement."""
        base = {
            "skip_finbert": True,
            "skip_setfit": True,
            "show_progress": False,
        }
        base.update(overrides)
        return cls(**base)
