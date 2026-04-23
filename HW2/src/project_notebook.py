from __future__ import annotations

import contextlib
import io
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.errors import error
from src.handcrafted_features import FEATURE_COLUMNS, handcrafted_features_frame
from src.model_config import ZooRunConfig
from src.paths import CACHE_DIR, GOLD_LABELED_PATH, PROJECT_ROOT
from src.train_split import split_metadata_to_json, stratified_supervision_split

GOLD_LOG_PREFIXES = (
    "max_transcripts ",
    "loaded pool rows from cache",
    "wrote pool rows",
    "pool rows ",
    "loaded gold sample from cache",
    "gold sample ",
    "sample rows ",
    "judge cache ",
    "haiku_api_usage_preview",
    "merge skipped",
    "loaded gold_labeled",
    "wrote ",
    "gold_labeled exists rows",
)

LEADERBOARD_COLUMNS = (
    "family_id",
    "test_macro_f1",
    "test_accuracy",
    "test_precision_boilerplate",
    "test_recall_boilerplate",
    "test_f1_boilerplate",
    "test_precision_substantive",
    "test_recall_substantive",
    "test_f1_substantive",
    "test_confusion_matrix",
    "eligible_oof",
    "threshold",
    "threshold_std_across_folds",
    "train_seconds",
    "approx_infer_sents_per_sec",
    "notes",
)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _print_json(title: str, obj: Any) -> None:
    print(f"{title}:")
    print(_json_dumps(obj))


def _selected_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": "not installed", "cuda_available": False}
    info: dict[str, Any] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def print_environment_report(cfg: ZooRunConfig | None = None) -> None:
    """Small reproducibility block for the notebook."""
    cfg = cfg or ZooRunConfig.for_8gb_gpu()
    info = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cache_dir": str(CACHE_DIR),
        "gold_labeled_exists": GOLD_LABELED_PATH.exists(),
        "target_profile": "8 GB CUDA GPU defaults, CPU fallback if CUDA is unavailable",
        "torch": _torch_environment(),
        "model_config": {
            "sentence_transformers_model_id": cfg.sentence_transformers_model_id,
            "embedding_batch_size": cfg.embedding_batch_size,
            "finbert_batch_size": cfg.finbert_batch_size,
            "finbert_eval_batch_size": cfg.finbert_eval_batch_size,
            "setfit_batch_size": cfg.setfit_batch_size,
            "setfit_num_epochs": cfg.setfit_num_epochs,
            "setfit_num_iterations": cfg.setfit_num_iterations,
            "skip_finbert": cfg.skip_finbert,
            "skip_setfit": cfg.skip_setfit,
        },
    }
    _print_json("Environment", info)


def run_gold_standard_pipeline_report(
    *,
    seed: int = 42,
    extra_argv: list[str] | None = None,
) -> None:
    """
    Run ``gold_standard.main`` and print the useful notebook-scale summary lines.

    Add ``extra_argv=["--use-api-check"]`` when you want the notebook to call
    Anthropic Haiku and rewrite ``cache/gold_labeled.parquet``.
    """
    import gold_standard

    argv = ["--mode", "full", "--random-seed", str(seed)]
    if extra_argv:
        argv.extend(list(extra_argv))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gold_standard.main(argv)
    log = buf.getvalue()

    print("Gold pipeline: completed.")
    for line in log.splitlines():
        s = line.strip()
        if not s or len(s) > 300:
            continue
        if any(s.startswith(prefix) for prefix in GOLD_LOG_PREFIXES):
            print(s)

    m = re.search(r'"frac_needs_haiku_api":\s*([0-9.]+)', log)
    if m:
        print("frac_needs_haiku_api:", m.group(1))

    if GOLD_LABELED_PATH.exists():
        n = len(pd.read_parquet(GOLD_LABELED_PATH))
        print(f"OK: {GOLD_LABELED_PATH.name} rows={n}")
    else:
        print(f"Missing {GOLD_LABELED_PATH.name}; pass --use-api-check or provide the cached parquet.")


def build_split_and_feature_preview(
    *,
    seed: int = 42,
    gold_labeled_path: Path | None = None,
    preview_rows: int = 5,
) -> dict[str, Any]:
    path = gold_labeled_path or GOLD_LABELED_PATH
    if not path.exists():
        error(f"Missing {path}: run the gold pipeline or place gold_labeled.parquet under cache/.")

    split_df, split_meta = stratified_supervision_split(gold_labeled_path=path, seed=seed)
    preview_cols = ["sentence_id", "source_file", "gold_final", "split"]
    feature_preview = handcrafted_features_frame(split_df["text"].head(max(preview_rows, 1)))
    return {
        "split_df": split_df,
        "split_meta": split_meta,
        "split_preview": split_df[preview_cols].head(preview_rows),
        "feature_preview": feature_preview.head(preview_rows),
    }


def run_stratified_split_and_handcrafted_features_report(
    *,
    seed: int = 42,
    gold_labeled_path: Path | None = None,
    preview_rows: int = 5,
) -> dict[str, Any]:
    """Run workflow steps 3 and 4, then print compact split and feature previews."""
    result = build_split_and_feature_preview(
        seed=seed,
        gold_labeled_path=gold_labeled_path,
        preview_rows=preview_rows,
    )
    meta = json.loads(split_metadata_to_json(result["split_meta"]))
    core = {
        k: meta[k]
        for k in (
            "rows_input",
            "rows_supervision",
            "seed",
            "sentences_per_split",
            "transcripts_per_split",
            "transcripts_total",
        )
    }
    print("Split and handcrafted features: completed.")
    _print_json("Split core metadata", core)
    print(result["split_preview"].to_string(index=False))
    print(result["feature_preview"].iloc[:, :8].to_string(index=False))
    print("feature_columns:", len(FEATURE_COLUMNS))
    return result


def leaderboard_frame(zoo_result: dict[str, Any]) -> pd.DataFrame:
    """Return a stable leaderboard frame from a model-zoo result dict."""
    lb = pd.DataFrame(zoo_result.get("leaderboard", []))
    if lb.empty:
        return lb
    return lb[_selected_columns(lb, LEADERBOARD_COLUMNS)]


def save_zoo_outputs(
    zoo_result: dict[str, Any],
    *,
    cfg: ZooRunConfig,
    out_dir: Path = CACHE_DIR,
) -> dict[str, str]:
    """Persist notebook-friendly leaderboard and winner summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    lb = leaderboard_frame(zoo_result)
    if not lb.empty:
        csv_path = out_dir / cfg.leaderboard_csv_name
        json_path = out_dir / cfg.leaderboard_json_name
        lb.to_csv(csv_path, index=False)
        json_path.write_text(lb.to_json(orient="records", indent=2), encoding="utf-8")
        paths["leaderboard_csv"] = str(csv_path)
        paths["leaderboard_json"] = str(json_path)

    winner_dir = Path(zoo_result.get("config", {}).get("winner_artifact_dir") or (CACHE_DIR / "zoo_winner"))
    winner_dir.mkdir(parents=True, exist_ok=True)
    winner_path = winner_dir / "winner_summary.json"
    winner_path.write_text(_json_dumps(zoo_result.get("winner")), encoding="utf-8")
    paths["winner_summary"] = str(winner_path)
    manifest_path = winner_dir / "winner_manifest.json"
    if manifest_path.exists():
        paths["winner_manifest"] = str(manifest_path)
    return paths


def run_classifier_zoo_report(
    cfg: ZooRunConfig | None = None,
    *,
    gold_labeled_path: Path | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Run workflow steps 5-7, print the leaderboard, save result tables, and return
    the structured model-zoo output.
    """
    from src.model_zoo import run_classifier_zoo_from_gold

    cfg = cfg or ZooRunConfig.for_8gb_gpu()
    print("Classifier zoo config:")
    _print_json(
        "Key hyperparameters",
        {
            "recall_floor": cfg.recall_floor,
            "embedding_batch_size": cfg.embedding_batch_size,
            "finbert_batch_size": cfg.finbert_batch_size,
            "finbert_eval_batch_size": cfg.finbert_eval_batch_size,
            "setfit_batch_size": cfg.setfit_batch_size,
            "setfit_num_epochs": cfg.setfit_num_epochs,
            "setfit_num_iterations": cfg.setfit_num_iterations,
            "skip_finbert": cfg.skip_finbert,
            "skip_setfit": cfg.skip_setfit,
        },
    )

    out = run_classifier_zoo_from_gold(gold_labeled_path=gold_labeled_path, cfg=cfg)
    lb = leaderboard_frame(out)
    if lb.empty:
        print("No leaderboard rows produced.")
    else:
        print(lb.to_string(index=False))

    print("Winner:")
    print(_json_dumps(out.get("winner")))

    if save_results:
        saved = save_zoo_outputs(out, cfg=cfg)
        _print_json("Saved result artifacts", saved)
        out["saved_result_artifacts"] = saved
    return out
