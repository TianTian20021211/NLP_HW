"""Gold-standard pipeline CLI. Part 1: pool → Part 2: Ollama judges → Part 3: Haiku preview/merge.

``notebooks/gold_standard_overview.ipynb`` calls ``main([])`` for the final demo; same as invoking the CLI with no arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run gold-standard pipeline (sentence pool → Ollama labels → merge preview / Haiku).",
    )
    p.add_argument(
        "--mode",
        choices=("full", "smoke"),
        default="full",
        help="full writes under cache/; smoke writes under cache/nb_smoke/ with caps.",
    )
    p.add_argument("--use-api-check", action="store_true", help="Call Anthropic Haiku merge (needs ANTHROPIC_API_KEY).")
    p.add_argument("--use-cache-lock", action="store_true", help="Enable exclusive cache lock during extraction/labeling.")
    p.add_argument("--skip-sentences", action="store_true", help="Skip Part 1 (require existing sentence_pool.parquet).")
    p.add_argument("--skip-labeling", action="store_true", help="Skip Part 2 (require existing gold_sample + judge caches).")
    p.add_argument("--skip-merge", action="store_true", help="Skip Part 3 (preview / merge).")
    p.add_argument("--smoke-max-transcripts", type=int, default=3)
    p.add_argument("--smoke-max-pool-rows", type=int, default=1200)
    p.add_argument("--smoke-gold-n", type=int, default=12)
    p.add_argument("--full-gold-n", type=int, default=3000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated Ollama model names; default uses src.gold_labeling.DEFAULT_OLLAMA_MODELS.",
    )
    p.add_argument(
        "--force-all",
        action="store_true",
        help="Ignore on-disk caches: rebuild pool, resample gold rows, and re-run Haiku merge.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run sentence pool extraction, Ollama judging, and optional Haiku merge per CLI flags."""
    args = _parse_args(argv)
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.errors import error
    from src.paths import CACHE_DIR, RUBRIC_PATH, TRANSCRIPTS_DIR
    from src.gold_labeling import (
        DEFAULT_OLLAMA_MODELS,
        DEFAULT_OLLAMA_URL,
        GoldRunConfig,
        atomic_write_gold_sample,
        atomic_write_parquet,
        disagreement_summary,
        haiku_api_usage_preview,
        labels_cache_path,
        load_rubric,
        merge_vote_and_arbitrate,
        run_ollama_labels_for_model,
        sample_gold_table,
        try_load_cached_gold_sample,
        write_gold_meta,
    )
    from src.sentence_extraction import cache_dir_lock, extract_sentence_pool

    import pandas as pd

    if args.mode == "smoke":
        work_cache = root / "cache" / "nb_smoke"
        max_transcripts = args.smoke_max_transcripts
        max_pool_rows = args.smoke_max_pool_rows
        gold_n_target = args.smoke_gold_n
    else:
        work_cache = CACHE_DIR
        max_transcripts = None
        max_pool_rows = None
        gold_n_target = args.full_gold_n

    work_cache.mkdir(parents=True, exist_ok=True)
    pool_path = work_cache / "sentence_pool.parquet"
    sample_path = work_cache / "gold_sample.parquet"
    labeled_path = work_cache / "gold_labeled.parquet"
    meta_path = work_cache / "gold_run_meta.json"
    lock_path = work_cache / ".lock"
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(DEFAULT_OLLAMA_MODELS)
    )
    if not models:
        error("--models must list at least one Ollama model when set")

    print("ROOT", root)
    print("WORK_CACHE", work_cache)
    print("max_transcripts", max_transcripts, "max_pool_rows", max_pool_rows, "gold_n_target", gold_n_target)
    print("models", models)

    if not args.skip_sentences:
        pool_df, pool_cached = extract_sentence_pool(
            TRANSCRIPTS_DIR,
            pool_path,
            max_transcripts=max_transcripts,
            max_pool_rows=max_pool_rows,
            use_cache_lock=args.use_cache_lock,
            lock_path=lock_path,
            force_rebuild=args.force_all,
        )
        if pool_cached:
            print("loaded pool rows from cache", len(pool_df), "->", pool_path.resolve())
        else:
            print("wrote pool rows", len(pool_df), "->", pool_path.resolve())
    elif not pool_path.exists():
        error(f"Missing pool at {pool_path}; run Part 1 or omit --skip-sentences")

    pool_df = pd.read_parquet(pool_path)
    print("pool rows", len(pool_df))
    print(pool_df.head(3).to_string(index=False))

    if not args.skip_labeling:
        gold_n = min(int(gold_n_target), int(len(pool_df)))
        if gold_n <= 0:
            error("pool empty; run Part 1 first")
        rubric = load_rubric(RUBRIC_PATH)
        seed = int(args.random_seed)
        if not args.force_all:
            cached_sample = try_load_cached_gold_sample(
                sample_path,
                pool_path=pool_path,
                gold_n=gold_n,
                seed=seed,
            )
        else:
            cached_sample = None
        if cached_sample is not None:
            sample_tbl = cached_sample
            print("loaded gold sample from cache", len(sample_tbl), "->", sample_path.resolve())
        else:
            sample_tbl = sample_gold_table(pool_df, gold_n, seed)
            atomic_write_gold_sample(sample_tbl, sample_path, pool_path=pool_path, gold_n=gold_n, seed=seed)
            print("gold sample", len(sample_tbl), "->", sample_path.resolve())
        with cache_dir_lock(lock_path, args.use_cache_lock):
            for model in models:
                out_path = labels_cache_path(model, labels_dir=work_cache)
                run_ollama_labels_for_model(
                    sample_tbl,
                    model,
                    out_path,
                    rubric=rubric,
                    base_url=DEFAULT_OLLAMA_URL,
                )
                print("judge cache", model, "->", out_path.resolve())
    elif not sample_path.exists():
        error(f"Missing sample at {sample_path}; run Part 2 or omit --skip-labeling")

    sample_tbl = pd.read_parquet(sample_path)
    print("sample rows", len(sample_tbl))

    if not args.skip_merge:
        _, api_prev = haiku_api_usage_preview(sample_tbl, models, work_cache)
        print("haiku_api_usage_preview")
        print(json.dumps(api_prev, indent=2, sort_keys=True))
        if args.use_api_check:
            sample_mtime = sample_path.stat().st_mtime_ns
            labeled_still_valid = (
                labeled_path.exists()
                and labeled_path.stat().st_mtime_ns >= sample_mtime
            )
            if labeled_still_valid and not args.force_all:
                print("loaded gold_labeled from cache", labeled_path.resolve())
            else:
                rubric = load_rubric(RUBRIC_PATH)
                merged = merge_vote_and_arbitrate(
                    sample_tbl,
                    models,
                    rubric=rubric,
                    labels_dir=work_cache,
                )
                atomic_write_parquet(merged, labeled_path)
                summ = disagreement_summary(merged, models)
                cfg = GoldRunConfig(
                    gold_n=int(len(sample_tbl)),
                    seed=int(args.random_seed),
                    models=tuple(models),
                    ollama_url=str(DEFAULT_OLLAMA_URL),
                )
                write_gold_meta(meta_path, cfg=cfg, summary=summ)
                print("wrote", labeled_path.resolve())
                print(json.dumps(summ, indent=2, sort_keys=True))
        else:
            print("merge skipped (pass --use-api-check to call Anthropic and write gold_labeled.parquet)")
    else:
        print("SKIP Part 3")

    if labeled_path.exists():
        df = pd.read_parquet(labeled_path)
        print("gold_labeled exists rows", len(df), "cols", sorted(df.columns.tolist()))
        if {"discord", "gold_final"}.issubset(df.columns):
            print(df[["sentence_id", "gold_final", "discord", "api_used"]].head(5).to_string(index=False))

    if meta_path.exists():
        print("meta excerpt")
        print(meta_path.read_text(encoding="utf-8")[:1200])


if __name__ == "__main__":
    main()
