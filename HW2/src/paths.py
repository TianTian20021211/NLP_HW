from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
DOCS_DIR = DATA_DIR / "docs"
RUBRIC_PATH = DOCS_DIR / "labeling_rubric.md"
CACHE_DIR = PROJECT_ROOT / "cache"
SENTENCE_POOL_PATH = CACHE_DIR / "sentence_pool.parquet"
GOLD_SAMPLE_PATH = CACHE_DIR / "gold_sample.parquet"
GOLD_LABELED_PATH = CACHE_DIR / "gold_labeled.parquet"
GOLD_META_PATH = CACHE_DIR / "gold_run_meta.json"
CACHE_LOCK_PATH = CACHE_DIR / ".lock"
OLLAMA_FAIL_LOG = CACHE_DIR / "logs" / "ollama_failed_rows.jsonl"
