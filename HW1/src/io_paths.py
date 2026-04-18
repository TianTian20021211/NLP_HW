"""Centralized filesystem paths for the HW1 pipeline.

All other modules read/write under these roots so that the notebook stays
free of hard-coded paths and caches can be relocated by editing one place.
"""

from __future__ import annotations

from pathlib import Path


HW1_ROOT = Path(__file__).resolve().parent.parent

DATA = HW1_ROOT / "data"
TRANSCRIPTS = DATA / "transcripts"
EXTERNAL = DATA / "external"
PRICES = DATA / "cache" / "prices"

CACHE = DATA / "cache"
UNITS = CACHE / "units"
SENTIMENT = CACHE / "sentiment"
SENTIMENT_LM = CACHE / "sentiment_lm"
EXTRACTIONS = CACHE / "extractions"
CALLS = CACHE / "calls"
CALLS_LM = CACHE / "calls_lm"
FEATURES = CACHE / "features"
LABELS = CACHE / "labels"
MODELS = CACHE / "models"
BACKTEST = CACHE / "backtest"
QC_DIR = CACHE / "qc"
PLOTS = CACHE / "plots"
ANALYSIS = CACHE / "analysis"
REPORT_DIR = HW1_ROOT


def error(msg: str) -> None:
    """Raise a RuntimeError with `msg` (pipeline convention: fail loud)."""
    raise RuntimeError(msg)


def ensure_dirs() -> None:
    """Create every cache directory the pipeline writes to.

    Idempotent; safe to call at the top of any stage driver.
    """
    for p in (
        UNITS,
        SENTIMENT,
        SENTIMENT_LM,
        EXTRACTIONS,
        CALLS,
        CALLS_LM,
        PRICES,
        FEATURES,
        LABELS,
        MODELS,
        BACKTEST,
        QC_DIR,
        PLOTS,
        ANALYSIS,
        EXTERNAL,
    ):
        p.mkdir(parents=True, exist_ok=True)


def transcript_path(ticker: str, quarter: str) -> Path:
    """Return the expected on-disk path for a `(ticker, quarter)` transcript.

    `quarter` is in the canonical `"Q<q>-<yyyy>"` form used in filenames.
    """
    return TRANSCRIPTS / f"{ticker}_{quarter}.txt"


def units_path(ticker: str, quarter: str) -> Path:
    return UNITS / f"{ticker}_{quarter}.jsonl"


def sentiment_path(unit_id: str) -> Path:
    return SENTIMENT / f"{unit_id}.json"


def sentiment_lm_path(unit_id: str) -> Path:
    return SENTIMENT_LM / f"{unit_id}.json"


def extraction_path(model: str, unit_id: str) -> Path:
    """Per-model, per-unit LLM extraction cache path.

    Model slugs like `gemma3:4b` get sanitized so they are safe as folder names.
    """
    model_dir = EXTRACTIONS / model.replace("/", "_").replace(":", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{unit_id}.json"


def extraction_failure_path(model: str, unit_id: str) -> Path:
    model_dir = EXTRACTIONS / model.replace("/", "_").replace(":", "_") / "_failures"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{unit_id}.txt"


def _qa_group_dir(model: str) -> Path:
    """Sub-folder that holds QA batched-per-officer extractions.

    Kept separate from the per-unit presenter extractions so a stale-cache
    sweep on one stage (`extraction` vs `extraction_qa_group`) never
    touches the other stage's files.
    """
    slug = model.replace("/", "_").replace(":", "_")
    return EXTRACTIONS / slug / "qa_groups"


def qa_group_extraction_path(model: str, call_id: str, role: str) -> Path:
    """Per-model, per-officer Q&A batched extraction cache path.

    ``call_id`` is ``"<TICKER>_<Q<q>-<yyyy>>"``, ``role`` is the speaker role
    slug (CEO / CFO / CTO / IR / Other_Exec / Unknown).
    """
    d = _qa_group_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    safe_role = role.replace("/", "_")
    return d / f"{call_id}__{safe_role}.json"


def qa_group_failure_path(model: str, call_id: str, role: str) -> Path:
    d = _qa_group_dir(model) / "_failures"
    d.mkdir(parents=True, exist_ok=True)
    safe_role = role.replace("/", "_")
    return d / f"{call_id}__{safe_role}.txt"


def call_path(ticker: str, quarter: str) -> Path:
    return CALLS / f"{ticker}_{quarter}.json"


def call_lm_path(ticker: str, quarter: str) -> Path:
    return CALLS_LM / f"{ticker}_{quarter}.json"


def price_path(ticker: str) -> Path:
    return PRICES / f"{ticker}.parquet"


def feature_table_path() -> Path:
    return FEATURES / "feature_table.parquet"


def labels_path() -> Path:
    return LABELS / "labels.parquet"


def _artifact_slug(extraction: str, model: str, horizon: str) -> str:
    return f"{extraction}__{model}__{horizon}"


def model_path(extraction: str, model: str, horizon: str) -> Path:
    """Pickled learned model (logreg / xgb / ridge only; rule has no pkl)."""
    return MODELS / f"{_artifact_slug(extraction, model, horizon)}.pkl"


def preds_path(extraction: str, model: str, horizon: str) -> Path:
    return MODELS / f"preds_{_artifact_slug(extraction, model, horizon)}.parquet"


def backtest_path(extraction: str, model: str, horizon: str) -> Path:
    return BACKTEST / f"{_artifact_slug(extraction, model, horizon)}.json"


def equity_path(extraction: str, model: str, horizon: str) -> Path:
    return BACKTEST / f"equity_{_artifact_slug(extraction, model, horizon)}.csv"


def backtest_summary_path() -> Path:
    return BACKTEST / "summary.parquet"


def plot_path(name: str) -> Path:
    """PNG output path under ``data/cache/plots/``."""
    return PLOTS / f"{name}.png"


def analysis_path(name: str) -> Path:
    """JSON / parquet output path under ``data/cache/analysis/``."""
    return ANALYSIS / name


def list_transcripts() -> list[Path]:
    """All transcript files under `data/transcripts/` sorted lexicographically."""
    if not TRANSCRIPTS.is_dir():
        error(f"transcripts dir missing: {TRANSCRIPTS}")
    return sorted(TRANSCRIPTS.glob("*.txt"))


def parse_stem(path: Path) -> tuple[str, str]:
    """Return `(ticker, quarter)` parsed from a transcript filename.

    Filenames follow `<TICKER>_Q<q>-<yyyy>.txt`; anything else raises.
    """
    stem = path.stem
    if "_" not in stem:
        error(f"bad transcript filename: {path.name}")
    ticker, tail = stem.split("_", 1)
    return ticker, tail
