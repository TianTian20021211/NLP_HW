from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import nltk

_NLTK_DATA_HOME = Path(os.environ.get("NLTK_DATA", "/home/tian/data")).expanduser().resolve()
_NLTK_DATA_HOME.mkdir(parents=True, exist_ok=True)
_resolved = str(_NLTK_DATA_HOME)
if _resolved not in {str(Path(p).resolve()) for p in nltk.data.path}:
    nltk.data.path.insert(0, _resolved)

import pandas as pd
from tqdm import tqdm

from src.errors import error
from src.paths import CACHE_DIR, CACHE_LOCK_PATH, PROJECT_ROOT, SENTENCE_POOL_PATH, TRANSCRIPTS_DIR

MIN_CHARS = 40

# Paragraph boundaries (README, confirmed): blank line, or whole line matching structural/role headers starts a new
# paragraph, then sent_tokenize on that segment. Includes Presentation/Presenter/Q&A headers, Question/Answer,
# Executives (optional suffix), Analysts (optional suffix), standalone Operator; case-insensitive. Sentence-level
# dedup keeps first occurrence per file. Parquet via atomic_write_parquet; optional exclusive cache/.lock.

_STRUCTURAL_LINE_RE = re.compile(
    r"^(?:"
    r"Presentation Operator Message|"
    r"Presenter Speech|"
    r"Question and Answer Operator Message|"
    r"Question|"
    r"Answer|"
    r"Executives(?:\s*-\s*.+)?|"
    r"Analysts(?:\s*-\s*.+)?|"
    r"Operator"
    r")\s*$",
    re.IGNORECASE,
)


def _ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
        return
    except LookupError:
        pass
    try:
        nltk.download("punkt_tab", download_dir=_resolved, quiet=True)
        nltk.data.find("tokenizers/punkt_tab/english")
        return
    except Exception:
        pass
    try:
        nltk.download("punkt", download_dir=_resolved, quiet=True)
    except Exception as exc:
        error(f"Failed to download NLTK punkt data: {exc}")


def dedupe_lines_first_occurrence(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def paragraph_line_breaks(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return bool(_STRUCTURAL_LINE_RE.match(stripped))


def lines_to_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in lines:
        if paragraph_line_breaks(line):
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf).strip())
    return [p for p in paragraphs if p]


def extract_sentences_from_paragraph(
    paragraph: str, source_file: str, start_seq: int
) -> tuple[list[dict], int]:
    from nltk.tokenize import sent_tokenize

    seq = start_seq
    rows: list[dict] = []
    for sent in sent_tokenize(paragraph):
        text = " ".join(sent.split())
        if len(text) < MIN_CHARS:
            continue
        sentence_id = f"{source_file}#{seq:07d}"
        rows.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "source_file": source_file,
            }
        )
        seq += 1
    return rows, seq


def extract_sentence_rows_from_text(
    text: str,
    source_file: str = "uploaded.txt",
) -> tuple[list[dict], dict[str, int | str]]:
    """
    Extract GUI/deployment sentences from one transcript string using the same
    dedupe -> paragraph -> punkt -> MIN_CHARS semantics as the training pool.
    """
    from nltk.tokenize import sent_tokenize

    _ensure_nltk_punkt()
    safe_source = Path(source_file).name or "uploaded.txt"
    raw_lines = text.splitlines()
    deduped_lines = dedupe_lines_first_occurrence(raw_lines)
    boundary_lines = sum(1 for line in deduped_lines if paragraph_line_breaks(line))
    paragraphs = lines_to_paragraphs(deduped_lines)

    rows: list[dict] = []
    seq = 0
    candidate_sentence_count = 0
    short_sentence_count = 0
    for paragraph in paragraphs:
        for sent in sent_tokenize(paragraph):
            clean = " ".join(sent.split())
            if not clean:
                continue
            candidate_sentence_count += 1
            if len(clean) < MIN_CHARS:
                short_sentence_count += 1
                continue
            rows.append(
                {
                    "sentence_id": f"{safe_source}#{seq:07d}",
                    "text": clean,
                    "source_file": safe_source,
                }
            )
            seq += 1

    meta: dict[str, int | str] = {
        "source_file": safe_source,
        "min_chars": MIN_CHARS,
        "raw_line_count": len(raw_lines),
        "deduped_line_count": len(deduped_lines),
        "duplicate_line_count": len(raw_lines) - len(deduped_lines),
        "paragraph_boundary_line_count": boundary_lines,
        "paragraph_count": len(paragraphs),
        "candidate_sentence_count": candidate_sentence_count,
        "short_sentence_count": short_sentence_count,
        "classified_sentence_count": len(rows),
    }
    return rows, meta


def iter_transcript_files(transcripts_dir: Path, max_transcripts: int | None) -> list[Path]:
    files = sorted(transcripts_dir.glob("*.txt"))
    if max_transcripts is not None:
        files = files[: max(0, max_transcripts)]
    return files


@contextmanager
def cache_dir_lock(lock_path: Path, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        error(
            f"Could not acquire exclusive lock at {lock_path}. "
            "Another pipeline instance may be running; use --no-cache-lock to override."
        )
    try:
        fh.write(str(os.getpid()))
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _sentence_pool_meta_path(out_path: Path) -> Path:
    return out_path.parent / "sentence_pool_meta.json"


def _sentence_pool_cache_matches(
    meta_path: Path,
    *,
    transcripts_dir: Path,
    out_path: Path,
    max_transcripts: int | None,
    max_pool_rows: int | None,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(meta.get("min_chars", -1)) != int(MIN_CHARS):
        return False
    if str(meta.get("out", "")) != str(out_path.resolve()):
        return False
    if str(meta.get("transcripts_dir", "")) != str(transcripts_dir.resolve()):
        return False
    if meta.get("max_transcripts") != max_transcripts:
        return False
    if meta.get("max_pool_rows") != max_pool_rows:
        return False
    return True


def _write_sentence_pool_meta(
    meta_path: Path,
    *,
    out_path: Path,
    transcripts_dir: Path,
    max_transcripts: int | None,
    max_pool_rows: int | None,
    row_count: int,
) -> None:
    _write_run_meta(
        meta_path,
        {
            "min_chars": MIN_CHARS,
            "rows": int(row_count),
            "out": str(out_path.resolve()),
            "transcripts_dir": str(transcripts_dir.resolve()),
            "max_transcripts": max_transcripts,
            "max_pool_rows": max_pool_rows,
            "project_root": str(PROJECT_ROOT.resolve()),
        },
    )


def extract_sentence_pool(
    transcripts_dir: Path,
    out_path: Path,
    *,
    max_transcripts: int | None = None,
    max_pool_rows: int | None = None,
    use_cache_lock: bool = True,
    lock_path: Path = CACHE_LOCK_PATH,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Build the full sentence pool from transcript .txt files and write a Parquet cache.

    Returns ``(dataframe, from_disk_cache)`` where ``from_disk_cache`` is true when an
    existing Parquet (and matching meta when present) was loaded instead of rebuilding.

    Sentences shorter than ``MIN_CHARS`` are dropped. Lines are deduplicated within each
    transcript using first-occurrence retention across the whole file (non-consecutive
    duplicates removed). Paragraph boundaries follow empty lines and frozen structural
    patterns sampled from WFC / PLTR / NVDA transcripts (see ``_STRUCTURAL_LINE_RE``).

    If ``out_path`` already exists, ``force_rebuild`` is false, and
    ``sentence_pool_meta.json`` beside the pool matches the same extraction parameters,
    returns ``(cached_df, True)`` without re-tokenizing transcripts. If only the Parquet
    exists (legacy runs without meta), returns ``(cached_df, True)`` as well.
    """
    meta_path = _sentence_pool_meta_path(out_path)
    if (
        not force_rebuild
        and out_path.exists()
        and _sentence_pool_cache_matches(
            meta_path,
            transcripts_dir=transcripts_dir,
            out_path=out_path,
            max_transcripts=max_transcripts,
            max_pool_rows=max_pool_rows,
        )
    ):
        return pd.read_parquet(out_path), True
    if not force_rebuild and out_path.exists() and not meta_path.exists():
        return pd.read_parquet(out_path), True
    _ensure_nltk_punkt()
    rows: list[dict] = []
    files = iter_transcript_files(transcripts_dir, max_transcripts)
    if not files:
        error(f"No .txt transcripts found under {transcripts_dir}")
    with cache_dir_lock(lock_path, use_cache_lock):
        for path in tqdm(files, desc="Transcripts", unit="file"):
            raw = path.read_text(encoding="utf-8", errors="strict")
            file_lines = raw.splitlines()
            deduped = dedupe_lines_first_occurrence(file_lines)
            paragraphs = lines_to_paragraphs(deduped)
            source_file = path.name
            seq = 0
            for para in paragraphs:
                new_rows, seq = extract_sentences_from_paragraph(para, source_file, seq)
                for r in new_rows:
                    rows.append(r)
                    if max_pool_rows is not None and len(rows) >= max_pool_rows:
                        df = pd.DataFrame(rows)
                        atomic_write_parquet(df, out_path)
                        _write_sentence_pool_meta(
                            meta_path,
                            out_path=out_path,
                            transcripts_dir=transcripts_dir,
                            max_transcripts=max_transcripts,
                            max_pool_rows=max_pool_rows,
                            row_count=int(len(df)),
                        )
                        return df, False
        df = pd.DataFrame(rows)
        atomic_write_parquet(df, out_path)
        _write_sentence_pool_meta(
            meta_path,
            out_path=out_path,
            transcripts_dir=transcripts_dir,
            max_transcripts=max_transcripts,
            max_pool_rows=max_pool_rows,
            row_count=int(len(df)),
        )
        return df, False


def _write_run_meta(out_path: Path, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract sentence pool from transcripts.")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help="Directory containing transcript .txt files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SENTENCE_POOL_PATH,
        help="Output Parquet path for the sentence pool.",
    )
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=None,
        help="Process only the first N transcripts after sorting by filename.",
    )
    parser.add_argument(
        "--max-pool-rows",
        type=int,
        default=None,
        help="Stop after writing this many sentence rows (smoke test cap).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Shortcut for --max-pool-rows (smoke test cap on sentence rows).",
    )
    parser.add_argument(
        "--no-cache-lock",
        action="store_true",
        help="Disable cache/.lock exclusive lock (not recommended for production).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the pool even if a matching cache exists.",
    )
    args = parser.parse_args(argv)
    max_transcripts = args.max_transcripts
    max_pool_rows = args.cap if args.cap is not None else args.max_pool_rows
    meta_path = _sentence_pool_meta_path(args.out)
    df, from_cache = extract_sentence_pool(
        args.transcripts_dir,
        args.out,
        max_transcripts=max_transcripts,
        max_pool_rows=max_pool_rows,
        use_cache_lock=not args.no_cache_lock,
        force_rebuild=args.force,
    )
    if from_cache:
        print(f"Loaded {len(df)} rows from cache {args.out.resolve()}")
    else:
        print(f"Wrote {len(df)} rows to {args.out.resolve()}")
    print(f"Meta at {meta_path.resolve()}")


if __name__ == "__main__":
    main()
