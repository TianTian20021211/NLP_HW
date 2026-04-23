from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentence_extraction import (
    MIN_CHARS,
    dedupe_lines_first_occurrence,
    extract_sentence_rows_from_text,
    extract_sentence_pool,
    lines_to_paragraphs,
    paragraph_line_breaks,
)


def test_dedupe_first_occurrence_keeps_order_and_skips_later_dupes() -> None:
    lines = ["a", "b", "a", "c", "b"]
    assert dedupe_lines_first_occurrence(lines) == ["a", "b", "c"]


def test_paragraph_line_breaks_empty_and_structural() -> None:
    assert paragraph_line_breaks("") is True
    assert paragraph_line_breaks("   ") is True
    assert paragraph_line_breaks("Presenter Speech") is True
    assert paragraph_line_breaks("Executives - CEO") is True
    assert paragraph_line_breaks("Operator") is True
    assert paragraph_line_breaks("This is a normal sentence.") is False


def test_lines_to_paragraphs_respects_blank_lines_and_headers() -> None:
    lines = [
        "Header line",
        "First sentence. Second sentence.",
        "",
        "Presenter Speech",
        "Only paragraph two here. Another one.",
    ]
    paras = lines_to_paragraphs(lines)
    assert len(paras) == 2
    assert "Header line" in paras[0]
    assert "Presenter Speech" not in paras[1]


def test_extract_sentence_pool_smoke(tmp_path: Path) -> None:
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    long_sent = ("word " * 12).strip()
    assert len(long_sent) >= MIN_CHARS
    text = (
        "Company, Q1 - \n\n"
        "Presenter Speech\n"
        f"{long_sent}. {'more ' * 20}words here.\n\n"
        "Question\n"
        f"{long_sent} continuation text here.\n"
    )
    (tdir / "AAA.txt").write_text(text, encoding="utf-8")
    out = tmp_path / "pool.parquet"
    df, from_cache = extract_sentence_pool(
        tdir,
        out,
        max_transcripts=10,
        max_pool_rows=1000,
        use_cache_lock=False,
        lock_path=tmp_path / "nolock",
    )
    assert out.exists()
    assert not from_cache
    assert set(df.columns) >= {"sentence_id", "text", "source_file"}
    assert len(df) >= 2
    assert all(len(str(t)) >= MIN_CHARS for t in df["text"].tolist())
    df2, from_cache2 = extract_sentence_pool(
        tdir,
        out,
        max_transcripts=10,
        max_pool_rows=1000,
        use_cache_lock=False,
        lock_path=tmp_path / "nolock",
    )
    assert from_cache2
    assert len(df2) == len(df)


def test_extract_sentence_rows_from_text_reports_gui_counts() -> None:
    long_sent = ("Revenue grew 12 percent while operating margin expanded " "during the quarter.")
    raw = "\n".join(
        [
            "Presenter Speech",
            "Hi.",
            long_sent,
            long_sent,
            "Question",
            "Short.",
        ]
    )

    rows, meta = extract_sentence_rows_from_text(raw, "../demo.txt")

    assert [row["source_file"] for row in rows] == ["demo.txt"]
    assert rows[0]["sentence_id"] == "demo.txt#0000000"
    assert rows[0]["text"] == long_sent
    assert meta["duplicate_line_count"] == 1
    assert meta["short_sentence_count"] == 2
    assert meta["classified_sentence_count"] == 1
