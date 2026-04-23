from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.train_split import stratified_supervision_split, tie_break_y_doc, transcript_y_doc


def test_tie_break_deterministic() -> None:
    assert tie_break_y_doc("a.txt", 42) == tie_break_y_doc("a.txt", 42)


def test_stratified_split_no_leakage() -> None:
    rows = []
    for fi in range(10):
        fn = f"t{fi}.txt"
        for _ in range(3):
            rows.append(
                {
                    "sentence_id": f"{fn}#x",
                    "text": "hello world " * 5,
                    "source_file": fn,
                    "gold_final": "boilerplate",
                }
            )
    df = pd.DataFrame(rows)
    out, meta = stratified_supervision_split(gold_df=df, seed=0)
    assert "split" in out.columns
    assert (out.groupby("source_file")["split"].nunique() == 1).all()
    assert len(out.drop_duplicates("source_file")) == 10
    assert set(meta["transcripts_per_split"].keys()) <= {"train", "val", "test"}
    for fn in df["source_file"].unique():
        sl = set(out.loc[out["source_file"] == fn, "split"].unique())
        assert len(sl) == 1


def test_transcript_y_doc_majority() -> None:
    df = pd.DataFrame(
        {
            "source_file": ["a.txt"] * 3,
            "gold_final": ["boilerplate", "boilerplate", "substantive"],
        }
    )
    tab = transcript_y_doc(df, seed=1)
    assert tab.iloc[0]["y_doc"] == "boilerplate"


def test_invalid_labels_dropped() -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "text": ["x" * 40, "y" * 40],
            "source_file": ["f.txt", "f.txt"],
            "gold_final": ["boilerplate", "maybe"],
        }
    )
    out, meta = stratified_supervision_split(gold_df=df, seed=42)
    assert len(out) == 1
    assert meta["rows_dropped_invalid_label"] == 1


def test_missing_parquet_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        stratified_supervision_split(gold_labeled_path=tmp_path / "none.parquet")
