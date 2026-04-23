from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.errors import error
from src.gold_labeling import largest_remainder_quotas
from src.paths import GOLD_LABELED_PATH

VALID_LABELS = frozenset({"boilerplate", "substantive"})
SPLIT_WEIGHTS = (60, 20, 20)


def normalize_sentence_label(raw: object) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    return s if s in VALID_LABELS else None


def tie_break_y_doc(source_file: str, seed: int) -> str:
    """When Boilerplate vs Substantive counts tie for a transcript, pick deterministically."""
    msg = f"{seed}\0{source_file}".encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return "boilerplate" if (digest[0] & 1) == 0 else "substantive"


def supervision_frame_from_gold(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Keep only rows with valid ``gold_final`` for supervised train/val/test.

    Returns the filtered frame (with normalized ``gold_final`` column) and simple counts
    for metadata.
    """
    if "gold_final" not in df.columns:
        error("Expected column gold_final in gold-labeled frame")
    if "source_file" not in df.columns:
        error("Expected column source_file in gold-labeled frame")
    n_in = len(df)
    norms: list[str | None] = []
    for v in df["gold_final"].tolist():
        norms.append(normalize_sentence_label(v))
    eff = np.array([x is not None for x in norms], dtype=bool)
    out = df.loc[eff].copy()
    out["gold_final"] = [norms[i] for i in np.flatnonzero(eff)]
    stats = {
        "rows_input": n_in,
        "rows_dropped_invalid_label": int(n_in - eff.sum()),
        "rows_supervision": int(len(out)),
    }
    return out, stats


def transcript_y_doc(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    One row per ``source_file`` with stratification key ``y_doc`` (majority sentence label).

    Transcripts with no rows should be omitted before calling; empty input raises.
    """
    rows: list[dict[str, Any]] = []
    for src, g in df.groupby("source_file", sort=False):
        labs = g["gold_final"].astype(str).tolist()
        n_bp = sum(1 for x in labs if x == "boilerplate")
        n_su = sum(1 for x in labs if x == "substantive")
        if n_bp > n_su:
            y_doc = "boilerplate"
        elif n_su > n_bp:
            y_doc = "substantive"
        else:
            y_doc = tie_break_y_doc(str(src), seed)
        rows.append({"source_file": str(src), "y_doc": y_doc, "n_sentences": len(g)})
    tab = pd.DataFrame(rows)
    if tab.empty:
        error("No transcripts left for stratified split after filtering")
    return tab


def _split_three_way(ids: list[str], rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    q = largest_remainder_quotas(list(SPLIT_WEIGHTS), n)
    a, b, c = q[0], q[1], q[2]
    train, val, test = ids[:a], ids[a : a + b], ids[a + b : a + b + c]
    assert len(train) + len(val) + len(test) == n
    return train, val, test


def stratified_train_val_test_files(y_doc_table: pd.DataFrame, seed: int) -> dict[str, str]:
    """
    Map each ``source_file`` to ``train``, ``val``, or ``test`` (60/20/20 by count,
    stratified by ``y_doc``). Same random seed fixes all randomness in this step.
    """
    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for y in ("boilerplate", "substantive"):
        sub = y_doc_table.loc[y_doc_table["y_doc"] == y, "source_file"].astype(str).tolist()
        t1, t2, t3 = _split_three_way(sub, rng)
        train.extend(t1)
        val.extend(t2)
        test.extend(t3)
    out: dict[str, str] = {}
    for x in train:
        out[x] = "train"
    for x in val:
        out[x] = "val"
    for x in test:
        out[x] = "test"
    return out


def build_split_metadata(
    df_sup: pd.DataFrame,
    y_doc_table: pd.DataFrame,
    file_to_split: dict[str, str],
    filter_stats: dict[str, int],
    *,
    seed: int,
    dropped_transcripts: int,
) -> dict[str, Any]:
    s = df_sup.assign(split=lambda d: d["source_file"].astype(str).map(file_to_split))
    uniq = s.drop_duplicates(subset=["source_file"])
    tx_per = uniq.groupby("split").size().to_dict()
    meta: dict[str, Any] = {
        "seed": seed,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        **filter_stats,
        "transcripts_no_valid_sentences_removed": dropped_transcripts,
        "transcripts_total": int(y_doc_table.shape[0]),
        "sentences_per_split": s["split"].value_counts().to_dict(),
        "transcripts_per_split": {k: int(tx_per.get(k, 0)) for k in ("train", "val", "test")},
        "sentence_class_fractions": {
            sp: {
                "boilerplate": float((s.loc[s["split"] == sp, "gold_final"] == "boilerplate").mean())
                if (s["split"] == sp).any()
                else 0.0,
                "substantive": float((s.loc[s["split"] == sp, "gold_final"] == "substantive").mean())
                if (s["split"] == sp).any()
                else 0.0,
            }
            for sp in ("train", "val", "test")
        },
        "transcript_y_doc_counts": y_doc_table["y_doc"].value_counts().to_dict(),
    }
    return meta


def stratified_supervision_split(
    *,
    gold_labeled_path: Path | None = None,
    gold_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load gold labels, keep valid ``gold_final`` rows, split by transcript (60/20/20)
    stratified by document-level majority label ``y_doc``. Returns sentences with ``split``
    column and metadata for reporting or JSON export.
    """
    if gold_df is None:
        path = gold_labeled_path or GOLD_LABELED_PATH
        if not path.exists():
            error(f"Missing gold-labeled parquet: {path}")
        gold_df = pd.read_parquet(path)
    df_sup, filt = supervision_frame_from_gold(gold_df)

    seen_files = set(df_sup["source_file"].astype(str).unique())
    all_files = set(gold_df["source_file"].astype(str).unique()) if "source_file" in gold_df.columns else seen_files
    dropped_transcripts = int(len(all_files - seen_files))

    y_tab = transcript_y_doc(df_sup, seed)
    fmap = stratified_train_val_test_files(y_tab, seed)
    df_out = df_sup.copy()
    df_out["split"] = df_out["source_file"].astype(str).map(fmap)
    if df_out["split"].isna().any():
        error("Internal error: missing split assignment for some rows")
    meta = build_split_metadata(df_sup, y_tab, fmap, filt, seed=seed, dropped_transcripts=dropped_transcripts)
    return df_out, meta


def split_metadata_to_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, sort_keys=True, default=str)
