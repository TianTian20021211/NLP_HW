from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import gold_labeling as gl


class _FakeAnthropicMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAnthropicClient:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.messages = self

    def create(self, **kwargs: Any) -> _FakeAnthropicMessage:
        return _FakeAnthropicMessage(self._payload)


def test_parse_json_label_accepts_plain_json() -> None:
    label, err = gl.parse_json_label('{"label":"boilerplate"}')
    assert label == "boilerplate"
    assert err is None


def test_parse_json_label_falls_back_to_embedded_object() -> None:
    label, err = gl.parse_json_label('prefix {"label":"substantive"} suffix')
    assert label == "substantive"
    assert err is None


def test_parse_json_label_invalid() -> None:
    label, err = gl.parse_json_label('{"label":"nope"}')
    assert label is None
    assert err == "invalid_label_value"


def test_majority_label_patterns() -> None:
    assert gl.majority_label("a", "a", "a") == ("a", "3-0", False)
    assert gl.majority_label("a", "a", "b") == ("a", "2-1", True)
    assert gl.majority_label("a", "b", "b") == ("b", "2-1", True)
    final, disc, needs = gl.majority_label("a", "b", "c")
    assert final is None
    assert disc == "1-1-1"
    assert needs is True


def test_merge_judge_tables_missing_sentence_errors(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "text": ["t1", "t2"],
            "source_file": ["f.txt", "f.txt"],
        }
    )
    pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "parsed_label": ["boilerplate"],
            "raw_response": ['{"label":"boilerplate"}'],
        }
    ).to_parquet(gl.labels_cache_path("m1", labels_dir=tmp_path), index=False)
    with pytest.raises(SystemExit):
        gl.merge_judge_tables(sample, ["m1"], labels_dir=tmp_path)


def test_merge_vote_unanimous_no_api(tmp_path: Path) -> None:
    rubric = "rubric"
    sample = pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "text": ["x" * 50],
            "source_file": ["f.txt"],
        }
    )
    models = ["m1", "m2", "m3"]
    for m in models:
        pd.DataFrame(
            {
                "sentence_id": ["s1"],
                "parsed_label": ["boilerplate"],
                "raw_response": ['{"label":"boilerplate"}'],
                "ollama_error": [None],
            }
        ).to_parquet(gl.labels_cache_path(m, labels_dir=tmp_path), index=False)

    merged = gl.merge_vote_and_arbitrate(
        sample,
        models,
        rubric=rubric,
        labels_dir=tmp_path,
        anthropic_client=_FakeAnthropicClient('{"label":"substantive"}'),
    )
    assert merged.iloc[0]["gold_final"] == "boilerplate"
    assert not bool(merged.iloc[0]["api_used"])


def test_merge_vote_disagreement_calls_haiku(tmp_path: Path) -> None:
    rubric = "rubric"
    sample = pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "text": ["x" * 50],
            "source_file": ["f.txt"],
        }
    )
    models = ["m1", "m2", "m3"]
    for m, lab in zip(models, ["boilerplate", "boilerplate", "substantive"]):
        pd.DataFrame(
            {
                "sentence_id": ["s1"],
                "parsed_label": [lab],
                "raw_response": [f'{{"label":"{lab}"}}'],
                "ollama_error": [None],
            }
        ).to_parquet(gl.labels_cache_path(m, labels_dir=tmp_path), index=False)

    merged = gl.merge_vote_and_arbitrate(
        sample,
        models,
        rubric=rubric,
        labels_dir=tmp_path,
        anthropic_client=_FakeAnthropicClient('{"label":"substantive"}'),
    )
    assert merged.iloc[0]["gold_final"] == "substantive"
    assert bool(merged.iloc[0]["api_used"])


def test_merge_vote_parse_fallback_to_substantive_changes_majority(tmp_path: Path) -> None:
    rubric = "rubric"
    sample = pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "text": ["x" * 50],
            "source_file": ["f.txt"],
        }
    )
    models = ["m1", "m2", "m3"]
    rows = [
        ("m1", "boilerplate"),
        ("m2", "boilerplate"),
        ("m3", None),
    ]
    for m, lab in rows:
        pd.DataFrame(
            {
                "sentence_id": ["s1"],
                "parsed_label": [lab],
                "raw_response": [None],
                "ollama_error": [None if lab else "json_decode_error"],
            }
        ).to_parquet(gl.labels_cache_path(m, labels_dir=tmp_path), index=False)

    merged = gl.merge_vote_and_arbitrate(
        sample,
        models,
        rubric=rubric,
        labels_dir=tmp_path,
        anthropic_client=_FakeAnthropicClient('{"label":"boilerplate"}'),
    )
    assert merged.iloc[0]["gold_final"] == "boilerplate"
    assert bool(merged.iloc[0]["api_used"])


def test_attach_vote_preview_matches_majority_label() -> None:
    vals: list[Any] = ["boilerplate", "substantive", "", np.nan]
    for v1 in vals:
        for v2 in vals:
            for v3 in vals:
                df = pd.DataFrame({"j1_label": [v1], "j2_label": [v2], "j3_label": [v3]})
                out = gl.attach_vote_preview(df)
                _, disc, need = gl.majority_label(v1, v2, v3)
                assert bool(out.iloc[0]["needs_haiku_api"]) == bool(need)
                assert str(out.iloc[0]["vote_discord"]) == str(disc)


def test_try_load_cached_gold_sample_hits_when_meta_matches(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.parquet"
    sample_path = tmp_path / "gold_sample.parquet"
    pd.DataFrame({"sentence_id": ["a"], "text": ["x" * 50], "source_file": ["f.txt"]}).to_parquet(
        pool_path, index=False
    )
    tbl = pd.DataFrame({"sentence_id": ["a"], "text": ["x" * 50], "source_file": ["f.txt"]})
    gl.atomic_write_gold_sample(tbl, sample_path, pool_path=pool_path, gold_n=1, seed=7)
    loaded = gl.try_load_cached_gold_sample(sample_path, pool_path=pool_path, gold_n=1, seed=7)
    assert loaded is not None
    assert len(loaded) == 1
    assert gl.try_load_cached_gold_sample(sample_path, pool_path=pool_path, gold_n=2, seed=7) is None


def test_call_ollama_chat_uses_post_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __init__(self) -> None:
            self.status_code = 200
            self.content = b'{"message":{"content":"{\"label\":\"boilerplate\"}"}}'

        def json(self) -> dict:
            return {"message": {"content": '{"label":"boilerplate"}'}}

    def post_fn(*args: Any, **kwargs: Any) -> _Resp:
        return _Resp()

    text, err = gl.call_ollama_chat("m", "prompt", post_fn=post_fn)
    assert err is None
    assert "boilerplate" in (text or "")
