"""Tests for artifact-backed Phase 3 feature parity helpers."""

from __future__ import annotations

import pandas as pd

from features.audit import (
    _build_streaming_targets_no_momentum,
    _read_feature_rows_for_dates,
)


def test_read_feature_rows_for_dates_streams_selected_rows(tmp_path):
    path = tmp_path / "features.parquet"
    df = pd.DataFrame({
        "availability_date": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-02",
        ]),
        "BESTTICKER": ["A", "B", "C", "D"],
        "feature": [1.0, 2.0, 3.0, 4.0],
    })
    df.to_parquet(path, index=False)

    result = _read_feature_rows_for_dates(
        path,
        [pd.Timestamp("2020-01-02")],
        ["availability_date", "BESTTICKER", "feature"],
        batch_size=2,
    )

    assert result["BESTTICKER"].tolist() == ["B", "D"]
    assert result["feature"].tolist() == [2.0, 4.0]


def test_streaming_targets_use_pruned_history_base():
    history_base = pd.DataFrame({
        "BESTTICKER": ["A", "A", "A"],
        "SECTOR": ["Tech", "Tech", "Tech"],
        "availability_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
        "call_entry_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
        "SignalType": ["Total", "Total", "Total"],
        "ATCClassifierScore": [1.0, 2.0, 4.0],
        "aspect_CurrentState_total": [10.0, 20.0, 40.0],
        "theme_Other_total": [5.0, 6.0, 7.0],
    })
    batch_target = history_base.iloc[[2]].copy()
    batch_target["raw_stretch_feature"] = [99.0]
    batch_target["qoq_delta_ATCClassifierScore"] = [2.0]

    result = _build_streaming_targets_no_momentum(
        batch_target,
        [pd.Timestamp("2020-07-01")],
        ["ATCClassifierScore", "raw_stretch_feature", "qoq_delta_ATCClassifierScore"],
        history_base=history_base,
    )

    assert result["raw_stretch_feature"].tolist() == [99.0]
    assert result["qoq_delta_ATCClassifierScore"].tolist() == [2.0]
