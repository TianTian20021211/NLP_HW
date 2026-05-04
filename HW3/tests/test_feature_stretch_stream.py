"""Tests for streaming stretch feature parquet construction."""

from __future__ import annotations

import pandas as pd

from features.engineer import write_stretch_features_from_enhanced


def test_write_stretch_features_from_enhanced_aligns_batches(tmp_path):
    signals_path = tmp_path / "signals.parquet"
    enhanced_path = tmp_path / "features_enhanced.parquet"
    output_path = tmp_path / "features_stretch.parquet"

    enhanced = pd.DataFrame({
        "BESTTICKER": ["A", "B", "C", "D", "E"],
        "ATCClassifierScore": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    signals = pd.DataFrame({
        "AspectTheme_CurrentState_CapitalAllocation - High - Negative": [1, 2, 3, 4, 5],
        "AspectTheme_Forecast_Other - Low - Positive": [5, 4, 3, 2, 1],
        "AspectTheme_Fluff_Other - Low - Positive": [9, 9, 9, 9, 9],
        "not_a_stretch_feature": [10, 11, 12, 13, 14],
    })
    enhanced.to_parquet(enhanced_path, index=False)
    signals.to_parquet(signals_path, index=False)

    rows, cols = write_stretch_features_from_enhanced(
        signals_path,
        enhanced_path,
        output_path,
        batch_size=2,
    )

    result = pd.read_parquet(output_path)
    assert rows == 5
    assert cols == 4
    assert list(result.columns) == [
        "BESTTICKER",
        "ATCClassifierScore",
        "AspectTheme_CurrentState_CapitalAllocation - High - Negative",
        "AspectTheme_Forecast_Other - Low - Positive",
    ]
    assert result["BESTTICKER"].tolist() == ["A", "B", "C", "D", "E"]
    assert (
        result["AspectTheme_CurrentState_CapitalAllocation - High - Negative"].tolist()
        == [1, 2, 3, 4, 5]
    )


def test_write_stretch_features_from_enhanced_can_limit_rows(tmp_path):
    signals_path = tmp_path / "signals.parquet"
    enhanced_path = tmp_path / "features_enhanced.parquet"
    output_path = tmp_path / "features_stretch_sample.parquet"

    enhanced = pd.DataFrame({
        "BESTTICKER": ["A", "B", "C", "D"],
        "ATCClassifierScore": [0.1, 0.2, 0.3, 0.4],
    })
    signals = pd.DataFrame({
        "AspectTheme_Surprise_Other - Medium - Neutral": [7, 8, 9, 10],
    })
    enhanced.to_parquet(enhanced_path, index=False)
    signals.to_parquet(signals_path, index=False)

    rows, _ = write_stretch_features_from_enhanced(
        signals_path,
        enhanced_path,
        output_path,
        batch_size=3,
        max_rows=3,
    )

    result = pd.read_parquet(output_path)
    assert rows == 3
    assert result["BESTTICKER"].tolist() == ["A", "B", "C"]
    assert result["AspectTheme_Surprise_Other - Medium - Neutral"].tolist() == [7, 8, 9]
