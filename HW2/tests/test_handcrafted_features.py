from __future__ import annotations

import numpy as np

from src.handcrafted_features import FEATURE_COLUMNS, SHORT_WORD_MAX, handcrafted_features_frame


def test_feature_count_and_names() -> None:
    df = handcrafted_features_frame(["Good morning everyone.", "EPS grew 12% YoY $1m"])
    assert df.shape == (2, 30)
    assert list(df.columns) == list(FEATURE_COLUMNS)


def test_good_morning_flag() -> None:
    df = handcrafted_features_frame(["Good morning and welcome."])
    assert df.iloc[0]["h03"] >= 1.0


def test_short_word_flag() -> None:
    words = ["x"] * SHORT_WORD_MAX
    ok = handcrafted_features_frame([" ".join(words)])
    assert ok.iloc[0]["h22"] >= 1.0
    long_w = handcrafted_features_frame([" ".join(words + ["extra"])])
    assert long_w.iloc[0]["h22"] == 0.0


def test_nan_text_safe() -> None:
    df = handcrafted_features_frame([np.nan])
    assert df.shape == (1, 30)
    assert df.iloc[0]["h21"] == 0.0
    assert df.iloc[0]["h22"] == 0.0


def test_empty_not_short_word_flag() -> None:
    df = handcrafted_features_frame([""])
    assert df.iloc[0]["h22"] == 0.0


def test_firm_regex_requires_capitalized_name() -> None:
    df = handcrafted_features_frame(
        [
            "from the filing we see margin expansion.",
            "I'm calling from Goldman Sachs regarding EPS.",
            "Non-GAAP EPS was up with revenue guidance.",
        ]
    )
    assert df.iloc[0]["h02"] == 0.0
    assert df.iloc[1]["h02"] >= 1.0
    assert df.iloc[2]["h02"] == 0.0


def test_dense_financial_sentence_semantics() -> None:
    df = handcrafted_features_frame(
        ["Non-GAAP EPS was $1.12, up 5.2% YoY with revenue guidance."]
    )
    row = df.iloc[0]
    assert row["h09"] >= 1.0
    assert row["h12"] >= 1.0
    assert row["h13"] >= 1.0
    assert row["h17"] >= 1.0
    assert row["h18"] >= 1.0
    assert row["h02"] == 0.0
