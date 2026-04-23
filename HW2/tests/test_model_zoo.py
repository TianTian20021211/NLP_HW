from __future__ import annotations

import numpy as np
from sklearn.metrics import recall_score

from src.model_zoo import (
    _setfit_positive_class_probability,
    classification_report_row,
    rank_average_scores,
    search_threshold_max_macro_f1,
)


class _FakeTorchTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def detach(self) -> "_FakeTorchTensor":
        return self

    def cpu(self) -> "_FakeTorchTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


def test_search_threshold_respects_recall_floor() -> None:
    y = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95], dtype=np.float64)
    t, _f1, ok = search_threshold_max_macro_f1(y, p, recall_floor=0.96, n_grid=200)
    assert ok
    assert t is not None
    pred = (p >= t).astype(np.int64)
    assert recall_score(y, pred, pos_label=1, zero_division=0) >= 0.96 - 1e-9


def test_rank_average_scores_shape() -> None:
    m = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.8]], dtype=np.float64)
    r = rank_average_scores(m)
    assert r.shape == (2,)
    assert np.all((r >= 0.0) & (r <= 1.0))


def test_rank_average_scores_preserves_sample_order() -> None:
    m = np.array(
        [
            [0.90, 0.80, 0.70],
            [0.10, 0.20, 0.30],
            [0.50, 0.40, 0.60],
        ],
        dtype=np.float64,
    )
    r = rank_average_scores(m)
    assert r[0] > r[2] > r[1]


def test_setfit_positive_probability_accepts_torch_like_tensor() -> None:
    proba = _FakeTorchTensor([[0.8, 0.2], [0.1, 0.9]])

    p_substantive = _setfit_positive_class_probability(proba, context="test")

    assert p_substantive.dtype == np.float64
    np.testing.assert_allclose(p_substantive, np.array([0.2, 0.9], dtype=np.float64))


def test_classification_report_row_includes_per_class_precision_and_recall() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1], dtype=np.int64)

    row = classification_report_row(y_true, y_pred)

    assert "precision_boilerplate" in row
    assert "recall_boilerplate" in row
    assert "precision_substantive" in row
    assert "recall_substantive" in row
    assert row["precision_boilerplate"] == 1.0
    assert row["recall_boilerplate"] == 0.5
    assert np.isclose(row["precision_substantive"], 2 / 3)
    assert row["recall_substantive"] == 1.0
