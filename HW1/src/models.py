"""LogReg / XGBoost classifiers + Ridge regressor, cached to pickle.

Per Part II §2.7 and §2.11:

* **LogReg**: L2, class-weight balanced. If the training set is degenerate
  (``n < 20`` or only one class present) fit silently falls back to a
  ``SingleClassSentinel`` whose score is the train-mean of ``y`` — so the
  backtest layer still sees "no preference" instead of a crash.
* **XGBoost**: 200 trees, max_depth=3, lr=0.05, reg_lambda=1,
  subsample=0.8, colsample=0.8. Uses the hist tree method for determinism
  on tiny tables (200×28).
* **Ridge**: α=1.0, targeting excess return as a float. Same sentinel
  fallback when ``n < 20``.

All models wrap a :class:`sklearn.impute.SimpleImputer(strategy="median")`
in a Pipeline so feature-table NaNs (first-call deltas, role-sentiment
missing, ret_21d_prior boundary) don't poison training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .io_paths import error, model_path


Target = Literal["binary", "regression"]
MIN_TRAIN = 20


@dataclass
class SingleClassSentinel:
    """Stand-in returned by degenerate fits; stores the train-mean of ``y``.

    Emits a constant score for every row — classifiers get the positive
    rate, regressors get the mean excess return. The backtest records the
    sentinel in the model metadata (``model.kind == "sentinel"``) so
    downstream reports can flag these runs.
    """

    kind: str = "sentinel"
    score: float = 0.0
    reason: str = ""
    feature_names: list[str] | None = None

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(self.score), dtype="float64")


def fit_logreg(X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit an L2 logistic regression with median imputation + scaling."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if _is_degenerate(X, y, binary=True):
        return SingleClassSentinel(
            score=float(y.mean()) if len(y) else 0.0,
            reason=_degenerate_reason(X, y, binary=True),
            feature_names=list(X.columns),
        )
    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    pipe.feature_names = list(X.columns)  # type: ignore[attr-defined]
    return pipe


def fit_xgb(X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit an XGBoost classifier (binary). Median-impute before training."""
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if _is_degenerate(X, y, binary=True):
        return SingleClassSentinel(
            score=float(y.mean()) if len(y) else 0.0,
            reason=_degenerate_reason(X, y, binary=True),
            feature_names=list(X.columns),
        )
    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=42,
                    eval_metric="logloss",
                    use_label_encoder=False,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    pipe.feature_names = list(X.columns)  # type: ignore[attr-defined]
    return pipe


def fit_ridge(X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit a Ridge regressor (α=1) with median imputation + scaling."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if _is_degenerate(X, y, binary=False):
        return SingleClassSentinel(
            score=float(y.mean()) if len(y) else 0.0,
            reason=_degenerate_reason(X, y, binary=False),
            feature_names=list(X.columns),
        )
    pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X, y)
    pipe.feature_names = list(X.columns)  # type: ignore[attr-defined]
    return pipe


def predict(model: Any, X: pd.DataFrame, target: Target = "binary") -> np.ndarray:
    """Return a 1-D score array aligned to ``X.index``.

    Classifiers emit ``P(y=1)`` via ``predict_proba``; regressors emit
    raw predictions; sentinels emit a constant. Always returns a numpy
    ``float64`` array of length ``len(X)``.
    """
    if isinstance(model, SingleClassSentinel):
        return model.predict_score(X)
    expected = getattr(model, "feature_names", None)
    if expected is not None:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            error(f"predict: X missing trained columns {missing[:5]}...")
        X = X[expected]
    if target == "binary":
        if not hasattr(model, "predict_proba"):
            error(f"model {type(model).__name__} has no predict_proba; is it a regressor?")
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype="float64")
    if target == "regression":
        return np.asarray(model.predict(X), dtype="float64")
    error(f"unknown target: {target}")
    return np.empty(0)


def save_model(model: Any, extraction: str, model_name: str, horizon: str) -> Path:
    """Pickle ``model`` to the canonical models/ slot."""
    path = model_path(extraction, model_name, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def load_model(extraction: str, model_name: str, horizon: str) -> Any:
    """Load a pickled model; fail loud if it doesn't exist."""
    path = model_path(extraction, model_name, horizon)
    if not path.is_file():
        error(f"model pickle missing: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _is_degenerate(X: pd.DataFrame, y: pd.Series, binary: bool) -> bool:
    if len(y) < MIN_TRAIN:
        return True
    if binary and y.nunique() < 2:
        return True
    return False


def _degenerate_reason(X: pd.DataFrame, y: pd.Series, binary: bool) -> str:
    if len(y) < MIN_TRAIN:
        return f"n_train={len(y)} < {MIN_TRAIN}"
    if binary and y.nunique() < 2:
        return f"single class in train (y={y.iloc[0] if len(y) else None})"
    return "unspecified"
