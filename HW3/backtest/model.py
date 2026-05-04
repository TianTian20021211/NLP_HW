"""Phase 4 — Model Training and Walk-Forward Backtest.

Inside each walk-forward fold:
- Fit imputation / scaling on training fold only
- Stretch tier: LassoCV for column selection (Phase 4.2)
- Fit Ridge / LightGBM / XGBoost with frozen hyperparameters
- Generate out-of-sample predictions

Orchestration entry-point: ``run_walk_forward``.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

from data.cache_utils import build_cache_manifest, write_cache_manifest
from data.config import RESULTS_DIR, AUDIT_DIR, SEED, PRICE_CACHE_DIR, CACHE_MANIFEST_DIR
from data.progress import progress

from backtest._stats import make_median_imputer, spearman

from backtest.splits import (
    HORIZONS,
    Fold,
    HPARAMS_DIR,
    ensure_forward_returns,
    generate_folds,
    purge_train_for_horizon,
    get_feature_cols,
    load_frozen_hparams,
    write_fold_manifest,
    write_sample_size_audit,
)

log = logging.getLogger("backtest.model")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _impute_and_scale(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """Fit median imputer + StandardScaler on train, transform both.

    Returns (X_train_scaled, X_test_scaled, imputer, scaler).
    """
    from sklearn.preprocessing import StandardScaler

    train_index = X_train.index if isinstance(X_train, pd.DataFrame) else None
    test_index = X_test.index if isinstance(X_test, pd.DataFrame) else None
    if isinstance(X_train, pd.DataFrame):
        input_cols = list(X_train.columns)
    else:
        input_cols = [f"x{i}" for i in range(X_train.shape[1])]

    imp = make_median_imputer()
    X_tr_arr = imp.fit_transform(X_train)
    X_te_arr = imp.transform(X_test)
    out_cols = input_cols if len(input_cols) == X_tr_arr.shape[1] else [
        f"x{i}" for i in range(X_tr_arr.shape[1])
    ]
    X_tr_imp = pd.DataFrame(X_tr_arr, index=train_index, columns=out_cols)
    X_te_imp = pd.DataFrame(X_te_arr, index=test_index, columns=out_cols)

    scl = StandardScaler()
    X_tr_scaled = pd.DataFrame(
        scl.fit_transform(X_tr_imp),
        index=train_index,
        columns=out_cols,
    )
    X_te_scaled = pd.DataFrame(
        scl.transform(X_te_imp),
        index=test_index,
        columns=out_cols,
    )

    return X_tr_scaled, X_te_scaled, imp, scl


# ---------------------------------------------------------------------------
# Stretch-tier feature selection  (Phase 4.2)
# ---------------------------------------------------------------------------

def _select_stretch_features(
    X_train: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    y_train: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    max_features: int = 200,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray, list[str], list[str]]:
    """LassoCV feature selection for Stretch tier.

    Fits ``LassoCV`` with purged ``TimeSeriesSplit(3)`` pairs on the training
    fold, keeps columns with nonzero coefficients. If more than
    *max_features* remain, truncates to the top *max_features* by absolute
    coefficient.

    Returns ``(selected_X_train, selected_indices, selected_feature_names,
    dropped_feature_names)``.  ``X_test`` must be subset with
    ``selected_indices``.

    Parameters
    ----------
    X_train: (n_train, n_features) — already imputed and scaled.
    feature_names: list of column names for all features.
    y_train: training targets.
    cv_splits: label-purged TimeSeriesSplit pairs, relative to X_train.
    max_features: max columns to retain after LassoCV.

    Returns (X_train_selected, selected_idx, selected_names, dropped_names).
    """
    from sklearn.linear_model import LassoCV

    log.info("  stretch: running LassoCV on %d features …", X_train.shape[1])
    t0 = time.time()

    lasso = LassoCV(
        cv=cv_splits,
        random_state=SEED,
        max_iter=5000,
        n_jobs=4,
    )
    lasso.fit(X_train, y_train)

    coef = np.abs(lasso.coef_)
    nonzero_mask = coef > 1e-10
    n_nonzero = int(nonzero_mask.sum())
    log.info("  LassoCV selected %d / %d features (%.1fs)",
             n_nonzero, X_train.shape[1], time.time() - t0)

    if n_nonzero == 0:
        log.warning("  LassoCV zeroed all features — falling back to top %d by |coef|",
                    max_features)
        top_idx = np.argsort(coef)[::-1][:min(max_features, X_train.shape[1])]
        selected = list(np.array(feature_names)[top_idx])
        keep_set = set(top_idx.tolist())
        dropped = [n for i, n in enumerate(feature_names) if i not in keep_set]
        if isinstance(X_train, pd.DataFrame):
            return X_train.iloc[:, top_idx], top_idx, selected, dropped
        return X_train[:, top_idx], top_idx, selected, dropped

    if n_nonzero > max_features:
        log.info("  truncating %d → %d by |coef|", n_nonzero, max_features)
        nonzero_idx = np.flatnonzero(nonzero_mask)
        top_k = nonzero_idx[np.argsort(coef[nonzero_idx])[::-1][:max_features]]
        selected = list(np.array(feature_names)[top_k])
        keep_set = set(top_k.tolist())
        dropped = [n for i, n in enumerate(feature_names) if i not in keep_set]
        if isinstance(X_train, pd.DataFrame):
            return X_train.iloc[:, top_k], top_k, selected, dropped
        return X_train[:, top_k], top_k, selected, dropped

    selected_idx = np.flatnonzero(nonzero_mask)
    selected = list(np.array(feature_names)[selected_idx])
    dropped = [n for i, n in enumerate(feature_names) if not nonzero_mask[i]]
    if isinstance(X_train, pd.DataFrame):
        return X_train.iloc[:, selected_idx], selected_idx, selected, dropped
    return X_train[:, selected_idx], selected_idx, selected, dropped


def _purged_time_series_splits(
    feature_dates: np.ndarray,
    target_available_dates: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit with G11 target-availability purge for inner CV."""
    from sklearn.model_selection import TimeSeriesSplit

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    base = np.arange(len(feature_dates))
    for tr, vl in TimeSeriesSplit(n_splits=n_splits).split(base):
        val_start = feature_dates[vl].min()
        keep = (
            (feature_dates[tr] < val_start)
            & (target_available_dates[tr] < val_start)
        )
        tr_purged = tr[keep]
        if len(tr_purged) == 0:
            log.warning("  stretch inner split dropped: empty purged train fold")
            continue
        splits.append((tr_purged, vl))
    return splits


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model(model_name: str, hparams: dict[str, Any]) -> Any:
    """Build a scikit-learn-compatible regressor from frozen hyperparameters.

    *model_name* is one of ``{"ridge", "lightgbm", "xgboost"}``.
    """
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(
            alpha=hparams.get("alpha", 1.0),
            random_state=SEED,
        )

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed") from None
        return lgb.LGBMRegressor(
            n_estimators=int(hparams.get("n_estimators", 300)),
            max_depth=int(hparams.get("max_depth", -1)),
            learning_rate=float(hparams.get("learning_rate", 0.05)),
            num_leaves=int(hparams.get("num_leaves", 31)),
            min_child_samples=int(hparams.get("min_child_samples", 50)),
            subsample=float(hparams.get("subsample", 0.8)),
            colsample_bytree=float(hparams.get("colsample_bytree", 0.8)),
            random_state=SEED,
            verbose=-1,
            n_jobs=4,
        )

    if model_name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise RuntimeError("xgboost not installed") from None
        return xgb.XGBRegressor(
            n_estimators=int(hparams.get("n_estimators", 300)),
            max_depth=int(hparams.get("max_depth", 5)),
            learning_rate=float(hparams.get("learning_rate", 0.05)),
            subsample=float(hparams.get("subsample", 0.8)),
            colsample_bytree=float(hparams.get("colsample_bytree", 0.8)),
            reg_alpha=float(hparams.get("reg_alpha", 0.1)),
            reg_lambda=float(hparams.get("reg_lambda", 10.0)),
            random_state=SEED,
            verbosity=0,
            n_jobs=4,
        )

    raise ValueError(f"unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Single-fold execution
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_id: int
    horizon: int
    model_name: str
    n_train: int
    n_test: int
    test_indices: np.ndarray
    y_pred: np.ndarray
    y_true: np.ndarray
    # audit fields
    selected_features: list[str] | None = None
    fit_time_s: float = 0.0
    fit_violations: list[str] | None = None
    fit_log: list[dict[str, Any]] | None = None

@dataclass
class FoldSample:
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    target_col: str
    target_date_col: str



def _prepare_fold_sample(
    df: pd.DataFrame,
    fold: Fold,
    horizon: int,
    availability_col: str,
    feature_names: list[str],
) -> FoldSample | None:
    """Apply G11 purge, sort training by availability date, and build X/y.

    Training rows still require realized targets.  Test rows do not: OOS
    predictions are tradable signals, so target availability must only affect
    evaluation metrics, never whether a signal is emitted.
    """
    # G11 purge
    train_idx = purge_train_for_horizon(fold, df, horizon)
    test_idx = fold.test_indices

    if len(train_idx) < 10 or len(test_idx) == 0:
        log.info("  fold %d h=%d: skip (train=%d test=%d)",
                 fold.fold_id, horizon, len(train_idx), len(test_idx))
        return None

    target_col = f"forward_return_{horizon}d"
    target_date_col = f"target_available_date_{horizon}d"

    # Convert availability dates once for reuse
    train_avail_dates = pd.to_datetime(
        df[availability_col].iloc[train_idx], errors="coerce"
    )

    # Sort training by availability date
    train_order = np.argsort(
        train_avail_dates.to_numpy(dtype="datetime64[ns]"), kind="mergesort"
    )
    train_idx = train_idx[train_order]
    train_avail_dates = train_avail_dates.iloc[train_order]

    # Drop NaN training targets
    y_train_all = df[target_col].iloc[train_idx].to_numpy(dtype="float64")
    train_valid = np.isfinite(y_train_all)
    if not train_valid.all():
        train_idx = train_idx[train_valid]
        train_avail_dates = train_avail_dates.iloc[train_valid]
        y_train_all = y_train_all[train_valid]

    # Keep every test event for OOS signal generation — right-censored test
    # events (NaN y_true because target_available_date_{h}d is beyond the
    # price-data extent) are retained so that the model emits a tradable
    # prediction for every event.  Missing future returns are excluded only
    # from IC/MSE evaluation (via eval_mask), never from portfolio signal
    # generation.  This is an intentional design choice (see Phase 4.4).
    y_test_all = df[target_col].iloc[test_idx].to_numpy(dtype="float64")

    if len(train_idx) < 10:
        log.info("  fold %d h=%d: skip after target filter (train=%d)",
                 fold.fold_id, horizon, len(train_idx))
        return None

    y_train = y_train_all
    y_test = y_test_all

    # Build X matrices with DatetimeIndex for fit monitoring
    X_train = df[feature_names].iloc[train_idx].copy()
    X_test = df[feature_names].iloc[test_idx].copy()
    X_train.index = pd.DatetimeIndex(train_avail_dates)
    X_test.index = pd.DatetimeIndex(
        pd.to_datetime(df[availability_col].iloc[test_idx], errors="coerce")
    )

    train_start = pd.Timestamp(train_avail_dates.min())
    train_end = pd.Timestamp(train_avail_dates.max())

    return FoldSample(
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_start=train_start,
        train_end=train_end,
        target_col=target_col,
        target_date_col=target_date_col,
    )


def _fit_select_predict_with_audit(
    sample: FoldSample,
    fold: Fold,
    model_name: str,
    hparams: dict[str, Any],
    tier: str,
    horizon: int,
    feature_names: list[str],
    df: pd.DataFrame,
    availability_col: str,
    model_checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, list[str] | None, float, list[str], list[dict[str, Any]]] | None:
    """Impute, scale, optionally select stretch features, fit model, predict, and audit fit calls.

    Returns None when the stretch inner purged CV is empty (can happen for
    small folds with many NaN targets after G11 purge).  Callers must handle
    this by skipping the fold.
    """
    from features.audit import monitor_fit_calls, get_fit_log, assert_fit_callstack, assert_fold_boundaries

    X_tr = sample.X_train
    X_te = sample.X_test
    y_train = sample.y_train
    fit_violations: list[str] = []

    with monitor_fit_calls():
        # Impute + scale (fit on train only)
        X_tr, X_te, imp, scl = _impute_and_scale(X_tr, X_te)

        # Stretch tier: LassoCV column selection
        selected_features = None
        if tier == "stretch":
            train_feature_dates = pd.to_datetime(
                df[availability_col].iloc[sample.train_idx], errors="coerce"
            ).to_numpy(dtype="datetime64[ns]")
            train_target_dates = pd.to_datetime(
                df[sample.target_date_col].iloc[sample.train_idx], errors="coerce"
            ).to_numpy(dtype="datetime64[ns]")
            lasso_cv = _purged_time_series_splits(
                train_feature_dates, train_target_dates, n_splits=3
            )
            if not lasso_cv:
                log.info("  fold %d h=%d: skip stretch selection (empty inner CV)",
                         fold.fold_id, horizon)
                return None
            X_tr, keep_idx, kept_names, _dropped = _select_stretch_features(
                X_tr, feature_names, y_train, lasso_cv, max_features=200,
            )
            # subset test to same columns
            if isinstance(X_te, pd.DataFrame):
                X_te = X_te.iloc[:, keep_idx]
            else:
                X_te = X_te[:, keep_idx]
            selected_features = kept_names
            log.info("  fold %d h=%d: stretch selection kept %d features",
                     fold.fold_id, horizon, len(kept_names))

        # Fit model
        t0 = time.time()
        model = _make_model(model_name, hparams)
        model.fit(X_tr, y_train)
        if model_checkpoint_path is not None:
            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_checkpoint_path, "wb") as f:
                pickle.dump(model, f)
        fit_time = time.time() - t0

    # Verify fit-call boundaries
    fit_log = get_fit_log()
    for entry in fit_log:
        entry["fold_id"] = fold.fold_id
        entry["horizon"] = horizon
        entry["train_start"] = str(sample.train_start.date())
        entry["train_end"] = str(sample.train_end.date())
    fit_violations = assert_fit_callstack(
        fit_log, sample.train_start, sample.train_end,
        fold_id=f"fold_{fold.fold_id}_h{horizon}d",
    )
    if fit_violations:
        log.warning("  fold %d h=%d: %d fit-call boundary violation(s)",
                    fold.fold_id, horizon, len(fit_violations))
        for v in fit_violations:
            log.warning("    %s", v)

    # Fold boundary validation
    max_train_target = fold.max_train_target_date.get(horizon)
    boundary_violations = assert_fold_boundaries(
        fold_train_start=fold.train_start,
        fold_train_end=fold.train_end,
        fold_test_start=fold.test_start,
        max_train_feature_date=sample.train_end,
        max_train_target_available=max_train_target,
    )
    if boundary_violations:
        log.warning("  fold %d h=%d: %d fold-boundary violation(s)",
                    fold.fold_id, horizon, len(boundary_violations))
        for v in boundary_violations:
            log.warning("    %s", v)
        fit_violations.extend(boundary_violations)

    y_pred = model.predict(X_te)

    return y_pred, selected_features, fit_time, fit_violations, fit_log


def _build_fold_result(
    fold: Fold,
    horizon: int,
    model_name: str,
    sample: FoldSample,
    y_pred: np.ndarray,
    selected_features: list[str] | None,
    fit_time_s: float,
    fit_violations: list[str],
    fit_log: list[dict[str, Any]],
) -> FoldResult:
    """Pack fold outputs into FoldResult."""
    return FoldResult(
        fold_id=fold.fold_id,
        horizon=horizon,
        model_name=model_name,
        n_train=len(sample.train_idx),
        n_test=len(sample.test_idx),
        test_indices=sample.test_idx,
        y_pred=y_pred,
        y_true=sample.y_test,
        selected_features=selected_features,
        fit_time_s=fit_time_s,
        fit_violations=fit_violations if fit_violations else None,
        fit_log=fit_log,
    )


def _build_oos_predictions_df(
    df: pd.DataFrame,
    fold_results: list[FoldResult],
    horizon: int,
    model_name: str,
    tier: str,
    universe_name: str,
    signal_type: str,
) -> tuple[pd.DataFrame | None, float, float]:
    """Concatenate fold predictions and compute aggregate OOS IC/MSE."""
    if not fold_results:
        return None, float("nan"), float("nan")

    all_pred = np.concatenate([f.y_pred for f in fold_results])
    all_true = np.concatenate([f.y_true for f in fold_results])
    all_indices = np.concatenate([f.test_indices for f in fold_results])
    eval_mask = np.isfinite(all_pred) & np.isfinite(all_true)
    if eval_mask.any():
        oos_ic = spearman(all_pred[eval_mask], all_true[eval_mask])
        oos_mse = float(np.mean((all_pred[eval_mask] - all_true[eval_mask]) ** 2))
    else:
        oos_ic = float("nan")
        oos_mse = float("nan")

    if "_orig_df_index" in df.columns:
        df_indices = df["_orig_df_index"].iloc[all_indices].to_numpy(dtype=np.int64)
    else:
        df_indices = all_indices

    oos_df = pd.DataFrame({
        "df_index": df_indices,
        "horizon": horizon,
        "model": model_name,
        "tier": tier,
        "universe": universe_name,
        "signal_type": signal_type,
        "y_pred": all_pred,
        "y_true": all_true,
    })
    log.info("  aggregate OOS: n_pred=%d  n_eval=%d  IC=%.6f  MSE=%.6f",
             len(all_pred), int(eval_mask.sum()), oos_ic, oos_mse)
    return oos_df, oos_ic, oos_mse


def _write_oos_predictions(
    oos_df: pd.DataFrame | None,
    audit_dir: Path,
    model_name: str,
    tier: str,
    universe_name: str,
    signal_type: str,
    horizon: int,
) -> Path | None:
    """Write one horizon's OOS prediction parquet using the existing filename scheme."""
    if oos_df is None:
        return None
    suffix_parts = [model_name, tier]
    if universe_name and universe_name != "all":
        suffix_parts.append(universe_name)
    if signal_type and signal_type != "all":
        suffix_parts.append(signal_type.lower())
    suffix_parts.append(f"h{horizon}d")
    pred_path = audit_dir / f"oos_pred_{'_'.join(suffix_parts)}.parquet"
    oos_df.to_parquet(pred_path, index=False)
    log.info("  wrote %s", pred_path)
    return pred_path


def _write_walk_forward_audit_outputs(
    df: pd.DataFrame,
    folds: list[Fold],
    results: dict[int, WalkForwardResult],
    model_name: str,
    tier: str,
    horizons: list[int],
    audit_dir: Path,
    universe_name: str,
    signal_type: str,
) -> None:
    """Write fold manifest and sample-size audit files."""
    manifest_suffix = "_".join(
        p for p in [model_name, tier, universe_name, signal_type.lower()] if p and p != "all"
    )
    fold_manifest_path = audit_dir / f"fold_manifest_{manifest_suffix}.parquet"
    sample_audit_path = audit_dir / f"sample_size_by_quarter_{manifest_suffix}.parquet"
    write_fold_manifest(
        folds,
        output_path=fold_manifest_path,
        model=model_name,
        tier=tier,
        horizons=horizons,
    )
    write_sample_size_audit(
        df,
        folds,
        output_path=sample_audit_path,
        universe=universe_name,
        signal_type=signal_type,
    )
    # Keep the generic evidence filenames populated for the one-page checklist;
    # these intentionally reflect the most recent walk-forward invocation.
    write_fold_manifest(folds, output_path=audit_dir / "fold_manifest.parquet",
                        model=model_name, tier=tier, horizons=horizons)
    write_sample_size_audit(df, folds, output_path=audit_dir / "sample_size_by_quarter.parquet",
                            universe=universe_name, signal_type=signal_type)


def _run_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    model_name: str,
    hparams: dict[str, Any],
    horizon: int,
    tier: str,
    availability_col: str,
    feature_names: list[str] | None = None,
    checkpoint_dir: Path | None = None,
    universe_name: str = "all",
) -> FoldResult | None:
    """Fit one model on one fold for one horizon.

    When *checkpoint_dir* is provided, per-fold model and prediction
    checkpoint files are saved.  On subsequent calls with the same
    parameters, the predictions are loaded from disk and training is
    skipped, enabling resume within a horizon.

    Returns a FoldResult with predictions, or None if the purged training set
    or test set is empty.
    """
    if feature_names is None:
        feature_names = get_feature_cols(df)

    # Compute checkpoint paths when caching is enabled
    model_checkpoint_path: Path | None = None
    pred_checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        stem = f"{model_name}_{tier}_{universe_name}_h{horizon}d_fold{fold.fold_id}"
        model_checkpoint_path = checkpoint_dir / f"{stem}.pkl"
        pred_checkpoint_path = checkpoint_dir / f"{stem}_pred.parquet"

    # Attempt checkpoint resume — skip training if prediction parquet exists
    if pred_checkpoint_path is not None and pred_checkpoint_path.exists():
        sample = _prepare_fold_sample(df, fold, horizon, availability_col, feature_names)
        if sample is None:
            return None
        try:
            pred_df = pd.read_parquet(pred_checkpoint_path)
            y_pred = pred_df["y_pred"].to_numpy(dtype="float64")
            log.info("  fold %d: loaded checkpoint (%d preds)",
                     fold.fold_id, len(y_pred))
            return _build_fold_result(
                fold, horizon, model_name, sample, y_pred,
                selected_features=None, fit_time_s=0.0,
                fit_violations=[], fit_log=[],
            )
        except Exception:
            log.warning("  fold %d: corrupt checkpoint, retraining", fold.fold_id)

    # Fresh training path
    sample = _prepare_fold_sample(df, fold, horizon, availability_col, feature_names)
    if sample is None:
        return None

    result = _fit_select_predict_with_audit(
        sample, fold, model_name, hparams, tier, horizon, feature_names,
        df, availability_col,
        model_checkpoint_path=model_checkpoint_path,
    )
    if result is None:
        return None

    y_pred, selected_features, fit_time_s, fit_violations, fit_log = result
    fr = _build_fold_result(
        fold, horizon, model_name, sample, y_pred, selected_features,
        fit_time_s, fit_violations, fit_log,
    )

    # Save prediction checkpoint
    if pred_checkpoint_path is not None:
        pred_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_pred": y_pred}).to_parquet(pred_checkpoint_path, index=False)

    return fr





# ---------------------------------------------------------------------------
# Full walk-forward loop
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    model_name: str
    tier: str
    horizon: int
    fold_results: list[FoldResult]
    oos_predictions: pd.DataFrame | None = None
    oos_ic: float = float("nan")
    oos_mse: float = float("nan")
    universe_name: str = "all"
    signal_type: str = "all"


def run_walk_forward(
    df: pd.DataFrame,
    model_name: str,
    tier: str = "enhanced",
    horizons: list[int] | None = None,
    hparams_dir: Path | None = None,
    audit_dir: Path | None = None,
    availability_col: str = "availability_date",
    universe_name: str = "all",
    signal_type: str = "all",
    features_path: Path | None = None,
) -> dict[int, WalkForwardResult]:
    """Run the full walk-forward backtest for one model + tier.

    Parameters
    ----------
    df: Feature DataFrame with forward_return_{h}d and
        target_available_date_{h}d columns already joined.
    model_name: "ridge" | "lightgbm" | "xgboost".
    tier: "enhanced" | "stretch".
    horizons: List of forward-return horizons (default: [1,3,5,10,20]).
    hparams_dir: Directory containing {tier}/h{horizon}d/frozen_hparams_{model}.json.
    audit_dir: Directory for audit output files.
    availability_col: Column for PIT availability filtering of training data
        (default "availability_date").

    Returns
    -------
    Dict mapping horizon to WalkForwardResult.
    """
    horizons = horizons or HORIZONS
    hparams_dir = hparams_dir or HPARAMS_DIR
    audit_dir = audit_dir or AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint directory for per-fold resume
    model_cache_dir = RESULTS_DIR / "cache" / "models"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate folds (identical across horizons)
    folds = generate_folds(df, availability_col=availability_col)

    # Cache feature names
    feature_names = get_feature_cols(df)
    log.info("feature matrix: %d columns", len(feature_names))

    results: dict[int, WalkForwardResult] = {}

    for horizon in horizons:
        # Load frozen hyperparameters for this specific tier x horizon.
        hparams = load_frozen_hparams(
            model_name, hparams_dir, tier=tier, horizon=horizon,
        )
        log.info("loaded frozen hparams for %s h=%dd: %s",
                 model_name, horizon, hparams)
        log.info("=== horizon=%dd  model=%s  tier=%s ===", horizon, model_name, tier)
        t_start = time.time()

        fold_results: list[FoldResult] = []
        for fold in progress(folds, desc=f"wf h={horizon}d", unit="fold"):
            fr = _run_one_fold(
                df, fold, model_name, hparams, horizon, tier,
                availability_col, feature_names,
                checkpoint_dir=model_cache_dir,
                universe_name=universe_name,
            )
            if fr is not None:
                fold_results.append(fr)
                log.info("  fold %d: train=%d test=%d  ic=%.4f  (%.1fs)",
                         fold.fold_id, fr.n_train, fr.n_test,
                         spearman(fr.y_pred, fr.y_true), fr.fit_time_s)

        # Build OOS predictions
        oos_df, oos_ic, oos_mse = _build_oos_predictions_df(
            df, fold_results, horizon, model_name, tier, universe_name, signal_type,
        )

        # Write per-horizon OOS predictions
        _write_oos_predictions(
            oos_df, audit_dir, model_name, tier, universe_name, signal_type, horizon,
        )

        wr = WalkForwardResult(
            model_name=model_name,
            tier=tier,
            horizon=horizon,
            fold_results=fold_results,
            oos_predictions=oos_df,
            oos_ic=oos_ic,
            oos_mse=oos_mse,
            universe_name=universe_name,
            signal_type=signal_type,
        )
        results[horizon] = wr

        # Per-horizon cache manifest
        if features_path is not None:
            _write_horizon_cache_manifest(
                model_name, tier, horizon, universe_name, signal_type,
                features_path, hparams_dir,
            )

        # Clean up per-fold checkpoint files — no longer needed after the
        # aggregated OOS parquet has been written.
        for fold in folds:
            stem = f"{model_name}_{tier}_{universe_name}_h{horizon}d_fold{fold.fold_id}"
            for suffix in [".pkl", "_pred.parquet"]:
                cp = model_cache_dir / f"{stem}{suffix}"
                if cp.exists():
                    cp.unlink()

        elapsed = time.time() - t_start
        log.info("  horizon=%dd done (%.0fs)  OOS_IC=%.6f", horizon, elapsed, oos_ic)

    # Summary
    log.info("=== walk-forward summary: %s / %s ===", model_name, tier)
    for h, wr in results.items():
        log.info("  h=%dd  IC=%.6f  MSE=%.6f  folds=%d",
                 h, wr.oos_ic, wr.oos_mse, len(wr.fold_results))

    # Write audit outputs
    _write_walk_forward_audit_outputs(
        df, folds, results, model_name, tier, horizons, audit_dir, universe_name, signal_type,
    )

    return results





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_horizon_cache_manifest(
    model_name: str,
    tier: str,
    horizon: int,
    universe_name: str,
    signal_type: str,
    features_path: Path,
    hparams_dir: Path,
) -> None:
    """Build and write a cache manifest for one completed horizon.

    Captures input file hashes (feature parquet, frozen hparams JSON, price
    manifest) and source-code hashes for the key walk-forward functions so
    that ``run_all.py`` can skip the phase when nothing changed.
    """
    phase = (
        f"4_{model_name}_{tier}_{universe_name}"
        f"_{signal_type.lower()}_h{horizon}d"
    )
    hparams_path = (
        Path(hparams_dir) / tier / f"h{horizon}d"
        / f"frozen_hparams_{model_name}.json"
    )
    price_manifest_path = PRICE_CACHE_DIR / "_manifest.json"

    manifest = build_cache_manifest(
        phase=phase,
        parameters={
            "model": model_name,
            "tier": tier,
            "universe": universe_name,
            "signal_type": signal_type,
            "horizon": horizon,
        },
        input_paths=[features_path, hparams_path, price_manifest_path],
        source_funcs=[run_walk_forward, _run_one_fold],
    )
    manifest_path = CACHE_MANIFEST_DIR / f"{phase}.json"
    write_cache_manifest(manifest, manifest_path)
    log.info("wrote cache manifest: %s", manifest_path)


# ---------------------------------------------------------------------------
# Fit-audit logging
# ---------------------------------------------------------------------------

def write_fit_audit_log(
    results: dict[int, WalkForwardResult],
    output_path: Path | None = None,
) -> Path:
    """Write one JSONL line per fold result for the audit trail."""
    first = next(iter(results.values()), None)
    if output_path is None:
        if first is None:
            output_path = AUDIT_DIR / "fit_audit_log.jsonl"
        else:
            suffix_parts = [first.model_name, first.tier]
            if first.universe_name and first.universe_name != "all":
                suffix_parts.append(first.universe_name)
            if first.signal_type and first.signal_type != "all":
                suffix_parts.append(first.signal_type.lower())
            output_path = (
                AUDIT_DIR / f"fit_audit_log_{'_'.join(suffix_parts)}.jsonl"
            )
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    for h, wr in results.items():
        for fr in wr.fold_results:
            lines.append(json.dumps({
                "fold_id": fr.fold_id,
                "horizon": fr.horizon,
                "model": fr.model_name,
                "tier": wr.tier,
                "universe": wr.universe_name,
                "signal_type": wr.signal_type,
                "n_train": fr.n_train,
                "n_test": fr.n_test,
                "fit_time_s": round(fr.fit_time_s, 3),
                "fold_ic": round(spearman(fr.y_pred, fr.y_true), 6),
                "n_selected_features": (
                    len(fr.selected_features) if fr.selected_features else -1
                ),
                "selected_features": fr.selected_features or [],
                "fit_violations": fr.fit_violations or [],
                "fit_calls": fr.fit_log or [],
            }))

    output_path.write_text("\n".join(lines) + "\n")
    log.info("wrote fit audit log: %s", output_path)
    if first is not None:
        legacy_path = AUDIT_DIR / f"fit_audit_log_{first.model_name}_{first.tier}.jsonl"
        if legacy_path != output_path:
            legacy_path.write_text("\n".join(lines) + "\n")
            log.info("wrote compatibility fit audit log: %s", legacy_path)
    return output_path


# ---------------------------------------------------------------------------
# Sample filtering
# ---------------------------------------------------------------------------

def filter_model_sample(
    df: pd.DataFrame,
    *,
    universe_name: str = "sp500",
    signal_type: str = "Total",
    universe_date_col: str = "call_entry_date",
) -> pd.DataFrame:
    """Restrict the model sample to the requested PIT universe and signal slice.

    ``_orig_df_index`` is preserved so OOS prediction rows can be joined back
    to the original feature parquet by ``portfolio.py``.
    """
    from backtest.universe import filter_to_universe

    out = df
    if "_orig_df_index" not in out.columns:
        out["_orig_df_index"] = np.arange(len(out), dtype=np.int64)

    if signal_type and signal_type.lower() != "all":
        if "SignalType" not in out.columns:
            raise KeyError("SignalType column is required for signal-type filtering")
        before = len(out)
        out = out[out["SignalType"] == signal_type].copy()
        log.info("SignalType filter %s: %d / %d rows kept", signal_type, len(out), before)

    if universe_name and universe_name.lower() != "all":
        before = len(out)
        out = filter_to_universe(
            out,
            universe_name,
            date_col=universe_date_col,
            ticker_col="BESTTICKER",
        )
        out = out[out["_in_universe"]].copy()
        log.info("Universe filter %s: %d / %d rows kept", universe_name, len(out), before)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_features_with_forward_returns(args: argparse.Namespace) -> pd.DataFrame:
    """Load feature parquet and ensure requested forward returns are present."""
    log.info("loading features from %s", args.features)
    feat_df = pd.read_parquet(args.features)
    log.info("features: %d rows x %d cols", len(feat_df), len(feat_df.columns))

    feat_df = ensure_forward_returns(
        feat_df, args.features, horizons=args.horizons,
        entry_date_col=args.availability_col,
    )

    feat_df = filter_model_sample(
        feat_df,
        universe_name=args.universe,
        signal_type=args.signal_type,
        universe_date_col="call_entry_date",
    )
    if feat_df.empty:
        raise RuntimeError(
            f"model sample is empty after universe={args.universe}, "
            f"signal_type={args.signal_type} filtering"
        )
    log.info(
        "model sample after filters: %d rows x %d cols",
        len(feat_df),
        len(feat_df.columns),
    )
    return feat_df


def _models_to_run(model_arg: str) -> list[str]:
    """Expand 'all' into ridge/lightgbm/xgboost."""
    if model_arg == "all":
        return ["ridge", "lightgbm", "xgboost"]
    return [model_arg]


def _tune_missing_hparams(
    args: argparse.Namespace,
    feat_df: pd.DataFrame,
    models_to_run: list[str],
) -> None:
    """Tune missing or requested frozen hyperparameters for all horizons."""
    from backtest.splits import tune_all_models

    hparams_dir = args.hparams_dir or HPARAMS_DIR
    horizons_to_tune = args.horizons if args.tune_only else args.horizons
    for tune_h in horizons_to_tune:
        target_col = f"forward_return_{tune_h}d"
        target_avail_col = f"target_available_date_{tune_h}d"

        if target_col not in feat_df.columns:
            raise RuntimeError(
                f"{target_col} not in feature columns — run "
                f"compute_forward_returns first or set --horizons"
            )

        missing_hparams = [
            m for m in models_to_run
            if not (hparams_dir / args.tier / f"h{tune_h}d" / f"frozen_hparams_{m}.json").exists()
        ]
        if not missing_hparams and not args.tune_only:
            continue

        to_tune = models_to_run if args.tune_only else missing_hparams
        log.info(
            "tuning models %s (tier=%s, horizon=%dd, availability_col=%s) …",
            to_tune, args.tier, tune_h, args.availability_col,
        )
        tune_all_models(
            feat_df,
            target_col,
            target_avail_col,
            output_dir=hparams_dir,
            availability_col=args.availability_col,
            models=to_tune,
            tier=args.tier,
            horizon=tune_h,
        )

    for h in args.horizons:
        for m in models_to_run:
            hparam_path = hparams_dir / args.tier / f"h{h}d" / f"frozen_hparams_{m}.json"
            log.info("frozen hparams for %s h=%dd: %s", m, h, hparam_path)


def _run_walk_forward_models(
    args: argparse.Namespace,
    feat_df: pd.DataFrame,
    models_to_run: list[str],
) -> None:
    """Run walk-forward and write fit audit logs for all selected models."""
    for m in models_to_run:
        results = run_walk_forward(
            feat_df,
            model_name=m,
            tier=args.tier,
            horizons=args.horizons,
            hparams_dir=args.hparams_dir,
            availability_col=args.availability_col,
            universe_name=args.universe,
            signal_type=args.signal_type,
            features_path=args.features,
        )
        write_fit_audit_log(results)


def main() -> None:
    import argparse

    from data.config import set_global_seed

    parser = argparse.ArgumentParser(
        description="Phase 4 — Walk-Forward Model Backtest",
    )
    parser.add_argument(
        "--features", type=Path,
        default=RESULTS_DIR / "features_enhanced.parquet",
        help="Path to feature parquet (enhanced or stretch tier)",
    )
    parser.add_argument(
        "--tier", choices=["enhanced", "stretch"], default="enhanced",
    )
    parser.add_argument(
        "--model", choices=["ridge", "lightgbm", "xgboost", "all"],
        default="ridge",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=HORIZONS,
    )
    parser.add_argument(
        "--tune-only", action="store_true",
        help="Tune hyperparameters on 2010-2019 and exit",
    )
    parser.add_argument(
        "--hparams-dir", type=Path, default=None,
        help="Directory with tier/horizon hparam subdirectories (default: results/hparams)",
    )
    parser.add_argument(
        "--tune-horizon", type=int, default=5,
        help="Deprecated; tuning now runs for every value in --horizons.",
    )
    parser.add_argument(
        "--availability-col", default="availability_date",
        choices=["call_entry_date", "availability_date"],
        help="Column for PIT availability filtering. "
             "'availability_date' (default) uses the unified operational "
             "availability date: call_entry_date+2bd before 2023-07-06, "
             "max(call,ingest) on/after. 'call_entry_date' uses the earnings "
             "call publication date (MOSTIMPORTANTDATEUTC).",
    )
    parser.add_argument(
        "--universe",
        choices=["sp500", "sp1500", "ru3k", "all"],
        default="sp500",
        help="PIT universe used for model training/backtest (default: sp500).",
    )
    parser.add_argument(
        "--signal-type",
        default="Total",
        help="SignalType slice for the predictive model (default: Total; use all to disable).",
    )
    args = parser.parse_args()

    set_global_seed()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load features with forward returns
    feat_df = _load_features_with_forward_returns(args)

    # Determine which models to run
    models_to_run = _models_to_run(args.model)

    # Tune missing hyperparameters
    _tune_missing_hparams(args, feat_df, models_to_run)

    if args.tune_only:
        log.info("--tune-only: done")
        return

    # Run walk-forward backtest
    _run_walk_forward_models(args, feat_df, models_to_run)

    log.info("Phase 4 done.")


if __name__ == "__main__":
    main()
