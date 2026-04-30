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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.config import RESULTS_DIR, AUDIT_DIR, SEED
from data.progress import progress

from backtest.splits import (
    HORIZONS,
    Fold,
    HPARAMS_DIR,
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

def _make_median_imputer() -> Any:
    from sklearn.impute import SimpleImputer

    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older scikit-learn fallback
        return SimpleImputer(strategy="median")


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

    imp = _make_median_imputer()
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
        n_jobs=-1,
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
            n_jobs=-1,
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
            n_jobs=-1,
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


def _run_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    model_name: str,
    hparams: dict[str, Any],
    horizon: int,
    tier: str,
    availability_col: str,
    feature_names: list[str] | None = None,
) -> FoldResult | None:
    """Fit one model on one fold for one horizon.

    Returns a FoldResult with predictions, or None if the purged training set
    or test set is empty.
    """
    # ----- G11 purge -----
    train_idx = purge_train_for_horizon(fold, df, horizon)
    test_idx = fold.test_indices

    if len(train_idx) < 10 or len(test_idx) == 0:
        log.info("  fold %d h=%d: skip (train=%d test=%d)",
                 fold.fold_id, horizon, len(train_idx), len(test_idx))
        return None

    # ----- Build feature matrix -----
    if feature_names is None:
        feature_names = get_feature_cols(df)

    target_col = f"forward_return_{horizon}d"
    target_date_col = f"target_available_date_{horizon}d"

    train_dates = pd.to_datetime(
        df[availability_col].iloc[train_idx], errors="coerce"
    ).to_numpy(dtype="datetime64[ns]")
    train_order = np.argsort(train_dates, kind="mergesort")
    train_idx = train_idx[train_order]

    y_train_all = df[target_col].iloc[train_idx].to_numpy(dtype="float64")
    train_valid = np.isfinite(y_train_all)
    if not train_valid.all():
        train_idx = train_idx[train_valid]
        y_train_all = y_train_all[train_valid]

    y_test_all = df[target_col].iloc[test_idx].to_numpy(dtype="float64")
    test_valid = np.isfinite(y_test_all)

    if len(train_idx) < 10:
        log.info("  fold %d h=%d: skip after target filter (train=%d)",
                 fold.fold_id, horizon, len(train_idx))
        return None
    if not test_valid.any():
        log.info("  fold %d h=%d: all test targets NaN — skip",
                 fold.fold_id, horizon)
        return None

    test_idx_valid = test_idx[test_valid]
    y_train = y_train_all
    y_test = y_test_all[test_valid]

    X_train_all = df[feature_names].iloc[train_idx].copy()
    X_test_all = df[feature_names].iloc[test_idx_valid].copy()
    X_train_all.index = pd.DatetimeIndex(
        pd.to_datetime(df[availability_col].iloc[train_idx], errors="coerce")
    )
    X_test_all.index = pd.DatetimeIndex(
        pd.to_datetime(df[availability_col].iloc[test_idx_valid], errors="coerce")
    )

    # ----- Fit audit: intercept every fit() call (Plan §3.2 assertion 3) -----
    from features.audit import monitor_fit_calls, get_fit_log, assert_fit_callstack, assert_fold_boundaries

    train_start = pd.Timestamp(
        pd.to_datetime(df[availability_col].iloc[train_idx], errors="coerce").min()
    )
    train_end = pd.Timestamp(
        pd.to_datetime(df[availability_col].iloc[train_idx], errors="coerce").max()
    )
    fit_violations: list[str] = []

    with monitor_fit_calls():
        # ----- Impute + scale (fit on train only) -----
        X_tr, X_te, imp, scl = _impute_and_scale(X_train_all, X_test_all)

        # ----- Stretch tier: LassoCV column selection -----
        selected_features = None
        if tier == "stretch":
            train_feature_dates = pd.to_datetime(
                df[availability_col].iloc[train_idx], errors="coerce"
            ).to_numpy(dtype="datetime64[ns]")
            train_target_dates = pd.to_datetime(
                df[target_date_col].iloc[train_idx], errors="coerce"
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

        # ----- Fit model -----
        t0 = time.time()
        model = _make_model(model_name, hparams)
        model.fit(X_tr, y_train)
        fit_time = time.time() - t0

    # ----- Verify fit-call boundaries -----
    fit_log = get_fit_log()
    for entry in fit_log:
        entry["fold_id"] = fold.fold_id
        entry["horizon"] = horizon
        entry["train_start"] = str(train_start.date())
        entry["train_end"] = str(train_end.date())
    fit_violations = assert_fit_callstack(
        fit_log, train_start, train_end,
        fold_id=f"fold_{fold.fold_id}_h{horizon}d",
    )
    if fit_violations:
        log.warning("  fold %d h=%d: %d fit-call boundary violation(s)",
                    fold.fold_id, horizon, len(fit_violations))
        for v in fit_violations:
            log.warning("    %s", v)

    # ----- Fold boundary validation (Plan §3.2 assertion 2) -----
    max_train_target = fold.max_train_target_date.get(horizon)
    boundary_violations = assert_fold_boundaries(
        fold_train_start=fold.train_start,
        fold_train_end=fold.train_end,
        fold_test_start=fold.test_start,
        fold_test_end=fold.test_end,
        max_train_feature_date=train_end,
        max_train_target_available=max_train_target,
        horizon_days=horizon,
    )
    if boundary_violations:
        log.warning("  fold %d h=%d: %d fold-boundary violation(s)",
                    fold.fold_id, horizon, len(boundary_violations))
        for v in boundary_violations:
            log.warning("    %s", v)
        fit_violations.extend(boundary_violations)

    y_pred = model.predict(X_te)

    return FoldResult(
        fold_id=fold.fold_id,
        horizon=horizon,
        model_name=model_name,
        n_train=len(train_idx),
        n_test=len(test_idx_valid),
        test_indices=test_idx_valid,
        y_pred=y_pred,
        y_true=y_test,
        selected_features=selected_features,
        fit_time_s=fit_time,
        fit_violations=fit_violations if fit_violations else None,
        fit_log=fit_log,
    )


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
) -> dict[int, WalkForwardResult]:
    """Run the full walk-forward backtest for one model + tier.

    Parameters
    ----------
    df: Feature DataFrame with ``forward_return_{h}d`` and
        ``target_available_date_{h}d`` columns already joined.
    model_name: ``"ridge"`` | ``"lightgbm"`` | ``"xgboost"``.
    tier: ``"enhanced"`` | ``"stretch"``.
    horizons: List of forward-return horizons (default: [1,3,5,10,20]).
    hparams_dir: Directory containing ``{tier}/h{horizon}d/frozen_hparams_{model}.json``.
    audit_dir: Directory for audit output files.
    availability_col: Column for PIT availability filtering of training data
        (default ``"availability_date"``).

    Returns
    -------
    Dict mapping horizon → WalkForwardResult.
    """
    horizons = horizons or HORIZONS
    hparams_dir = hparams_dir or HPARAMS_DIR
    audit_dir = audit_dir or AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)

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
            )
            if fr is not None:
                fold_results.append(fr)
                log.info("  fold %d: train=%d test=%d  ic=%.4f  (%.1fs)",
                         fold.fold_id, fr.n_train, fr.n_test,
                         _spearman(fr.y_pred, fr.y_true), fr.fit_time_s)

        # Aggregate OOS predictions
        if fold_results:
            all_pred = np.concatenate([f.y_pred for f in fold_results])
            all_true = np.concatenate([f.y_true for f in fold_results])
            all_indices = np.concatenate([f.test_indices for f in fold_results])
            oos_ic = _spearman(all_pred, all_true)
            oos_mse = float(np.mean((all_pred - all_true) ** 2))

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
            log.info("  aggregate OOS: n=%d  IC=%.6f  MSE=%.6f",
                     len(all_pred), oos_ic, oos_mse)
        else:
            oos_df = None
            oos_ic = float("nan")
            oos_mse = float("nan")
            log.warning("  no valid fold results for horizon=%d", horizon)

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

        # Write per-horizon OOS predictions
        if oos_df is not None:
            suffix_parts = [model_name, tier]
            if universe_name and universe_name != "all":
                suffix_parts.append(universe_name)
            if signal_type and signal_type != "all":
                suffix_parts.append(signal_type.lower())
            suffix_parts.append(f"h{horizon}d")
            pred_path = audit_dir / f"oos_pred_{'_'.join(suffix_parts)}.parquet"
            oos_df.to_parquet(pred_path, index=False)
            log.info("  wrote %s", pred_path)

        elapsed = time.time() - t_start
        log.info("  horizon=%dd done (%.0fs)  OOS_IC=%.6f", horizon, elapsed, oos_ic)

    # Summary
    log.info("=== walk-forward summary: %s / %s ===", model_name, tier)
    for h, wr in results.items():
        log.info("  h=%dd  IC=%.6f  MSE=%.6f  folds=%d",
                 h, wr.oos_ic, wr.oos_mse, len(wr.fold_results))

    # Write audit outputs
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
    )
    # Keep the generic evidence filenames populated for the one-page checklist;
    # these intentionally reflect the most recent walk-forward invocation.
    write_fold_manifest(folds, output_path=audit_dir / "fold_manifest.parquet",
                        model=model_name, tier=tier, horizons=horizons)
    write_sample_size_audit(df, folds, output_path=audit_dir / "sample_size_by_quarter.parquet")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


# ---------------------------------------------------------------------------
# Fit-audit logging
# ---------------------------------------------------------------------------

def write_fit_audit_log(
    results: dict[int, WalkForwardResult],
    output_path: Path | None = None,
) -> Path:
    """Write one JSONL line per fold result for the audit trail."""
    if output_path is None:
        first = next(iter(results.values()), None)
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
                "fold_ic": round(_spearman(fr.y_pred, fr.y_true), 6),
                "n_selected_features": (
                    len(fr.selected_features) if fr.selected_features else -1
                ),
                "selected_features": fr.selected_features or [],
                "fit_violations": fr.fit_violations or [],
                "fit_calls": fr.fit_log or [],
            }))

    output_path.write_text("\n".join(lines) + "\n")
    log.info("wrote fit audit log: %s", output_path)
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

    out = df.copy()
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

    # Load features
    log.info("loading features from %s", args.features)
    feat_df = pd.read_parquet(args.features)
    log.info("features: %d rows x %d cols", len(feat_df), len(feat_df.columns))

    # Compute forward returns if not already present
    has_returns = any(
        c.startswith("forward_return_") for c in feat_df.columns
    )
    if not has_returns:
        log.info("computing forward returns (cached) …")
        from backtest.splits import get_forward_returns_cached
        fwd = get_forward_returns_cached(
            args.features, feat_df, horizons=args.horizons,
            entry_date_col=args.availability_col,
        )
        feat_df = pd.concat([feat_df.reset_index(drop=True),
                             fwd.reset_index(drop=True)], axis=1)
        log.info("forward returns joined: %d cols", len(feat_df.columns))

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

    models_to_run = (
        ["ridge", "lightgbm", "xgboost"] if args.model == "all"
        else [args.model]
    )

    # ------------------------------------------------------------------
    # Hyperparameter tuning (one-time, on 2010-2019)
    # ------------------------------------------------------------------
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

        from backtest.splits import tune_all_models

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

    if args.tune_only:
        log.info("--tune-only: done")
        return

    # ------------------------------------------------------------------
    # Walk-forward backtest
    # ------------------------------------------------------------------
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
        )
        write_fit_audit_log(results)

    log.info("Phase 4 done.")


if __name__ == "__main__":
    main()
