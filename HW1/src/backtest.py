"""Backtest: (ticker, call_date) signals → metrics + equity curve.

Per Part II §2.8, extended with long-short support (report.md §5.2):

* **Directional accuracy**: ``mean(sign(score - s0) == sign(excess))``
  across all test rows (``s0 = 0.5`` for classifier scores, ``0`` for
  regression / rule scores). Unaffected by leg direction — the score
  already carries sign.
* **Hit rate**: fraction of *traded* rows where ``direction · excess > 0``.
  Long trades win on positive excess, short trades win on negative
  excess. ML models are long-only (direction ≡ +1 on traded rows), so
  their hit rate is unchanged from the long-only contract.
* **IC**: Spearman rank correlation between continuous ``score`` and
  ``excess_{h}``, after dropping NaN scores. Does not depend on the
  leg decision (only on the score ranking).
* **PnL**: Sum of ``direction · excess_{h}`` across traded rows
  (equal-weight, unit notional per leg). Long-short trades cancel
  market beta at the row level, not across rows.
* **Sharpe**: ``mean(bets) / std(bets) · sqrt(N_trades)`` on the signed
  bet stream, with a minimum of 3 trades; otherwise ``None``.
* **Equity**: Cumulative product of ``(1 + direction · excess)`` across
  traded rows, **chronologically ordered** by ``call_date``.
* **n_long / n_short**: per-leg trade counts alongside ``n_trades``.
"""

from __future__ import annotations

import json
import math
from typing import Literal

import numpy as np
import pandas as pd

from .io_paths import (
    backtest_path,
    backtest_summary_path,
    ensure_dirs,
    equity_path,
    error,
    preds_path,
)


ScoreKind = Literal["binary", "regression", "rule"]


def run_backtest(
    meta: pd.DataFrame,
    score: pd.Series,
    action: pd.Series | None,
    horizon: str,
    extraction: str,
    model_name: str,
    score_kind: ScoreKind,
) -> dict:
    """Evaluate a signal and persist artifacts; return the metrics dict.

    ``meta`` must carry ``ticker``, ``call_date``, ``raw_{h}``, and
    ``excess_{h}`` (produced by :func:`dataset.build_xy`). ``score`` is
    always required (drives IC); ``action`` only exists for the rule
    baseline (drives PnL coverage). For ML variants, ``action`` may be
    ``None`` — every row with a finite score trades.
    """
    _validate_inputs(meta, score, action, horizon)

    n = len(meta)
    raw_col = f"raw_{horizon}"
    excess_col = f"excess_{horizon}"
    excess = meta[excess_col].to_numpy(dtype="float64")
    score_arr = score.to_numpy(dtype="float64")

    trade_mask, direction = _trade_mask(score, action)
    n_trades = int(trade_mask.sum())
    n_long = int((direction[trade_mask] == 1).sum())
    n_short = int((direction[trade_mask] == -1).sum())

    s0 = 0.5 if score_kind == "binary" else 0.0
    dir_acc = _directional_accuracy(score_arr, excess, s0)
    hit = _hit_rate(trade_mask, direction, excess)
    ic = _information_coefficient(score_arr, excess)
    pnl_stats = _pnl(trade_mask, direction, excess)

    order = meta["call_date"].astype(str).argsort()
    equity_df = _equity_curve(
        meta.iloc[order], trade_mask[order], direction[order], excess[order], horizon,
    )

    metrics = {
        "extraction": extraction,
        "model": model_name,
        "horizon": horizon,
        "score_kind": score_kind,
        "n_rows": n,
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "directional_accuracy": dir_acc,
        "hit_rate": hit,
        "IC_spearman": ic,
        "pnl_sum": pnl_stats["sum"],
        "pnl_mean_per_trade": pnl_stats["mean"],
        "sharpe_annualized": pnl_stats["sharpe"],
        "avg_win": pnl_stats["avg_win"],
        "avg_loss": pnl_stats["avg_loss"],
        "final_equity": float(equity_df["equity"].iloc[-1]) if len(equity_df) else 1.0,
    }

    _persist(
        metrics,
        equity_df,
        meta=meta,
        score=score,
        action=action,
        score_arr=score_arr,
        trade_mask=trade_mask,
        direction=direction,
        raw_col=raw_col,
        excess_col=excess_col,
        extraction=extraction,
        model_name=model_name,
        horizon=horizon,
    )
    _append_summary(metrics)
    return metrics


def _validate_inputs(meta: pd.DataFrame, score: pd.Series, action: pd.Series | None, horizon: str) -> None:
    raw_col = f"raw_{horizon}"
    excess_col = f"excess_{horizon}"
    for c in ("ticker", "call_date", raw_col, excess_col):
        if c not in meta.columns:
            error(f"run_backtest: meta missing column {c!r}")
    if len(meta) != len(score):
        error(f"run_backtest: meta ({len(meta)}) and score ({len(score)}) length mismatch")
    if action is not None and len(action) != len(meta):
        error(f"run_backtest: action ({len(action)}) length mismatch vs meta ({len(meta)})")
    if meta[excess_col].isna().any():
        error(f"run_backtest: meta has NaN in {excess_col} — build_xy should have dropped these")


def _trade_mask(score: pd.Series, action: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(traded_bool, direction_int)``.

    ``direction`` is ``+1`` on long, ``-1`` on short, ``0`` on flat.
    ML models call with ``action=None`` and get direction ``+1`` on every
    finite-score row (long-only, matching the pre-long-short contract).
    """
    if action is not None:
        act = action.fillna("flat").to_numpy()
        traded = (act == "long") | (act == "short")
        direction = np.where(act == "long", 1, np.where(act == "short", -1, 0)).astype(int)
        return traded, direction
    traded = score.notna().to_numpy()
    direction = traded.astype(int)
    return traded, direction


def _directional_accuracy(score_arr: np.ndarray, excess: np.ndarray, s0: float) -> float:
    """Fraction of rows where the directional call matched the excess return.

    Applied to **every** test row (even flat trades), because the signal
    still expressed a directional belief about the residual return.
    ``sign(0) == sign(0)`` would pollute the count, so we drop rows whose
    score is NaN or whose excess is zero.
    """
    mask = ~np.isnan(score_arr) & (excess != 0.0)
    if not mask.any():
        return float("nan")
    pred_dir = np.sign(score_arr[mask] - s0)
    actual_dir = np.sign(excess[mask])
    match = (pred_dir == actual_dir).sum()
    return float(match) / float(mask.sum())


def _hit_rate(trade_mask: np.ndarray, direction: np.ndarray, excess: np.ndarray) -> float:
    if not trade_mask.any():
        return float("nan")
    signed = direction[trade_mask] * excess[trade_mask]
    return float((signed > 0).sum()) / float(trade_mask.sum())


def _information_coefficient(score_arr: np.ndarray, excess: np.ndarray) -> float:
    """Spearman rank correlation (NaN-safe). Requires ≥ 3 finite pairs."""
    mask = ~np.isnan(score_arr) & ~np.isnan(excess)
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return _spearman_fallback(score_arr[mask], excess[mask])
    rho, _p = spearmanr(score_arr[mask], excess[mask])
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return float("nan")
    return float(rho)


def _spearman_fallback(a: np.ndarray, b: np.ndarray) -> float:
    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty_like(order, dtype="float64")
        r[order] = np.arange(len(v))
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pnl(trade_mask: np.ndarray, direction: np.ndarray, excess: np.ndarray) -> dict:
    bets = direction[trade_mask] * excess[trade_mask]
    if len(bets) == 0:
        return {
            "sum": 0.0,
            "mean": float("nan"),
            "sharpe": None,
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
        }
    pnl_sum = float(bets.sum())
    pnl_mean = float(bets.mean())
    std = float(bets.std(ddof=1)) if len(bets) > 1 else 0.0
    sharpe: float | None
    if len(bets) < 3 or std == 0.0:
        sharpe = None
    else:
        sharpe = pnl_mean / std * math.sqrt(len(bets))
    wins = bets[bets > 0]
    losses = bets[bets < 0]
    return {
        "sum": pnl_sum,
        "mean": pnl_mean,
        "sharpe": sharpe,
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
    }


def _equity_curve(
    meta: pd.DataFrame,
    trade_mask: np.ndarray,
    direction: np.ndarray,
    excess: np.ndarray,
    horizon: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    equity = 1.0
    for i, (_, r) in enumerate(meta.iterrows()):
        trade = bool(trade_mask[i])
        sign = int(direction[i])
        x = float(excess[i])
        delta = sign * x if trade else 0.0
        equity *= (1.0 + delta)
        rows.append({
            "ticker": r["ticker"],
            "call_date": r["call_date"],
            "traded": int(trade),
            "direction": sign,
            f"excess_{horizon}": x,
            "equity": equity,
        })
    return pd.DataFrame(rows)


def _persist(
    metrics: dict,
    equity_df: pd.DataFrame,
    *,
    meta: pd.DataFrame,
    score: pd.Series,
    action: pd.Series | None,
    score_arr: np.ndarray,
    trade_mask: np.ndarray,
    direction: np.ndarray,
    raw_col: str,
    excess_col: str,
    extraction: str,
    model_name: str,
    horizon: str,
) -> None:
    ensure_dirs()

    metrics_path = backtest_path(extraction, model_name, horizon)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    eq_path = equity_path(extraction, model_name, horizon)
    equity_df.to_csv(eq_path, index=False)

    preds_cols = {
        "ticker": meta["ticker"].values,
        "call_date": meta["call_date"].values,
        "score": score_arr,
        "traded": trade_mask.astype(int),
        "direction": direction.astype(int),
        raw_col: meta[raw_col].values,
        excess_col: meta[excess_col].values,
    }
    if action is not None:
        preds_cols["action"] = action.values
    preds_df = pd.DataFrame(preds_cols)
    preds_df.to_parquet(preds_path(extraction, model_name, horizon), index=False)


def _append_summary(metrics: dict) -> None:
    """Upsert-replace this (extraction, model, horizon) row in summary.parquet."""
    path = backtest_summary_path()
    if path.is_file():
        existing = pd.read_parquet(path)
        mask = (
            (existing["extraction"] == metrics["extraction"])
            & (existing["model"] == metrics["model"])
            & (existing["horizon"] == metrics["horizon"])
        )
        existing = existing[~mask]
        out = pd.concat([existing, pd.DataFrame([metrics])], ignore_index=True)
    else:
        out = pd.DataFrame([metrics])
    out = out.sort_values(["extraction", "horizon", "model"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
