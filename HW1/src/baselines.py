"""Rule-based baseline: event-gated long/short on ``sentiment_call``.

Per Part II §2.7 + the long-short extension documented in ``report.md §5.2``:

* **Long leg** (event-aware): ``long if sentiment_call > τ_long AND is_guidance_raised``.
* **Short leg** (event-aware): ``short if sentiment_call < τ_short AND is_guidance_lowered``.
* **ext_finbert_only**: event gate is dropped on both legs —
  ``long if sentiment_call > τ_long``, ``short if sentiment_call < τ_short``.
* Continuous IC score is unchanged from the long-only version:
  ``sentiment_call + 0.5 · guidance_ordinal`` (event-aware) or
  ``sentiment_call`` (``ext_finbert_only``).

**τ selection** (independent per leg, same quantile grid):

* Grid: ``{p20, p40, p60, p80}`` of the **train** ``sentiment_call``
  distribution, deduplicated.
* ``τ_long*`` maximizes train ``hit_rate`` on predicted-long trades
  (``excess_train > 0``), subject to
  ``n_long_train >= max(3, ceil(0.1 · n_train))``.
* ``τ_short*`` maximizes train ``hit_rate`` on predicted-short trades
  (``excess_train < 0``), subject to the same coverage floor.
* Ties broken by leg count.
* Coverage fallback: if no candidate meets the floor, relax to
  ``>0`` count. If the relaxed pool is still empty, fall back to
  "event-only" on that leg: ``τ_long* = -inf`` (trade whenever event
  long gate fires) or ``τ_short* = +inf`` (trade whenever event short
  gate fires). Under ``ext_finbert_only`` the corresponding fallback
  is "trade every row with finite sentiment on that leg".
* The two legs are selected independently; nothing prevents
  ``τ_long* ≤ τ_short*`` on tiny samples (possible dead-zone overlap
  is handled at predict time by preferring long when both gates match).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_paths import error


Extraction = str
_EVENT_AWARE_VARIANTS = {"ext_finbert_llm", "ext_lm_llm"}
_EVENT_LESS_VARIANTS = {"ext_finbert_only"}


@dataclass
class RuleParams:
    """Fitted rule state needed to replay the decision on new data."""
    tau: float
    tau_short: float
    variant: Extraction
    grid_empty_long: bool
    grid_empty_short: bool

    def to_dict(self) -> dict:
        return {
            "tau": float(self.tau),
            "tau_short": float(self.tau_short),
            "variant": self.variant,
            "grid_empty_long": bool(self.grid_empty_long),
            "grid_empty_short": bool(self.grid_empty_short),
        }


def e0_rule_params(variant: Extraction) -> RuleParams:
    """Forced τ for the E0 smoke (Part II §2.9 — 4-call sample)."""
    return RuleParams(
        tau=0.0, tau_short=0.0, variant=variant,
        grid_empty_long=False, grid_empty_short=False,
    )


def pick_tau(
    train_df: pd.DataFrame,
    variant: Extraction,
    horizon: str,
) -> RuleParams:
    """Select (τ_long*, τ_short*) on ``train_df``.

    Each leg picks its τ independently on the shared
    ``{p20, p40, p60, p80}`` quantile grid of train ``sentiment_call``,
    maximizing hit-rate on its own leg (long: ``excess > 0``;
    short: ``excess < 0``) subject to the coverage floor.
    ``train_df`` must be canonicalized for ``variant``.
    """
    _require_variant(variant)
    excess_col = f"excess_{horizon}"
    if excess_col not in train_df.columns:
        error(f"pick_tau: train_df missing {excess_col!r}")
    if "sentiment_call" not in train_df.columns:
        error("pick_tau: train_df missing 'sentiment_call' (canonicalize first)")

    clean = train_df[train_df[excess_col].notna() & train_df["sentiment_call"].notna()]
    if clean.empty:
        return RuleParams(
            tau=-math.inf, tau_short=math.inf, variant=variant,
            grid_empty_long=True, grid_empty_short=True,
        )

    sent = clean["sentiment_call"].to_numpy()
    excess = clean[excess_col].to_numpy()
    event_long = _event_mask(clean, variant, leg="long")
    event_short = _event_mask(clean, variant, leg="short")

    grid = _dedupe(np.quantile(sent, [0.2, 0.4, 0.6, 0.8]).tolist())
    n_train = len(clean)
    coverage_floor = max(3, math.ceil(0.1 * n_train))

    tau_long, empty_long = _pick_leg(
        grid, sent, excess, event_long,
        compare=np.greater, hit_fn=lambda x: x > 0, coverage_floor=coverage_floor,
        empty_value=-math.inf,
    )
    tau_short, empty_short = _pick_leg(
        grid, sent, excess, event_short,
        compare=np.less, hit_fn=lambda x: x < 0, coverage_floor=coverage_floor,
        empty_value=math.inf,
    )
    return RuleParams(
        tau=tau_long, tau_short=tau_short, variant=variant,
        grid_empty_long=empty_long, grid_empty_short=empty_short,
    )


def rule_predict(
    df: pd.DataFrame,
    params: RuleParams,
) -> tuple[pd.Series, pd.Series]:
    """Apply a fitted rule to ``df``; return ``(action, score)`` as Series.

    ``action ∈ {"long", "short", "flat"}``. When both long and short
    gates fire on the same row (only possible on tiny samples where
    τ_long* ≤ τ_short*), long wins — the event gates are mutually
    exclusive under ``ext_finbert_llm / ext_lm_llm`` (a call cannot
    simultaneously be ``is_guidance_raised == 1`` and
    ``is_guidance_lowered == 1``), so overlap only happens under
    ``ext_finbert_only``. ``score`` is the continuous IC score
    (Part II §2.7), unchanged from long-only.
    """
    _require_variant(params.variant)
    if "sentiment_call" not in df.columns:
        error("rule_predict: df missing 'sentiment_call' (canonicalize first)")

    sent = df["sentiment_call"]
    score = _score_series(df, params.variant)

    long_gate = _leg_gate(df, sent, params.variant, leg="long", tau=params.tau, grid_empty=params.grid_empty_long)
    short_gate = _leg_gate(df, sent, params.variant, leg="short", tau=params.tau_short, grid_empty=params.grid_empty_short)

    action = np.where(long_gate, "long", np.where(short_gate, "short", "flat"))
    return pd.Series(action, index=df.index, name="action"), score.rename("score")


def _leg_gate(
    df: pd.DataFrame,
    sent: pd.Series,
    variant: Extraction,
    *,
    leg: str,
    tau: float,
    grid_empty: bool,
) -> np.ndarray:
    """Boolean gate for a single leg, respecting τ + event + NaN semantics."""
    if variant in _EVENT_AWARE_VARIANTS:
        event_col = "is_guidance_raised" if leg == "long" else "is_guidance_lowered"
        if event_col not in df.columns:
            error(f"rule_predict: event-aware variant missing {event_col!r}")
        event = df[event_col].astype(int) == 1
        if grid_empty:
            gate = event
        else:
            gate = event & (sent > tau if leg == "long" else sent < tau)
    else:
        if grid_empty:
            gate = sent.notna()
        else:
            gate = (sent > tau) if leg == "long" else (sent < tau)
    gate = gate & sent.notna()
    return gate.fillna(False).to_numpy()


def _pick_leg(
    grid: list[float],
    sent: np.ndarray,
    excess: np.ndarray,
    event_mask: np.ndarray,
    *,
    compare,
    hit_fn,
    coverage_floor: int,
    empty_value: float,
) -> tuple[float, bool]:
    """Pick τ for one leg; return (τ, grid_empty_flag)."""
    candidates: list[tuple[float, int, float]] = []
    for tau in grid:
        mask = event_mask & compare(sent, tau)
        n = int(mask.sum())
        if n == 0:
            candidates.append((tau, n, float("nan")))
            continue
        hit = float(hit_fn(excess[mask]).mean())
        candidates.append((tau, n, hit))

    under_floor = [c for c in candidates if c[1] >= coverage_floor]
    pool = under_floor if under_floor else [c for c in candidates if c[1] > 0]
    if not pool:
        return empty_value, True
    pool.sort(key=lambda c: (c[2] if not math.isnan(c[2]) else -math.inf, c[1]), reverse=True)
    return float(pool[0][0]), False


def _event_mask(df: pd.DataFrame, variant: Extraction, leg: str = "long") -> np.ndarray:
    """Boolean event gate array used by :func:`pick_tau` for the given leg."""
    if variant in _EVENT_AWARE_VARIANTS:
        col = "is_guidance_raised" if leg == "long" else "is_guidance_lowered"
        if col not in df.columns:
            error(f"pick_tau: event-aware variant requires {col!r} column")
        return df[col].to_numpy().astype(bool)
    return np.ones(len(df), dtype=bool)


def _score_series(df: pd.DataFrame, variant: Extraction) -> pd.Series:
    """Continuous IC score (matches :func:`rule_predict`'s return value)."""
    sent = df["sentiment_call"].astype(float)
    if variant in _EVENT_AWARE_VARIANTS:
        if "guidance_ordinal" not in df.columns:
            error("score: event-aware variant requires 'guidance_ordinal' column")
        return sent + 0.5 * df["guidance_ordinal"].astype(float)
    return sent


def _dedupe(values: list[float]) -> list[float]:
    """Remove consecutive duplicates from a quantile grid (order-preserving)."""
    seen: set[float] = set()
    out: list[float] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _require_variant(variant: Extraction) -> None:
    if variant not in _EVENT_AWARE_VARIANTS and variant not in _EVENT_LESS_VARIANTS:
        error(f"unknown extraction variant: {variant}")
