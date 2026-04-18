"""Backtest figure generation (PNG, headless ``Agg`` backend).

PDF §6 explicitly requires "at least one equity-curve plot" in the
deliverable. Backtests already persist 24 ``equity_*.csv`` files under
``data/cache/backtest/``; this module renders them into multi-panel
figures saved to ``data/cache/plots/``.

Conventions:

- All equity curves are on **excess returns** (raw - SPY), so the SPY
  buy-and-hold reference is a horizontal line at ``1.0`` (annotated).
- Curves are chronologically ordered by ``call_date`` to match
  ``backtest._equity_curve``.
- Markers on traded rows make low-coverage strategies (e.g. the
  long-short rule on +21d, 21 trades) visible against high-coverage
  ML strategies (56 trades).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_paths import (
    PLOTS,
    analysis_path,
    backtest_summary_path,
    ensure_dirs,
    equity_path,
    error,
    plot_path,
    preds_path,
)
from .prices import BENCHMARK_TICKER, load_prices


_DEFAULT_PALETTE = {
    "ext_finbert_llm__rule": "#1f77b4",
    "ext_lm_llm__rule": "#2ca02c",
    "ext_finbert_only__rule": "#d62728",
    "ext_finbert_llm__ridge": "#ff7f0e",
    "ext_finbert_llm__logreg": "#9467bd",
    "ext_finbert_llm__xgb": "#8c564b",
    "ext_lm_llm__ridge": "#17becf",
}


def _load_equity(extraction: str, model: str, horizon: str) -> pd.DataFrame:
    p = equity_path(extraction, model, horizon)
    if not p.is_file():
        error(f"equity csv missing: {p}")
    df = pd.read_csv(p, parse_dates=["call_date"])
    return df.sort_values("call_date").reset_index(drop=True)


def _palette(label: str) -> str:
    return _DEFAULT_PALETTE.get(label, "#7f7f7f")


def _draw_one(
    ax,
    df: pd.DataFrame,
    label: str,
    color: str,
    linestyle: str = "-",
    show_markers: bool = True,
) -> None:
    """Plot one equity series with optional traded-row markers."""
    ax.plot(df["call_date"], df["equity"], label=label, color=color,
            linestyle=linestyle, linewidth=1.6, alpha=0.95)
    if show_markers and "traded" in df.columns:
        traded = df[df["traded"].astype(int) == 1]
        if not traded.empty:
            ax.scatter(traded["call_date"], traded["equity"],
                       s=18, color=color, alpha=0.55, zorder=3)


def plot_equity_curves(
    horizons: Sequence[str] = ("5d", "21d"),
    strategies: Sequence[tuple[str, str, str]] = (
        ("ext_finbert_llm", "rule", "long-short rule (FinBERT+LLM)"),
        ("ext_finbert_llm", "ridge", "Ridge (FinBERT+LLM)"),
        ("ext_lm_llm", "ridge", "Ridge (LM+LLM)"),
        ("ext_finbert_only", "rule", "rule (FinBERT only, ablation)"),
    ),
    out_name: str = "equity_curves",
) -> Path:
    """Render the headline equity-curve figure (one subplot per horizon).

    Each subplot overlays the listed ``strategies`` against an SPY
    reference line at 1.0 (excess return baseline). Saves a single PNG
    under ``data/cache/plots/<out_name>.png``.
    """
    ensure_dirs()
    n = len(horizons)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 4.6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--",
                   alpha=0.55, label="SPY (zero excess)")
        for extraction, model_name, display in strategies:
            try:
                df = _load_equity(extraction, model_name, horizon)
            except RuntimeError:
                continue
            color = _palette(f"{extraction}__{model_name}")
            _draw_one(ax, df, label=f"{display}", color=color,
                      show_markers=(model_name == "rule"))
        ax.set_title(f"Equity curve, +{horizon} excess return", fontsize=12)
        ax.set_xlabel("call date (chronological)")
        ax.set_ylabel("cumulative excess equity (start = 1.0)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    fig.suptitle(
        "Out-of-sample equity (excess vs SPY) — test set, "
        "per-ticker first 5 calls held out as train",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = plot_path(out_name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_raw_vs_spy(
    extraction: str = "ext_finbert_llm",
    model_name: str = "rule",
    horizon: str = "21d",
    out_name: str = "raw_vs_spy",
) -> Path:
    """Compounded raw return of the strategy vs an SPY buy-and-hold.

    PDF §4 asks "compare to a buy-and-hold of SPY over the same window".
    Equity files only carry excess returns, so we recompute raw equity
    from the predictions parquet (which carries ``raw_h``) and overlay
    SPY's compounded close-to-close return between the test span's first
    entry date and the last exit date as the buy-and-hold reference.
    """
    preds_p = preds_path(extraction, model_name, horizon)
    if not preds_p.is_file():
        error(f"preds parquet missing: {preds_p}")
    preds = pd.read_parquet(preds_p).copy()
    preds["call_date"] = pd.to_datetime(preds["call_date"])
    preds = preds.sort_values("call_date").reset_index(drop=True)
    raw_col = f"raw_{horizon}"
    if raw_col not in preds.columns:
        error(f"preds missing {raw_col!r}")

    direction = preds.get("direction", pd.Series(np.ones(len(preds), dtype=int))).astype(int)
    traded = preds.get("traded", (direction != 0).astype(int)).astype(int)
    bet = direction * preds[raw_col].astype(float)
    bet = np.where(traded == 1, bet, 0.0)
    raw_equity = (1.0 + pd.Series(bet)).cumprod().to_numpy()

    spy = load_prices(BENCHMARK_TICKER).copy()
    spy["Date"] = pd.to_datetime(spy["Date"])
    first = preds["call_date"].min()
    last = preds["call_date"].max()
    spy_window = spy[(spy["Date"] >= first) & (spy["Date"] <= last)].reset_index(drop=True)
    if spy_window.empty:
        error("spy window empty for raw_vs_spy plot")
    spy_idx = pd.to_datetime(preds["call_date"])
    spy_lookup = spy_window.set_index("Date")["Close"]
    spy_at_call = spy_lookup.reindex(spy_idx, method="ffill").to_numpy()
    spy_equity = spy_at_call / spy_at_call[0]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(preds["call_date"], raw_equity,
            color="#1f77b4", linewidth=1.8,
            label=f"Strategy (raw, {extraction.replace('ext_','')}/{model_name} +{horizon})")
    ax.plot(preds["call_date"], spy_equity,
            color="black", linewidth=1.4, linestyle="--",
            label="SPY buy-and-hold")
    if "traded" in preds.columns:
        td = preds[preds["traded"].astype(int) == 1]
        ax.scatter(td["call_date"], raw_equity[traded == 1],
                   s=20, color="#1f77b4", alpha=0.6, zorder=3)
    ax.set_title(f"Strategy raw equity vs SPY buy-and-hold ({horizon} horizon)", fontsize=12)
    ax.set_xlabel("call date")
    ax.set_ylabel("cumulative wealth (start = 1.0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.tight_layout()
    out = plot_path(out_name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_llm_agreement(
    agreement_df: pd.DataFrame,
    out_name: str = "llm_agreement",
) -> Path:
    """Render the per-call gemma3:4b vs llama3.1:8b guidance agreement bar chart.

    Expects ``agreement_df`` with columns
    ``[ticker, n_calls, n_agree, agree_rate, n_disagree]`` per ticker.
    """
    df = agreement_df.sort_values("agree_rate", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.barh(df["ticker"], df["agree_rate"],
                   color="#4c72b0", edgecolor="black", linewidth=0.4)
    overall_rate = float(agreement_df["n_agree"].sum() / max(1, agreement_df["n_calls"].sum()))
    ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1.0,
               label=f"corpus avg = {overall_rate:.2f}")
    for b, n_ag, n_ca in zip(bars, df["n_agree"], df["n_calls"]):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{int(n_ag)}/{int(n_ca)}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("guidance agreement rate (gemma3:4b vs llama3.1:8b)")
    ax.set_title("Per-ticker LLM agreement on call-level guidance label",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = plot_path(out_name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_sectional(
    cs_equity: pd.DataFrame,
    out_name: str = "cross_sectional",
) -> Path:
    """Bar / line of cumulative cross-sectional long-short PnL by quarter.

    ``cs_equity`` columns: ``[quarter, n_long, n_short, ret_long_avg,
    ret_short_avg, ret_ls, equity]`` (one row per quarter).
    """
    df = cs_equity.copy()
    df["quarter"] = df["quarter"].astype(str)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.0),
                                   gridspec_kw={"height_ratios": [1.0, 1.5]},
                                   sharex=True)

    width = 0.4
    x = np.arange(len(df))
    ax1.bar(x - width / 2, df["ret_long_avg"], width=width,
            color="#2ca02c", label="long basket avg excess")
    ax1.bar(x + width / 2, -df["ret_short_avg"], width=width,
            color="#d62728", label="short basket avg excess (sign-flipped for chart)")
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_ylabel("avg excess (+5d)")
    ax1.set_title("Cross-sectional long-short on top-3 / bottom-3 of each quarter "
                  "(score = sentiment_call + 0.5·guidance_ordinal)", fontsize=11)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    ax2.plot(x, df["equity"], marker="o", color="#1f77b4",
             linewidth=1.8, label="cross-sectional L/S equity")
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                label="SPY-neutral baseline (1.0)")
    ax2.set_ylabel("cumulative L/S wealth")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["quarter"], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("quarter")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    out = plot_path(out_name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ic_grid(
    out_name: str = "ic_grid",
) -> Path:
    """Heatmap of Spearman IC across (extraction, model, horizon) cells.

    Reads ``backtest_summary_path()`` directly so it stays in sync with
    the latest run.
    """
    summary_p = backtest_summary_path()
    if not summary_p.is_file():
        error(f"summary parquet missing: {summary_p}")
    df = pd.read_parquet(summary_p)
    df["cell"] = df["extraction"] + " | " + df["model"]
    pivot = df.pivot_table(index="cell", columns="horizon",
                           values="IC_spearman", aggfunc="first")
    pivot = pivot.reindex(sorted(pivot.index))

    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(pivot) + 1.8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"+{h}" for h in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.15 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label="Spearman IC (out-of-sample test set)")
    ax.set_title("Information coefficient by (extraction, model, horizon)",
                 fontsize=11)
    fig.tight_layout()
    out = plot_path(out_name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(
    horizons: Iterable[str] = ("5d", "21d"),
    cs_csv: Path | None = None,
    llm_agreement_csv: Path | None = None,
) -> dict[str, Path]:
    """Render every figure used by ``report.md``; returns ``{name: path}``.

    Convenience wrapper for the notebook so a single cell can refresh
    every figure after a backtest re-run. ``cs_csv`` and
    ``llm_agreement_csv`` paths are resolved lazily so the equity
    curves still render even if the analysis stage hasn't been run.
    """
    out: dict[str, Path] = {}
    out["equity_curves"] = plot_equity_curves(horizons=tuple(horizons))
    out["raw_vs_spy"] = plot_raw_vs_spy()
    out["ic_grid"] = plot_ic_grid()
    if llm_agreement_csv is None:
        llm_agreement_csv = analysis_path("llm_agreement_per_ticker.csv")
    if Path(llm_agreement_csv).is_file():
        out["llm_agreement"] = plot_llm_agreement(
            pd.read_csv(llm_agreement_csv)
        )
    if cs_csv is None:
        cs_csv = analysis_path("cross_sectional_quarterly.csv")
    if Path(cs_csv).is_file():
        out["cross_sectional"] = plot_cross_sectional(
            pd.read_csv(cs_csv)
        )
    return out
