#!/usr/bin/env python3
"""One-command pipeline: Phase 1 -> 2 -> 3 -> 4 -> 5.

Usage:
  python run_all.py                           # run everything
  python run_all.py --dry-run                 # print what would run
  python run_all.py --from-phase 2            # skip Phase 1
  python run_all.py --stop-at-phase 3         # stop after Phase 3
  python run_all.py --tier enhanced           # only enhanced tier
  python run_all.py --force                   # accepted for compatibility
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from data.config import UNIVERSE_CACHE_DIR

ARTIFACTS = {
    "1.1_signals": ROOT / "data/cache/signals.parquet",
    "1.2_universes": ROOT / "data/cache/universes/sp500_pit.parquet",
    "1.3_prices": ROOT / "data/cache/prices/_manifest.json",
    "1.4_shares": ROOT / "data/cache/shares/_manifest.json",
    "2_enhanced": ROOT / "results/features_enhanced.parquet",
    "2_stretch": ROOT / "results/features_stretch.parquet",
    "3_enhanced": ROOT / "results/audit/lookahead_checklist_onepager.md",
    "3_stretch": ROOT / "results/audit/lookahead_checklist_onepager.md",
    "4_tune_enhanced": ROOT / "results/hparams/enhanced/h5d/frozen_hparams_ridge.json",
    "4_tune_stretch": ROOT / "results/hparams/stretch/h5d/frozen_hparams_ridge.json",
    "4_enhanced": ROOT / "results/audit/fit_audit_log_ridge_enhanced.jsonl",
    "4_stretch": ROOT / "results/audit/fit_audit_log_ridge_stretch.jsonl",
    # Phase 5 outputs are now tier x universe x model x cadence.
    # These example paths reference the first tier, first universe, first model, first cadence.
    "5a_ic": ROOT / "results/ic/ic_summary_sp500.parquet",
    "5b_quintile": ROOT / "results/quintile/decile_summary_sp500.parquet",
    "5c_portfolio": ROOT / "results/portfolio/daily_returns_sp500_ridge_enhanced_weekly_5d.parquet",
    "5d_robustness": ROOT / "results/robustness/robustness_subperiod_ic.parquet",
}

TIER_COLORS = {"enhanced": "\033[36m", "stretch": "\033[35m"}
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def _run(cmd: list[str], desc: str, *, dry_run: bool = False) -> bool:
    print(f"{BOLD}  → {desc}{RESET}")
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        print(f"    {YELLOW}DRY RUN{RESET} (not executed)\n")
        return True
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode == 0:
        print(f"    {GREEN}OK{RESET} ({elapsed:.1f}s)\n")
        return True
    else:
        print(f"    \033[31mFAILED (exit {result.returncode}){RESET}\n")
        return False


def _skip(artifact_key: str, force: bool) -> bool:
    # Deliberately run every orchestrated step.  The phase loaders still keep
    # their own resumable network caches, but run_all should not treat a single
    # artifact as proof that a whole tier/model/universe phase is complete.
    return False


def _universe_populated(universe_name: str) -> bool:
    """Check whether a universe PIT parquet exists and has rows.

    Uses pyarrow parquet metadata (O(1), no data loaded). Results are cached
    so repeated checks across tiers and sub-phases read metadata only once.
    """
    pit_path = UNIVERSE_CACHE_DIR / f"{universe_name}_pit.parquet"
    if not pit_path.exists():
        return False
    try:
        import pyarrow.parquet as pq  # already a transitive dependency
        pf = pq.ParquetFile(pit_path)
        return pf.metadata.num_rows > 0
    except Exception:
        return False


def _write_coverage_constrained_artifact(
    universe: str, tier: str, module: str, output_dir: Path
) -> None:
    """Write a JSON sentinel for a coverage-constrained universe (e.g. RU3K)."""
    marker = {
        "universe": universe,
        "tier": tier,
        "module": module,
        "status": "coverage_constrained",
        "note": (
            f"No historical PIT data available for {universe}. "
            "Historical Russell 3000 snapshots were checked on WRDS, "
            "but the Baruch account did not have access to the required "
            "historical constituent/snapshot data."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{module}_{universe}_coverage_constrained.json"
    path.write_text(json.dumps(marker, indent=2))


def phase1(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 1 — Data Loading{RESET}\n{'='*60}")

    for key, cmd, desc in [
        ("1.1_signals", ["python", "-m", "data.load_signals"], "1.1 Signals CSV → Parquet"),
        ("1.2_universes", ["python", "-m", "data.load_universes"], "1.2 PIT universe membership"),
        ("1.3_prices", ["python", "-m", "data.load_prices"], "1.3 Prices & volume (yfinance)"),
        ("1.4_shares", ["python", "-m", "data.load_shares"], "1.4 Shares outstanding (yfinance)"),
    ]:
        if _skip(key, args.force):
            continue
        if not _run(cmd, desc):
            return False
    return True


def phase2(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 2 — Feature Engineering{RESET}\n{'='*60}")

    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        key = f"2_{tier}"
        output = ROOT / f"results/features_{tier}.parquet"
        if _skip(key, args.force):
            continue
        cmd = ["python", "-m", "features.engineer", "--tier", tier, "--output", str(output)]
        if not _run(cmd, f"2.x {color}{tier}{RESET} features → {output.name}"):
            return False
    return True


def phase3(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 3 — Look-Ahead Audit{RESET}\n{'='*60}")

    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        key = f"3_{tier}"
        if _skip(key, args.force):
            continue
        features = ROOT / f"results/features_{tier}.parquet"
        cmd = [
            "python", "-m", "features.audit",
            "--tier", tier,
            "--signals", str(ROOT / "data/cache/signals.parquet"),
            "--full-dates", "15",
        ]
        if not _run(cmd, f"3.x {color}{tier}{RESET} audit ({features.name})"):
            return False
    return True


def phase4(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 4 — Walk-Forward Backtest{RESET}\n{'='*60}")

    model_horizons = ["1", "3", "5", "10", "20"]
    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        features = ROOT / f"results/features_{tier}.parquet"
        tune_universe = next((u for u in args.universes if _universe_populated(u)), "sp500")

        # 4a: hyperparameter tuning.  Hparams are shared across universes, so
        # tune once per tier/horizon on the first populated universe, then
        # reuse those frozen files for every universe backtest below.
        tune_key = f"4_tune_{tier}"
        if not _skip(tune_key, args.force):
            cmd = [
                "python", "-m", "backtest.model",
                "--features", str(features),
                "--tier", tier,
                "--model", "all",
                "--tune-only",
                "--horizons", *model_horizons,
                "--universe", tune_universe,
                "--signal-type", "Total",
            ]
            if not _run(cmd, f"4a {color}{tier}{RESET} hyperparameter tuning ({tune_universe})"):
                return False

        # 4b: walk-forward backtest
        backtest_key = f"4_{tier}"
        if _skip(backtest_key, args.force):
            continue
        for univ in args.universes:
            if not _universe_populated(univ):
                print(f"  {YELLOW}SKIP{RESET} Phase 4 {univ} (empty universe)")
                continue
            cmd = [
                "python", "-m", "backtest.model",
                "--features", str(features),
                "--tier", tier,
                "--model", "all",
                "--horizons", *model_horizons,
                "--universe", univ,
                "--signal-type", "Total",
            ]
            if not _run(cmd, f"4b {color}{tier}{RESET} walk-forward backtest ({univ})"):
                return False
    return True


def phase5(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 5 — Experiment Execution{RESET}\n{'='*60}")

    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        features = ROOT / f"results/features_{tier}.parquet"

        if not features.exists() and not args.dry_run:
            print(f"  {YELLOW}SKIP{RESET} — Features not found: {features}\n")
            continue

        # ---------------------------------------------------------------
        # 5a: Single-feature IC analysis
        # ---------------------------------------------------------------
        ic_key = f"5a_ic_{tier}"
        if not _skip(ic_key, args.force):
            print(f"\n  {color}{tier}{RESET} — 5a Single-feature IC analysis")
            for univ in args.universes:
                if not _universe_populated(univ):
                    if args.skip_empty_universe:
                        print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                        continue
                    _write_coverage_constrained_artifact(univ, tier, "ic", ROOT / "results/ic")
                    print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                    continue
                cmd = [
                    "python", "-m", "backtest.single_feature_ic",
                    "--features", str(features),
                    "--universe", univ,
                    "--output-dir", str(ROOT / "results/ic"),
                ]
                if not _run(cmd, f"5a IC analysis ({color}{tier}{RESET}, {univ})", dry_run=args.dry_run):
                    return False

        # ---------------------------------------------------------------
        # 5b: Quintile / decile portfolios
        # ---------------------------------------------------------------
        quintile_key = f"5b_quintile_{tier}"
        if not _skip(quintile_key, args.force):
            print(f"\n  {color}{tier}{RESET} — 5b Quintile/decile portfolios")
            for univ in args.universes:
                if not _universe_populated(univ):
                    if args.skip_empty_universe:
                        print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                        continue
                    _write_coverage_constrained_artifact(univ, tier, "quintile", ROOT / "results/quintile")
                    print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                    continue
                cmd = [
                    "python", "-m", "backtest.quintile",
                    "--features", str(features),
                    "--universe", univ,
                    "--output-dir", str(ROOT / "results/quintile"),
                ]
                if not _run(cmd, f"5b Quintile/decile portfolios ({color}{tier}{RESET}, {univ})", dry_run=args.dry_run):
                    return False

        # ---------------------------------------------------------------
        # 5c: Portfolio simulation
        # ---------------------------------------------------------------
        # Auto-detect available OOS prediction files for this tier across
        # all models and horizons.
        pred_glob = sorted((ROOT / "results/audit").glob(f"oos_pred_*_{tier}*_h*.parquet"))
        if not pred_glob:
            print(f"\n  {color}{tier}{RESET} — 5c Portfolio simulation")
            print(f"    {YELLOW}SKIP{RESET} — no Phase 4 OOS predictions found for {tier}; run Phase 4 first\n")
        else:
            print(f"\n  {color}{tier}{RESET} — 5c Portfolio simulation")
            for pred_path in pred_glob:
                # Filenames:
                # - old: oos_pred_{model}_{tier}_h{horizon}d.parquet
                # - new: oos_pred_{model}_{tier}_{universe}_{signal}_h{horizon}d.parquet
                stem = pred_path.stem
                match = re.match(
                    rf"^oos_pred_(?P<model>.+?)_{tier}"
                    r"(?:_(?P<universe>sp500|sp1500|ru3k))?"
                    r"(?:_(?P<signal>[a-z]+))?"
                    r"_h(?P<horizon>\d+)d$",
                    stem,
                )
                if not match:
                    print(f"    {YELLOW}SKIP{RESET} unrecognized OOS filename: {pred_path.name}")
                    continue
                model_name = match.group("model")
                pred_universe = match.group("universe")
                pred_horizon = match.group("horizon")
                tag = f"{model_name}_{tier}_predh{pred_horizon}d"
                for univ in args.universes:
                    if pred_universe and univ != pred_universe:
                        continue
                    if not _universe_populated(univ):
                        if args.skip_empty_universe:
                            continue
                        _write_coverage_constrained_artifact(univ, tier, "portfolio", ROOT / "results/portfolio")
                        print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — {model_name}, wrote sentinel")
                        continue
                    for cadence, lookback in [("daily", 1), ("weekly", 5), ("monthly", 21)]:
                        cmd = [
                            "python", "-m", "backtest.portfolio",
                            "--signals", str(pred_path),
                            "--features", str(features),
                            "--universe", univ,
                            "--cadence", cadence,
                            "--lookback", str(lookback),
                            "--tag", tag,
                            "--signal-type", "Total",
                            "--date-col", "availability_date",
                            "--output-dir", str(ROOT / "results/portfolio"),
                        ]
                        if not _run(
                            cmd,
                            f"5c Portfolio sim ({color}{tier}{RESET}, {tag}, {univ}, {cadence})",
                            dry_run=args.dry_run,
                        ):
                            return False

        # ---------------------------------------------------------------
        # 5d: Robustness checks
        # ---------------------------------------------------------------
        robustness_key = f"5d_robustness_{tier}"
        if not _skip(robustness_key, args.force):
            print(f"\n  {color}{tier}{RESET} — 5d Robustness checks")
            for univ in args.universes:
                if not _universe_populated(univ):
                    if args.skip_empty_universe:
                        print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                        continue
                    _write_coverage_constrained_artifact(univ, tier, "robustness", ROOT / "results/robustness")
                    print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                    continue
                cmd = [
                    "python", "-m", "backtest.robustness",
                    "--features", str(features),
                    "--universe", univ,
                    "--output-dir", str(ROOT / "results/robustness"),
                ]
                if not _run(cmd, f"5d Robustness checks ({color}{tier}{RESET}, {univ})", dry_run=args.dry_run):
                    return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1–5 end-to-end")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated; run_all now runs every selected phase by default",
    )
    parser.add_argument("--from-phase", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--stop-at-phase", type=int, choices=[1, 2, 3, 4, 5], default=5)
    parser.add_argument("--tier", choices=["enhanced", "stretch", "both"], default="enhanced")
    parser.add_argument(
        "--universes", nargs="+", default=["sp500", "sp1500", "ru3k"],
        help="Universes for Phase 5 (default: sp500 sp1500 ru3k)",
    )
    parser.add_argument(
        "--skip-empty-universe",
        action="store_true",
        help="Skip empty universes (e.g., RU3K with no PIT data) instead of writing"
        " coverage-constrained sentinel artifacts",
    )
    args = parser.parse_args()

    args.tiers = ["enhanced", "stretch"] if args.tier == "both" else [args.tier]

    if args.from_phase > args.stop_at_phase:
        print("Error: --from-phase > --stop-at-phase", file=sys.stderr)
        sys.exit(2)

    phases = {
        1: ("Phase 1 — Data Loading", phase1),
        2: ("Phase 2 — Feature Engineering", phase2),
        3: ("Phase 3 — Look-Ahead Audit", phase3),
        4: ("Phase 4 — Walk-Forward Backtest", phase4),
        5: ("Phase 5 — Experiment Execution", phase5),
    }

    print(f"{BOLD}Pipeline: Phase {args.from_phase} → Phase {args.stop_at_phase}{RESET}")
    print(f"Tier(s): {', '.join(args.tiers)}")
    if args.dry_run:
        print(f"{YELLOW}DRY RUN — no commands executed{RESET}")

    t_start = time.perf_counter()

    for ph in range(args.from_phase, args.stop_at_phase + 1):
        label, fn = phases[ph]
        if args.dry_run:
            print(f"\n{BOLD}{label}{RESET}")
            if ph == 5:
                fn(args)  # Phase 5 prints expanded sub-tasks in dry-run mode
            continue
        ok = fn(args)
        if not ok:
            print(f"\n\033[31mPipeline stopped at Phase {ph}{RESET}")
            sys.exit(1)

    elapsed = time.perf_counter() - t_start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n{GREEN}{BOLD}Pipeline complete{RESET} ({mins}m {secs}s)")


if __name__ == "__main__":
    main()
