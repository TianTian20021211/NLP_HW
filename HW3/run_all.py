#!/usr/bin/env python3
"""One-command pipeline: Phase 1 -> ... -> Phase 8.

Usage:
  python run_all.py                           # run everything
  python run_all.py --dry-run                 # print what would run
  python run_all.py --from-phase 2            # skip Phase 1
  python run_all.py --stop-at-phase 7         # stop after Phase 7
  python run_all.py --tier enhanced           # only enhanced tier
  python run_all.py --force                   # accepted for compatibility
  python run_all.py --from-phase 7 --stop-at-phase 8  # charts + PDF only
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parent

from data.cache_utils import validate_cache_manifest
from data.config import (
    AUDIT_DIR,
    CACHE_MANIFEST_DIR,
    PRICE_CACHE_DIR,
    REPORT_PDF,
    SHARES_CACHE_DIR,
    UNIVERSE_CACHE_DIR,
    set_global_seed,
    utc_now_iso,
)

ARTIFACTS = {
    "1.1_signals": ROOT / "data/cache/signals.parquet",
    "1.2_universes": [
        ROOT / "data/cache/universes/sp500_pit.parquet",
        ROOT / "data/cache/universes/sp1500_pit.parquet",
        ROOT / "data/cache/universes/ru3k_pit.parquet",
    ],
    "1.3_prices": ROOT / "data/cache/prices/_manifest.json",
    "1.4_shares": ROOT / "data/cache/shares/_manifest.json",
    "2_enhanced": ROOT / "results/features_enhanced.parquet",
    "2_stretch": ROOT / "results/features_stretch.parquet",
    "3_enhanced": ROOT / "results/audit/feature_parity_summary_enhanced.json",
    "3_stretch": ROOT / "results/audit/feature_parity_summary_stretch.json",
    "4_tune": ROOT / "results/hparams/enhanced/h5d/frozen_hparams_ridge.json",
    "4_enhanced": ROOT / "results/audit/fold_manifest_ridge_enhanced_sp500_total.parquet",
    "4_stretch": ROOT / "results/audit/fold_manifest_ridge_stretch_sp500_total.parquet",
    "5a_ic": ROOT / "results/ic/ic_summary_sp500.parquet",
    "5b_quintile": ROOT / "results/quintile/decile_summary_sp500.parquet",
    "5c_portfolio": ROOT / "results/portfolio/daily_returns_sp500_ridge_enhanced_predh5d_weekly_5d.parquet",
    "5d_robustness": ROOT / "results/robustness/robustness_subperiod_ic.parquet",
}

TIER_COLORS = {"enhanced": "\033[36m", "stretch": "\033[35m"}
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# Phase 5 now reads column-pruned feature slices, so the default fan-out can
# keep CPU cores busy without repeating full stretch feature-frame loads.
PHASE5_DEFAULT_OUTER_WORKERS = 4
PHASE5_DEFAULT_INNER_WORKERS = 6
PHASE5_DEFAULT_ROBUSTNESS_WORKERS = 4


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


def _artifact_exists(path_or_paths) -> bool:
    if path_or_paths is None:
        return False
    if isinstance(path_or_paths, list):
        return all(Path(p).exists() for p in path_or_paths)
    return Path(path_or_paths).exists()


def _skip(artifact_key: str, force: bool) -> bool:
    if force:
        return False
    path = ARTIFACTS.get(artifact_key)
    if path is None:
        return False
    if not _artifact_exists(path):
        return False
    # Phase 1: simple existence check (price/shares add coverage check in phase1())
    if artifact_key.startswith("1."):
        return True
    # Phase 2+: content-addressed — must have a valid cache manifest
    manifest_path = CACHE_MANIFEST_DIR / f"{artifact_key}.json"
    return validate_cache_manifest(manifest_path)


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


def _manifest_coverage_sufficient(
    manifest_path: Path, label: str, min_success: int = 400
) -> bool:
    """Return True if manifest has at least ``min_success`` entries with status 'success'.

    When coverage is too low, prints a warning and returns False so the caller
    can fall through to re-run the loader.
    """
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        n_success = sum(1 for v in manifest.get("tickers", {}).values() if v.get("status") == "success")
        if n_success >= min_success:
            return True
        print(
            f"  {YELLOW}WARN{RESET} {label} manifest: {n_success} of {len(manifest)} tickers "
            f"successful (expected >= {min_success}), re-running"
        )
        return False
    except Exception:
        return False


def _build_signals_cmd(args) -> list[str]:
    cmd = ["python", "-m", "data.load_signals"]
    if args.signals_zip is not None:
        cmd.extend(["--zip", str(args.signals_zip)])
    return cmd


def _run_parallel(jobs: list[tuple[str, list[str]]], max_workers: int = 3) -> bool:
    """Run multiple subprocess commands sequentially.

    Each *job* is ``(description, cmd_list)``.  All must succeed — the first
    failure stops remaining jobs and returns ``False``.

    Runs sequentially because each subprocess already uses internal
    ProcessPoolExecutor parallelism — nesting would cause CPU oversubscription
    and memory-pressure spikes.
    """
    if not jobs:
        return True

    for desc, cmd in jobs:
        if not _run(cmd, desc):
            return False
    return True


def phase1(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 1 — Data Loading{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would run: signals CSV -> Parquet, PIT universes, prices, shares")
        return True

    for key, cmd, desc in [
        ("1.1_signals", _build_signals_cmd(args), "1.1 Signals CSV → Parquet"),
        ("1.2_universes", ["python", "-m", "data.load_universes"], "1.2 PIT universe membership"),
        ("1.3_prices", ["python", "-m", "data.load_prices"], "1.3 Prices & volume (yfinance)"),
        ("1.4_shares", ["python", "-m", "data.load_shares"], "1.4 Shares outstanding (yfinance)"),
    ]:
        skip = _skip(key, args.force)
        if skip:
            # Price/shares loaders: verify manifest coverage before skipping
            if key == "1.3_prices" and not _manifest_coverage_sufficient(
                PRICE_CACHE_DIR / "_manifest.json", "prices", 400
            ):
                skip = False
            elif key == "1.4_shares" and not _manifest_coverage_sufficient(
                SHARES_CACHE_DIR / "_manifest.json", "shares", 400
            ):
                skip = False
        if skip:
            continue
        if not _run(cmd, desc):
            return False
    return True


def phase2(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 2 — Feature Engineering{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would build features for tiers: {args.tiers}")
        return True

    enhanced_output = ROOT / "results/features_enhanced.parquet"
    built_enhanced = False
    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        output = ROOT / f"results/features_{tier}.parquet"

        if _skip(f"2_{tier}", args.force):
            print(f"    {GREEN}SKIP{RESET} (cached) 2.x {color}{tier}{RESET} features")
            if tier == "enhanced":
                built_enhanced = True
            continue

        if tier == "stretch":
            if not built_enhanced and not _artifact_exists(enhanced_output):
                cmd = [
                    "python", "-m", "features.engineer",
                    "--tier", "enhanced",
                    "--output", str(enhanced_output),
                ]
                desc = (
                    f"2.x {TIER_COLORS.get('enhanced', '')}enhanced{RESET} "
                    f"prerequisite → {enhanced_output.name}"
                )
                if not _run(cmd, desc):
                    return False
            built_enhanced = True

            cmd = [
                "python", "-m", "features.engineer",
                "--tier", "stretch",
                "--signals", str(ROOT / "data/cache/signals.parquet"),
                "--enhanced-input", str(enhanced_output),
                "--output", str(output),
            ]
        else:
            cmd = ["python", "-m", "features.engineer", "--tier", tier, "--output", str(output)]

        if not _run(cmd, f"2.x {color}{tier}{RESET} features → {output.name}"):
            return False
        if tier == "enhanced":
            built_enhanced = True
    return True


def phase3(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 3 — Look-Ahead Audit{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would run look-ahead audit for tiers: {args.tiers}")
        return True

    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        features = ROOT / f"results/features_{tier}.parquet"

        if _skip(f"3_{tier}", args.force):
            print(f"    {GREEN}SKIP{RESET} (cached) 3.x {color}{tier}{RESET} audit")
            continue

        cmd = [
            "python", "-m", "features.audit",
            "--tier", tier,
            "--signals", str(ROOT / "data/cache/signals.parquet"),
            "--full-dates", "15",
        ]
        if not _run(cmd, f"3.x {color}{tier}{RESET} audit ({features.name})"):
            return False
    return True


def _phase4_manifests_valid(tier: str, universe: str) -> bool:
    """Return True if all 15 expected Phase 4 model manifests exist and are valid."""
    models = ["ridge", "lightgbm", "xgboost"]
    horizons = ["1", "3", "5", "10", "20"]
    for model in models:
        for h in horizons:
            manifest_path = CACHE_MANIFEST_DIR / f"4_{model}_{tier}_{universe}_total_h{h}d.json"
            if not validate_cache_manifest(manifest_path):
                return False
    return True


def phase4(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 4 — Walk-Forward Backtest{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would tune: ridge, lightgbm, xgboost")
        print(f"  [DRY RUN]   horizons: 1, 3, 5, 10, 20")
        print(f"  [DRY RUN]   tiers: {args.tiers}")
        print(f"  [DRY RUN]   universes: {args.universes}")
        print(f"  [DRY RUN]   tuning sample: 2010-01 to 2019-12")
        print(f"  [DRY RUN]   walk-forward: 2020Q1 to 2026Q2")
        return True

    model_horizons = ["1", "3", "5", "10", "20"]
    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        features = ROOT / f"results/features_{tier}.parquet"
        tune_universe = next((u for u in args.universes if _universe_populated(u)), "sp500")

        if tune_universe == "sp500" and not _universe_populated("sp500"):
            print(f"  {YELLOW}WARN{RESET} No universe has PIT data; tuning fallback to 'sp500' will likely fail")

        # 4a: hyperparameter tuning.  Hparams are shared across universes, so
        # tune once per tier/horizon on the first populated universe, then
        # reuse those frozen files for every universe backtest below.
        hparams_dir = ROOT / "results/hparams" / tier
        if not args.force and hparams_dir.exists() and any(hparams_dir.rglob("*.json")):
            print(f"    {GREEN}SKIP{RESET} (cached) 4a {color}{tier}{RESET} hyperparameter tuning")
        else:
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
        for univ in args.universes:
            if not _universe_populated(univ):
                sentinel = AUDIT_DIR / f"tuning_{univ}_coverage_constrained.json"
                sentinel.write_text(json.dumps({
                    "universe": univ,
                    "reason": "no PIT members available for tuning range",
                    "created_at": utc_now_iso(),
                }))
                print(f"  {YELLOW}SKIP{RESET} Phase 4 {univ} (empty universe, sentinel: {sentinel.name})")
                continue
            if not args.force and _phase4_manifests_valid(tier, univ):
                print(f"    {GREEN}SKIP{RESET} (cached) 4b {color}{tier}{RESET} walk-forward ({univ})")
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
        print(f"\n  {color}{tier}{RESET} — 5a Single-feature IC analysis")
        jobs_ic = []
        for univ in args.universes:
            if not _universe_populated(univ):
                if args.skip_empty_universe:
                    print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                    continue
                _write_coverage_constrained_artifact(univ, tier, "ic", ROOT / "results/ic")
                print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                continue
            if not args.force and validate_cache_manifest(CACHE_MANIFEST_DIR / f"5a_{tier}_{univ}.json"):
                print(f"    {GREEN}SKIP{RESET} (cached) 5a IC ({color}{tier}{RESET}, {univ})")
                continue
            jobs_ic.append((
                f"5a IC ({color}{tier}{RESET}, {univ})",
                ["python", "-m", "backtest.single_feature_ic",
                 "--features", str(features), "--universe", univ,
                 "--output-dir", str(ROOT / "results/ic"),
                 "--tier", tier,
                 "--n-jobs", str(args.phase5_inner_workers)],
            ))
        if jobs_ic:
            if args.dry_run:
                for desc, cmd in jobs_ic:
                    print(f"    {YELLOW}QUEUED{RESET} {desc}")
                    print(f"      $ {' '.join(cmd)}")
            elif not _run_parallel(jobs_ic, max_workers=args.max_workers):
                return False

        # ---------------------------------------------------------------
        # 5b: Quintile / decile portfolios
        # ---------------------------------------------------------------
        print(f"\n  {color}{tier}{RESET} — 5b Quintile/decile portfolios")
        jobs_quintile = []
        for univ in args.universes:
            if not _universe_populated(univ):
                if args.skip_empty_universe:
                    print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                    continue
                _write_coverage_constrained_artifact(univ, tier, "quintile", ROOT / "results/quintile")
                print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                continue
            if not args.force and validate_cache_manifest(CACHE_MANIFEST_DIR / f"5b_{tier}_{univ}.json"):
                print(f"    {GREEN}SKIP{RESET} (cached) 5b Quintile ({color}{tier}{RESET}, {univ})")
                continue
            jobs_quintile.append((
                f"5b Quintile ({color}{tier}{RESET}, {univ})",
                ["python", "-m", "backtest.quintile",
                 "--features", str(features), "--universe", univ,
                 "--output-dir", str(ROOT / "results/quintile"),
                 "--tier", tier,
                 "--n-jobs", str(args.phase5_inner_workers)],
            ))
        if jobs_quintile:
            if args.dry_run:
                for desc, cmd in jobs_quintile:
                    print(f"    {YELLOW}QUEUED{RESET} {desc}")
                    print(f"      $ {' '.join(cmd)}")
            elif not _run_parallel(jobs_quintile, max_workers=args.max_workers):
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
            portfolio_jobs: list[dict[str, object]] = []
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
                        manifest_key = f"5c_{univ}_{tag}_{cadence}_{lookback}d"
                        if not args.force and validate_cache_manifest(CACHE_MANIFEST_DIR / f"{manifest_key}.json"):
                            print(f"    {GREEN}SKIP{RESET} (cached) 5c Portfolio ({color}{tier}{RESET}, {univ}, {model_name}, h{pred_horizon}d, {cadence})")
                            continue
                        portfolio_jobs.append(
                            {
                                "signals": str(pred_path),
                                "features": str(features),
                                "universe": univ,
                                "cadence": cadence,
                                "lookback": lookback,
                                "tag": tag,
                                "signal_type": "Total",
                                "date_col": "availability_date",
                                "output_dir": str(ROOT / "results/portfolio"),
                                "audit_dir": str(ROOT / "results/audit"),
                                "price_cache": str(ROOT / "data/cache/prices"),
                                "weekly_day": "monday",
                            }
                        )
            if portfolio_jobs:
                jobs_path = ROOT / f"results/cache/portfolio_jobs_{tier}.json"
                if not args.dry_run:
                    jobs_path.parent.mkdir(parents=True, exist_ok=True)
                    jobs_path.write_text(json.dumps({"jobs": portfolio_jobs}, indent=2))
                cmd = [
                    "python", "-m", "backtest.portfolio_batch",
                    "--jobs-file", str(jobs_path),
                    "--max-workers", str(args.max_workers),
                ]
                if not _run(
                    cmd,
                    f"5c Portfolio batch ({color}{tier}{RESET}, {len(portfolio_jobs)} jobs)",
                    dry_run=args.dry_run,
                ):
                    return False
            else:
                print(f"    {YELLOW}SKIP{RESET} — no portfolio jobs after universe filters\n")

        # ---------------------------------------------------------------
        # 5d: Robustness checks
        # ---------------------------------------------------------------
        print(f"\n  {color}{tier}{RESET} — 5d Robustness checks")
        jobs_robustness = []
        for univ in args.universes:
            if not _universe_populated(univ):
                if args.skip_empty_universe:
                    print(f"    {YELLOW}SKIP{RESET} {univ} (empty universe, --skip-empty-universe)")
                    continue
                _write_coverage_constrained_artifact(univ, tier, "robustness", ROOT / "results/robustness")
                print(f"    {YELLOW}COVERAGE-CONSTRAINED{RESET} {univ} — wrote sentinel")
                continue
            if not args.force and validate_cache_manifest(CACHE_MANIFEST_DIR / f"5d_subperiod_ic_{tier}_{univ}.json"):
                print(f"    {GREEN}SKIP{RESET} (cached) 5d Robustness ({color}{tier}{RESET}, {univ})")
                continue
            jobs_robustness.append((
                f"5d Robustness ({color}{tier}{RESET}, {univ})",
                ["python", "-m", "backtest.robustness",
                 "--features", str(features), "--universe", univ,
                 "--output-dir", str(ROOT / "results/robustness"),
                 "--tier", tier,
                 "--n-jobs", str(args.phase5_robustness_workers)],
            ))
        if jobs_robustness:
            if args.dry_run:
                for desc, cmd in jobs_robustness:
                    print(f"    {YELLOW}QUEUED{RESET} {desc}")
                    print(f"      $ {' '.join(cmd)}")
            elif not _run_parallel(jobs_robustness, max_workers=args.max_workers):
                return False

    return True


def phase6(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 6 — Audit Verification{RESET}\n{'='*60}")

    audit_dir = ROOT / "results/audit"
    required = [
        "lookahead_checklist_onepager.md",
        "fold_manifest.parquet",
        "feature_parity_summary_enhanced.json",
        "feature_parity_summary_stretch.json",
        "trade_execution_log.parquet",
        "trade_execution_violations.parquet",
        "universe_coverage_by_date.csv",
        "sample_size_by_quarter.parquet",
        "marketcap_capacity_coverage.csv",
        "validation_summary.json",
    ]
    if args.dry_run:
        print(f"  [DRY RUN] Would validate {len(required)} audit artifacts, "
              f"fit_audit_log files, and trade execution violations")
        return True

    missing = [f for f in required if not (audit_dir / f).exists()]
    if missing:
        print(f"  \033[31mMISSING:\033[0m {', '.join(missing)}")
        return False

    fit_logs = sorted(audit_dir.glob("fit_audit_log_*.jsonl"))
    if not fit_logs:
        print("  \033[31mMISSING:\033[0m fit_audit_log_*.jsonl")
        return False

    vs_path = audit_dir / "validation_summary.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text())
        summary = vs.get("summary", {})
        checks = vs.get("checks", {})
        print(f"  Validation summary: {summary.get('passed', '?')}/{summary.get('total', '?')} passed")
        failed_checks = []
        pending_checks = []
        for name, info in checks.items():
            status = info.get("status", "?")
            symbol = {"pass": GREEN + "✓", "pending": YELLOW + "?", "fail": "\033[31m✗"}.get(status, "?")
            print(f"    {symbol}{RESET} {name} ({status})")
            if status == "fail":
                failed_checks.append(name)
            elif status == "pending":
                pending_checks.append(name)
        if failed_checks or pending_checks:
            print(
                f"  \033[31mAUDIT NOT CLEAN:\033[0m "
                f"failed={failed_checks}, pending={pending_checks}"
            )
            return False

    violation_paths = sorted(
        {
            *(audit_dir.glob("trade_execution_violations*.parquet")),
            audit_dir / "trade_execution_violations.parquet",
        }
    )
    nonempty_trade_violations: list[str] = []
    for violations_path in violation_paths:
        if not violations_path.exists():
            continue
        try:
            import pyarrow.parquet as pq
            n_violations = pq.ParquetFile(violations_path).metadata.num_rows
        except Exception as exc:
            print(
                f"  \033[31mTRADE LOG VIOLATION FILE UNREADABLE:\033[0m "
                f"{violations_path.name}: {exc}"
            )
            return False
        if n_violations:
            nonempty_trade_violations.append(
                f"{violations_path.name} ({n_violations} rows)"
            )
    if nonempty_trade_violations:
        print(
            "  \033[31mTRADE LOG VIOLATIONS:\033[0m "
            + ", ".join(nonempty_trade_violations)
        )
        return False

    fit_violation_count = 0
    fit_parse_errors = 0
    for fit_log_path in fit_logs:
        for line_no, line in enumerate(fit_log_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"  \033[31mFIT AUDIT LOG MALFORMED:\033[0m "
                    f"{fit_log_path.name}:{line_no}"
                )
                fit_parse_errors += 1
                continue
            fit_violation_count += len(entry.get("fit_violations") or [])
    if fit_parse_errors:
        return False
    if fit_violation_count:
        print(f"  \033[31mFIT AUDIT VIOLATIONS:\033[0m {fit_violation_count}")
        return False

    # -----------------------------------------------------------------------
    # C1 — Re-generate checklist items 7-9 with Phase 6 results
    # -----------------------------------------------------------------------
    checklist_path = audit_dir / "lookahead_checklist_onepager.md"
    if checklist_path.exists():
        content = checklist_path.read_text()
        item7_status = "PASS" if (audit_dir / "fold_manifest.parquet").exists() else "FAIL"
        item8_status = "PASS" if (fit_violation_count == 0 and fit_parse_errors == 0) else "FAIL"
        item9_status = "PASS" if not nonempty_trade_violations else "FAIL"

        content = content.replace(
            "| 7 | Fold boundary + label purge | PEND |",
            f"| 7 | Fold boundary + label purge | {item7_status} |"
        )
        content = content.replace(
            "| 8 | fit() call-stack monitoring | PEND |",
            f"| 8 | fit() call-stack monitoring | {item8_status} |"
        )
        content = content.replace(
            "| 9 | Trade execution log validation | PEND |",
            f"| 9 | Trade execution log validation | {item9_status} |"
        )

        # Update overall summary line
        n_failed = sum(1 for s in [item7_status, item8_status, item9_status] if s == "FAIL")
        n_pending = sum(1 for s in [item7_status, item8_status, item9_status] if s == "PEND")
        if n_failed:
            overall = "SOME FAILED"
        else:
            overall = "ALL PASSED"
        content = content.replace(
            "Overall: PHASE 3 PASSED; 3 PHASE 4/5 ITEM(S) PENDING",
            f"Overall: {overall}"
        )
        checklist_path.write_text(content)

    # -----------------------------------------------------------------------
    # C2 — Append Phase 4/5 entries to validation_summary.json
    # -----------------------------------------------------------------------
    vs_path = audit_dir / "validation_summary.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text())
        vs["checks"]["fold_boundary_label_purge"] = {
            "status": "pass" if (audit_dir / "fold_manifest.parquet").exists() else "fail",
            "evidence": "fold_manifest.parquet",
        }
        vs["checks"]["fit_callstack_monitoring"] = {
            "status": "pass" if (fit_violation_count == 0 and fit_parse_errors == 0) else "fail",
            "evidence": ", ".join(f.name for f in fit_logs),
        }
        vs["checks"]["trade_execution_log"] = {
            "status": "pass" if not nonempty_trade_violations else "fail",
            "evidence": "trade_execution_violations.parquet",
        }
        # Update summary counts
        passed = sum(1 for c in vs["checks"].values() if c.get("status") == "pass")
        total = len(vs["checks"])
        vs["summary"] = {"passed": passed, "total": total, "failed": total - passed}
        vs_path.write_text(json.dumps(vs, indent=2))

    print(f"  {GREEN}All {len(required)} required audit artifacts present and clean.{RESET}")
    return True


def phase7(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 7 — Charts + PDF Report{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would generate charts for tiers: {args.tiers}, universes: {args.universes}")
        print(f"  [DRY RUN] Would compile PDF report")
        return True

    for tier in args.tiers:
        color = TIER_COLORS.get(tier, "")
        figures_dir = ROOT / "reports/figures"
        if not args.force and figures_dir.exists() and any(figures_dir.glob(f"*{tier}*.png")):
            print(f"    {GREEN}SKIP{RESET} (cached) 7a {color}{tier}{RESET} charts")
            continue
        cmd = [
            "python", "-m", "reports.charts",
            "--tier", tier,
            "--universes", *args.universes,
        ]
        if not _run(cmd, f"7a {color}{tier}{RESET} charts → reports/figures/", dry_run=args.dry_run):
            return False

    pdf_path = ROOT / "reports/final_report.pdf"
    if not args.force and _artifact_exists(pdf_path):
        print(f"    {GREEN}SKIP{RESET} (cached) 7b PDF → reports/final_report.pdf")
    else:
        cmd = ["python", "-m", "reports.pdf"]
        if not _run(cmd, "7b PDF → reports/final_report.pdf", dry_run=args.dry_run):
            return False
    return True


def phase8(args) -> bool:
    print(f"\n{'='*60}\n{BOLD}Phase 8 — Reproducibility Check{RESET}\n{'='*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would verify PDF exists and print reproducibility instructions")
        return True

    pdf_path = REPORT_PDF
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        print(f"  {GREEN}✓{RESET} {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    else:
        print(f"  {YELLOW}WARN{RESET} PDF not found or empty — run Phase 7 first")

    print(f"\n  Reproducible with:")
    print(f"    {BOLD}python run_all.py --tier enhanced{RESET}")
    print(f"  Customize:")
    print(f"    {BOLD}python run_all.py --tier both --universes sp500 sp1500{RESET}")
    return True


def main():
    set_global_seed()

    parser = argparse.ArgumentParser(description="Run Phase 1–5 end-to-end")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all phases even when output artifacts already exist",
    )
    parser.add_argument("--from-phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=1)
    parser.add_argument("--stop-at-phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=8)
    parser.add_argument("--tier", choices=["enhanced", "stretch", "both"], default="enhanced")
    parser.add_argument(
        "--signals-zip", type=Path, default=None,
        help="Path to the signal data ZIP file. "
             "Default: data/Earnings_ATC_until_2026-04-21.csv.zip",
    )
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
    parser.add_argument(
        "--max-workers", type=int, default=PHASE5_DEFAULT_OUTER_WORKERS,
        help=(
            "Max parallel subprocesses for Phase 5 universe/module tasks "
            f"(default: {PHASE5_DEFAULT_OUTER_WORKERS})"
        ),
    )
    parser.add_argument(
        "--phase5-inner-workers", type=int, default=PHASE5_DEFAULT_INNER_WORKERS,
        help=(
            "Workers inside Phase 5 IC/quintile subprocesses "
            f"(default: {PHASE5_DEFAULT_INNER_WORKERS})"
        ),
    )
    parser.add_argument(
        "--phase5-robustness-workers", type=int,
        default=PHASE5_DEFAULT_ROBUSTNESS_WORKERS,
        help=(
            "Section workers inside Phase 5 robustness subprocesses "
            f"(default: {PHASE5_DEFAULT_ROBUSTNESS_WORKERS})"
        ),
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
        6: ("Phase 6 — Audit Verification", phase6),
        7: ("Phase 7 — Charts + PDF Report", phase7),
        8: ("Phase 8 — Reproducibility Check", phase8),
    }

    print(f"{BOLD}Pipeline: Phase {args.from_phase} → Phase {args.stop_at_phase}{RESET}")
    print(f"Tier(s): {', '.join(args.tiers)}")
    if args.dry_run:
        print(f"{YELLOW}DRY RUN — no commands executed{RESET}")

    t_start = time.perf_counter()

    for ph in range(args.from_phase, args.stop_at_phase + 1):
        label, fn = phases[ph]
        if args.dry_run:
            fn(args)  # each phase prints its own header and dry-run details
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
