"""Batch runner for Phase 5 portfolio simulations.

Runs multiple ``backtest.portfolio`` jobs.  When ``--no-parallel`` is given (or
only a single job exists) jobs run sequentially in-process with shared price
matrices.  Otherwise jobs are distributed across subprocesses via
``ProcessPoolExecutor``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from data.config import AUDIT_DIR, PRICE_CACHE_DIR, RESULTS_DIR

log = logging.getLogger("backtest.portfolio_batch")


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        jobs = payload.get("jobs")
    else:
        jobs = payload
    if not isinstance(jobs, list):
        raise ValueError("jobs file must contain a list or an object with a 'jobs' list")
    return [dict(job) for job in jobs]


def _run_single_job(job: dict[str, Any]) -> Path:
    """Process a single portfolio job — entry point for subprocess workers."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _log = _logging.getLogger("backtest.portfolio_batch")

    from backtest.portfolio import (
        PortfolioSimulator,
        _normalize_signals,
        _persist_portfolio_results,
        _portfolio_output_suffix,
    )

    signals_path = Path(job["signals"])
    features_path = Path(job["features"]) if job.get("features") else None
    universe = str(job.get("universe", "sp500"))
    price_cache = Path(job.get("price_cache", PRICE_CACHE_DIR))
    output_dir = Path(job.get("output_dir", RESULTS_DIR / "portfolio"))
    audit_dir = Path(job.get("audit_dir", AUDIT_DIR))
    cadence = str(job.get("cadence", "weekly"))
    lookback = int(job.get("lookback", 5))
    weekly_day = str(job.get("weekly_day", "monday"))
    cost_bps = float(job.get("cost_bps", 5.0))
    long_frac = float(job.get("long_frac", 0.2))
    tag = job.get("tag")

    _log.info(
        "Portfolio job: %s %s %sd %s",
        universe, cadence, lookback, tag or signals_path.stem,
    )

    signals = _normalize_signals(
        signals_path=signals_path,
        features_path=features_path,
        signal_type=str(job.get("signal_type", "Total")),
        date_col=str(job.get("date_col", "availability_date")),
    )

    sim = PortfolioSimulator(price_dir=price_cache, universe_name=universe)

    result = sim.run(
        signals=signals,
        cadence=cadence,
        lookback=lookback,
        long_frac=long_frac,
        transaction_cost_bps=cost_bps,
        weekly_day=weekly_day,
    )

    suffix = _portfolio_output_suffix(
        universe=universe,
        cadence=cadence,
        lookback=lookback,
        tag=str(tag) if tag is not None else None,
        signals=signals,
        weekly_day=weekly_day,
    )
    return _persist_portfolio_results(
        result=result,
        output_dir=output_dir,
        audit_dir=audit_dir,
        suffix=suffix,
        cost_bps=cost_bps,
        price_cache_dir=price_cache,
        tag=str(tag) if tag is not None else None,
        signals_path=Path(job["signals"]),
        features_path=Path(job["features"]) if job.get("features") else None,
        signal_type=str(job.get("signal_type", "Total")),
        date_col=str(job.get("date_col", "availability_date")),
    )


def run_jobs(
    jobs: list[dict[str, Any]],
    max_workers: int | None = None,
    parallel: bool = True,
) -> list[Path]:
    """Run portfolio jobs, optionally in parallel subprocesses.

    Parameters
    ----------
    jobs:
        List of job dicts (same format as the JSON jobs file).
    max_workers:
        Max subprocesses for parallel mode. Defaults to a small memory-safe cap.
    parallel:
        Set ``False`` to force sequential in-process execution.
    """
    if not jobs:
        return []

    if not parallel or len(jobs) == 1:
        from backtest.portfolio import (
            PortfolioSimulator,
            _normalize_signals,
            _persist_portfolio_results,
            _portfolio_output_suffix,
        )

        simulators: dict[tuple[str, str], PortfolioSimulator] = {}
        outputs: list[Path] = []

        for idx, job in enumerate(jobs, start=1):
            signals_path = Path(job["signals"])
            features_path = Path(job["features"]) if job.get("features") else None
            universe = str(job.get("universe", "sp500"))
            price_cache = Path(job.get("price_cache", PRICE_CACHE_DIR))
            output_dir = Path(job.get("output_dir", RESULTS_DIR / "portfolio"))
            audit_dir = Path(job.get("audit_dir", AUDIT_DIR))
            cadence = str(job.get("cadence", "weekly"))
            lookback = int(job.get("lookback", 5))
            weekly_day = str(job.get("weekly_day", "monday"))
            cost_bps = float(job.get("cost_bps", 5.0))
            long_frac = float(job.get("long_frac", 0.2))
            tag = job.get("tag")

            log.info(
                "Portfolio batch job %d/%d: %s %s %sd %s",
                idx, len(jobs), universe, cadence, lookback,
                tag or signals_path.stem,
            )

            signals = _normalize_signals(
                signals_path=signals_path,
                features_path=features_path,
                signal_type=str(job.get("signal_type", "Total")),
                date_col=str(job.get("date_col", "availability_date")),
            )

            sim_key = (universe, str(price_cache.resolve()))
            sim = simulators.get(sim_key)
            if sim is None:
                sim = PortfolioSimulator(price_dir=price_cache, universe_name=universe)
                simulators[sim_key] = sim

            result = sim.run(
                signals=signals,
                cadence=cadence,
                lookback=lookback,
                long_frac=long_frac,
                transaction_cost_bps=cost_bps,
                weekly_day=weekly_day,
            )

            suffix = _portfolio_output_suffix(
                universe=universe,
                cadence=cadence,
                lookback=lookback,
                tag=str(tag) if tag is not None else None,
                signals=signals,
                weekly_day=weekly_day,
            )
            outputs.append(
                _persist_portfolio_results(
                    result=result,
                    output_dir=output_dir,
                    audit_dir=audit_dir,
                    suffix=suffix,
                    cost_bps=cost_bps,
                    price_cache_dir=price_cache,
                    tag=str(tag) if tag is not None else None,
                    signals_path=signals_path,
                    features_path=features_path,
                    signal_type=str(job.get("signal_type", "Total")),
                    date_col=str(job.get("date_col", "availability_date")),
                )
            )

        return outputs

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 2, len(jobs))

    log.info(
        "Dispatching %d portfolio jobs across %d workers",
        len(jobs), max_workers,
    )

    outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_run_single_job, job): i for i, job in enumerate(jobs)}
        for f in as_completed(future_to_idx):
            idx = future_to_idx[f]
            try:
                outputs.append(f.result())
            except Exception:
                log.exception("Portfolio job %d failed", idx + 1)
                for future in future_to_idx:
                    future.cancel()
                raise

    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Batch portfolio simulations")
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max subprocesses (default: cpu_count)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force sequential in-process execution",
    )
    args = parser.parse_args()

    outputs = run_jobs(
        _load_jobs(args.jobs_file),
        max_workers=args.max_workers,
        parallel=not args.no_parallel,
    )
    log.info("Portfolio batch complete: %d summary files written", len(outputs))


if __name__ == "__main__":
    main()
