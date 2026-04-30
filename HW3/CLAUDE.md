# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation rules

- `docs/requirement.md` — everything that must be done. Strictly follow it.
- `ideas/plan.md` — what I plan to do (execution checklist).
- `docs/report.md` — everything that needs to go into the final report PDF (methodology choices, rationale, transparency statements).
- When code changes are made, update `docs/report.md` and `ideas/plan.md` accordingly.

## Project

Backtesting the ProntoNLP earnings-call ATC signal across S&P 500 / S&P 1500 / Russell 3000 at daily / weekly / monthly rebalance cadences. Hard constraint: zero look-ahead bias.

## Run

```bash
python run_all.py                          # Phases 1–5, enhanced tier only
python run_all.py --tier both              # Both tiers
python run_all.py --from-phase 3           # Skip data loading
python run_all.py --dry-run                # Preview without executing
python -m features.engineer --tier enhanced --output results/features_enhanced.parquet
python -m features.audit --tier enhanced --full-dates 15
python -m backtest.model --features results/features_enhanced.parquet --model all
python -m backtest.model --features results/features_enhanced.parquet --tune-only
python -m backtest.single_feature_ic --features results/features_enhanced.parquet --universe sp500
python -m backtest.quintile --features results/features_enhanced.parquet --universe sp500
python -m backtest.portfolio --signals results/audit/oos_pred_ridge_enhanced_h5d.parquet --features results/features_enhanced.parquet --universe sp500 --cadence weekly --lookback 5
python -m backtest.robustness --features results/features_enhanced.parquet --universe sp500
```

## Architecture

- `data/config.py` — single source of truth for SEED (42), all cache/output paths
- `data/load_*.py` — Phase 1: CSV→parquet, PIT universes, yfinance prices, shares
- `features/engineer.py` — Phase 2: `build_features()` — timestamps → row (60) → time-series (16) → PIT percentiles (6) → momentum (3) → optional stretch join (~405 cols)
- `features/audit.py` — Phase 3: streaming-vs-batch regression, 10 assertion classes
- `backtest/splits.py` — Phase 4: walk-forward folds (2020Q1→2026Q2), forward returns (joblib-cached), G11 purge, hparam tuning
- `backtest/model.py` — Phase 4/5: `run_walk_forward()` — Ridge/LightGBM/XGBoost
- `backtest/single_feature_ic.py` — Phase 5.1: Spearman IC with Newey-West t-stats
- `backtest/quintile.py` — Phase 5.2: decile/quintile portfolios
- `backtest/portfolio.py` — Phase 5.4: rebalanced portfolio simulation (cohort-based, 3-day quote tolerance, gap accounting)
- `backtest/robustness.py` — Phase 5.5: subperiod, sector neutralization, mcap buckets, bootstrap, OFAT
- `backtest/universe.py` — shared PIT universe filter with auto-detected tolerance

## Key rules

- Import `from data.config import SEED` and call `set_global_seed()` — never hardcode paths or seeds.
- `availability_date` is the strategy-facing timestamp: `call_entry_date + 2bd` before 2023-07-06, `max(call_entry_date, ingest_entry_date)` after.
- Two feature tiers: **enhanced** (85 cols), **stretch** (enhanced + ~405 AspectTheme cols, in-fold LassoCV selection).
- Forward returns (`forward_return_{h}d`, `target_available_date_{h}d`) are targets, never features.
- All imputation/scaling/selection fit on training fold only. Use `monitor_fit_calls()` context manager from audit.
- Same frozen hparams across all three universes — no per-universe tuning.
- Never hit yfinance mid-experiment; use `data/cache/prices/{TICKER}.parquet`.
- Don't read the 4.5 GB CSV directly; use `data/cache/signals.parquet`.
- `data/cache/` and `results/` are gitignored.
- Sharpe > 2.5 → suspect leakage, run the audit.
