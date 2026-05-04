# Phase 0-5 Debugging Discussion Plan

Date: 2026-04-30

This document records the issues found in the Phase 0-5 code review, evaluates the proposed fixes, and defines the recommended implementation plan to discuss before code changes. The main goal is to keep the project aligned with `ideas/plan.md` and the no-look-ahead requirement.

## Review Summary

The project structure is reasonable and the Phase 2 feature pipeline is much stronger than a typical first pass. The highest-risk remaining problems are not syntax errors; they are methodology mismatches in Phase 4/5, incomplete audit wiring, and cache invalidation. The small Phase 3 audit passed with zero strict feature mismatches, but several later audit items are still pending.

Validation already run:

- `python -m compileall backtest features data run_all.py` passed.
- `python -m features.audit --small-only --full-dates 3` passed with 611 rows compared, zero strict mismatches, and 32 expected cross-sectional momentum mismatches.

## Confirmed Direction After Discussion

- Use full hyperparameter granularity: `tier x model x horizon`.
- Produce explicit RU3K coverage-constrained artifacts. Add a methodology note that historical Russell 3000 snapshots were checked on WRDS, but the Baruch account did not have access to the required historical constituent/snapshot data.
- Replace the previous `call_entry_date` main assumption with a unified operational `availability_date`:
  - calls before the ProntoNLP launch date `2023-07-06`: `availability_date = call_entry_date + 2 business days`;
  - calls on or after `2023-07-06`: `availability_date = max(call_entry_date, ingest_entry_date)`.
- Use this unified `availability_date` everywhere that the strategy needs signal availability: feature history, PIT percentile cutoffs, forward-return entry, model folds, tuning sample construction, portfolio signal dates, and audit checks.
- For one- or two-day quote gaps inside portfolio P&L, use forward-filled prices/returns, but report the percentage of filled observations and filled portfolio weight.
- For longer quote gaps, do not automatically classify the case as delisting solely because the gap is longer than two days. Use subsequent price/history evidence to classify whether the ticker later resumed trading, had a known delisting/merger exit, or remained unavailable. This classification is for audit and execution accounting only; it must not be used to make ex-ante trade-selection decisions.
- Use `joblib.Memory` plus explicit dependency signatures for forward-return cache invalidation.

## 1. Hyperparameter Granularity

### Finding

Frozen hyperparameters are currently written as `frozen_hparams_{model}.json`. This means one file is shared by all horizons and both feature tiers. The plan requires one frozen set per `feature tier x model x horizon`, shared across universes but not across horizons or tiers.

Current risk:

- Enhanced and Stretch tuning can overwrite each other.
- A 5-day tuning run can silently supply parameters for 1d, 3d, 10d, and 20d models.
- The report methodology would claim stricter frozen-parameter discipline than the code actually enforces.

### Your Proposed Fix

"Follow the plan for hyperparameters and try all of them."

### Assessment

This is the right direction. The main cost is runtime: tuning `2 tiers x 3 models x 5 horizons = 30` model groups is expensive, especially for LightGBM/XGBoost. But methodologically it is the cleanest approach and matches the written plan.

### Recommended Plan

- Tune all horizons `{1, 3, 5, 10, 20}` for each `(tier, model)` pair.
- Keep hyperparameters shared across universes.
- Store hparams with explicit tier and horizon keys, for example:
  - `results/hparams/enhanced/h1d/frozen_hparams_ridge.json`
  - `results/hparams/enhanced/h5d/frozen_hparams_lightgbm.json`
  - `results/hparams/stretch/h20d/frozen_hparams_xgboost.json`
- Update `load_frozen_hparams()` to require `tier` and `horizon`.
- Update `run_walk_forward()` so each horizon loads its own frozen parameters.
- Keep a manifest such as `results/hparams/hparams_manifest.json` with tuning sample dates, target horizon, feature tier, model name, and git/source fingerprint.

### Decision

Use the full `2 x 3 x 5 = 30` hparam groups despite runtime. If runtime later becomes impractical, any shortcut must be documented as a deviation, but the implementation target is the full grid.

### Fix Applied (2026-04-30)

Hyperparameter storage has been upgraded from shared ``frozen_hparams_{model}.json`` files to a tier x model x horizon granular structure.

- **``backtest/splits.py``**:
  - Added ``HPARAMS_DIR`` constant (``results/hparams``).
  - ``_write_frozen_hparams()`` now takes ``tier`` and ``horizon`` parameters and writes to ``{HPARAMS_DIR}/{tier}/h{horizon}d/frozen_hparams_{model}.json``.
  - ``load_frozen_hparams()`` now requires ``tier`` and ``horizon`` parameters and loads from the tier/horizon-specific path.
  - ``tune_all_models()`` now accepts ``tier`` and ``horizon`` parameters and passes them through to ``_write_frozen_hparams()``. Output directory defaults to ``HPARAMS_DIR``.
  - Added ``_update_hparams_manifest()`` which appends an entry to ``results/hparams/hparams_manifest.json`` after each hparam file write, including model, tier, horizon, CV IC, git commit hash, and creation timestamp.
  - Added ``write_hparams_manifest()`` public function that scans the entire hparams directory tree and rebuilds the manifest (useful for batch updates).
- **``backtest/model.py``**:
  - ``run_walk_forward()`` now loads frozen hparams inside the horizon loop, calling ``load_frozen_hparams(model, hparams_dir, tier=tier, horizon=horizon)`` separately for each horizon.
  - ``--hparams-dir`` CLI argument defaults to ``None`` (resolved to ``HPARAMS_DIR`` internally). Hparam existence checks and path logging use tier/horizon-specific paths.
  - ``tune_all_models()`` call in ``main()`` now passes ``tier=args.tier`` and ``horizon=tune_h``.
- **``run_all.py``**: Updated artifact paths to ``results/hparams/{tier}/h5d/frozen_hparams_ridge.json``.
- **``docs/report.md``**: Sections 3.2 and 3.8 updated to reflect tier x model x horizon granularity.
- **``ideas/plan.md``**: Phase 4.1 updated to describe the new hparam storage structure.

Verification: ``python -m compileall backtest features data run_all.py`` passes. ``from backtest.splits import load_frozen_hparams, tune_all_models, HPARAMS_DIR`` verifies new signatures with ``tier`` and ``horizon`` parameters. ``python run_all.py --dry-run`` passes.

## 2. Phase 5 Coverage Across Universes, Tiers, and Models

### Finding

Phase 5 does not actually run every requested dimension. IC and quintile analysis use `args.universes`, but portfolio simulation and robustness checks are hardcoded to SP500. Phase 5 also uses only `args.tiers[0]`, so `--tier both` does not run both Enhanced and Stretch experiments.

Current risk:

- SP1500 and RU3K do not receive full Phase 5 treatment.
- Stretch results may be missing from Phase 5 even when `--tier both` is requested.
- Final conclusions could accidentally be based only on SP500 Enhanced.

### Your Proposed Fix

"Use all models."

### Assessment

Using all models is necessary, but not sufficient. The loop must cover all relevant tiers, universes, models, and cadences. Because SP1500/RU3K are coverage-constrained in the current data, the code should still run them when requested but clearly mark low-coverage or empty results instead of silently skipping.

### Recommended Plan

- Update `run_all.py` Phase 5 to loop over all selected tiers.
- For each tier, run IC and quintile for every requested universe.
- For portfolio simulation, loop over every available OOS prediction file for that tier, every requested universe, and all cadences `{daily, weekly, monthly}`.
- For robustness, loop over requested universes, but allow `--skip-empty-universe` or write explicit empty/coverage-constrained artifacts.
- Do not tune or optimize per universe. Universes are evaluation dimensions only.

### Decision

Produce explicit RU3K coverage-constrained artifacts instead of silently skipping it. The report should state that historical Russell 3000 snapshots were checked through WRDS, but the Baruch account did not have access to the required historical snapshot/constituent files. Therefore RU3K is reported as coverage-constrained rather than backfilled with a survivorship-biased current snapshot.

### Fix Applied (2026-04-30)

Phase 5 in ``run_all.py`` has been expanded to loop over all selected tiers, universes, models, and cadences:

- **``run_all.py``**:
  - ``phase5()`` now iterates over every tier in ``args.tiers`` (not just ``args.tiers[0]``), so ``--tier both`` runs both Enhanced and Stretch experiments.
  - For each tier, 5a (IC) and 5b (quintile) now iterate over every universe in ``args.universes`` (default: ``sp500 sp1500 ru3k``).
  - 5c portfolio simulation loops over every available OOS prediction file for that tier (glob ``results/audit/oos_pred_*_{tier}_h*.parquet``), every requested universe, and all three cadences (daily, weekly, monthly).
  - 5d robustness checks now iterate over every requested universe (previously hardcoded to sp500).
  - Added ``_universe_populated()`` helper (cached parquet-metadata check) to detect empty PIT universes.
  - Added ``_write_coverage_constrained_artifact()`` which writes a JSON sentinel ``{module}_{universe}_coverage_constrained.json`` for universes with empty PIT data.
  - Added ``--skip-empty-universe`` flag: when set, empty universes are silently skipped; by default, coverage-constrained sentinels are written.
  - ``_run()`` now accepts a ``dry_run`` keyword argument. In dry-run mode (``python run_all.py --dry-run --tier both``), each sub-task description and command is printed but not executed.
  - The main loop calls ``phase5()`` even in dry-run mode so all sub-tasks are enumerated.

Verification: ``python -m compileall run_all.py`` passes. ``python run_all.py --dry-run --tier both`` shows both enhanced and stretch sub-tasks, all three universes (sp500, sp1500, ru3k as coverage-constrained), and all three portfolio cadences (daily, weekly, monthly).

## 3. Portfolio Cohort Aggregation

### Finding

The current portfolio implementation collapses the lookback window to the latest score per ticker and then builds one rank book. The plan says to build independent stock selections by cohort, net by stock after aggregation, and then scale to fixed gross=200% and net=0%.

Current risk:

- Weekly/monthly portfolios are not true aggregations of event cohorts.
- Multiple events in the lookback window do not contribute independently.
- Cross-cadence comparison is less faithful to the planned capital convention.

### Your Proposed Fix

"Execute according to the plan."

### Assessment

Correct. This is one of the most important Phase 5 fixes. The implementation should be careful to avoid accidental leverage changes when many cohorts overlap.

### Recommended Plan

- Define a cohort as the set of eligible events on one signal date or one event availability date.
- At each rebalance date, collect eligible cohorts in the trailing N trading-day window.
- For each cohort:
  - deduplicate to one signal per ticker inside that cohort if needed;
  - rank by score;
  - create equal-weight long/short raw weights for that cohort.
- Concatenate cohort weights.
- Net by ticker across cohorts.
- Re-center to net zero if needed.
- Scale final weights so gross exposure equals 2.0.
- Persist both:
  - raw cohort weights;
  - final aggregated weights.

### Discussion Point

Need to decide whether a "cohort" means exact `call_entry_date` or exact selected `date_col` (`call_entry_date` for main, `availability_date` for strict robustness). The clean choice is to use the portfolio input `date` column consistently.

### Fix Applied (2026-04-30)

The portfolio weight construction has been redesigned to use cohort-based aggregation instead of collapsing the lookback window to the latest score per ticker.

- **``backtest/portfolio.py``**:
  - Replaced ``_build_weights()`` (single-cohort ranking) with ``_build_cohort_weights()`` that implements the full cohort pipeline:
    - Groups eligible signals in the lookback window by signal date (``date`` column, which is ``availability_date``).
    - For each cohort: drops NaN scores, deduplicates per ticker (keeps highest absolute score), ranks by score, selects top/bottom ``long_frac`` for equal-weight long/short raw weights.
    - Concatenates all cohort raw weights, nets by ticker (sum weights across cohorts), re-centers to net zero (subtract mean), and scales to gross = 2.0.
  - ``PortfolioResult`` gains a ``cohort_weights_history`` field mapping rebalance date to raw cohort weights before cross-cohort aggregation.
  - The ``run()`` rebalance block now calls ``_build_cohort_weights()`` instead of the old collapse-and-rank approach.
  - Entry-price lookups are performed only for tickers with non-zero aggregated weights (not all tickers with signals).
  - Non-tradeable tickers are removed from the aggregated weights and remaining weights are re-scaled to gross = 2.0.
  - ``main()`` persists both ``cohort_weights_{suffix}.parquet`` and ``weights_{suffix}.parquet``.

Verification: ``python -m compileall backtest`` passes. ``python -m backtest.portfolio --signals results/audit/oos_pred_ridge_enhanced_h5d.parquet --features results/features_enhanced.parquet --universe sp500 --cadence weekly`` produces both output files. Gross exposure is exactly 2.0 and net exposure is exactly 0.0 for every rebalance date. Cohort weights file has schema ``[cohort_date, ticker, raw_weight, rebalance_date]``.

## 4. Entry and Exit Quote Handling

### Finding

Portfolio entry currently requires a quote exactly on the rebalance date. If the quote is missing, the trade is skipped immediately. The plan requires `next_valid_close_on_or_after(planned_entry_date)` with up to 5 consecutive trading days of roll-forward before skipping.

Current risk:

- Some valid trades are skipped.
- The execution log does not fully support the planned corporate-action / missing-quote rule.

### Your Proposed Fix

"Follow the plan."

### Assessment

Correct. This should be implemented both for entry and exit, and the trade log must preserve planned vs actual dates.

### Recommended Plan

- Add a helper such as `next_valid_close(ticker, planned_date, max_bdays=3)`.
- Entry:
  - planned entry = rebalance date;
  - actual entry = first valid close on or after planned entry, within 5 business days;
  - if none, mark `skip_no_entry_quote`.
- Exit:
  - planned exit = next rebalance date or target holding date;
  - actual exit = first valid close on or after planned exit, within 5 business days;
  - if none, mark `right_censored_no_exit_quote`.
- Record planned/actual entry and exit dates, prices, and skip reason.
- Keep trades with missing exits out of ordinary return statistics but summarize them separately.

### Discussion Point

For a rebalanced portfolio, exiting usually happens through weight changes at the next rebalance rather than through independent fixed-horizon trades. We should define whether the trade log is a position-accounting audit or a literal per-trade holding-period audit. The portfolio P&L engine should follow daily holdings; the log can still record planned next rebalance as the intended exit checkpoint.

### Fix Applied (2026-04-30)

Entry and exit quote handling has been fully implemented in ``backtest/portfolio.py`` and ``features/audit.py``:

- **``backtest/portfolio.py``**:
  - ``_next_valid_quote()`` scans up to 5 **trading days** forward in the trading calendar (not calendar days), checking the planned date plus offsets of 1 through 5 trading days in the price-derived trading calendar.
  - **Entry**: At each rebalance, every ticker with non-zero aggregated weight is looked up via ``_next_valid_quote()``. ``planned_entry_date`` is the rebalance date. If no valid quote is found within 5 trading days, the ticker's weight is removed, re-scaling remaining weights to gross=2.0, and a trade log entry is recorded with ``skip_reason="skip_no_entry_quote"``.
  - **Exit**: For each entered position, ``planned_exit_date`` is the next rebalance date. ``_next_valid_quote()`` looks up the exit price at that date with the same 5-trading-day tolerance. If no valid exit quote is found, the trade is marked ``right_censored_no_exit_quote``.
  - **Trade log persistence**: ``trade_execution_log_{suffix}.parquet`` and ``trade_execution_log.parquet`` contain ``planned_entry_date``, ``actual_entry_date``, ``entry_price``, ``planned_exit_date``, ``actual_exit_date``, ``exit_price``, and ``skip_reason`` (``None`` for normal trades).
  - **Per-rebalance-period logging**: Each rebalance period creates a separate trade log entry. Positions that persist across rebalances receive a new entry at each rebalance with that period's entry and the next rebalance as planned exit.
  - ``validate_trade_log()`` is called inside ``PortfolioSimulator.run()`` after the simulation loop; violations are persisted to ``trade_execution_violations_{suffix}.parquet``.
  - ``PortfolioResult`` gained a ``trade_log_violations`` field.
- **``features/audit.py``**:
  - ``validate_trade_log()`` checks: (1) ``actual_entry_date >= planned_entry_date``, (2) ``actual_exit_date >= planned_exit_date``, (3) skip_no_entry_quote entries have no entry_price, (4) right_censored_no_exit_quote entries have no exit_price.
  - ``TRADE_LOG_COLUMNS`` now includes the ``weight`` column.
- **Verification**: ``python -m compileall backtest features`` passes. Portfolio smoke test on SP500 weekly, Ridge model, produces ``trade_execution_log.parquet`` with all required columns — 0 date-order violations, 0 skip-reason-contradiction violations. ``skip_reason`` is correctly ``null`` for normal trades.

## 5. Missing Daily Returns and Delisting Treatment

### Finding

Daily P&L currently computes returns and then drops missing ticker returns before summing. That means a held name with a missing next-day quote can disappear from P&L while gross/net exposure still includes the original weight.

Current risk:

- Missing or delisted names can be silently ignored.
- Portfolio returns may be biased upward or downward.
- This conflicts with the plan's instruction not to silently delete missing-exit trades.

### Your Proposed Fix

"Use forward-fill for one- or two-day missing cases. For longer missing periods, treat them as delisting or unavailable data. Report the proportion of both cases."

### Assessment

This is acceptable if the forward-fill window is tightly bounded and explicitly reported. The main risk is that forward-fill can hide a real price move if the missing quote is not merely a data gap. Limiting the rule to one or two missing trading days and reporting affected observations/weights makes it a defensible practical compromise.

### Recommended Plan

- During daily P&L, classify every held ticker:
  - valid next-day return;
  - one-day forward-filled quote gap;
  - two-day forward-filled quote gap;
  - longer quote gap / possible delisting / unavailable data;
  - delisting exit used, if evidence exists.
- For one- or two-day quote gaps:
  - forward-fill the price for return continuity;
  - mark the affected return observation as filled;
  - include it in daily P&L;
  - report counts and portfolio weight affected.
- For gaps longer than two trading days:
  - do not keep forward-filling mechanically;
  - inspect subsequent price/history evidence in the local price cache;
  - use the planned exit date or next rebalance date as the main accounting window;
  - additionally run a supplemental 30-trading-day recovery audit to identify cases that resumed trading later;
  - if a valid quote resumes inside the accounting window or supplemental recovery window, classify as `long_quote_gap_recovered`;
  - if a known delisting/merger exit price is available, classify as `delisting_exit_used`;
  - if no later quote or corporate-action evidence exists, classify as `possible_delisting_or_data_unavailable`;
  - censor the position from headline fully-observable portfolio returns once the gap exceeds two trading days;
  - report recovered-gap returns only in a supplemental table; do not include them in headline Sharpe;
  - report frequency, affected weight, recovery rate, and universe/year breakdown.
- Add daily return/audit columns:
  - `ffill_1d_weight`
  - `ffill_2d_weight`
  - `long_gap_recovered_weight`
  - `possible_delisting_or_unavailable_weight`
  - `n_ffill_1d_positions`
  - `n_ffill_2d_positions`
  - `n_long_gap_recovered_positions`
  - `n_possible_delisting_or_unavailable_positions`
- Add summary percentages:
  - share of position-days filled for one day;
  - share of position-days filled for two days;
  - share of position-days with longer gaps that later recovered;
  - share of position-days classified as possible delisting or unavailable data.

### Decision

Use forward-fill only for one- or two-day quote gaps and report the proportion of those filled observations. For longer gaps, do not use the two-day threshold as an automatic delisting rule. Censor these positions from headline fully-observable portfolio returns once the gap exceeds two trading days. Use subsequent local price/history evidence only for post-trade audit classification: first through the planned exit/next rebalance accounting window, and then through a supplemental 30-trading-day recovery audit. Recovered-gap returns are reported separately and do not enter headline Sharpe.

### Fix Applied (2026-04-30)

The daily P&L computation in ``backtest/portfolio.py`` has been upgraded with bounded forward-fill for quote gaps and a 30-trading-day recovery audit for longer gaps.

- **``backtest/portfolio.py``**:
  - Added ``_audit_long_gap_recovery()`` module-level function that checks whether tickers with long quote gaps (>2 trading days) later resume trading within a 30-trading-day window. Returns a dict mapping ``(gap_date, ticker)`` to a boolean recovery flag.
  - Modified the daily P&L loop in ``PortfolioSimulator.run()`` to classify every held ticker each day:
    - **Normal return**: valid price at ``next_date`` — standard P&L contribution.
    - **ffill_1d**: no quote at ``next_date`` but a valid quote at ``date+2`` — forward-filled (0% return), included in headline P&L.
    - **ffill_2d**: no quote at ``next_date`` or ``date+2``, but a valid quote at ``date+3`` — same logic as ffill_1d.
    - **Long gap (>2 trading days)**: no quote within 5 trading days ahead — censored from headline P&L (0% contribution); recorded for the 30-day recovery audit.
  - After the simulation loop, runs the recovery audit via ``_audit_long_gap_recovery()`` and updates each long-gap row with either ``long_gap_recovered_weight`` (if a quote resumed within 30 trading days) or ``possible_delisting_or_unavailable_weight`` (if not).
  - Eight new columns added to ``daily_returns`` parquet: ``ffill_1d_weight``, ``ffill_2d_weight``, ``long_gap_recovered_weight``, ``possible_delisting_or_unavailable_weight``, ``n_ffill_1d_positions``, ``n_ffill_2d_positions``, ``n_long_gap_recovered_positions``, ``n_possible_delisting_or_unavailable_positions``.
  - ``PortfolioResult`` gained a ``gap_records`` field (DataFrame with per-gap-event details).
  - ``summary()`` now computes four gap summary statistics (share of position-days in each category).
  - ``main()`` persists ``gap_accounting_{suffix}.parquet`` with per-gap-event details (gap date, ticker, weight, recovered flag).
- **``ideas/plan.md``**: Phase 5.4 updated to describe the gap accounting approach, reporting columns, and recovery audit.
- **``docs/report.md``**: Sections 4.5 and 5.3 (G6) updated to document bounded forward-fill and the 30-trading-day recovery audit.

Verification: ``python -m compileall backtest features data`` passes. ``python -m backtest.portfolio --signals results/audit/oos_pred_ridge_enhanced_h5d.parquet --features results/features_enhanced.parquet --universe sp500 --cadence weekly`` produces daily returns with all 8 gap accounting columns (15 total columns including date/pnl/exposure/turnover). Portfolio summary JSON includes four gap position-day share statistics. ``gap_accounting_sp500_ridge_enhanced_weekly_5d.parquet`` is persisted when gap events exist.

## 6. Audit Integration

### Finding

Several audit helpers are defined but not actually wired into Phase 4/5 execution:

- `monitor_fit_calls`
- `assert_fit_callstack`
- `assert_fold_boundaries`
- `validate_trade_log`

The checklist currently marks these as pending.

Current risk:

- The project cannot honestly claim all audit items passed automatically.
- Fit-time leakage checks are weaker than the plan states.

### Your Proposed Fix

"Add these validations."

### Assessment

Correct. This is required before final submission. The existing helper functions are a good start, but they need to be invoked and their outputs persisted.

### Recommended Plan

- Wrap Phase 4 model fitting in `monitor_fit_calls()`.
- After each fold, validate:
  - max training feature date < test start;
  - max training target availability date < test start;
  - fit calls only used training-fold data.
- Persist:
  - `fit_audit_log.jsonl`;
  - `fold_manifest.parquet`;
  - validation summary JSON.
- After each portfolio run, call `validate_trade_log()` and write:
  - `trade_execution_log.parquet`;
  - `trade_execution_violations.parquet`;
  - summary counts by violation type.
- Update `lookahead_checklist_onepager.md` only after these checks run.

### Discussion Point

The current `monitor_fit_calls()` can only infer date ranges if fit inputs carry a date index. Many model inputs are NumPy arrays, so the validation may need explicit metadata passed from the fold rather than relying only on monkey-patching.

### Fix Applied (2026-04-30)

All four audit helpers are now wired into Phase 4/5 execution, and the outputs are persisted to ``results/audit/``:

- **``backtest/model.py``**:
  - ``monitor_fit_calls()`` context manager already wraps the impute/scale -> LassoCV -> model.fit section in ``_run_one_fold()`` (wired in A1 from previous round).
  - **NEW**: ``assert_fold_boundaries()`` is called after each fold, validating ``max(train_feature_date) < test_start`` and ``max(train_target_available_date_h) < test_start``. Violations are appended to the fold's fit violation list and logged.
  - ``write_fit_audit_log()`` persists ``fit_audit_log_{model}_{tier}.jsonl`` with fold/horizon metadata and violation lists.
  - ``write_fold_manifest()`` now includes ``model`` and ``tier`` columns in each row, and per-horizon ``max_train_target_date_h`` columns populated from ``fold.max_train_target_date``.

- **``backtest/splits.py``**:
  - ``write_fold_manifest()`` accepts optional ``model``, ``tier``, ``horizons`` parameters and includes them in the parquet output.

- **``backtest/portfolio.py``**:
  - ``validate_trade_log()`` is called after the simulation loop (wired in A2 from previous round).
  - **NEW**: Violation summary counts by type are logged after validation.
  - **NEW**: ``trade_execution_violations.parquet`` is always written (both suffixed and generic).

- **``results/audit/`` artifacts**:
  - ``fold_manifest.parquet``: 26 folds with train/test boundaries, model/tier metadata, per-horizon target dates.
  - ``fit_audit_log_*.jsonl``: 390 entries across ridge/lightgbm/xgboost enhanced models.
  - ``trade_execution_log.parquet``: planned/actual entry/exit for every trade.
  - ``trade_execution_violations.parquet``: date-ordering and skip-reason consistency violations.
  - ``validation_summary.json``: per-check status (pass/fail/pending), evidence file paths, timestamp.
  - ``lookahead_checklist_onepager.md``: updated - items 7/8/9 marked PASSED with evidence file paths.

- **Verification**: ``python -m compileall backtest features`` passes. Fit-audit logs contain actual entries. Fold manifest parquet has complete metadata with model/tier columns.

## 7. Unified Operational `availability_date`

### Finding

The model code supports `--availability-col availability_date` for fold construction, but forward returns are still always computed from `call_entry_date`. That means the strict availability robustness changes sample eligibility but not the actual tradable entry date.

Current risk:

- Strict robustness is incomplete.
- If `availability_date` is later than `call_entry_date`, the target return can still start too early.
- The previous main-vs-strict split leaves two competing date concepts in the code, making it easier to accidentally mix feature availability and trade entry timing.

### Your Proposed Fix

"For calls before 2023-07-06, use `call_entry_date + 2 days` as the available date, assuming the company/system needs two days to process the call. For calls after that, use the real available date. Use this available date everywhere."

### Assessment

This is a coherent and practical compromise. It avoids the unrealistic result where all pre-launch historical signals become unavailable until the 2023 ProntoNLP backfill, while still avoiding same-day look-ahead. It should be described as an operational availability assumption: if the NLP system had existed historically, signals would be available two business days after the tradable call-entry date.

Important implementation detail: interpret "+2 days" as **two business/trading days**, not two calendar days. Calendar days can land on weekends or holidays and would create inconsistent close-to-close entry timing. The exact exchange-holiday roll should still be handled by price-aware execution.

### Recommended Plan

- Preserve raw timestamp fields:
  - `call_entry_date`
  - `ingest_entry_date`
  - optionally `vendor_availability_date = max(call_entry_date, ingest_entry_date)` for transparency.
- Redefine the strategy-facing `availability_date` as:
  - if `call_entry_date < 2023-07-06`: `call_entry_date + 2 business days`;
  - else: `max(call_entry_date, ingest_entry_date)`.
- Use this strategy-facing `availability_date` everywhere:
  - QoQ/time-series feature history;
  - PIT percentile history and cutoffs;
  - forward-return entry date;
  - tuning sample date filter;
  - walk-forward fold generation;
  - label purge;
  - portfolio signal date;
  - rebalance eligibility;
  - audit fixtures.
- Update `compute_forward_returns()` to accept `entry_date_col`, then pass `entry_date_col="availability_date"` in all main runs.
- Update cache keys to include `entry_date_col` and the availability-rule version.
- Add a report caveat: the pre-2023 `+2 business days` availability is an operational simulation assumption, not an observed ProntoNLP vendor timestamp.

### Decision

Use the unified operational `availability_date` as the main date everywhere. Keep raw and vendor availability dates for transparency/audit, but do not use `call_entry_date` as the main tradable date after this fix.

Confirmed details:

- `call_entry_date + 2 days` means two business/trading days, not two calendar days.
- Use `call_entry_date < 2023-07-06` for the pre-launch simulated-processing rule and `call_entry_date >= 2023-07-06` for real ingest-based availability.

### Fix Applied (2026-04-30)

The unified operational `availability_date` rule has been implemented in all relevant components:

- **`features/engineer.py`**: ``compute_timestamps()`` now implements the unified rule (``+2bd`` before 2023-07-06, ``max(call, ingest)`` on/after) using ``np.busday_offset`` for business-day arithmetic.
- **`backtest/splits.py`**: ``compute_forward_returns()`` added ``entry_date_col`` parameter (default ``"availability_date"``). Cache keys include ``entry_date_col`` and an ``AVAILABILITY_RULE_VERSION`` string (``"v1"``). Default ``availability_col`` changed to ``"availability_date"`` in ``generate_folds()``, ``get_tuning_sample()``, and ``tune_all_models()``.
- **`backtest/model.py`**: ``--availability-col`` default changed from ``"call_entry_date"`` to ``"availability_date"``. Forward-return computation passes ``entry_date_col``. Help text updated.
- **`backtest/portfolio.py`**: ``--date-col`` default changed from ``"call_entry_date"`` to ``"availability_date"``.
- **`run_all.py`**: Phase 5 portfolio calls pass ``--date-col availability_date``.
- **All Phase 5 IC/quintile/robustness call sites**: pass ``entry_date_col="availability_date"`` to ``get_forward_returns_cached()``.
- **Documentation**: ``ideas/plan.md`` Phase 2.1 and Phase 4.1 updated. ``docs/report.md`` Section 3.8 and Section 5.6 (G16) updated.

## 8. SP1500 Monthly SP500 Component

### Finding

SP1500 construction maps every daily SP500 row to month-end and deduplicates. This creates a union of all SP500 members that appeared during the month, not necessarily the true month-end SP500 membership.

Current risk:

- A stock removed mid-month can remain in the month-end SP1500 snapshot.
- A stock added mid-month can also appear for the full month-end snapshot. The effect is probably small but methodologically avoidable.

### Your Proposed Fix

"Use the suggested way."

### Assessment

Correct. The monthly SP500 component should use the latest actual SP500 daily snapshot on or before each month-end.

### Recommended Plan

- Build a month-end grid.
- For each month-end date, call the latest SP500 PIT snapshot date `<= month_end`.
- Use only that snapshot's tickers.
- Combine that month-end SP500 snapshot with IJH and IJR month-end components.
- Keep component-level coverage gaps for IJH/IJR.

### Discussion Point

No major disagreement here. This should be a straightforward correctness fix.

### Fix Applied (2026-04-30)

The SP1500 month-end SP500 component construction has been corrected in `data/load_universes.py` `build_sp1500()`:

- **Before**: Every daily SP500 PIT row was mapped to its calendar month-end via `MonthEnd(0)`, then deduplicated. This created a union of all SP500 members that appeared on any trading day during the month, so a stock added mid-month and removed before month-end would still appear in the month-end snapshot.
- **After**: For each month-end date in the grid, the code finds the **latest** SP500 daily snapshot date `<= month_end` and uses **only that snapshot's tickers**. A stock removed before the trading day closest to month-end is correctly excluded.
- **Implementation**: Replaced the per-row `MonthEnd(0)` + `drop_duplicates()` approach with a loop over the month-end grid, sorting SP500 dates, scanning for eligible snapshots, and collecting members from the single latest snapshot per month-end.
- **Verification**: For sampled month-ends (2020-03-31, 2020-06-30, 2020-12-31, 2024-12-31), tickers that appeared during the month but were absent from the latest snapshot are correctly excluded from SP1500 membership. All PASS.
- **Smoke test**: `python -m compileall data` and `python -m data.load_universes --only sp1500` both pass.

## 9. Shares Coverage Denominator

### Finding

Market-cap coverage currently uses the union of all historical members for every year. It does not restrict the denominator to actual PIT members in that year.

Current risk:

- Early-year coverage can be understated because future members are included in the denominator.
- Coverage reports can misstate which periods are quantitative vs qualitative.

### Your Proposed Fix

"Only look at actual members."

### Assessment

Correct. The coverage denominator should be based on PIT membership during the relevant year or date.

### Recommended Plan

- For each `(universe, year)`, compute `members` as the union of PIT members on snapshots within that year.
- If a universe has no snapshots in a year, write an explicit missing/empty row instead of using all historical members.
- Count a ticker as covered only if it has valid shares data overlapping that year.
- Keep the existing 70% floor for quantitative market-cap bucket eligibility.
- For event-level market-cap buckets, continue using event-date PIT membership and strict T-1 price/shares.

### Discussion Point

For monthly universes, yearly members should be the union of monthly snapshots in that calendar year. For SP500 daily, yearly members should be the union of daily snapshots in that calendar year.

### Fix Applied (2026-04-30)

The coverage denominator has been corrected in `data/load_shares.py` `write_coverage()`:

- **Before**: Used the union of all historical PIT members (e.g., 817 tickers for SP500) as the denominator for every year. Early-year coverage was understated because future members were included.
- **After**: For each `(universe, year)`, the denominator is the union of PIT members on snapshots within that calendar year only (521-537 per year for SP500). Years with no PIT snapshots (e.g., all RU3K years) get explicit zero-member rows instead of being silently skipped.
- **Implementation**: Added `_yearly_pit_members()` helper that reads the PIT parquet, extracts the year from each snapshot date, groups by year, and collects unique ticker sets. `write_coverage()` uses these per-year sets instead of the all-historical ticker set.
- **Coverage impact**: Old coverage for SP500 2015 was `491/817 = 60.1%` (below floor); new coverage is `491/534 = 91.9%` (above floor). 2010-2014 remain below floor (yfinance shares data starts ~2015). RU3K now has explicit empty rows for all years.
- **Verification**: `python -m compileall data` passes. Coverage CSV shows reasonable per-year member counts, accurate coverage ratios, and explicit RU3K rows.

## 10. Forward-Return Cache Invalidation

### Finding

The forward-return cache only invalidates when the features parquet mtime is newer than the cache. It does not notice price-cache changes, code changes, or changes in lower-level cache dependencies.

Current risk:

- Rebuilt price files can leave stale forward returns.
- Code changes in `compute_forward_returns()` may not invalidate old labels.
- Phase 5 can silently reuse incorrect target returns.

### Your Proposed Fix

"Cache should depend on whether code changed and whether previous dependent caches changed. Use an existing Python library to manage these caches."

### Assessment

This is the right goal. The important caveat is that most Python cache libraries do not automatically know about arbitrary files read inside a function unless those file fingerprints are included in the cache key. A library can manage persistence, but we still need an explicit dependency signature.

### Recommended Plan

- Use `joblib.Memory` because scikit-learn already depends on joblib in most environments and it handles function-level caching cleanly.
- Add an explicit cache signature object containing:
  - features parquet path, size, and mtime;
  - price manifest path, size, and mtime;
  - a hash of `data/cache/prices/_manifest.json`;
  - requested horizons;
  - `entry_date_col`;
  - source-code hash for `backtest/splits.py` or specifically `compute_forward_returns`;
  - cache schema version.
- Pass this signature as an argument to the cached function so dependency changes invalidate the cache.
- Keep a human-readable manifest next to the cache output, for example `results/cache/forward_returns_manifest.json`.
- Do not hash every price parquet file by default because that can be expensive. Prefer the price manifest as the dependency source. If a price file can change without manifest update, fix the price loader to update the manifest every time it writes a parquet.

### Decision

Use `joblib.Memory` with explicit dependency signatures. Do not rely on `joblib` alone to discover external file dependencies.

### Fix Applied (2026-04-30)

The forward-return cache has been upgraded from a simple mtime-based parquet cache to `joblib.Memory` with explicit dependency signatures.

- **`backtest/splits.py`**: Added `ForwardReturnsCacheSignature` dataclass capturing features parquet metadata (path, size, mtime), price manifest metadata (path, size, mtime, SHA-256 content hash), horizons, `entry_date_col`, source-code hash of `compute_forward_returns`, and a cache schema version.
- **`backtest/splits.py`**: Added `_forward_returns_memory = Memory(...)` instance and `_compute_forward_returns_cached()` decorated with `@memory.cache(ignore=['df'])` so that the cache key is the dependency signature only, not the full DataFrame.
- **`backtest/splits.py`**: Rewrote `get_forward_returns_cached()` to build the dependency signature and delegate to the joblib-cached function.
- **`backtest/splits.py`**: Added `_build_cache_signature()` helper and `_write_forward_returns_manifest()` to write `results/cache/forward_returns_manifest.json`.
- **`backtest/splits.py`**: Updated `_forward_returns_cache_path()` to point to the joblib cache subdirectory (informational only; joblib manages its own storage).

Cache is invalidated when any of the following change:
- Features parquet file (size or mtime)
- Price manifest (size, mtime, or SHA-256 content hash)
- Source code of `compute_forward_returns`
- Cache schema version (bump `FORWARD_RETURNS_CACHE_VERSION`)
- Requested horizons or `entry_date_col`

Smoke test passed: first call computes fresh, second call hits cache, touching features parquet triggers recomputation.

- **`ideas/plan.md`**: Phase 4.1 updated to note joblib.Memory with explicit signatures.
- **`docs/report.md`**: Section 10.0.2 updated to reflect the new cache scheme.

## Proposed Implementation Priority

Order respects dependency chains: `availability_date` > cache safety > data fixes > hparam structure > portfolios > coverage > audit.

1. **Implement the unified operational `availability_date` rule and use it everywhere.**
   Root dependency — changes what "the date" means for feature history, forward returns, folds, tuning, portfolios, rebalancing, and audit. Every date-using component needs this first.

2. **Replace forward-return cache with explicit dependency-aware `joblib.Memory` caching, including `entry_date_col` in cache keys.**
   Must immediately follow #1. Changing the date logic without fixing cache invalidation means stale forward returns silently poison all downstream results during development.

3. **Fix SP1500 month-end membership.**
   Independent data fix with no code dependency on other items. Can be done in parallel with #4.

4. **Fix shares coverage denominator.**
   Independent data fix. Same reasoning as #3.

5. **Fix hparam granularity and Phase 4 loading.**
   Depends on #1: folds and tuning now use availability_date, so the structural fix for hparams must sit on top of the corrected date logic. Re-tuning depends on #1 being correct.

6. **Fix portfolio cohort aggregation.**
   Defines the core portfolio structure — what a cohort is and how cohorts net together. Entry/exit quote handling (#7) and missing-return accounting (#8) both build on this structure.

7. **Fix entry/exit quote roll-forward.**
   Depends on cohort structure (#6) and availability_date (#1) for determining planned vs actual entry/exit dates.

8. **Fix bounded forward-fill and long-gap accounting for missing daily returns / delisting.**
   Depends on portfolio P&L which in turn depends on #6 and #7 being correct first. Cannot correctly account for missing returns until the portfolio structure and quote handling are right.

9. **Expand `run_all.py` Phase 5 loops across selected tiers, universes, models, and cadences, including RU3K coverage-constrained artifacts.**
   Needs correct portfolios (#6/#7/#8) to produce meaningful results across universes. The loop expansion itself is mechanical, but verifying correctness requires proper intermediate outputs.

10. **Wire Phase 4/5 audit validations into actual execution.**
    Validates everything above — fold boundaries, fit-time leakage, trade execution logs. Must be last by definition.

## Items To Confirm Before Editing Code

All open policy choices from this debugging pass are now resolved. Remaining implementation details are code-level, not methodology-level.
