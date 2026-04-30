## Implementation Plan (Execution Order)

> This file is the **execution checklist**: every step from receiving the raw zip file to producing the PDF.
> Methodology choices, decision rationale, and transparency statements belong in `docs/report.md`.
> Project requirements and hard constraints are in `docs/requirement.md`.

---

### Phase 0 - Project Skeleton

- Create directories: `data/ features/ backtest/ reports/ results/ results/audit/ data/cache/ data/universe_raw/`
- Add `results/`, `data/cache/`, `*.parquet`, and `*.csv.zip` to `.gitignore`
- Add a README placeholder; fill in the final "one command to reproduce" entry point in the last phase
- Fix the global random seed at `seed=42`

---

### Phase 1 - Raw Data Loading and Cleaning

#### 1.1 ATC signal CSV -> Parquet

`data/load_signals.py`

- Read `Earnings_ATC_until_2026-04-21.csv` in chunks of 100k rows
- Drop `SignalType == 'delete'` (2,231 rows)
- **Do not** pre-filter on `COUNTRY == 'US'`; keep the full cleaned signal set
- Parse `MOSTIMPORTANTDATEUTC`, extract the hour, and persist `call_hour_utc`
- The main cache keeps all non-`Fluff`/`Filler` `AspectTheme_*` columns for both Enhanced and Stretch tiers; optionally persist a slim cache with only identifiers, headline fields, and EventScore fields
- Slim cache is derived from the main Arrow table via `main_table.select(slim_columns)` to avoid a second `from_pandas()` conversion per chunk
- `coerce_chunk_types` batch-converts all ~400 AspectTheme columns at once (`.apply(pd.to_numeric)` on the masked DataFrame) instead of one column at a time
- Outputs: `data/cache/signals.parquet`, `data/cache/signals_slim.parquet`

#### 1.2 PIT universe membership

`data/load_universes.py`

- **SP500**: Wikipedia historical constituent changes + current constituents -> add/remove ledger -> daily PIT parquet
- **SP1500**: iShares `IJH` (SP400) + `IJR` (SP600) monthly holdings + SP500 PIT, monthly frequency
  - SP500 component at each month-end uses the **latest SP500 daily snapshot on or before that month-end date** (not the union of all daily SP500 members that appeared during the month). This prevents a stock added-and-removed mid-month from appearing in the month-end snapshot.
- **RU3K**: iShares `IWV` monthly holdings, monthly frequency
- Store raw snapshots in `data/universe_raw/`; compiled artifacts in `data/cache/universes/{sp500,sp1500,ru3k}_pit.parquet`
- Any segment that cannot be recovered automatically -> write separately to `results/audit/universe_coverage_gaps.csv`; do not silently patch gaps and do not fall back to the current snapshot
- Provide `members_at(universe, date)`: raise directly when `date > today`; return an empty set when `date < min(snapshot)`
- `build_sp500_pit` and `expand_to_month_grid` use `list.extend(generator)` instead of per-ticker `append` to cut python-level loop overhead for ~2.2M row constructions

#### 1.3 Prices and Volume

`data/load_prices.py`

- Default ticker set = union of `{sp500, sp1500, ru3k}_pit.parquet` historical members; signal-side `BESTTICKER` is opt-in via `--include-signal-tickers` (and filtered to US-format symbols `^[A-Z]{1,5}([.\-][A-Z])?$` so non-US numeric codes are not retried)
- yfinance batch download (50 tickers per call) + `ThreadPoolExecutor` with 4 workers; sequential mode (`--no-parallel`) available for debugging
- `auto_adjust=True` so `Close` is the adjusted close
- Exponential backoff (base 1s, cap 120s, max 3 retries per batch); a small jittered sleep between batches; never rotate User-Agent
- Resumable manifest at `data/cache/prices/_manifest.json` keyed by ticker with `{status, rows, first_date, last_date, error, fetched_at}`
- Freshness rules in `is_cached_fresh()`: `success` is fresh when `target - last_date <= 5 days`; `empty` is cached for **90 days** so delisted/invalid tickers don't trigger yfinance ERROR spam every run; `error` is never cached
- yfinance + peewee loggers are pinned to `CRITICAL` after the lazy import to silence "possibly delisted; no timezone found" noise
- Maintain `data/cache/prices/failed_tickers.csv` with `{ticker, status, rows, first_date, last_date, error}`; status vocabulary is `success | empty | error`; a later success automatically clears stale failure rows for that ticker. New failure rows are accumulated in a list and concatenated once per batch to avoid O(n^2) `pd.concat` overhead.
- Output: one file per ticker at `data/cache/prices/{ticker}.parquet` (columns: `date, adj_close, volume`)

#### 1.4 Historical shares outstanding (market-cap buckets only)

`data/load_shares.py`

- Prefer Yahoo historical shares outstanding via `Ticker.get_shares_full(start, end)`; sequential per ticker (no batch endpoint) with exponential backoff (base 0.6s, cap 60s, max 4 retries), small jittered sleep between calls
- Default ticker set = union of `{sp500, sp1500, ru3k}_pit.parquet` historical members (no signal-side opt-in); `--only T1 T2 ...` for spot fetches
- Resumable manifest at `data/cache/shares/_manifest.json`; `is_cached_fresh()` requires status=`success` and `target - last_date <= 14 days` (shares change slowly)
- Output schema: `data/cache/shares/{ticker}.parquet` with columns `date, shares` (drop NaN, drop non-positive, dedupe by date keep-last, sort ascending)
- Mark missing series instead of imputing; record coverage in `results/audit/marketcap_capacity_coverage.csv` (one row per `universe x year` with `members / covered / coverage_ratio / below_floor`)
- **Coverage denominator**: per-calendar-year PIT membership (union of tickers on snapshots within that year only), not all-historical membership. Years with no PIT snapshots get an explicit zero row. This prevents early-year coverage from being understated by future members.
- If coverage for any universe x year is below 70%, downgrade market-cap buckets for that subperiod to qualitative robustness only and flag this in red in the PDF

---

### Phase 2 - Feature Engineering

`features/engineer.py`

#### 2.1 Timestamp fields (build first; all later features depend on them)

```
entry_rule(ts):  hour < 13 UTC -> same business-day close entry (BMO)
                 hour >= 13 UTC -> next business-day close entry (AMC; gray zone treated conservatively as AMC)

call_entry_date   = entry_rule(MOSTIMPORTANTDATEUTC)
ingest_entry_date = entry_rule(INGESTDATEUTC)

# Unified operational availability_date rule:
# - call_entry_date < 2023-07-06: call_entry_date + 2 business days
# - call_entry_date >= 2023-07-06: max(call_entry_date, ingest_entry_date)
availability_date = unified_rule(call_entry_date, ingest_entry_date)
```

- `INGESTDATEUTC` is used only for availability calculation and never enters features
- Implementation rolls weekends forward with `np.busday_offset`; exact exchange holidays are handled later by price-aware execution / return joins
- The `+2 business days` pre-launch rule simulates operational processing delay before ProntoNLP went live on 2023-07-06. On/after the launch date, real `INGESTDATEUTC` timestamps are available.
- Counterexample to avoid: `INGESTDATEUTC = 2020-03-15 22:30 UTC`; naive date max gives 03-15, while the correct date is 03-16
- **Phase 1 data independence**: Phase 1 stores `MOSTIMPORTANTDATEUTC` and `INGESTDATEUTC` as raw columns without applying any availability logic. The `availability_date` computation is entirely in Phase 2. Therefore, changing the availability rule never requires re-running Phase 1 — only Phase 2+ need to be re-run.

#### 2.2 Enhanced features (85 columns)

**Row-level 60 columns** (direct row derivations with zero look-ahead risk)


| Subgroup                     | Count | Contents                                                                           |
| ------------------------------ | ------: | ------------------------------------------------------------------------------------ |
| Headline                     |     1 | `ATCClassifierScore`                                                               |
| EventScore variants          |     4 | `EventsScore_{1_1_1, 4_2_1, 3_1_0, 1_1_0}`                                         |
| Per-Aspect totals (x4 views) | 5+5+5 | totals / net sentiment / magnitude-weighted (**Fluff/Filler always excluded**)     |
| Per-Theme totals (x3 views)  | 9+9+9 | totals / net sentiment / magnitude-weighted (after excluding Fluff/Filler aspects) |
| Call-length controls         |     2 | `DOCSENTENCECOUNT` / `Sentences`                                                   |
| Sector one-hot               |    11 | GICS 11 sectors                                                                    |

Magnitude weights: `High x 3 + Med x 2 + Low x 1`

**Time-series 16 columns**


| Subgroup                       | Count | Contents                                 |
| -------------------------------- | ------: | ------------------------------------------ |
| QoQ delta of ATC               |     1 | `current_q - previous_q`                 |
| QoQ delta of per-Aspect totals |     5 |                                          |
| QoQ delta of per-Theme totals  |     9 |                                          |
| 4Q rolling trend slope of ATC  |     1 | OLS slope using only the past 4 quarters |

QoQ joins use `(BESTTICKER, SignalType, availability_date)` with a **strict prior availability date**, not `QTR_YEAR < QTR_YEAR`. Rows with the same availability date never chain into each other.

**Cross-sectional PIT percentiles: 6 columns (highest leak-risk area)**


| Subgroup                                                  | Count |
| ----------------------------------------------------------- | ------: |
| Sector-relative expanding percentile of ATC               |     1 |
| Sector-relative expanding percentile of per-Aspect totals |     5 |

Implementation: compare each row against historical events in the same `(SECTOR, SignalType)` bucket with `availability_date < call_entry_date`. Current rows and same-day rows are excluded; early backfilled rows can legitimately have `NaN` PIT percentiles under the strict `INGESTDATEUTC` availability rule.
**Never** use full-sample ranking such as `df.groupby('SECTOR')['ATC'].rank(pct=True)`.

**Pre-event price momentum: 3 columns**


| Subgroup                            | Count | Contents                                                                                                                                        |
| ------------------------------------- | ------: | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 21d pre-event return                |     1 | T-1 through T-21                                                                                                                                |
| Sector-relative pre-event return    |     1 | company 21d return minus median sector 21d return                                                                                               |
| 5d pre-event idiosyncratic residual |     1 | `residual_5d = stock_ret_5d - beta_T x sector_median_ret_5d`, where `beta_T` is 60-day rolling OLS on window `[T-65, T-5]`; set NaN when bars < 30 |

Window anchor: T is the date part of `MOSTIMPORTANTDATEUTC` (**not** `entry_date`); all return bars must be strictly `<= T-1`. `beta_T` is shifted so the 60-day regression ends at T-5 and excludes T-4..T-1. Sector returns use the median across loaded sector tickers to avoid yfinance outlier contamination.

#### 2.3 Stretch features (interaction candidate pool)

- Among 567 raw `AspectTheme_*` columns, Fluff/Filler account for ~162 columns (2 Aspects x 9 Themes x 3 Magnitudes x 3 Sentiments); exclude all of them -> ~405 columns remain
- **No full-sample pre-screening**; perform feature selection inside each walk-forward step's training fold (see Phase 4)

#### 2.4 Exclusion list

- `QTR_YEAR` does not enter features
- Any `Return_*d` column does not enter features
- `INGESTDATEUTC` does not enter features

#### 2.5 Implementation and verification log — 2026-04-29

All 85 enhanced features and 405 stretch columns are implemented. `features/engineer.py` exposes `build_features(df, price_cache_dir, tier, include_momentum)` and runs: timestamps -> row features -> strict time-series features -> strict PIT percentiles -> vectorized momentum -> optional stretch join.

**Leakage fixes after review.**
- Timestamp entry dates now roll weekends forward to the next business day; `availability_date` uses a pandas row-wise max so a missing ingest timestamp does not erase a valid call timestamp.
- QoQ deltas and the 4Q ATC trend are grouped by `(BESTTICKER, SignalType)` and use only the previous **distinct** availability date. Same-day rows never chain into each other.
- PIT percentiles are exact historical empirical percentiles inside `(SECTOR, SignalType)` with `availability_date < call_entry_date`, implemented with a Fenwick tree over discretized values. The old self-inclusive `expanding().rank()` approach was removed.
- Momentum beta is shifted by four trading rows so the 60-day beta estimate ends at T-5. `merge_asof` has a 10-calendar-day tolerance to avoid stale delisted-price metrics.
- Final feature output is restored to input row order after sorted PIT/time-series calculations.

**Vectorized momentum (deviation from per-row loop).** The per-row momentum computation was replaced with a vectorized design: pre-load ticker price histories, compute daily returns, compute median sector daily returns and n-day sector returns via log-return cumsum, compute rolling beta by ticker, then join events to per-ticker metrics with `merge_asof`.

**Pandas 3.0 workaround.** `pd.merge_asof(..., by=)` is broken in pandas 3.0.2 for all tested dtypes, so merging is done per ticker without `by=`.

**PIT percentile acceleration — 2026-04-29.** `_strict_historical_percentile()` now uses an optional Numba `njit(cache=True)` fast path when `numba` is installed, with the original Python Fenwick-tree implementation retained as a fallback. The compiled path preserves the exact strict-history rule (`history_date < cutoff_date`) and was checked against a naive O(n^2) fixture implementation on randomized values/timestamps.

**Full enhanced rebuild results (`python -m features.engineer --tier enhanced --output results/features_enhanced.parquet`).**

| Stage | Time | Output |
|---|---:|---|
| 2.1 Timestamps | 3.0s | business-day `call_entry_date`, `ingest_entry_date`, `availability_date` |
| 2.2 Row features | 15.7s | 60 cols |
| 2.2 Time-series | 3.4s | 16 cols |
| 2.2 PIT percentiles | 54.4s | 6 strict historical percentile cols |
| 2.2 Momentum | 6.8s | 3 cols; 657 / 17,636 tickers had cached prices |
| **Total** | **85.1s** | 2,738,206 rows x 91 cols |

**Sanity checks passed on the rebuilt full artifact:**
- Output aligns 1:1 with input `signals.parquet` row order
- No `QTR_YEAR`, `INGESTDATEUTC`, or `Return_*d` columns in output
- `availability_date >= call_entry_date` for all rows
- Sector one-hot sum is exactly 1 for all 2,738,206 rows
- No QoQ value appears on a ticker/slice's first availability date
- PIT percentiles are either `NaN` or in `[0, 1]`; `NaN` is expected for rows with no strictly prior available history
- Momentum valid counts: `pre_event_ret_21d` 279,400; sector-relative 279,400; idiosyncratic residual 279,233

#### 2.6 Performance fixes — 2026-04-29

Performance fixes applied during Phase 2:

| # | Location | Fix |
|---|---|---|
| 1 | `ticker_sector_map` | Replaced per-ticker full-frame filters with `df.groupby(ticker_col)["SECTOR"].first().to_dict()` |
| 2 | 4Q slope | Replaced per-row `np.polyfit` with closed-form 4-point OLS slope |
| 3 | Anchor trading day | Replaced per-row calendar scans with `DatetimeIndex.searchsorted()` |
| 4 | Per-ticker merges | Pre-grouped events and metrics by ticker before `merge_asof` |
| 5 | QoQ deltas | Replaced 15 repeated groupby diffs with one date-block calculation across all QoQ columns |
| 6 | `merge_asof` dtype | Forced both join keys to `datetime64[ns]` after Parquet round trips |
| 7 | PIT percentiles | Replaced self-inclusive expanding rank with exact strict-history Fenwick percentile |
| 8 | PIT percentile hot loop | Added optional Numba-compiled Fenwick core; fallback remains the exact Python implementation when Numba is unavailable |

---

### Phase 3 - Automated Look-Ahead Tests

`features/audit.py`

#### 3.1 Streaming vs batch regression test

- Fit once on the full sample -> compare the feature matrix against per-day streaming fits
- Tolerance: `np.allclose(rtol=1e-9, atol=1e-12)`
- Start with a unit test on a small ticker subset (~50 tickers) and a 1-year window, then run full-sample regression

#### 3.1 implementation notes — 2026-04-29

- The full regression uses the complete `signals.parquet`; `--full-dates N` controls how many availability dates are sampled for streaming replay, not how many signal rows are loaded.
- Full-regression memory was first reduced by trimming the batch feature matrix to only sampled comparison dates, avoiding extra DataFrame copies in `_build_safe()`, and releasing per-date streaming frames after comparison.
- The no-momentum full-regression path now uses a target-only streaming audit: row-level features are reused from the full batch output because they are row-local; QoQ deltas, 4Q trend, and strict PIT percentiles are recomputed only for rows whose `availability_date` is one of the sampled audit dates. PIT queries use the complete row-level history but only emit sampled-date rows, so future rows cannot enter because each query still enforces `history_date < cutoff_date`.
- `--full-dates` is now wired through the CLI to `run_all_audits()`. Validation passed with `--small-only`, an equivalence check comparing the target-only path against old full-prefix streaming on sampled small-subset dates, and the full-data 15-date command (`110,592` rows compared, zero strict mismatches).
- The expected full command is `python -m features.audit --signals data/cache/signals.parquet --tier enhanced --full-dates 15`; smaller `--full-dates` values remain useful as diagnostics.

#### 3.2 Eight assertion classes (any failure turns CI red)

1. **Feature parity**: streaming vs batch columns align exactly; key targets = 6 PIT percentile columns / QoQ / R8 idiosyncratic beta / pre-event sector-relative features
2. **Fold boundary + label purge**: `max(train_feature_date) < min(test_feature_date)` and `max(train_target_available_date_h) < min(test_feature_date)`
3. **`fit()` call-stack monitoring**: monkey-patch `StandardScaler.fit / SimpleImputer.fit / LassoCV.fit / model.fit`; record `caller_fold_id` and input date range; raise directly if any row outside the fold is included; R8 rolling beta estimation must also receive data only up to `T-5`
4. **PIT universe defense**: `members_at(date)` raises for `date > today`; returns an empty set for `date < min(snapshot)`
5. **Forward-return isolation**: `assert set(feature_cols).isdisjoint(return_cols)`; regex `^Return_\d+d$` must not appear in feature names
6. **Timestamp boundary fixtures**: `entry_rule()` / `availability_date()` cover `12:59 / 13:00 / 15:59 / 16:00 / 22:30 UTC` plus a cross-day sample where the call is same day and ingest is next day after market close
7. **Rebalance eligibility**: in a hand-built sample, `eligible_events(rebalance_date)` returns only events with `availability_date <= rebalance_date`
8. **Trade execution log**: `actual_entry_date >= planned_entry_date` and `actual_exit_date >= planned_exit_date`; `skip_no_entry_quote / right_censored_no_exit_quote / delisting_exit_used` must have matching price evidence

---

### Phase 4 - Walk-Forward Splits and Frozen Hyperparameters

`backtest/splits.py`, `backtest/model.py`

#### 4.1 Hyperparameter tuning (one-time run, then freeze)

- Tuning sample = pooled PIT training sample from 2010-01 through 2019-12 (**never touch 2020Q1+**)
- Inner CV: `TimeSeriesSplit(n_splits=5)` (strictly no `KFold` / `StratifiedKFold`)
- Run Ridge / LightGBM / XGBoost once each per `(tier, horizon)` -> `results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json`
- Storage follows a tier x model x horizon directory structure:
  - `results/hparams/enhanced/h1d/frozen_hparams_ridge.json`
  - `results/hparams/enhanced/h5d/frozen_hparams_lightgbm.json`
  - `results/hparams/stretch/h20d/frozen_hparams_xgboost.json`
  - Each write updates `results/hparams/hparams_manifest.json` with tuning metadata
- The same `feature tier x model x horizon` shares one hparam set across all three universes
- **Availability column**: uses ``availability_date`` (unified operational rule: ``call_entry_date+2bd`` before 2023-07-06, ``max(call_entry_date, ingest_entry_date)`` on/after). This is the strategy-facing availability timestamp used for fold construction, tuning sample filtering, G11 label purge, and downstream portfolio signal dates. ``call_entry_date`` (derived from ``MOSTIMPORTANTDATEUTC``, the actual earnings call publication time) is available via ``--availability-col call_entry_date`` for backward compatibility.

#### 4.2 Main walk-forward backtest

- Start: 2020Q1; quarterly step through 2026Q2
- Initial training set = 2010-01 through 2019-12, expanding quarterly up to the day before each test fold
- Read frozen hparams; no further tuning
- Inside each fold:
  - Fit imputation / scaling only on the training fold
  - Stretch tier: run `LassoCV(cv=TimeSeriesSplit(n_splits=3))` for column selection; keep nonzero-coefficient columns; if >200 columns remain, truncate to top 200 by `|coef|`; write selected feature names to the audit log

#### 4.5 Forward-return cache (joblib.Memory with explicit signatures)

- Forward returns are cached using `joblib.Memory` (scikit-learn dependency) with an explicit `ForwardReturnsCacheSignature` dataclass that captures all dependencies:
  - Features parquet path, size, and mtime
  - Price manifest path, size, mtime, and SHA-256 content hash
  - Requested horizons and `entry_date_col`
  - Source-code hash of `compute_forward_returns` (via `inspect.getsource`)
  - Cache schema version integer
- The `df` argument is excluded from the joblib cache key (`ignore=["df"]`) since it is fully determined by the features parquet path in the signature.
- A human-readable manifest is written to `results/cache/forward_returns_manifest.json` on every call.
- The old mtime-only parquet cache is replaced; joblib manages storage in `results/cache/joblib/`.

#### 4.3 Label availability purge (G11)

- For horizon `h`, keep training rows only when `target_available_date_h <= fold_train_end`
- `target_available_date_h` = the date when the forward return is fully realized and readable from the price table (using the roll-forward rule in audit item 9)
- Any sample whose `entry_date` is in the training fold but whose `target_available_date_h` crosses into validation/test is removed from training
- Apply purge to both the outer main backtest and inner `TimeSeriesSplit`

#### 4.4 Partial-period handling

- Samples with `target_available_date_h > price_data_end` -> `right_censored_target`; exclude them from all IC / training / test / portfolio return calculations
- Report censored counts separately by universe x horizon

---

### Phase 5 - Experiment Execution (Run Every Universe)

#### 5.1 Single-feature IC analysis

`backtest/single_feature_ic.py`

- 14-column short list, frozen during the planning phase to avoid cherry-picking:
  1. `ATCClassifierScore`
  2. `EventsScore_4_2_1` (production version)
  3. `EventsScore_1_1_1` / `EventsScore_3_1_0` / `EventsScore_1_1_0`
  4. per-Aspect Surprise net sentiment
  5. per-Theme FinancialPerformance net sentiment
  6. per-Theme StrategicInitiatives net sentiment
  7. QoQ delta of ATC
  8. Sector-relative expanding percentile of ATC
  9. 21d pre-event return
  10. Sector-relative pre-event return
  11. 5d pre-event idiosyncratic residual
  12. 4Q rolling trend slope of ATC
- horizons = `{1, 3, 5, 10, 20}` days
- Spearman IC, split by year and sector
- **SignalType slices**: run one ATC IC curve each for `Total / CEO / CFO / Analysts / Executives`
- Uncertainty: mean IC / t-stat / **Newey-West adjusted t-stat (lag=horizon)** / hit rate

#### 5.2 Quintile / Decile portfolios

`backtest/quintile.py`

- Required baseline: `ATCClassifierScore` x universe x horizon `{1,3,5,10,20}` x SignalType `{Total, CEO, CFO, Analysts, Executives}` -> **decile spread**
- Outputs: cumulative L/S equity / drawdown / **rolling Sharpe**
- Sizing: equal-weight + dollar-neutral (long $1 / short $1, equal-weight within each leg)
- Other single features / model signals use at least quintiles; keep decile sensitivity in the appendix
- Report long-only / short-only / long-short

#### 5.3 Walk-forward model predictions

`backtest/model.py`

- Run Ridge / LightGBM / XGBoost for both Enhanced and Stretch
- Output OOS predictions for each fold plus audit logs (selected features / fit calls / sample dates)

#### 5.4 Rebalanced portfolio simulation

`backtest/portfolio.py`

- Run **all three cadences**: daily / weekly / monthly
- Lookback N: daily=1 / weekly=5 / monthly=21 trading days
- Rebalance time: weekly = Monday close; monthly = first trading-day close of each month
- Candidate events: `availability_date <= rebalance_date`, trailing N trading-day window (**never** aggregate all events from "this week" or "this month" after the fact)
- Capital convention: independent stock selection by cohort -> net by stock after aggregation -> scale to fixed gross=200% / net=0%
- **Cohort aggregation algorithm**:
  1. Define a cohort as the set of eligible events on one signal date (``date`` column = ``availability_date``).
  2. At each rebalance date, collect eligible cohorts in the trailing N trading-day window.
  3. For each cohort:
     - Drop NaN scores.
     - Deduplicate to one signal per ticker (if multiple events for same ticker on same date, keep the one with highest absolute score).
     - Rank by score descending.
     - Select top ``long_frac`` for long leg, bottom ``long_frac`` for short leg with ``n_min_positions=5`` floor per leg.
     - Assign equal-weight raw weights (``1/n_long`` long, ``-1/n_short`` short).
  4. Concatenate all cohort raw weights.
  5. Net by ticker across cohorts (sum weights for each ticker).
  6. Re-center to net zero (subtract mean weight from all tickers).
  7. Scale final weights so gross exposure equals 2.0 (``sum|w| = 2``).
  8. Look up entry prices only for tickers with non-zero aggregated weights; remove non-tradeable tickers and re-scale.
  9. **Entry quote lookup**: Use ``_next_valid_quote(ticker, planned_entry_date, max_forward_days=3)`` which scans up to 3 **trading days** forward in the price-derived trading calendar. ``planned_entry_date`` is the rebalance date. If no valid close is found within 3 trading days, mark ``skip_no_entry_quote`` and remove the ticker from weights, re-scaling remaining weights to gross=2.0.
  10. **Exit quote lookup**: For each entered position, ``planned_exit_date`` is the next rebalance date. Use ``_next_valid_quote(ticker, planned_exit_date, max_forward_days=3)`` with the same 3-trading-day tolerance. If no valid close is found, mark ``right_censored_no_exit_quote``.
- **Daily return gap accounting**: During daily P&L computation, each held ticker is classified every day:
  - Valid next-day return — normal case, included in P&L
  - One-day forward-filled quote gap (``ffill_1d``) — price forward-filled for return continuity, marked in daily return columns, contributes 0% return to headline P&L that day. The cumulative return is captured when the price reappears.
  - Two-day forward-filled quote gap (``ffill_2d``) — same logic as ``ffill_1d`` for two consecutive missing days
  - Longer quote gap (>2 trading days) — position is censored from headline fully-observable portfolio returns; a supplemental 30-trading-day recovery audit runs after the simulation to check if the ticker resumed trading. If recovered, classified as ``long_quote_gap_recovered``; if not, classified as ``possible_delisting_or_data_unavailable``.
- Gap accounting columns in daily returns parquet:
  - ``ffill_1d_weight``, ``ffill_2d_weight``, ``long_gap_recovered_weight``, ``possible_delisting_or_unavailable_weight``
  - ``n_ffill_1d_positions``, ``n_ffill_2d_positions``, ``n_long_gap_recovered_positions``, ``n_possible_delisting_or_unavailable_positions``
- Gap summary statistics in portfolio summary JSON (position-day shares for each category)
- Separate ``gap_accounting_*.parquet`` persisted with per-gap-event details (gap date, ticker, weight, recovered flag)
- Persist both raw cohort weights and aggregated weights
- Metrics: turnover / gross / net / 5 bps post-cost Sharpe
- Capacity proxy:
  - `20d ADDV_{T-1} = mean(price_{T-1..T-20} x volume_{T-1..T-20})`
  - holding name count
  - top-10 weight concentration
  - `%ADV consumed` under AUM grid `{10M, 50M, 100M}`

#### 5.5 Robustness checks

- Subperiods: pre-2020 / 2020-2022 / 2023-2026
- **Sector neutralization**: signal-stage within-sector ranking (rank quintiles inside each GICS sector, then merge long/short legs)
- **Market-cap bucket**: `market_cap_{T-1} = adj_close_{T-1} x shares_{T-1}`; cross-sectional percentiles inside the same-day universe -> top10% mega / 10-40% large / 40-70% mid / bottom30% small; flag days with coverage <85% as coverage-constrained
- **Sector-classification sensitivity**: rerun core Enhanced/Stretch without sector one-hot / sector-relative features, or switch to all-market percentiles
- **Weighting scheme**: equal-weight vs ATC-score-weighted
- **Uncertainty**: portfolio Sharpe / spread use monthly block bootstrap for 95% CI; all IC results include Newey-West t-stat
- **OFAT parameter sensitivity** (fixed to the main backtest's recommended cadence + recommended model):
  - **A required (run all 3 universes)**
    1. Quantile cutoff `{top-quintile, top-decile, top-20, top-50, top-100}` (post-processing)
    2. Cadence lookback N: daily `{1,3}` / weekly `{3,5,10}` / monthly `{15,21,30}` (rerun portfolio sim)
    3. One-way transaction cost `{3, 5, 7, 10}` bps (post-processing)
    4. Weekly rebalance timing `{Monday close, Friday close}` (post-processing)
    5. Label purge gap `{0, 5, 21}` days (rerun walk-forward; gap=0 should trigger a warning and serves as a negative control)
  - **B sampled (SP500 + recommended cadence + LightGBM only)**
    6. R8 idiosyncratic beta rolling window `{40, 60, 90}` days; if IC sign flips, remove this column from the final model and document it in transparency

#### 5.1-5.5 implementation notes — 2026-04-30

**5.1 `backtest/single_feature_ic.py`** (422 lines) — 14 features x 5 horizons x 5 SignalTypes. Cross-sectional Spearman IC computed per month (grouped by `year_month`), then summarized with mean IC / t-stat / Newey-West t-stat (Bartlett kernel, manual implementation since statsmodels is not installed) / hit rate. Universe filtering uses the shared global PIT snapshot helper in `backtest/universe.py`, so empty or stale universes mark rows out-of-universe instead of silently keeping all rows. Sector-split IC computed for `ATCClassifierScore` only. Outputs: `results/ic/ic_summary_{universe}.parquet`, `results/ic/ic_sector_split_{universe}.parquet`.

**5.2 `backtest/quintile.py`** (422 lines) — Decile baseline (ATCClassifierScore) and quintile analysis (all 14 short-list features) per universe x horizon x SignalType. Monthly cross-sectional bucketing via `pd.qcut`; equal-weight dollar-neutral L/S portfolios; cumulative equity curves with long-only / short-only / long-short legs; rolling Sharpe (12-month window); max drawdown. Outputs: `results/quintile/decile_summary_{universe}.parquet`, `results/quintile/quintile_summary_{universe}.parquet`, `results/quintile/quintile_equity_curves_{universe}.parquet`.

**5.3 `backtest/model.py`** (679 lines, shared with Phase 4) — Walk-forward model training and OOS prediction generation. `run_walk_forward()` runs Ridge/LightGBM/XGBoost per tier with frozen hyperparameters, G11 label-purged folds, and `TimeSeriesSplit` inner CV. Outputs: `results/audit/oos_pred_{model}_{tier}_h{h}d.parquet`, `results/audit/fold_manifest.parquet`, `results/audit/sample_size_by_quarter.csv`, `results/audit/fit_audit_log.jsonl`. CLI supports `--tune-only` for hyperparameter tuning on 2010–2019.

**5.4 `backtest/portfolio.py`** (952 lines) — Rebalanced portfolio simulator accepting generic `[date, ticker, score]` DataFrame input, plus OOS prediction format (`df_index` column) via `--features` join. Three cadences (daily/weekly/monthly) with configurable lookback; trading calendar derived from the loaded price matrix; rebalance dates: weekly=Monday close, monthly=first trading day; equal-weight dollar-neutral weight construction (gross=200%, net=0%); daily P&L from next-day close returns on every trading day; turnover tracking; PIT universe filtering before construction; model-tagged outputs; capacity proxy (`20d ADDV` using T-1 data, holding count, top-10 concentration, `%ADV consumed` max across positions under AUM grid `{10M, 50M, 100M}`). Outputs: `results/portfolio/daily_returns_*.parquet`, `results/portfolio/weights_*.parquet`, `results/portfolio/summary_*.json`, `results/audit/trade_execution_log*.parquet`, `results/audit/universe_coverage_by_date*.csv`.

**5.5 `backtest/robustness.py`** (680 lines) — Eight robustness check categories:
1. **Subperiod analysis**: IC and quintile L/S recomputed for pre-2020 / 2020–2022 / 2023–2026
2. **Sector neutralization**: within-sector quintile ranking, then merged across sectors
3. **Market-cap buckets**: `shares × adj_close` at T-1, four buckets (mega/large/mid/small), coverage flag at <85%
4. **Weighting scheme**: equal-weight vs ATC-score-weighted quintile portfolios
5. **Block bootstrap**: monthly block bootstrap (21-day blocks, 2000 resamples) for Sharpe 95% CI
6. **OFAT quantile cutoff**: top-{5, 10, 20, 50, 100} stocks per leg
7. **OFAT transaction cost**: {3, 5, 7, 10} bps post-cost Sharpe (consumes portfolio daily returns)
8. **R8 beta window IC check**: if `pre_event_idio_resid_5d` IC flips sign for any SignalType, flag for removal

Outputs: `results/robustness/robustness_subperiod_ic.parquet`, `results/robustness/robustness_subperiod_quintile.parquet`, `results/robustness/robustness_sector_neutral.parquet`, `results/robustness/robustness_mcap_buckets.parquet`, `results/robustness/robustness_weighting.parquet`, `results/robustness/robustness_ofat_quantile.parquet`, `results/robustness/robustness_ofat_cost.parquet`, `results/robustness/robustness_bootstrap_ci.json`, `results/robustness/robustness_r8_beta_window.parquet`.

**Full pipeline validation (SP500 + SP1500, 2026-04-30, 14m32s).** All five sub-phases completed end-to-end via `python run_all.py --from-phase 5 --force`:
- 5a: IC analysis on sp500 + sp1500 (~83s each)
- 5b: Quintile/decile on sp500 + sp1500 (~127s each)
- 5c: Portfolio simulation — 3 models (ridge, lightgbm, xgboost) × 3 cadences (daily, weekly, monthly) = 9 runs (~8-15s each)
- 5d: Robustness checks on sp500 (~356s including forward-return computation + universe filter + 8 check categories)

**Key robustness findings (SP500).**
- **Signal decay**: ATCClassifierScore IC drops from 0.062 (pre-2020) to −0.012 (2023–2026); the signal weakened materially in the walk-forward period
- **Sector neutralization hurts**: raw quintile Sharpe > sector-neutral Sharpe for all horizons except h=5d (near zero for both)
- **Score-weighting helps mid-horizons**: improves Sharpe at h=3–10d vs equal-weight; hurts at h=1d and h=20d
- **Top-50 cutoff optimal**: Sharpe 0.84 at h=5d vs 0.20 for top-5; top-100 is negative — signal concentrated in tails
- **R8 warning**: `pre_event_idio_resid_5d` IC is negative for all CFO signal types (4/25 horizon×SignalType combinations); per plan, remove this column from the final model and document in transparency

**`run_all.py` Phase 5 integration.** 5a (IC) / 5b (quintile) / 5c (portfolio) / 5d (robustness) all wired with full tier × universe × model × cadence loops:
- Phase 5 iterates over **all selected tiers** (not just ``args.tiers[0]``), so ``--tier both`` runs both Enhanced and Stretch experiments.
- For each tier, 5a (IC) and 5b (quintile) iterate over **every requested universe** (default: ``sp500 sp1500 ru3k``).
- 5c (portfolio) auto-detects available OOS prediction files via ``glob("results/audit/oos_pred_*_{tier}_h*.parquet")``, then loops over every OOS file × every universe × all three cadences (daily, weekly, monthly).
- 5d (robustness) iterates over every requested universe.
- Empty PIT universes (e.g., RU3K) produce explicit ``{module}_{universe}_coverage_constrained.json`` sentinel artifacts by default; use ``--skip-empty-universe`` to skip silently.
- ``--from-phase 5`` / ``--stop-at-phase 5`` wired. Dry-run (``--dry-run --tier both``) enumerates all sub-tasks.

**Post-review fixes — 2026-04-30.** Corrected the shared PIT universe filter for IC/quintile/portfolio, fixed portfolio daily accounting so weekly/monthly runs produce daily P&L rather than rebalance-only rows, added trade log and universe coverage artifacts, changed forward-return entry matching from a 10-calendar-day tolerance to a maximum 3-business-day entry gap, made the audit checklist distinguish Phase 3 pass from Phase 4/5 pending items, fixed sector-neutral robustness to compare `Total` vs `Total`, and batched failed-ticker log writes in `data/load_shares.py`.

#### 5.6 Correctness + speed fixes — 2026-04-30 (round 2)

After a deep code review of Phase 0–5, three issues were addressed:

| # | Type | Location | Fix |
|---|---|---|---|
| M6 | Correctness | `backtest/robustness.py` `block_bootstrap` (~line 438) and call site (~line 1002) | Added `periods_per_year` parameter (default 252). The bootstrap inputs are *monthly* L/S returns from `_build_equity_curves`, but the function previously hard-coded `*252 / sqrt(252)`, overstating reported Sharpe 95% CI by `sqrt(252/12) ≈ 4.58×`. Call site now passes `periods_per_year=12`. |
| S1 | Speed | new `get_forward_returns_cached()` in `backtest/splits.py`; call sites in `single_feature_ic.py`, `quintile.py`, `robustness.py`, `model.py` | Forward returns were recomputed from per-ticker parquet files in every Phase 5 sub-task (5a/5b/5c/5d). The new helper writes a parquet cache to `results/cache/forward_returns_<features_stem>_h<horizons>.parquet`, keyed on the features-parquet path + horizon set, and invalidated automatically when the features parquet mtime moves forward. Across a `run_all.py --from-phase 5` run this eliminates 3+ duplicate full passes (~30–60s each). |
| S2 | Speed + correctness | `backtest/robustness.py` `assign_market_cap_buckets` (~line 299) | Replaced per-month-per-ticker parquet reads (~100k reads on full SP500) and per-row `df.loc[idx, ...]` bucket assigns with: (a) one up-front pass that loads PIT-universe members' price + shares histories via `_load_ticker_history`, (b) `np.searchsorted` per event date for strict T-1 price and strict T-1 shares, (c) quantile cutoffs computed from covered members in the latest PIT universe snapshot on or before that event date, and (d) `np.select` for vectorized event-row bucket assignment. This fixes the old event-month-ticker threshold approximation while keeping missing data as `"unknown"` and reporting coverage as covered PIT members / PIT members. |

Validation: `python -m compileall backtest features data run_all.py` passes; smoke test confirms `block_bootstrap(monthly_returns, periods_per_year=12)` returns sensible Sharpe CIs vs the previously inflated `=252` default.

#### 5.7 Correctness fixes — 2026-04-30 (round 3)

After a third code review, four issues were addressed:

| # | Type | Location | Fix |
|---|---|---|---|
| A1 | Correctness | `backtest/model.py` `_run_one_fold` (~line 311) | Wired `monitor_fit_calls()` from `features/audit.py` around the impute/scale → LassoCV → model.fit section. After the context exits, `get_fit_log()` entries are annotated with fold boundaries and validated via `assert_fit_callstack()`. Violations are recorded in `FoldResult.fit_violations` and persisted to `fit_audit_log.jsonl`. This closes the Plan §3.2 assertion 3 gap — the audit infrastructure existed but was never connected to Phase 4. |
| A2 | Correctness | `backtest/portfolio.py` `PortfolioSimulator.run` (~line 515, ~line 565) | Added `_next_valid_quote()` helper that scans up to 3 **trading days** forward in the trading-calendar-derived list for a valid close (not calendar days). Entry and exit quote lookups now use this helper instead of checking only the single planned date. ``skip_reason`` is ``None`` for normal trades (was ``""``). ``PortfolioResult`` gained a ``trade_log_violations`` field. ``validate_trade_log()`` is called after the simulation loop; violations are persisted to ``trade_execution_violations_{suffix}.parquet``. Plan §4.5 requires this tolerance before marking ``skip_no_entry_quote`` / ``right_censored_no_exit_quote``. Actual entry/exit dates and prices are recorded in ``trade_execution_log.parquet``. |
| A3 | Correctness | `backtest/single_feature_ic.py` `_filter_to_universe`, `backtest/quintile.py` `_filter_to_universe` | Changed `tolerance_days` default from `65` to `None` so the shared `backtest/universe.py` module auto-detects the appropriate tolerance (7 days for daily SP500 snapshots, 65 for monthly SP1500/RU3K). The previous hard-coded 65 caused SP500 events to match stale snapshots up to 2 months old. |
| A4 | Hygiene | `data/config.py` (new `utc_now_iso`), `data/load_prices.py` (3 call sites), `data/load_shares.py` (2 call sites) | Replaced deprecated `datetime.utcnow()` with a compat helper that uses `datetime.now(dt.UTC)` on Python 3.12+ and falls back to `utcnow()` on older interpreters. |

Validation: `python -m compileall backtest features data` passes. `from backtest.model import FoldResult` and `from backtest.portfolio import PortfolioSimulator, _next_valid_quote` import cleanly. The fit-monitoring wiring produces date-aware fit logs; the portfolio 3-day tolerance produces more realistic entry/exit fill rates.

#### 5.8 Correctness fixes — 2026-04-30 (round 4)

After a fourth code review, two issues were addressed:

| # | Type | Location | Fix |
|---|---|---|---|
| C1 | Correctness | `backtest/model.py` `run_walk_forward` (~line 432) | Changed `availability_col` default from `"call_entry_date"` to `"availability_date"`. The CLI `main()` correctly defaulted to `"availability_date"`, but the function signature itself used `"call_entry_date"`, so a programmatic caller that omitted the argument would silently use the wrong date column for fold construction, training filters, and G11 label purge. |
| C2 | Correctness | `backtest/splits.py` `_build_cache_signature` (~line 173) | Replaced `hash(source)` with `int(hashlib.sha256(source.encode()).hexdigest()[:16], 16)`. Python's built-in `hash()` is randomised via `PYTHONHASHSEED` across interpreter restarts, so the `source_hash` field in `ForwardReturnsCacheSignature` could differ between runs even when the source code was unchanged, causing unnecessary forward-return cache recomputation. SHA-256 is deterministic and already used for the price-manifest hash in the same function. |

Validation: `python -m compileall backtest/model.py backtest/splits.py` passes.

#### 5.9 Correctness fixes — 2026-04-30 (round 5)

After a follow-up code review of the current Phase 0-5 implementation, six additional issues were addressed. These changes invalidate the old Phase 4/5 artifacts and require a rerun from at least Phase 1.2 onward for final reported results.

| # | Type | Location | Fix |
|---|---|---|---|
| D1 | Correctness | `data/load_universes.py` `build_sp500_pit` (~line 187) | Fixed SP500 reverse-replay membership dates. The old code recorded the pre-change state at `change_date - 1` and then forward-filled it, so added names could become active only at a later snapshot and removed names could remain active after the effective date. The corrected logic records the post-change state at the actual effective date, then undoes all additions/removals for that date to continue walking backward. Same-day multiple changes are handled as a date block. |
| D2 | Correctness | `backtest/model.py` new `filter_model_sample()` and CLI args | Phase 4 predictive models now filter the modeling sample by PIT universe and `SignalType` before tuning/walk-forward. Default model sample is `--universe sp500 --signal-type Total`. The filter preserves `_orig_df_index`, so OOS prediction rows can be joined back to the original feature parquet by `backtest/portfolio.py`. |
| D3 | Correctness + reproducibility | `backtest/model.py` CLI and `run_all.py` Phase 4 | Hyperparameter tuning now runs once per requested horizon in `--horizons` and writes `results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json`. `run_all.py` tunes on the first populated universe and then runs walk-forward separately for every requested populated universe, reusing the shared frozen hparams. |
| D4 | Correctness | `backtest/model.py` OOS output naming and `run_all.py` Phase 5 portfolio tags | OOS prediction filenames now include model, tier, universe, signal slice, and prediction horizon: `oos_pred_{model}_{tier}_{universe}_{signal}_h{horizon}d.parquet`. Phase 5 portfolio tags include `predh{horizon}d`, preventing h1/h3/h5/h10/h20 runs for the same model/cadence from overwriting one another. |
| D5 | Audit correctness | `features/audit.py` `monitor_fit_calls`; `backtest/model.py` `_impute_and_scale` / `_run_one_fold` | Fit-call monitoring now patches `Ridge.fit()` in addition to `RidgeCV`, `LassoCV`, LightGBM, and XGBoost. The model fit path keeps training matrices as pandas DataFrames with `DatetimeIndex` through imputation/scaling/model fitting, so fit logs record real `min_date` / `max_date` instead of only array shapes. Each `FoldResult` stores the recorded fit log and `write_fit_audit_log()` persists it. |
| D6 | Execution correctness | `backtest/portfolio.py` `PortfolioSimulator.run` | Delayed-entry positions returned by `_next_valid_quote()` no longer enter daily P&L before their `actual_entry_date`. The simulator tracks `current_entry_dates` and only includes active weights whose actual entry date is on or before the current P&L date. |

Validation performed:
- `python -m compileall data features backtest reports run_all.py` passes.
- Main module import smoke test passes.
- Synthetic SP500 add/remove fixture now returns old member before the change date and new member on/after the effective date.
- `filter_model_sample(results/features_enhanced.parquet, universe_name="sp500", signal_type="Total")` returns `32,136` rows and preserves unique `_orig_df_index` values.
- Synthetic walk-forward smoke writes `oos_pred_ridge_enhanced_sp500_total_h1d.parquet`; OOS `df_index` maps back to original feature indices; fit logs include dated entries for `SimpleImputer`, `StandardScaler`, and `Ridge`.
- Portfolio smoke confirms positions with delayed `actual_entry_date` are excluded from gross/P&L before the actual entry date.

Required rerun after this round:

```bash
python -m data.load_universes --only sp500
python run_all.py --from-phase 4 --stop-at-phase 5 --tier enhanced
```

---

### Phase 6 - Persist Audit Outputs

`results/audit/` must contain the following files; logs alone are not sufficient:

- `lookahead_checklist_onepager.md`: sign-off for the 10 requirement section 3 items (rule / code location / evidence file)
- `fold_manifest.parquet`: train/test start and end for each outer/inner fold + `max_train_target_available_date_h`
- `feature_parity_summary.json` + `feature_parity_mismatches.parquet`
- `trade_execution_log.parquet`: planned/actual entry-exit for every trade + skip reason + quote lookup
- `universe_coverage_by_date.csv`: `pit_members / tradeable_members / coverage_ratio / missing_quote_ratio`
- `sample_size_by_quarter.csv`: event count / tradeable count / censored count by universe x SignalType x horizon x quarter; automatically flag quarters with <100 events as low-sample
- `fit_audit_log.jsonl`: metadata for every scaler / imputer / selector / model `fit()`
- `marketcap_capacity_coverage.csv`: shares coverage + bucket-eligible samples + ADDV-eligible samples
- `validation_summary.json`: per-check status with evidence file paths

#### Phase 6 implementation notes — 2026-04-30

All audit wiring is complete. Here is the mapping of checks to code and artifacts:

| # | Check | Code Location | Evidence File |
|---|-------|-------------|--------------|
| 1 | Feature parity (small) | `features/audit.py : run_all_audits()` | `results/audit/feature_parity_summary.json` |
| 2 | Feature parity (full) | `features/audit.py : run_all_audits()` | `results/audit/feature_parity_summary.json` (PEND — use --full-dates 15) |
| 3 | PIT universe defense | `features/audit.py : assert_pit_universe_defense()` | `results/audit/feature_parity_summary.json` |
| 4 | Forward-return isolation | `features/audit.py : assert_forward_return_isolation()` | `results/audit/feature_parity_summary.json` |
| 5 | Timestamp boundary fixtures | `features/audit.py : assert_timestamp_boundaries()` | `results/audit/feature_parity_summary.json` |
| 6 | Rebalance eligibility | `features/audit.py : assert_rebalance_eligibility()` | `results/audit/feature_parity_summary.json` |
| 7 | Fold boundary + label purge | `backtest/model.py : _run_one_fold()` calls `assert_fold_boundaries()` | `results/audit/fold_manifest.parquet` |
| 8 | fit() call-stack monitoring | `backtest/model.py : monitor_fit_calls()` context manager | `results/audit/fit_audit_log_*.jsonl` |
| 9 | Trade execution log | `backtest/portfolio.py : validate_trade_log()` post-simulation | `results/audit/trade_execution_log.parquet` + `trade_execution_violations.parquet` |
| 10 | R8 rolling beta window | `features/engineer.py` (beta shifted T-5 end) | Enforced in code |

**Key additions in Bug #10 fix**:
- `assert_fold_boundaries()` now called after each fold in `_run_one_fold()`, checking that `max(train_feature_date) < test_start` and `max(train_target_available_date_h) < test_start`.
- `write_fold_manifest()` accepts `model`, `tier`, `horizons` parameters and includes them in the output.
- `validate_trade_log()` violation summary counts by type are logged.
- `trade_execution_violations.parquet` always written (even empty) so downstream consumers can reference it.
- `results/audit/validation_summary.json` written with per-check status and evidence paths.

---

### Phase 7 - Charts and PDF Report

`reports/charts.py`, `reports/pdf.py`

#### 7.1 Required charts

- Cumulative L/S equity curve for each universe
- IC heatmap (feature x horizon)
- Quintile spread chart
- Drawdown chart
- Rolling Sharpe chart
- Turnover bar chart

#### 7.2 PDF (15-25 pages)

The structure comes from the `docs/report.md` draft, in this order:

1. Data description
2. Methodology
3. Look-ahead audit
4. Per-universe results
5. Recommended deployment (cadence + position sizing)
6. Risks & limitations
7. Future work

Append the 1-page audit checklist (`lookahead_checklist_onepager.md`).

---

### Phase 8 - Reproducibility Entry Point

- One command, `python run_all.py`: raw zip -> universes -> prices -> shares -> features -> audit tests -> walk-forward -> experiments -> charts -> PDF
- `python run_all.py --tier enhanced` runs only the Enhanced tier; `--tier both` runs both Enhanced and Stretch
- `python run_all.py --from-phase N` / `--stop-at-phase N` for partial reruns; `--force` to ignore cached artifacts; `--dry-run` to preview
- Each phase auto-skips when its output artifacts already exist (idempotent); Phase 1 price/shares loaders are self-resuming via their own manifest files
- README lists dependencies, data sources, expected runtime, and artifact paths
- Verification: run successfully from a clean clone

---

### Guardrail Notes (Recheck During Execution)

- "Sharpe 4" -> definitely leakage; go back to the audit
- Do not draw quintile conclusions from quarters with <100 events
- Long-only looks good almost everywhere in 2010-2021 -> always report L/S spread
- Do not tune per-universe just to make RU3K look better
- Do not remove "money-losing sectors" or delete samples that "look abnormal" -> that is look-ahead
- Do not hit the yfinance API repeatedly mid-run; always use the local cache
- Any NaN handling / scaling / selection must be fit inside the training fold
