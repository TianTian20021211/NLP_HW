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
- Any segment that cannot be recovered automatically -> write separately to `results/audit/universe_coverage_gaps.csv`. When at least one iShares snapshot is available but does not cover the full historical range, the earliest snapshot is used as a static-universe fallback with an explicit survivorship-bias note in the coverage gaps file (per requirement §6.3: documented survivorship bias is acceptable; reported alpha is an upper bound).
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

- Prefer Yahoo historical shares outstanding via `Ticker.get_shares_full(start, end)`; yfinance has no batch endpoint for `get_shares_full`, so each ticker is one HTTP request, but `ThreadPoolExecutor` with 3 workers parallelises the per-ticker calls (sequential mode via `--no-parallel`)
- Prefilter: before any network call, scan all tickers against the manifest and split into `cached` (skip) and `stale` (fetch) groups; report `cached=N  to_fetch=M` upfront
- Per-ticker fetch with exponential backoff (base 0.6s, cap 60s, max 4 retries), small jittered sleep between calls
- Default ticker set = union of `{sp500, sp1500, ru3k}_pit.parquet` historical members (no signal-side opt-in); `--only T1 T2 ...` for spot fetches; `--workers N` to control parallelism
- Resumable manifest at `data/cache/shares/_manifest.json` with `threading.Lock`-protected atomic writes; `is_cached_fresh()` requires status=`success` and `target - last_date <= 14 days` (shares change slowly)
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

#### 2.7 Performance fixes from review — 2026-05-01

| # | Location | Fix |
|---|---|---|
| 1 | `_aspect_features`, `_theme_features` | Compute `_group_aspect_theme_cols(df)` once in `compute_row_features`, pass dict to both functions |
| 2 | `compute_qoq_deltas`, `compute_4q_trend` | Shared `_prepare_timeseries_base` helper for sort + groupby-collapse |
| 3 | `_merge_momentum_metrics` | Stream groupby loop instead of materializing to dict |
| 4 | `compute_pit_percentiles` | Pre-process dates once per group in `_prepare_percentile_dates`, per-column Fenwick walk in `_percentile_column` |
| 5 | `_sector_onehot` | Replace 11 string comparisons with `pd.get_dummies` + `reindex` |

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

#### 3.3 Bug fix — 2026-05-02: missing id columns in no-momentum artifact batch

- **Bug**: `_build_artifact_features_for_audit` loaded batch rows with only
  `["availability_date"] + strict_cols + xsectional_cols`. The no-momentum
  streaming target builder (`_build_streaming_targets_no_momentum`) expects
  `BESTTICKER`, `SECTOR`, `SignalType`, and `call_entry_date` in the batch
  DataFrame to recompute QoQ deltas and PIT percentiles. Missing columns
  caused `KeyError: "['BESTTICKER', 'SignalType'] not in index"`.
- **Fix**: Added these id columns to `compare_cols` in
  `_build_artifact_features_for_audit` using `dict.fromkeys` to preserve
  `strict_cols + xsectional_cols` order without duplicating names.

#### 3.4 Performance fixes from review — 2026-05-01

Performance fixes applied per `docs/review/phase3.md`:

| # | Location | Fix |
|---|---|---|
| 1 | `_compare_one_streaming_date` (per-date body) | Removed `gc.collect()` call after deleting local DataFrames. Python's reference counting frees pandas DataFrames immediately. Added a single `gc.collect()` after the streaming loop in `run_streaming_vs_batch_test` instead. |
| 2 | `_compute_target_pit_percentiles` | Swapped loop nesting so groups are outer, columns are inner. Each group's history is fetched once and the Fenwick tree is built once per group instead of once per column per group. |
| 3 | `run_streaming_vs_batch_test` / `_compare_one_streaming_date` | Pre-grouped the batch DataFrame by `availability_date` before the per-date loop (`dict(list(batch.groupby("availability_date")))`), replacing repeated `batch.loc[batch["availability_date"] == d]` linear scans with O(1) dict lookups. |
| 4 | `_sample_availability_dates` | Removed redundant `sorted()` call — `pd.DatetimeIndex.unique()` already returns sorted values. |

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
- Implementation fix — 2026-05-01: LightGBM random-search samples were carrying `numpy` scalar types (`np.int64`, `np.float64`) into the manifest append path. The per-model frozen hparam JSON was already sanitized, but `hparams_manifest.json` reused the unsanitized dict and crashed with `TypeError: Object of type int64 is not JSON serializable`. `backtest/splits.py` now normalizes sampled params and recursively converts numpy/pandas scalar containers before writing both frozen hparam files and the manifest.
- Speed fix — 2026-05-01: `tune_lightgbm` and `tune_xgboost` inner trial loops (30 random hyperparameter combinations per CV fold) were serial, leaving most CPU cores idle and memory underutilized. Extracted `_fit_lgbm_trial` / `_fit_xgb_trial` module-level helpers and parallelized the per-fold trial evaluation with `joblib.Parallel(n_jobs=-1)` (default `loky` backend). Each trial's model still uses `n_jobs=1` to avoid nested OpenMP oversubscription. Ridge tuning is left serial (6 alphas × 5 folds, each fit is <100ms — overhead of Parallel would dominate).
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
- Final sync — 2026-05-01: before a signature is built, `backtest/splits.py`
  updates `data/cache/prices/_manifest.json` with a deterministic fingerprint
  over current price parquet names, sizes, and mtimes. This forced sync keeps
  joblib cache invalidation tied to local price-file changes even when the
  original price loader was not rerun.
- The old mtime-only parquet cache is replaced; joblib manages storage in `results/cache/joblib/`.

#### 4.3 Label availability purge (G11)

- For horizon `h`, keep training rows only when `target_available_date_h <= fold_train_end`
- `target_available_date_h` = the date when the forward return is fully realized and readable from the price table (using the roll-forward rule in audit item 9)
- Any sample whose `entry_date` is in the training fold but whose `target_available_date_h` crosses into validation/test is removed from training
- Apply purge to both the outer main backtest and inner `TimeSeriesSplit`

#### 4.4 Partial-period handling

- Samples with `target_available_date_h > price_data_end` -> `right_censored_target`; exclude them from all IC / training / test / portfolio return calculations
- Report censored counts separately by universe x horizon

#### 4.6 Performance fixes from review — 2026-05-01

Performance recommendations 1-6 from `docs/review/phase4.md` implemented in `backtest/splits.py` and `backtest/model.py`. Items 7 and 8 are skipped (higher effort, ticker/horizon parallelism).

| # | Location | Fix |
|---|---|---|
| 1 | `_prepare_fold_sample` in `backtest/model.py` | Convert `df[availability_col].iloc[train_idx]` to datetime once; reuse for sorting, DatetimeIndex, min, max. Removed 3 redundant conversions. |
| 2 | `generate_folds` in `backtest/splits.py` | Removed `df.reset_index(drop=True)` which copied the full DataFrame. Extracted only needed date columns via `.to_numpy()`. |
| 3 | `tune_ridge` in `backtest/splits.py` | Extracted `_fit_ridge_alpha` module-level helper. Parallelized 6-alpha grid search per fold with `Parallel(n_jobs=-1)`. Each Ridge uses `n_jobs=1` to avoid nested parallelism. |
| 4 | `filter_model_sample` in `backtest/model.py` | Removed the first `df.copy()` that created a redundant deep copy before filtering. The `_orig_df_index` column assignment is safe on the original df. |
| 5 | `_compute_ticker_forward_return_matches` in `backtest/splits.py` | Removed `.copy()` after `.sort_values()` — `sort_values` already returns a new DataFrame. |
| 6 | `purge_train_for_horizon` in `backtest/splits.py` | Added dtype guard: if `df[date_col]` is already `datetime64[ns]`, use `.to_numpy()` directly instead of calling `pd.to_datetime` again. |

Validation: `python -m compileall backtest/` passes; `python -m pytest tests/ -v` — 47/47 passed.

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
  9. **Entry quote lookup**: Use ``_next_valid_quote(ticker, planned_entry_date, max_forward_days=5)`` which scans up to 5 **trading days** forward in the price-derived trading calendar. ``planned_entry_date`` is the rebalance date. If no valid close is found within 5 trading days, mark ``skip_no_entry_quote`` and remove the ticker from weights, re-scaling remaining weights to gross=2.0.
  10. **Exit quote lookup**: For each entered position, ``planned_exit_date`` is the next rebalance date. Use ``_next_valid_quote(ticker, planned_exit_date, max_forward_days=5)`` with the same 5-trading-day tolerance. If no valid close is found, mark ``right_censored_no_exit_quote``.
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

**5.5 `backtest/robustness.py`** (680+ lines) — Robustness check categories:
1. **Subperiod analysis**: IC and quintile L/S recomputed for pre-2020 / 2020–2022 / 2023–2026
2. **Sector neutralization**: within-sector quintile ranking, then merged across sectors
3. **Market-cap buckets**: `shares × adj_close` at T-1, four buckets (mega/large/mid/small), coverage flag at <85%
4. **Weighting scheme**: equal-weight vs ATC-score-weighted quintile portfolios
5. **Block bootstrap**: monthly block bootstrap (21-day blocks, 2000 resamples) for Sharpe 95% CI
6. **OFAT quantile cutoff**: top-{5, 10, 20, 50, 100} stocks per leg
7. **OFAT transaction cost**: {3, 5, 7, 10} bps post-cost Sharpe (consumes portfolio daily returns)
8. **OFAT cadence/lookback**: daily `{1,3}`, weekly `{3,5,10}`, monthly `{15,21,30}` portfolio reruns
9. **Weekly timing**: Monday-close versus Friday-close weekly rebalance comparison
10. **Label purge gap**: `{0,5,21}` business-day gap sample-impact audit, with gap=0 as the negative control
11. **R8 beta window IC check**: recompute idiosyncratic residual with beta windows `{40,60,90}`; if IC sign flips for any SignalType, flag for removal

Outputs: `results/robustness/robustness_subperiod_ic.parquet`, `results/robustness/robustness_subperiod_quintile.parquet`, `results/robustness/robustness_sector_neutral.parquet`, `results/robustness/robustness_mcap_buckets.parquet`, `results/robustness/robustness_weighting.parquet`, `results/robustness/robustness_ofat_quantile.parquet`, `results/robustness/robustness_ofat_cost.parquet`, `results/robustness/robustness_ofat_lookback.parquet`, `results/robustness/robustness_weekly_timing.parquet`, `results/robustness/robustness_label_purge_gap.parquet`, `results/robustness/robustness_bootstrap_ci.json`, `results/robustness/robustness_r8_beta_window.parquet`.

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
- 5c (portfolio) auto-detects available OOS prediction files via ``glob("results/audit/oos_pred_*_{tier}_h*.parquet")``, builds a JSON job list, and runs ``backtest.portfolio_batch`` once per tier so prices/calendar/coverage are cached across every OOS file × universe × cadence job.
- 5d (robustness) iterates over every requested universe.
- Empty PIT universes (e.g., RU3K) produce explicit ``{module}_{universe}_coverage_constrained.json`` sentinel artifacts by default; use ``--skip-empty-universe`` to skip silently.
- ``--from-phase 5`` / ``--stop-at-phase 5`` wired. Dry-run (``--dry-run --tier both``) enumerates all sub-tasks.

**Post-review fixes — 2026-04-30.** Corrected the shared PIT universe filter for IC/quintile/portfolio, fixed portfolio daily accounting so weekly/monthly runs produce daily P&L rather than rebalance-only rows, added trade log and universe coverage artifacts, changed forward-return entry matching from a 10-calendar-day tolerance to a maximum 5-business-day entry gap, made the audit checklist distinguish Phase 3 pass from Phase 4/5 pending items, fixed sector-neutral robustness to compare `Total` vs `Total`, and batched failed-ticker log writes in `data/load_shares.py`.

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
| A2 | Correctness | `backtest/portfolio.py` `PortfolioSimulator.run` (~line 515, ~line 565) | Added `_next_valid_quote()` helper that scans up to 5 **trading days** forward in the trading-calendar-derived list for a valid close (not calendar days). Entry and exit quote lookups now use this helper instead of checking only the single planned date. ``skip_reason`` is ``None`` for normal trades (was ``""``). ``PortfolioResult`` gained a ``trade_log_violations`` field. ``validate_trade_log()`` is called after the simulation loop; violations are persisted to ``trade_execution_violations_{suffix}.parquet``. Plan §4.5 requires this tolerance before marking ``skip_no_entry_quote`` / ``right_censored_no_exit_quote``. Actual entry/exit dates and prices are recorded in ``trade_execution_log.parquet``. |
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

#### 5.10 Refactor test-plan hardening — 2026-04-30

Before starting the behavior-preserving codespace reconstruction in
`ideas/rewrite.md`, the unit-test plan was tightened so the new tests preserve
current contracts without locking in accidental or unsafe assumptions.

| ID | Type | Change |
|---|---|---|
| RUT-01 | Reproducibility | Replaced `/tmp` as the only rewrite-baseline location with persistent baselines under `tests/fixtures/rewrite_baseline/` when small enough, or `results/audit/rewrite_baseline/` for larger local artifacts. Added a required `manifest.json` with git SHA, commands, package versions, input metadata, hashes, and creation time. |
| RUT-02 | Test isolation | Required every file-writing unit test to use `tmp_path`; tests must not read or write real `data/cache/`, `results/`, or user baseline paths. |
| RUT-03 | Failed-log semantics | Added explicit same-ticker edge cases for shares and price failed-log helpers: share failures replace the previous row, later share successes clear stale failures, later share failures are retained, and price batches clear stale successes while appending same-batch non-success rows. |
| RUT-04 | Spearman NaN behavior | Changed the Spearman test from generic "NaN propagation" to pandas-compatible pairwise NaN dropping, plus a separate insufficient-valid-pairs test that expects NaN. This matches the existing `pd.Series.corr(..., method="spearman")` behavior. |
| RUT-05 | Observed-bucket schema | Split equity-curve column testing into full-bucket and sparse-bucket fixtures so tests do not require unobserved `bucket_*` columns when `pd.qcut(..., duplicates="drop")` removes buckets. |
| RUT-06 | Portfolio stats edge cases | Added one-row equity and zero-volatility Sharpe tests to preserve current summary behavior (`{}` for too-short equity; NaN Sharpe when volatility is zero). |
| RUT-07 | Lock wording | Renamed failed-log update tests from "atomic" to "locked read-apply-write" unless the implementation later uses temp-file plus `Path.replace()` crash-atomic writes. |
| RUT-08 | Test-runner gate | Added `python -m pytest --version` as Step 1.0; pytest must be installed in the project environment before writing or running red tests. |

---

### Phase 5.11 - Code Restructuring (2026-04-30)

Per the restructuring plan in `ideas/rewrite/`, a behavior-preserving refactor was executed across 14 modules (A1–C7). See `ideas/rewrite/main.md` for the full module index and cross-cutting rules.

**Summary of changes:**

| Step | Module | Files Changed | Summary |
|------|--------|--------------|---------|
| 1 | A1 | New: `data/_utils.py`, `tests/test_data_utils.py` | Shared leaf module: `FetchResult`, `US_TICKER_RE`, `setup_logger`, `suppress_yfinance_logging`, manifest/failed-log helpers, `exponential_backoff`. 22 unit tests. |
| 1 | A2 | New: `backtest/_stats.py`, `tests/test_backtest_stats.py` | Shared leaf module: `spearman`, `make_median_imputer`, `dedup_latest_per_ticker`, `bucket_returns`, `build_equity_curves`, `max_drawdown_from_equity`, `portfolio_stats`. 25 unit tests. |
| 2 | B1 | `data/load_prices.py`, `load_shares.py`, `load_universes.py`, `load_signals.py` | Replaced ~200 lines of duplicated code with imports from `data/_utils.py`. |
| 3 | B2 | `backtest/single_feature_ic.py`, `quintile.py`, `robustness.py` | Removed 3 redundant `_filter_to_universe` wrappers; use `backtest.universe.filter_to_universe` directly. |
| 3 | B3 | `backtest/model.py`, `splits.py`, `quintile.py`, `robustness.py`, `single_feature_ic.py` | Replaced ~170 lines of duplicated stats helpers with imports from `backtest/_stats.py`. |
| 3 | B4 | `backtest/splits.py` (+ 4 call sites) | Added `ensure_forward_returns()` — single entry point for forward-return column guarantees with filtered-DataFrame misuse detection. |
| 4 | C5 | `backtest/splits.py` | Split `compute_forward_returns` into 6 helpers. Bumped `FORWARD_RETURNS_CACHE_VERSION` to 2. Added deterministic SHA-256 source hash. |
| 4 | C1 | `backtest/portfolio.py` | Split `PortfolioSimulator.run()` (was ~470 lines → 92 lines) into `SimulationState` dataclass + 11 helpers. Fixed bug: `_append_entered_trade_records` return-value handling. |
| 4 | C2 | `features/engineer.py` | Split `compute_momentum_features` into 8 helpers. |
| 4 | C4 | `backtest/model.py` | Split `_run_one_fold` and `run_walk_forward` into `FoldSample` dataclass + 7 helpers. |
| 4 | C3 | `backtest/robustness.py` | Split `run_all_robustness` (was ~239 lines → ~40 lines) into 10 section helpers. |
| 4 | C6 | `backtest/portfolio.py`, `backtest/model.py` | Extracted 9 CLI helpers from `main()` functions. |
| 4 | C7 | `features/audit.py` | Split `run_streaming_vs_batch_test` into 4 helpers. |

**Deviations from the rewrite plan:**

| Module | Deviation | Reason |
|--------|-----------|--------|
| A1 | Added internal `_utc_now_iso()` to `data/_utils.py` | `save_manifest` needs UTC timestamps; leaf module cannot import from `data.config`. Mirrors existing helper that was already duplicated in `load_prices.py`/`load_shares.py`. |
| C1 | `_lookup_tradeable_entries` gained `lookback` parameter | Skip records store `"horizon_days": lookback` — the parameter is required to preserve this field. |
| C1 | `_rebalance_lookback_start` takes `pd.DatetimeIndex` not `list` | Integer-based indexing requires the DatetimeIndex type. |
| C4 | `_fit_select_predict_with_audit` gained `df`, `availability_col` parameters | Stretch tier's `_purged_time_series_splits` needs access to these for inner CV. |
| C5 | Minor `n_with_prices` counting difference in edge case | When price files exist but no valid matches are found (all gaps >3 BD), `n_with_prices` may differ from the old code. Forward return values are identical; only the warning log message may be slightly less precise. |

**RU3K / SP1500 static-universe fallback (2026-04-30):**

`expand_to_month_grid` in `data/load_universes.py` was changed to use the
earliest available iShares snapshot as a static universe for all earlier months,
instead of logging per-month gaps and returning an empty PIT. A single
coverage gap is logged per component noting the survivorship-bias assumption.
This matches the requirement's explicit allowance (§6.3): "If you cannot obtain
one, document the survivorship-bias caveat explicitly in your research PDF and
apply your model on a 'current-membership' universe — your reported alpha will
be an upper bound."

After rebuild:
- RU3K PIT: 506,268 rows, 2,583 tickers (static, today's IWV)
- SP1500 PIT: 289,185 rows, ~1,500 tickers (IJH + IJR static + SP500 PIT rolling)

**Bug found and fixed during refactoring:**

- **C1**: `_process_rebalance` had `state.trade_records.extend(self._append_entered_trade_records(...) or [])` but the function appends directly to `state.trade_records` and returns `None`. Fixed to call the function directly without `extend`.

**Verification (2026-04-30):**
- `python -m compileall data features backtest reports run_all.py` — clean
- `python -m pytest tests/ -v` — 47/47 passed
- `python run_all.py --dry-run --tier enhanced --from-phase 5 --stop-at-phase 5` — all sub-tasks enumerated correctly
- Import smoke tests pass for all new public symbols

---

#### 5.12 Performance fixes from review — 2026-05-01

Seven performance recommendations from `docs/review/phase5.md` (items 1-6, 8) were applied across 6 backtest modules:

| # | Type | Location | Fix |
|---|---|---|---|
| 1 | I/O | `backtest/splits.py` (new `_cached_read_features`); call sites in `single_feature_ic.py`, `quintile.py`, `robustness.py` | Module-level `@lru_cache(maxsize=1)` caches the features parquet in memory after the first read, eliminating redundant ~1 GB reads across IC, quintile, and robustness modules. |
| 2 | Speed | `backtest/single_feature_ic.py` | Replaced three separate worker functions (`_ic_combo_worker`, `_ic_yearly_worker`, `_ic_sector_worker`) with a single `_ic_full_worker` that computes monthly IC, derives yearly IC from monthly, and computes sector IC in one pass. Eliminates ~67% of ProcessPoolExecutor submissions. |
| 3 | Speed | `backtest/portfolio.py` `_process_rebalance` | Pre-index signals as `pd.DatetimeIndex` in `run()`, then use `self._signals.loc[lookback_start:date]` for O(log N) date-range lookups vs O(N) boolean mask scan. |
| 4 | Speed | `backtest/single_feature_ic.py` `_compute_cross_sectional_ic`, `_compute_sector_monthly_ic` | Replaced `dedup_latest_per_ticker(gdf)` (which calls `sort_values` + `drop_duplicates`) with `gdf.groupby("BESTTICKER").last()` — O(n) and preserves input's chronological ordering assertion. |
| 5 | Speed | `backtest/_stats.py` `spearman()` | Replaced `pd.Series(a).corr(pd.Series(b), method="spearman")` with `scipy.stats.spearmanr(a, b, nan_policy="omit")` — 3-5x faster, no Series allocation. |
| 6 | I/O | `backtest/portfolio.py` `compute_capacity_metrics` | Added `price_table: pd.DataFrame | None = None` parameter. When provided, extracts `dollar_volume` from the pre-loaded price table via per-ticker mask filtering instead of loading individual ticker parquet files. |
| 8 | Speed | `backtest/robustness.py` `block_bootstrap` | Replaced Python list accumulation (`boot_means: list[float] = []` with `.append()` in loop + `np.array()` at end) with pre-allocated `np.empty(n_boot)` and `np.concatenate(blocks)` for the resampled block. |

Validation: `python -m compileall backtest/` — no syntax errors. `python -m pytest tests/ -v` — 47/47 passed. Item 7 (vectorize daily PnL loop) was deferred to §5.13.

#### 5.13 Vectorize daily PnL loop — 2026-05-01

Item 7 from `docs/review/phase5.md` and the full plan in `docs/review/faster.md`.
Implemented in 6 steps (Steps 0–5):

**Step 0 — Extract `_advance_daily_state` free function + 12 unit tests.**

- `backtest/portfolio.py`: extracted `_advance_daily_state()` — a pure function over
  plain Python dicts that implements the gap-state machine (setdefault →
  censored-check → valid-PnL → missing-streaks) independently of the simulator.
  The existing `_compute_daily_pnl` method became a thin wrapper that slices the
  close matrix into per-ticker dicts and delegates.
- `tests/test_portfolio_pnl.py`: 12 unit tests covering empty weights, valid
  both days, 1d/2d/3d gaps, censored skip, setdefault initialization,
  missing-without-last-price edge case, mixed tickers, cumulative gap recovery,
  zero-weight, and short positions.

**Step 1 — Build flat price matrix once (upstream of the simulation loop).**

- After `_load_calendar_prices_and_coverage` returns `close_matrix`, build:
  `_ticker_to_idx: dict[str, int]`, `_idx_to_ticker: list[str]`,
  `_day_index: dict[pd.Timestamp, int]`, `_prices_2d: np.ndarray`
  (shape `(n_tickers, n_days)` float64, `close_matrix.values.T`).
  Two integer-indexed lookups replace per-day `.loc` + `.reindex`.

**Step 2 — Add numpy array fields to `SimulationState`.**

- New fields: `last_prices_arr`, `missing_streaks_arr`, `is_censored_arr`,
  `weights_arr`, `entry_dates_arr` (all `np.ndarray | None`, default `None`).
- Initialized in `run()` to size `n_tickers`.
- Populated in `_process_rebalance`: reset all to neutral, then set active
  positions by ticker index.

**Step 3 — `_advance_daily_state_numpy` with 4 boolean-mask passes.**

- `backtest/portfolio.py`: new `_advance_daily_state_numpy()` — pure numpy
  vectorized kernel. Four passes over boolean masks:
  1. `setdefault` — `np.isnan(last_prices) & np.isfinite(p_today) & (p_today > 0)`
  2. Valid PnL — `np.isfinite(p_next) & (p_next > 0) & ...` → `np.dot(weights, ret)`
  3. Missing streaks — increment, classify 1d/2d/long_gap
  4. Gap accounting — `np.abs(weights[mask]).sum()` for weight columns
- Censored tickers handled in a separate sub-pass before the active-PnL passes.
- `_compute_daily_pnl` auto-selects: uses the vectorized path when
  `state.last_prices_arr is not None` (production), falls back to the dict
  kernel otherwise (tests/bootstrap).

**Step 4 — Dual-implementation equivalence tests.**

- `tests/test_portfolio_pnl.py`: 10 additional parametrized tests
  (tests 13–22) that run identical scenarios through both
  `_advance_daily_state` (dict) and `_advance_daily_state_numpy` (numpy),
  asserting identical `pnl`, `gross_exposure`, `net_exposure`, `n_positions`,
  all four gap-weight columns, gap counts, `long_gap_tickers` membership,
  and mutated state arrays.
- Full test suite: 29 tests, all passing.

**Step 5 — Numba `@njit(cache=True)` single fused loop.**

- `backtest/portfolio.py`: new `_advance_daily_state_numba` — `@njit`-compiled
  single loop over all tickers. Fuses all 4 passes into one register-level
  iteration with zero temporary boolean arrays.
- `_advance_daily_state_numba_wrapper`: Python wrapper that allocates a
  pre-sized `long_gap_out` array, calls the numba kernel, and builds the
  result dict from the returned scalars + long-gap indices.
- `_compute_daily_pnl_vectorized` auto-selects numba when `_NUMBA_AVAILABLE`,
  falls back to pure numpy.
- `tests/test_portfolio_pnl.py`: 7 numba-specific equivalence tests
  (class `TestNumbaEquivalence`, skipped when numba not installed) covering
  the same scenarios as the dual tests.

**Expected speedup (SP500, ~2500 trading days, ~320K ticker-day iterations):**

| Stage | PnL loop time |
|---|---|
| Before (Python for-loop + dicts + `.loc`) | ~5–15 s |
| After Steps 1+2+3 (pure numpy) | ~0.5–2 s |
| After Steps 1+2+3+5 (numba) | < 0.3 s |

**Deliverables:**

| File | Change |
|---|---|
| `backtest/portfolio.py` | `_advance_daily_state`, `_advance_daily_state_numpy`, `_advance_daily_state_numba`, `_advance_daily_state_numba_wrapper` free functions; `SimulationState` numpy array fields; `_compute_daily_pnl` dual-path dispatch; `_compute_daily_pnl_vectorized` numba/pure-numpy dispatch; flat price matrix + index mappings in `run()`; array init + populate in `_process_rebalance` |
| `tests/test_portfolio_pnl.py` | 29 tests: 12 dict semantics + 10 dual numpy + 7 numba equivalence |

Validation: `python -m compileall backtest/` — clean. `python -m pytest tests/ -v` — 76/76 passed (47 existing + 29 new).


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

`reports/charts.py` reads `results/` parquet/JSON files and generates PNG images
using matplotlib. Output directory: `reports/figures/`.

- Cumulative L/S equity curve for each universe
- IC heatmap (feature x horizon)
- Quintile spread chart
- Drawdown chart
- Rolling Sharpe chart
- Turnover bar chart

#### 7.2 PDF (15-25 pages) — route B: markdown → HTML → weasyprint

The PDF is generated from `docs/report.md` rather than built programmatically.
This keeps the report prose in a single source of truth and lets charts be
embedded via standard markdown image syntax.

**Pipeline:**

```
docs/report.md   ──→  markdown  ──→  HTML  ──→  weasyprint  ──→  reports/final_report.pdf
                          ↑
reports/figures/*.png ───┘  (referenced as ![](figures/xxx.png) in the .md)
```

**Step-by-step in `reports/pdf.py`:**

1. Read `docs/report.md` as a string.
2. Use `markdown.markdown(text, extensions=["tables", "fenced_code"])` to convert
   to a complete HTML document (or inject the HTML body into a template with
   `<style>` for PDF-appropriate CSS: page size A4, font-size 11pt, margins,
   page-break rules).
3. Use `weasyprint.HTML(string=html).write_pdf(output_path)` to render the PDF.
4. The 1-page audit checklist (`results/audit/lookahead_checklist_onepager.md`)
   is appended as the last section of the report.

**Why route B over reportlab/fpdf2:**
- `docs/report.md` already contains all methodology, transparency statements,
  and result placeholders. Route B removes duplication: the .md is the report.
- Charts are referenced naturally in markdown (`![](figures/ic_heatmap.png)`)
  and weasyprint resolves relative paths from the HTML's base directory.
- Tables in `docs/report.md` (GFM pipe tables) render natively via the
  `tables` extension.

**`reports/charts.py` implementation notes:**
- One function per chart type, each loads data from `results/`, builds the
  matplotlib figure, and saves to `reports/figures/<name>.png` at 150 dpi.
- Consistent style via a shared `set_style()` helper (font size, color cycle,
  figure size).

---

### Phase 8 - Reproducibility Entry Point

- One command, `python run_all.py`: raw zip -> universes -> prices -> shares -> features -> audit tests -> walk-forward -> experiments -> charts -> PDF
- `python run_all.py --tier enhanced` runs only the Enhanced tier; `--tier both` runs both Enhanced and Stretch
- `python run_all.py --from-phase N` / `--stop-at-phase N` for partial reruns; `--force` to ignore cached artifacts; `--dry-run` to preview
- Only Phase 1 raw-loader artifacts are skipped from representative outputs;
  Phase 2 and later are intentionally rerun by `run_all.py` so a single stale
  artifact cannot hide missing tier/model/universe/cadence jobs. Phase 1
  price/shares loaders remain self-resuming through their own manifest files.
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

---

### Logic Review Fixes - 2026-05-01

Static review against `docs/requirement.md` found several logic issues after the
full-scale code was written. The code was patched as follows:

- **Strict pre-event price anchoring**: `features/engineer.py` now anchors
  momentum features to the last trading day strictly before the call calendar
  date. BMO calls no longer include the call-day close in `pre_event_*`
  features.
- **No target-based OOS signal filtering**: `backtest/model.py` now emits
  predictions for every test event with available features. Missing future
  returns are ignored only in IC/MSE evaluation, not in portfolio signal
  generation.
- **Volume-aware tradability**: `backtest/splits.py` and
  `backtest/portfolio.py` now require `volume > 0` in addition to a valid close
  for forward-return entry/exit bars and portfolio execution.
- **IC requirement coverage**: `backtest/single_feature_ic.py` now writes
  explicit yearly IC and sector IC outputs for every feature/horizon/SignalType
  combination, not only aggregate monthly summaries or ATC-only sector splits.
- **T-0 bucket construction**: `backtest/quintile.py` now forms quintile/decile
  buckets by `availability_date`, preventing later events in the same month from
  affecting earlier event ranks.
- **Artifact compatibility**: `reports/charts.py` now discovers model-tagged
  portfolio outputs such as `ridge_enhanced_predh5d`; `backtest/model.py` writes
  compatibility fit-audit logs; `features/audit.py` writes
  `validation_summary.json` for Phase 6.
- **Universe transparency**: `data/load_universes.py` wording now matches the
  implemented iShares fallback: static early-history snapshots are allowed only
  with explicit survivorship-bias coverage-gap records.

#### Follow-up fixes applied after the final static logic review

The final pass applied the remaining review recommendations before the
full-scale run artifacts are interpreted:

- **Recovered short quote gaps now affect PnL**: `backtest/portfolio.py` keeps
  per-position last valid prices through 1- and 2-trading-day quote gaps. Those
  days are recorded as bounded forward-fill days with 0% interim return, and
  the cumulative return is booked once the quote resumes. Gaps longer than two
  trading days remain censored from headline PnL and flow into the 30-day
  recovery audit.
- **Portfolio quote-state resets at every rebalance**: each fresh portfolio now
  resets entry prices, last prices, missing-streak counters, and censored
  tickers so stale gap state cannot leak from the previous holdings.
- **Audit verification is fail-closed**: `run_all.py` Phase 6 now fails when
  `validation_summary.json` contains failed or pending checks, when trade-log
  violations exist, when the trade-violation parquet is missing or unreadable,
  or when fit-audit logs contain violations or malformed JSON lines.
- **Forward-return isolation audit checks generated features**:
  `features/audit.py` now rebuilds a no-momentum feature sample and checks the
  generated non-identifier columns for raw `Return_*`, `forward_return_*`, and
  `target_available_date_*` leakage.
- **Sector IC is a true within-sector time series**:
  `backtest/single_feature_ic.py` computes monthly IC inside each sector and
  summarizes that sector-month IC series, instead of computing one all-history
  sector correlation.
- **Decile baseline reports all required legs**: `backtest/quintile.py` now
  records long-only, short-only, and long-short stats for the
  `ATCClassifierScore` decile baseline.
- **Robustness drawdowns use equity curves**: `backtest/robustness.py` now
  passes cumulative `cum_long_short` equity to max-drawdown calculations in the
  subperiod, OFAT cutoff, and weighting-scheme checks.
- **Raw zip input is accepted directly**: `data/load_signals.py` extracts
  `Earnings_ATC_until_2026-04-21.csv` from the configured raw zip when the CSV
  is not already present.
- **`_in_universe` is excluded from model features**: `backtest/splits.py`
  treats the PIT membership marker as an identifier column, not as a constant
  predictive feature after filtering.
- **Charts are regenerated instead of trusting stale PNGs**:
  `reports/charts.py` no longer skips chart functions just because the expected
  output file already exists.

Validation after these changes:

```bash
python -m compileall data/load_signals.py features/audit.py backtest/splits.py backtest/single_feature_ic.py backtest/quintile.py backtest/robustness.py backtest/portfolio.py reports/charts.py run_all.py
python -m pytest -q
```

Both commands passed locally (`47 passed` for pytest).

### Final Pipeline Synchronization Pass - 2026-05-01

This pass records the last consistency and speed changes applied before the
final report is generated.

- **Unified 5-trading-day quote roll-forward**:
  `backtest/splits.py` uses `MAX_ENTRY_GAP_BDAYS = 5` for forward-return entry
  matching and bumped the forward-return cache schema to version 3.
  `backtest/portfolio.py` uses `MAX_QUOTE_FORWARD_DAYS = 5` for portfolio
  entry and exit quote lookup. The previous 3-day wording in the plan/report
  has been replaced with the 5-day standard.
- **Forward-return price-manifest forced sync**:
  `_build_cache_signature()` now calls a lightweight manifest sync that writes
  `_price_files_fingerprint`, `_price_files_count`, and
  `_price_files_synced_at` into `data/cache/prices/_manifest.json` from the
  current parquet files before hashing the manifest. This prevents stale
  manifests from masking price-cache changes.
- **Phase 2 PIT percentile grouping speedup**:
  `compute_pit_percentiles()` now groups once by `(SECTOR, SignalType)` and
  computes all available percentile columns within that group pass. The strict
  historical percentile semantics are unchanged.
- **Phase 5 portfolio batch runner**:
  New `backtest/portfolio_batch.py` runs a JSON list of portfolio jobs in one
  process. `PortfolioSimulator` caches loaded close matrices, calendars,
  rebalance dates, and universe coverage by ticker set, date range, cadence,
  lookback, and weekly timing. `run_all.py` writes
  `results/cache/portfolio_jobs_{tier}.json` and invokes the batch runner once
  per tier.
- **`run_all.py` overwrite semantics**:
  `_skip()` now only skips Phase 1 loader artifacts. Later phases always rerun
  from the orchestrator so a representative artifact cannot incorrectly skip an
  incomplete tier/model/universe/cadence matrix. `1.2_universes` now requires
  all three PIT universe files (`sp500`, `sp1500`, `ru3k`) when deciding
  whether the universe loader can be skipped.
- **Phase 6 audit clarification and stronger checks**:
  The audit concern was generic "last run wins" evidence. For example,
  `trade_execution_violations.parquet` could be empty after the last portfolio
  run even if an earlier scenario-specific violation file was non-empty.
  Phase 6 now scans every `trade_execution_violations*.parquet`, requires at
  least one `fit_audit_log_*.jsonl`, and fails on pending/failed validation
  checks, non-empty trade violations, malformed fit logs, or fit violations.
- **Robustness coverage expanded**:
  `backtest/robustness.py` now persists:
  `robustness_ofat_lookback.parquet` for daily `{1,3}`, weekly `{3,5,10}`,
  monthly `{15,21,30}`; `robustness_weekly_timing.parquet` for Monday vs
  Friday weekly timing; `robustness_label_purge_gap.parquet` for `{0,5,21}`
  business-day purge-gap sample impact; and
  `robustness_r8_beta_window.parquet` after recomputing idiosyncratic residuals
  with beta windows `{40,60,90}`.
- **Weekly rebalance timing exposed**:
  `backtest/portfolio.py` adds `--weekly-day {monday,friday}`. Monday remains
  the default; Friday is used for robustness and is included in output suffixes
  only when non-default.

Validation after this pass:

```bash
python -m compileall backtest features run_all.py
python -m pytest -q
python run_all.py --dry-run --from-phase 5 --stop-at-phase 5 --tier enhanced --universes sp500 --skip-empty-universe
python run_all.py --from-phase 6 --stop-at-phase 6 --tier enhanced --universes sp500 --skip-empty-universe
```

All commands passed locally; pytest reports `47 passed`.

---

### Post-Review Fixes — 2026-05-01

Systematic code review across all 9 phases (`docs/review/phase_0.md` through `phase_8.md`). All recommended changes applied. See `docs/report.md` §10.0.14 for the full per-phase change log.

Key fixes by severity:

**CRITICAL:**
- Sharpe annualization: quintile/decile and subperiod robustness now use `252/horizon` (was `252`, overstating Sharpe by `sqrt(h)`)
- PDF chart images: weasyprint `HTML()` now has `base_url` so embedded images resolve correctly

**HIGH:**
- Rolling Sharpe window: 12 → 252 periods (daily-indexed equity curve)
- Block bootstrap: `block_size=3` → `block_size=1` per plan spec
- Test-fold assignment uses `availability_date` instead of `call_entry_date`
- `--dry-run` extended to all phases (0-8)
- Consistent empty-universe sentinel files across Phases 4/5

**Audit completeness (Phase 6):**
- Checklist items 7-9 now resolved at runtime (not stuck at PEND)
- `validation_summary.json` includes Phase 4/5 entries
- `sample_size_by_quarter` enriched with universe × signal_type × quarter breakdown and `low_sample_flag`
- Per-check evidence paths; item 10 labeled `ENFORCED AT BUILD TIME`

**Other:**
- Ticker normalization (dot→dash) in signals loader
- Crash-atomic manifest writes; jitter in exponential backoff
- `delisting_exit_used` skip-reason in portfolio + audit validation
- Dead code removal (`assert_feature_parity`, `IDENTIFIER_COLS`, `_FEATURES_14`)
- README filled with reproduction command, dependencies, runtime, artifact paths
- `REPORT_PDF` from config; manifest coverage check before skipping loaders

Validation: `python -m pytest tests/ -v` — 47/47 passed. `python -m compileall` — clean. `python run_all.py --dry-run` — all phases enumerate.

### IC dedup sort-order fix — 2026-05-01

- **Bug**: `backtest/single_feature_ic.py` `_compute_cross_sectional_ic` (line 122) and `_compute_sector_monthly_ic` (line 164) asserted `gdf["call_entry_date"].is_monotonic_increasing` before using `groupby("BESTTICKER").last()` to deduplicate. The features parquet has 40,484 rows out of chronological order (1.5% of 2.74M; `features/engineer.py` does not sort by `call_entry_date`). sp500 happened to dodge the assertion because its unsorted groups fell below `min_samples` after SignalType/NaN filtering; sp1500 hit one.
- **Fix**: replaced the assertion + `groupby.last()` with `gdf.sort_values("call_entry_date").groupby("BESTTICKER").last()` in both functions. Semantically equivalent regardless of input order.

### Memory + Cache Fixes — 2026-05-01

Three issues surfaced by an OOM failure in Phase 5 and a review of the Phase 1 cache-resumption logic.

**M1 — Phase 5.1 OOM: cap workers and drop columns before ProcessPoolExecutor fork.**

- **Location**: `backtest/single_feature_ic.py` `run_single_feature_ic` (lines 321, 325–331).
- **Symptom**: `run_all.py --from-phase 5` OOM-killed during `5a IC (enhanced, sp1500)` with 2 GB swap. Machine: 28 CPU cores, 25 GB RAM. The full features DataFrame (~2.7M rows × ~100 cols, ~3–5 GB) was shared with 28 forked workers via a module-level `_IC_GLOBAL_DF`. Each worker's filtering triggered copy-on-write page duplication, exhausting physical memory.
- **Fix 1**: Capped default `_n_jobs` at 2 (`min(os.cpu_count() or 4, 2)`) and made `run_all.py` pass `--n-jobs 2` for Phase 5a/5b.
- **Fix 2**: Before setting `_IC_GLOBAL_DF`, select only the columns needed by the IC workers (`SignalType`, `year_month`, `call_entry_date`, `BESTTICKER`, `SECTOR`, the 14 short-list features, and 5 `forward_return_*` columns) — roughly 24 columns instead of ~100. The full filtered frame is replaced and `_cached_read_features` is cleared before the `ProcessPoolExecutor` fork so child processes do not inherit unused feature columns.
- **Fix 3**: `run_all.py` defaults Phase 5 outer universe/module fan-out to one subprocess at a time (`--max-workers 1`) on the 28-core / 25GB machine. This prevents `3 universes x inner workers` from multiplying resident feature frames and triggering the Linux OOM killer.

**M2 — Phase 1.4 `is_cached_fresh` did not cache "empty" results.**

- **Location**: `data/load_shares.py` `is_cached_fresh` (lines 258–289).
- **Symptom**: Delisted/invalid tickers (status `empty`) hit the yfinance API on every invocation, wasting calls and re-triggering yfinance ERROR noise. The function checked parquet file existence before manifest status, so `empty` tickers (no parquet file) were never considered fresh.
- **Fix**: Rewrote to match `data/load_prices.py`'s `is_cached_fresh`: check manifest entry first, dispatch by status. `success` still requires a parquet file with `last_date` within 14 days. `empty` now cached 90 days (these tickers won't come back). `error` never cached.

**M3 — `_manifest_coverage_sufficient` never returned `True`.**

- **Location**: `run_all.py` `_manifest_coverage_sufficient` (line 156).
- **Symptom**: Iterated `manifest.values()` counting `status == "success"`. Manifest structure `{"updated_at": "<iso>", "tickers": {...}}` means `manifest.values()` includes a string; `.get("status")` on it raises `AttributeError`, caught by bare `except`, always returns `False`. Phase 1.3/1.4 could **never** be skipped even with a fully-populated cache.
- **Fix**: Changed to `manifest.get("tickers", {}).values()`.

### Portfolio Batch Parallelization — 2026-05-01

The Phase 5.4 portfolio batch runner (`backtest/portfolio_batch.py`) processed all 135 jobs sequentially in a single process despite 28 available CPU cores, making it the longest serial step in the pipeline (~67 minutes). Parallelized with `ProcessPoolExecutor`.

**Changes to `backtest/portfolio_batch.py`:**

- New `_run_single_job(job)` module-level function — self-contained worker entry point for `ProcessPoolExecutor`. Each worker independently loads signals, creates a `PortfolioSimulator`, runs the simulation loop, and persists results. Logging is configured inside the worker so subprocess output is not silenced.
- `run_jobs()` now accepts `max_workers` (default `min(os.cpu_count(), 2, len(jobs))`) and `parallel` parameters. When `parallel=True` and `len(jobs) > 1`, jobs are dispatched via `ProcessPoolExecutor` with `as_completed` collection. When `parallel=False` or only one job, the original sequential path is used (including simulator-sharing cache across same-universe jobs).
- Added `--max-workers` and `--no-parallel` CLI arguments.

**Changes to `run_all.py`:**

- Phase 5c portfolio batch command now passes `--max-workers` from `args.max_workers` (default 1) to the batch runner, consistent with the Phase 5 memory-safe outer fan-out.

**Design constraints for `max_workers` selection:**

1. **Memory** (hardest): each subprocess loads its own price matrix (~32 MB for close matrix, ~50–100 MB for enhanced features, ~400–500 MB for stretch tier). 8 workers at stretch tier = ~4–5 GB.
2. **Disk I/O**: all workers read the same 2583 price parquet files and features parquet simultaneously. Beyond ~8–12 workers, I/O contention dominates.
3. **CPU**: simulation inner loop is numpy/numba (releases GIL), so ProcessPoolExecutor uses true multi-core. Diminishing returns after physical core count.
4. **Shared-file writes**: `audit/trade_execution_log.parquet` and similar convenience copies may race between workers; suffix-specific outputs are safe.

Default cap is 2 workers for standalone `portfolio_batch.py`; `run_all.py` passes the Phase 5 outer cap (default 1) unless explicitly overridden. This favors finishing reliably on the 25 GB / 28-core dev machine over saturating CPU.

### Phase 5d Robustness Performance Optimizations — 2026-05-01

The original `run_all_robustness()` ran 12 analysis modules mostly with redundant data passes, sequential inner loops, and a duplicate portfolio simulation. Six optimizations were applied (P0=high impact/low effort, P1=high impact/medium effort):

**P0-1 — Pre-split df by SignalType (avoid ~700 redundant boolean-mask filters).**
- `run_all_robustness()` now builds `signal_dfs = {st: df[df["SignalType"] == st].copy() for st in SIGNAL_TYPES}` once after loading features.
- Sections 1–5, 10, and 11 receive pre-filtered data (`signal_dfs[sig_type]` or `total_df`) instead of the full `df`, eliminating O(N) boolean-mask + copy on every loop iteration.
- Functions updated: `_monthly_ic_by_subperiod` (removed `signal_type` param), `run_subperiod_ic`, `run_subperiod_quintile`, all 9 section-wrapper functions, and `_baseline_total_signals`.

**P0-2 — Merge OFAT lookback + weekly timing into one section, eliminate duplicate `weekly_5d_monday`.**
- New `_run_portfolio_combined_section()` replaces `_run_ofat_lookback_section` and `_run_weekly_timing_section`.
- Shares a single `PortfolioSimulator` instance across all 11 unique combos (9 OFAT + 2 weekly timing, with `weekly_5d_monday` computed only once).
- Saves one full portfolio backtest + one price-table load (~5–15 minutes on SP500).

**P0-3 — Reuse existing `pre_event_idio_resid_5d` column for beta_window=60 baseline.**
- `_run_beta_window_section` now uses `df["pre_event_idio_resid_5d"]` (already computed during Phase 2) instead of recomputing `compute_momentum_features` with `beta_window=60`.
- Results for `beta_window=40` and `beta_window=90` are cached to `results/cache/momentum_beta_{window}.parquet` and loaded on subsequent runs.
- Eliminates 1 of 3 `compute_momentum_features` calls (and 2 of 3 after the first run).

**P0-4 — Memory-safe section-level parallelism.**
- `_n_jobs` defaults to 1 for full robustness sections; the subperiod IC/quintile inner loops default to 2 threads.
- `run_all.py` Phase 5d passes `--n-jobs 1` by default via `--phase5-robustness-workers`.
- This avoids running market-cap matrices, beta-window momentum recomputation, bootstrap, and portfolio reruns concurrently in the same robustness process.

**P1-1 — Parallelize inner 350-combo loops in subperiod IC and quintile.**
- Added module-level global `_ROBUSTNESS_DFS` and worker functions `_subperiod_ic_worker` / `_subperiod_quintile_worker` (same pattern as `quintile.py`).
- `run_subperiod_ic` and `run_subperiod_quintile` now accept optional `n_jobs` and dispatch the 14×5×5=350 combos via `ThreadPoolExecutor(max_workers=2)` by default with `as_completed`.
- Chose `ThreadPoolExecutor` over `ProcessPoolExecutor` to avoid fork-safety issues inside the section-level `ThreadPoolExecutor`.

**P1-2 — Vectorize market-cap bucket inner per-ticker searchsorted loop.**
- `assign_market_cap_buckets` now builds aligned `(n_tickers × n_dates)` price and shares matrices after pre-loading per-ticker histories. Matrices are forward-filled once with `pd.DataFrame.ffill(axis=1)`.
- The per-date inner loop uses vectorised indexing (`px_mat[:, date_pos]`, `sh_mat[:, date_pos]`) to get all tickers' T-1 values in a single operation, then `np.quantile` for cross-sectional cutoffs and `np.select` for bucket assignment.
- Eliminates ~1M Python-level `np.searchsorted` + dict-lookup calls per SP500 run (~500 tickers × ~2000 unique dates).
- Logs matrix dimensions and memory footprint at startup.

**Involved files:** `backtest/robustness.py` (bulk of changes), `run_all.py` (Phase 5d `--n-jobs 1` by default).
