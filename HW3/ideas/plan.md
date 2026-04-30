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
availability_date = max(call_entry_date, ingest_entry_date)
```

- `INGESTDATEUTC` is used only for availability calculation and never enters features
- Implementation rolls weekends forward with `np.busday_offset`; exact exchange holidays are handled later by price-aware execution / return joins
- Counterexample to avoid: `INGESTDATEUTC = 2020-03-15 22:30 UTC`; naive date max gives 03-15, while the correct date is 03-16

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

---

### Phase 3 - Automated Look-Ahead Tests

`features/audit.py`

#### 3.1 Streaming vs batch regression test

- Fit once on the full sample -> compare the feature matrix against per-day streaming fits
- Tolerance: `np.allclose(rtol=1e-9, atol=1e-12)`
- Start with a unit test on a small ticker subset (~50 tickers) and a 1-year window, then run full-sample regression

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
- Run Ridge / LightGBM / XGBoost once each -> `frozen_hparams_<model>.json`
- The same `feature tier x model x horizon` shares one hparam set across all three universes

#### 4.2 Main walk-forward backtest

- Start: 2020Q1; quarterly step through 2026Q2
- Initial training set = 2010-01 through 2019-12, expanding quarterly up to the day before each test fold
- Read frozen hparams; no further tuning
- Inside each fold:
  - Fit imputation / scaling only on the training fold
  - Stretch tier: run `LassoCV(cv=TimeSeriesSplit(n_splits=3))` for column selection; keep nonzero-coefficient columns; if >200 columns remain, truncate to top 200 by `|coef|`; write selected feature names to the audit log

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

- One command, `make all` or `python run_all.py`: raw zip -> universes -> prices -> shares -> features -> audit tests -> walk-forward -> experiments -> charts -> PDF
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
