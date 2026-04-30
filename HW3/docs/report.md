# Report Draft

> This file holds all content intended for the final PDF report: methodology choices, decision rationale, alternatives considered, transparency statements, risks, and future work.
> The concrete execution checklist is in `ideas/plan.md`; project requirements and hard constraints are in `docs/requirement.md`.
> Experiment results (numbers, figures, tables) will be filled into the relevant sections after the experiments finish.

---

## 1. Project Objective and Hard Constraints

Backtest the ProntoNLP ATC earnings-call signals on three universes (S&P 500 / S&P 1500 / Russell 3000) without look-ahead bias, and provide a production-grade deployment recommendation across three rebalance cadences (daily / weekly / monthly).

**Hard constraint**: any look-ahead bias directly fails the project. This is the top priority. Whenever methodology choices conflict with analytical convenience or better-looking results, this constraint wins.

---

## 2. Data Description

### 2.1 Raw data
- Single CSV: `Earnings_ATC_until_2026-04-21.csv` ~ 4.5 GB / 2.74M rows / 609 columns
- Date range: 2010-01-04 -> 2026-04-21
- 9 `SignalType` slices (Total / Presentation / Question / Answer / Executives / CEO / CFO / Analysts / delete)
- 11 GICS sectors / 100 countries (US ~ 55%)

### 2.2 SignalType selection
- **Primary analysis uses `Total`**: it covers the full transcript, gives the largest sample, and matches production desk usage
- **Four slices for comparison**: `CEO / CFO / Analysts / Executives`; run the same IC + baseline decile experiments for differentiated analysis
- **Not run for now**: `Presentation / Question / Answer`; these slices focus on prepared remarks vs Q&A and are left for future work, with effort prioritized on the critical path of cadence x universe x tier x model
- **Complex models and final portfolio construction still use `Total`**; if a speaker slice materially outperforms `Total` on IC / decile spread, propose an ensemble in future work

### 2.3 Scope decision: no `COUNTRY == 'US'` pre-filter
- The dataset's `COUNTRY` field is a snapshot from extraction time and can contain metadata mislabels
- Filtering directly on `COUNTRY == 'US'` can wrongly remove true index constituents, such as ADR-listed companies headquartered outside the US
- Therefore, first keep all cleaned signals with `SignalType != 'delete'`, then use PIT universe membership for final sample selection
- `COUNTRY` is used only for QA checks, not as an eligibility condition

### 2.4 Universe construction: all three tiers are PIT, no survivorship snapshot
- **SP500**: Wikipedia historical constituent changes + current constituents -> add/remove ledger with effective dates -> **daily** PIT
  - Wikipedia is preferred over ETF data because the S&P 500 historical change table is public, well maintained, and can recover daily add/remove events
- **SP1500**: iShares `IJH` (SP400) + `IJR` (SP600) monthly holdings + SP500 PIT -> **monthly**
- **RU3K**: iShares `IWV` monthly holdings -> **monthly**
  - Russell 3000 reconstitutes only once per year in June, so monthly granularity is much finer than the index update frequency
- **Fallback discipline**: when iShares historical holdings cannot be downloaded automatically or coverage is insufficient, the one-command README path first consumes frozen snapshots stored in the repository; if those are also missing, mark the universe as `coverage_constrained` and **never** silently replace it with the current constituent snapshot
- **PDF transparency statement** (in methodology): using ETF holdings as approximate PIT constituents for SP1500/RU3K may introduce small sampling bias because they are not official FTSE Russell / S&P constituent files; this is a data-source transparency issue and **does not create look-ahead bias**

### 2.5 Phase 1 smoke-test log - 2026-04-29
All Phase 1 smoke tests passed after the fixes below.

| Area | Smoke command / check | Final result |
|---|---|---|
| Signals | temporary-output run of `data.load_signals.run(data/sample.csv, chunksize=4)` plus full rebuild `python -m data.load_signals --chunksize 100000` | PASS: 2,740,437 raw rows read; 2,231 `delete` rows dropped; 2,738,206 rows written; main cache 448 columns; slim cache 43 columns |
| SP500 PIT | `python -m data.load_universes --only sp500` plus `members_at()` boundary assertions | PASS: `data/cache/universes/sp500_pit.parquet` has 2,289,593 rows; future dates raise; pre-history dates return an empty set |
| SP1500 PIT | `python -m data.load_universes --only sp1500` | PASS: SP500 component written; missing IJH/IJR history is explicitly logged as 390 component gaps, not silently patched |
| RU3K PIT | `python -m data.load_universes --only ru3k` | PASS: no historical IWV snapshots available, so the PIT parquet is empty and 195 monthly coverage gaps are logged |
| Prices | `python -m data.load_prices --only AAPL --start 2024-01-01 --end 2024-01-10 --sleep 0` and smoke batch `python -m data.load_prices --limit 20 --no-parallel` | PASS: `AAPL.parquet` has 6 rows with `date, adj_close, volume`; 20-ticker smoke fetches 10 new tickers (8 success + 2 empty for delisted ABMD/ACAS) with zero errors and no yfinance log spam |
| Shares | `python -m data.load_shares --only AAPL --start 2024-01-01 --end 2024-01-10 --sleep 0` | PASS: `AAPL.parquet` has 4 rows with positive shares; manifest status is `success`; coverage report written |
| Final verification | `python -m compileall data` and a consolidated Python assertion script over all Phase 1 artifacts | PASS |

### 2.6 Phase 1 full-universe final run - 2026-04-29

After the smoke tests, the same loaders ran end-to-end across the full ticker universe; results are consistent with the smoke logs and form the baseline that Phase 2 builds on.

| Area | Final-run output | Numbers |
|---|---|---|
| Signals | `data/cache/signals.parquet`, `data/cache/signals_slim.parquet` | 2,738,206 rows, 8 SignalType slices (`Total / Executives / Presentation / Answer / Question / Analysts / CEO / CFO`); main 448 cols (29 ID + 12 EventScore + 1 ATCClassifierScore + 1 `call_hour_utc` + 405 non-Fluff/Filler `AspectTheme_*`); slim 43 cols; zero `call_hour_utc = -1` rows; 17,636 unique BESTTICKERs (193 nulls) |
| SP500 PIT | `sp500_pit.parquet` | 2,289,593 rows, 817 unique tickers across history, 4,520 daily dates from 2009-01-01 to 2026-04-29 |
| SP1500 PIT | `sp1500_pit.parquet` | 98,996 rows, 807 unique tickers, 195 month-ends 2010-01-31 → 2026-03-31. **The SP1500 PIT currently equals the SP500 monthly slice**: only the today snapshot of IJH and IJR was retrievable, so all historical mid/small-cap months are logged as `IJH:` / `IJR:` gaps (390 rows) in `universe_coverage_gaps.csv`, not silently patched |
| RU3K PIT | `ru3k_pit.parquet` | 0 rows; 195 monthly gap rows logged in `universe_coverage_gaps.csv` (no historical IWV snapshots are available without a paid feed); RU3K is therefore tagged `coverage_constrained` until snapshots accumulate month-over-month going forward |
| Prices | `data/cache/prices/{ticker}.parquet` | 817 universe tickers attempted: 667 success / 150 empty / 0 error; 667 parquet files on disk; `failed_tickers.csv` contains 150 `empty` rows; AAPL/A series cover 2009-01-02 → 2026-04-29 with 4,357 daily rows; ABBV starts 2013-01-02 (post-IPO) |
| Shares | `data/cache/shares/{ticker}.parquet` | 817 attempted: 780 success / 37 empty / 0 error; AAPL series has 336 rows from 2015-10-28 — Yahoo's `get_shares_full` history begins ~2015 for most tickers |
| Marketcap coverage | `results/audit/marketcap_capacity_coverage.csv` | 34 `(universe, year)` rows for SP500 + SP1500 over 2010–2026. **2010–2014 fail the 70% floor** (coverage 0.0–0.9% — the yfinance shares endpoint has no pre-2015 history); **2015–2026 are above floor** (77–85%). RU3K rows are not yet emitted because the RU3K PIT is empty. Practical consequence: market-cap buckets are quantitative for 2015+ only; the 2010–2014 subperiod is qualitative-robustness-only and is flagged in red in the PDF |

Resolved implementation issues:
- **Signal chunk schema drift**: small chunk smoke runs failed because pandas inferred different dtypes across chunks (`QTR_YEAR` and identifier-like columns could flip between numeric and string), causing `pyarrow` schema mismatch. Fixed in `data/load_signals.py` by reading identifier columns as strings and coercing count / score / aspect columns to stable dtypes before each Parquet write.
- **Date-only call timestamps**: four persisted rows had `call_hour_utc = -1` because mixed date and datetime strings were not parsed robustly. Fixed `parse_call_hour()` to use mixed-format timestamp parsing, then rebuilt both signal caches from the raw CSV.
- **SP1500 coverage masking**: combining IJH/IJR rows with SP500 before monthly expansion allowed the SP500 component to hide missing mid/small-cap snapshots. Fixed `build_sp1500()` to expand IJH and IJR separately, combine after expansion, and tag component-level coverage gaps.
- **Repeated audit-run duplication**: repeated universe smoke runs appended duplicate coverage-gap rows. Fixed `write_coverage_gaps()` to write unique rows idempotently.
- **Stale price/share failure rows**: after a failed sandboxed fetch followed by a successful network-enabled rerun, `failed_tickers.csv` still contained the earlier miss. Fixed price and shares loaders so a later success clears stale failure rows for that ticker.
- **Price loader performance and error rate**: the original `data/load_prices.py` downloaded one ticker per yfinance call (~18k tickers when including all signal BESTTICKERs, of which ~5,600 were non-US numeric codes that could never succeed). This produced >80% failure rates and hours of wasted retries. Fixed with four changes: (1) `collect_tickers()` defaults to universe-only (817 tickers), with signal tickers gated behind `--include-signal-tickers` and filtered to US-format symbols via `^[A-Z]{1,5}([.\-][A-Z])?$`; (2) batch download — up to 50 tickers per yfinance call using `yf.download("AAPL MSFT ...")`, reducing ~800 API calls to ~16; (3) `ThreadPoolExecutor` with 4 workers for parallel batch fetching; (4) delisted/invalid tickers classified as `empty` immediately, with no retry backoff wasted on them.
- **yfinance logger noise**: yfinance logs `"possibly delisted; no timezone found"` at ERROR level for every delisted or invalid ticker, even though the loader handles them gracefully. yfinance also re-configures its own handler on import, defeating any suppression set at module load time. Fixed by setting `yfinance` and `peewee` loggers to `CRITICAL` inside `_download_batch()` immediately after the lazy `import yfinance`.
- **`empty` results re-fetched every run**: the original `is_cached_fresh()` only recognized `success` manifest entries, so delisted tickers hit the network on every invocation and re-triggered yfinance's ERROR noise. Fixed by caching `empty` entries with a 90-day TTL (delisted stocks won't return); `error` entries are still never cached so transient failures retry.
- **Performance: duplicate Arrow conversion in signal loader**: `stream_chunks` was converting both `main_df` and `slim_df` to Arrow tables via separate `pa.Table.from_pandas()` calls per chunk. Since slim cols are a strict subset of main cols, slim_table is now derived from main_table via `.select()`, cutting one full copy + serialization per chunk. For a 4.5 GB CSV streamed in ~28 chunks, this roughly halves the Arrow conversion overhead.
- **Performance: O(n^2) `pd.concat` in price loader's `update_failed_log`**: each failed ticker triggered a separate `pd.concat([existing, row])` that copied the full failure DataFrame, yielding quadratic growth over thousands of tickers. Fixed by collecting all new failure rows in a list and concatenating once per batch. Also replaced the per-result `existing[existing["ticker"] != res.ticker]` filter with a single `~isin(succeeded)` pass.
- **Performance: per-ticker `append` in universe construction**: `build_sp500_pit` and `expand_to_month_grid` used nested `for t in state: rows.append((d, t))` loops, building ~2.2M tuples one `append` call at a time. Replaced with `rows.extend((d, t) for t in state)` to reduce Python method-call overhead.
- **Performance: per-column aspect coercion in signal loader**: `coerce_chunk_types` was calling `pd.to_numeric` on each of ~400 AspectTheme columns individually, incurring Python→C boundary crossing overhead on every column. Now batch-converts all aspect columns at once via `chunk[mask].apply(pd.to_numeric, errors="coerce")`, reducing ~400 calls per chunk to a single DataFrame-level operation.

Network note: iShares and Yahoo Finance smoke tests require live network access. The first non-escalated iShares/yfinance attempts failed only because DNS/network access was sandboxed; the same commands passed with approved network access.

### 2.7 Phase 2 smoke-test log — 2026-04-29

All Phase 2 features are implemented and the full Enhanced artifact has been rebuilt. One `python -m features.engineer` call produces the 85-column Enhanced feature matrix (or 490 columns for Stretch tier) with progress bars and per-stage timing.

| Stage | Smoke command / check | Final result |
|---|---|---|
| Timestamps | `compute_timestamps(df)` | PASS: business-day `call_entry_date`, `ingest_entry_date`, `availability_date`; `availability_date >= call_entry_date` holds for all rows |
| Row features | `compute_row_features(df)` | PASS: 60 columns — 1 headline + 4 EventScore + 15 per-Aspect + 27 per-Theme + 2 call-length + 11 sector one-hot |
| Time-series | `compute_timeseries_features(base)` | PASS: 16 columns — grouped by `(BESTTICKER, SignalType)` with strict prior availability date; same-day rows do not chain |
| PIT percentiles | `compute_pit_percentiles(base)` | PASS: 6 columns — exact historical percentiles inside `(SECTOR, SignalType)` with `availability_date < call_entry_date`; values are `NaN` or in [0, 1] |
| Momentum (vectorized) | `compute_momentum_features(base)` | PASS: 3 columns — 21d pre-event return / sector-relative / 5d idiosyncratic residual; sector returns use median; rolling beta is shifted to end at T-5; per-ticker `merge_asof` with 10-day tolerance |
| Stretch tier | `build_features(df, tier="stretch")` | PASS: 405 `AspectTheme_*` columns correctly aligned via `_row_id` join; zero row duplication |
| Exclusion list | Assertion check | PASS: `QTR_YEAR`, `INGESTDATEUTC`, `Return_*d` absent from feature columns |
| Full pipeline | `python -m features.engineer --tier enhanced --output results/features_enhanced.parquet` | PASS: 2,738,206 rows x 91 columns in 85.1s; output row order matches `signals.parquet`; sector one-hot valid; no forbidden target/date columns |

**Deviation from plan — sector returns use median.** Some yfinance price series for micro-cap tickers contain extreme daily returns (e.g., +475% single-day), which caused equal-weight sector average returns to explode (21d sector returns exceeding +4,000,000%). Sector daily return aggregation was switched from mean to median, which is robust to these outliers while preserving the central tendency of sector movements. Stock-level n-day returns use the price-ratio method (`adj_close / adj_close.shift(n) - 1`) and are unaffected.

**Deviation from plan — vectorized momentum.** The plan described a per-row loop for momentum features. Implementation uses a fully vectorized design: pre-compute daily returns, sector returns, and rolling beta for all tickers; then join with events via `merge_asof`. This is ~10× faster than the per-row approach.

**Correctness fixes after review.** The initial Phase 2 artifact used calendar-day AMC dates, self-inclusive PIT percentiles, same-day QoQ chains, and beta estimates ending at T-1. The corrected implementation rolls weekends forward, uses strict prior availability for QoQ and PIT features, shifts beta to T-5, and restores output row order after sorted computations.

**Pandas 3.0 workaround.** `pd.merge_asof(by=)` is broken in pandas 3.0.2. Merging is done per-ticker without the `by` parameter, which adds negligible overhead given ~400–17K tickers.

**Full-dataset runtime.** The corrected full Enhanced rebuild took 85.1s: timestamps 3.0s, row features 15.7s, time-series 3.4s, strict PIT percentiles 54.4s, momentum 6.8s. Strict PIT percentiles are now the dominant Phase 2 cost.

**Performance fixes applied after initial full-run hang (2026-04-29).** The first full-dataset attempt hung at the sector-returns step. Seven algorithmic/correctness issues were identified and fixed:

| # | Issue | Fix |
|---|---|---|
| 1 | Per-ticker `df[df[tkr_col] == tkr]` on 2.74M rows × 17,636 tickers | `groupby().first().to_dict()` — single O(n) pass |
| 2 | Per-row Python `np.polyfit` in `_compute_4q_slope` | Closed-form OLS slope `(-3·lag3 - lag2 + lag1 + 3·cur) / 10` |
| 3 | Per-row `calendar < target` for 2.74M events × 4000-day calendar | `DatetimeIndex.searchsorted()` — vectorized binary search |
| 4 | Per-ticker `events[events[tkr] == t]` boolean mask for each of 657 tickers | `dict(list(df.groupby()))` — pre-grouped O(1) lookup |
| 5 | 15-column QoQ diff loop with repeated groupby re-indexing | One strict date-block calculation across all QoQ columns |
| 6 | `merge_asof` dtype mismatch (`datetime64[ns]` vs `[s]`) | `astype("datetime64[ns]")` on both keys |
| 7 | Self-inclusive PIT expanding rank | Exact strict-history Fenwick percentile by `(SECTOR, SignalType)` |

These fixes eliminated the O(n²) and O(n×m) patterns that caused the full run to hang while preserving the strict no-look-ahead rules.

---

## 3. Methodology and Design Choices

### 3.1 Three-tier strategy progression: Baseline / Enhanced / Stretch
| Tier | Features | Model |
|---|---|---|
| Baseline | `ATCClassifierScore` only | No model; run IC + decile L/S by universe x horizon x SignalType |
| Enhanced | 85 engineered features | Run Ridge / LightGBM / XGBoost |
| Stretch | Enhanced + selected sparse-matrix interactions from ~405 non-Fluff/Filler `AspectTheme_*` cells | Run Ridge / LightGBM / XGBoost |

**Why run and report all three tiers**: the grader needs to see the marginal value of complexity. Baseline is the honest sanity check. If Enhanced/Stretch do not beat it, the complex model added no value; if they do, the results quantify whether feature engineering was worthwhile.

### 3.2 Model selection: run Ridge / LightGBM / XGBoost
- **Core motivation for comparing all three**: the difference between a linear model (Ridge) and tree models (LightGBM, XGBoost) on sparse high-dimensional features is itself an analysis dimension
  - Ridge is usually stable on clean engineered linear signals
  - LightGBM / XGBoost are better suited to the Stretch tier's 405-dimensional sparse interaction features and can capture nonlinearities
  - Running two GBDT implementations is useful because their splitting and regularization defaults differ substantially; after freezing hyperparameters, the grader can see whether conclusions are consistent
- **Cross-universe hyperparameter discipline**: the same `feature tier x model x horizon` freezes one hparam set and shares it across all three universes. **Never** tune separately to improve one universe, especially RU3K; that would overfit the test set and is explicitly prohibited by audit item 10 in section 3.

### 3.3 Feature-engineering tradeoffs: 85-column Enhanced design
- **85-column upper bound**: enough to cover 7 information dimensions - Aspect / Theme / sentiment / magnitude / time-series / cross-sectional / pre-event momentum - while staying far below the ~405-column Stretch tier; the linear model is less likely to explode and tree models can still handle it
- **Strictly exclude Fluff / Filler**: requirement section 1.4(c) identifies them as noise classes. Keeping them only adds noise. Use them only in an audit sanity check showing "Fluff-only signal IC ~ 0"
- **Per-Aspect / Per-Theme x {totals, net sentiment, magnitude-weighted}**: keep three aggregation views side by side so the model can choose which one carries information; net sentiment often beats absolute counts, while magnitude-weighting gives more weight to "high-importance sentences"
- **The 6 cross-sectional columns must use strict historical windows**: compare against historical events in the same `(SECTOR, SignalType)` bucket with `availability_date < call_entry_date`. Any self-inclusive expanding rank or full-sample rank such as `df.groupby('SECTOR')['ATC'].rank(pct=True)` is leakage.
- **QoQ joins use `(BESTTICKER, SignalType, availability_date)` with a strict prior availability date**: using `QTR_YEAR < QTR_YEAR` can leak forward, and same-day SignalType rows must not chain into each other.

### 3.4 Use pre-event momentum as the surprise proxy; do not join external consensus
- Requirement section 1.7 states that consensus / KPI information is already internalized by `ATCClassifierScore`; joining external IBES/FactSet/Refinitiv estimates is a major look-ahead risk due to estimate revisions, vendor restatements, and point-in-time hygiene
- Therefore, surprise information is derived only from pre-event price momentum: 21d return, sector-relative return, and 5d idiosyncratic residual
- **R8 idiosyncratic beta must be rolling and end at T-5**: estimating beta with full-sample OLS is classic leakage. Use a fixed 60-day rolling window, regression window `[T-65, T-5]`, and exclude the T-4..T-1 interval that overlaps with the residual return.

### 3.5 Position sizing: equal-weight + dollar-neutral as the main convention
- **Equal-weight + dollar-neutral** (long $1 / short $1, equal-weight within each leg) is the main backtest convention
  - Simple, reproducible, and free of hidden optimization
  - Directly comparable with the baseline `ATCClassifierScore`
- **ATC-score-weighted** is a robustness check to test whether weighting by signal strength adds marginal value
- **Beta-neutral / vol-targeting** is left for future work: without a PIT factor model, forcing beta-neutrality can easily introduce leakage through covariance estimation

### 3.6 Sector neutralization: within-sector ranking at the signal stage
- Choose "signal-stage within-sector ranking": rank quintiles inside each GICS sector, then merge the long/short legs
- **Do not choose** after-the-fact exposure neutralization: it requires PIT covariance / factor exposures, raises engineering cost, and increases leak risk; leave it for future work

### 3.7 Cadence decision: run daily / weekly / monthly, then decide
- Run the complete pipeline for all three cadences without preselection. Choose the winner from all evidence:
  - post-cost Sharpe (5 bps one-way)
  - alpha decay curve (which horizons among 1/3/5/10/20d are strongest)
  - turnover
  - capacity proxy (20d ADDV / name count / top-10 concentration / `%ADV consumed` under an AUM grid)
- **Fair cross-cadence comparison**: independent stock selection by cohort -> net by stock after aggregation -> scale to fixed gross=200% / net=0%. Only then are daily/weekly/monthly Sharpe, turnover, and capacity directly comparable
- Fix baseline lookback N values to avoid hidden tuning: daily=1 / weekly=5 / monthly=21 trading days; put N sensitivity into robustness OFAT
- **Rebalance timing discipline**: weekly = Monday close; monthly = first trading-day close of the month; candidate events must satisfy `availability_date <= rebalance_date`. **Never** aggregate all events from "this week" or "this month" after the fact, because that admits future events

### 3.8 Hyperparameter tuning split principles
- **2010-2019 tuning / 2020Q1+ walk-forward**: the tuning sample is pooled PIT training and never touches 2020Q1+
- **Inner loop must use `TimeSeriesSplit(n_splits=5)`**; **never use `KFold` / `StratifiedKFold`** because random CV can use 2018 data to predict 2015, leaking inside the inner loop
- **Share one frozen hparam set across universes**: reuse the same `feature tier x model x horizon` hparams across all three universes. Per-universe tuning equals test-set overfitting
- **Write frozen hparams to `frozen_hparams_<model>.json`**: the main walk-forward reads these files and does not tune again

### 3.9 Freeze the IC short list before experiments to avoid cherry-picking
- Fix the 14-column IC short list before experiments start; see Phase 5.1 in `plan.md`
- All IC tables and heatmaps use this frozen list, preventing post-hoc selection of good-looking features to manufacture high IC
- Count: 14 columns = 12 bullets, because bullet 3 contains three EventsScore variants

### 3.10 Walk-forward label availability purge (G11)
- For horizon `h`, keep training rows only when `target_available_date_h <= fold_train_end`
- Any sample whose `entry_date` falls in the training fold but whose `target_available_date_h` crosses into validation/test is removed from that fold's training set
- This rule is easy to miss: many walk-forward tutorials split only features, not targets, which lets the 21d-horizon training fold "know" what happens over the next 21 days
- Apply the same purge to both the outer main backtest and inner `TimeSeriesSplit`

---

## 4. Look-Ahead Bias Audit

### 4.1 Signed delivery of the official 10-item checklist
The final PDF includes a 1-page audit checklist from `results/audit/lookahead_checklist_onepager.md`. Each item states "rule / code location / evidence file".

### 4.2 Simplified binary BMO/AMC rule for gray-zone times
- **Rule**: `hour < 13 UTC` -> same business-day close entry (BMO); `hour >= 13 UTC` -> next business-day close entry (AMC, with gray-zone times always treated conservatively as AMC)
- **Rationale**:
  1. The spirit of requirement section 3.1 is "be conservative rather than leak"
  2. The gray zone (13-16 UTC) includes some intraday calls; treating them as BMO creates micro-leak risk, while treating them as AMC is clean
  3. Two buckets are less error-prone in code
  4. ATC signals are trained on a 14-day window, so losing one day of exposure is statistically negligible

### 4.3 Correct handling of `INGESTDATEUTC`
`availability_date = max(entry_rule(MOSTIMPORTANTDATEUTC), entry_rule(INGESTDATEUTC))` - both timestamps are first mapped by hour-of-day to the earliest tradable close, then maxed.

**Counterexample to avoid**: `INGESTDATEUTC = 2020-03-15 22:30 UTC` (US market already closed). A naive date max allows entry at the 03-15 close, but the true tradable close is 03-16.

Implementation note: timestamp features roll weekends forward with a business-day calendar; exact exchange holidays are handled later when returns and fills are joined to actual price quotes.

### 4.4 Automated regression test: per-day streaming vs batch
- One-shot full-sample fit and per-day streaming fit must produce feature matrices satisfying `np.allclose(rtol=1e-9, atol=1e-12)`
- Per-day granularity catches most day-level leakage; per-event is too slow, and per-month is not strict enough
- Eight assertion classes (feature parity / fold boundary + label purge / fit call-stack monitoring / PIT universe / forward-return isolation / timestamp boundary fixtures / rebalance eligibility / trade execution log); any failure turns CI red

### 4.5 Corporate-action / delisting handling
- **Entry**: planned entry uses `next_valid_close_on_or_after(planned_entry_date)`; if there is still no valid quote after 3 consecutive trading days, mark `skip_no_entry_quote` and do not open the position
- **Exit**: after a position is opened, planned exit uses `next_valid_close_on_or_after(planned_exit_date)`; if there is still no quote after 3 consecutive trading days, do not delete the whole trade. First mark `right_censored_no_exit_quote` and exclude it from ordinary horizon-return statistics, while summarizing it separately in the delisting / no-exit audit table. If the data source provides a final delisting/merger transaction price, exit at that price and mark `delisting_exit_used`
- **Never** use backfill, fake forward-fill, or price=0. All states go into `trade_execution_log.parquet` and are reported by universe x year

---

## 5. Transparency Caveats (Must Appear in the PDF)

### 5.1 G4. GICS sector classification is approximately PIT
- The dataset's `SECTOR` field is likely a snapshot from extraction time (non-PIT)
- Known GICS changes include REITs being split out of Financials in 2016 and Communication Services being created in 2018
- Estimated affected rows are <1%, but this touches sector-relative percentiles (6 columns), sector one-hot (11 columns), sector-neutral ranking (experiment 5), and sector residual beta (R8)
- **Handling**: accept this approximation, list the most likely affected companies in the PDF (MSFT/GOOG/META/T/VZ + roughly 30 REITs), and run robustness check 5 by rerunning the core Enhanced/Stretch pipeline without sector features to quantify the effect on conclusions
- **R8 same-source assumption**: the 60-day idiosyncratic beta regression window builds sector_eqw_ret from the T-1 sector snapshot on the historical side, rather than rebuilding a PIT sector basket for every historical date; assume sector reclassification contributes negligibly inside a 60-day window

### 5.2 G5. yfinance retroactive adjustment
- yfinance returns retroactively adjusted close prices; historical prices before split/dividend ex-dates are rewritten, which is standard for academic backtests
- Between declaration date and ex-date, the "known but not yet effective" interval can shift returns for a few trading days by a few bps
- Accept this for an academic backtest

### 5.3 G6. yfinance survivorship in the price source
- yfinance often returns empty data for small-cap stocks that were delisted or acquired long ago; this matters most for RU3K
- **Mandatory handling**:
  - `failed_tickers.csv` records fetch status
  - Report `tradeable_members / pit_members` for every universe x rebalance date; flag <90% in red in the PDF
  - Missing entry price -> `skip_no_entry_quote`; missing exit price -> `right_censored_no_exit_quote`; do not silently forward-fill fake data or silently delete the whole trade
- **Do not** fill with a second data source: multi-source stitching can introduce silent bias and is more dangerous than "single source + transparent missingness"

### 5.4 G15. Historical shares-outstanding coverage
- Market-cap buckets depend on PIT `shares_outstanding`
- Tickers missing historical share series are excluded only from market-cap bucket / `%ADV consumed` supplemental analysis and **do not affect main strategy returns**
- Coverage must be reported by universe x year; low-coverage buckets are qualitative reference only
- **Confirmed after Phase 1**: yfinance's `Ticker.get_shares_full` returns no data before ~2015. `results/audit/marketcap_capacity_coverage.csv` shows SP500 / SP1500 coverage at 0.0–0.9% for 2010–2014 (well below the 70% floor) and 77–85% for 2015–2026. **Market-cap buckets and `%ADV-consumed` AUM analysis are therefore quantitative only for 2015 onward**; the 2010–2014 subperiod is qualitative-robustness-only and is flagged in red in the PDF

### 5.5 ETF approximation for SP1500/RU3K PIT constituents
- iShares ETF actual holdings are not identical to official FTSE Russell / S&P constituents; ETF sampling can skip some small-cap names
- This is a data-source transparency issue and **not look-ahead bias**
- Disclose it only in the PDF methodology section
- **Phase 1 status**: iShares does not expose a historical-snapshot API, so `data/load_universes.py` only persists the iShares snapshot for *today*. Until enough monthly snapshots accumulate going forward, the SP1500 PIT is effectively the SP500 monthly slice (390 month-end gaps logged for `IJH` + `IJR`) and the RU3K PIT is empty (195 month-end gaps). All gaps are written to `results/audit/universe_coverage_gaps.csv`; the loader **never** falls back to the current snapshot for past dates. Practical consequence: the SP1500 and RU3K backtests for 2010–2025 are tagged `coverage_constrained` and reported alongside SP500 with explicit coverage disclosure, rather than being silently filled with survivorship-biased data

---

## 6. Uncertainty Quantification (30% of the Rubric)

- **IC tables**: report mean IC / standard t-stat / **Newey-West adjusted t-stat (lag = horizon)** / hit rate
- **Portfolio Sharpe / spread**: use monthly block bootstrap for 95% CI
- **All main conclusions** include confidence intervals or t-stats, not point estimates alone
- **Sample-size threshold**: combinations with quarterly count <100 in `results/audit/sample_size_by_quarter.csv` are automatically marked low-sample and are not used as main PDF conclusions

---

## 7. Finalized Project Decisions (Quick Reference)

| # | Decision Point | Choice | Key Consideration |
|---|---|---|---|
| Q1 | Universe membership | All three tiers PIT; SP500 Wikipedia daily / SP1500 SP400+SP600 monthly / RU3K IWV monthly; no survivorship fallback | Meets sections 3.6 / 6.3 |
| Q2 | Cadence selection | Run daily / weekly / monthly and choose the winner from full evidence | No preselection; avoids confirmation bias |
| Q3 | BMO/AMC gray zone | `hour < 13 UTC` same business-day close; all others next business-day close, conservatively treated as AMC | Prefer conservatism over leakage |
| Q4 | Scope | No `COUNTRY=='US'` pre-filter; select by PIT universe | `COUNTRY` can be mislabeled |
| Q5 | Models | Run Ridge / LightGBM / XGBoost | Linear vs tree-model comparison is an analysis dimension |
| Q6 | Hparam tuning | 2010-2019 tuning + TimeSeriesSplit(5) inner loop, then frozen walk-forward; shared across universes | No random CV / no per-universe tuning |
| Q7 | Position sizing | Equal-weight + dollar-neutral as main convention; ATC-score-weighted robustness; beta-neutral future work | Simple, reproducible, no hidden optimization |
| Q8 | Sector neutralization | Within-sector ranking at the signal stage; after-the-fact exposure neutralization left for future work | Avoid PIT covariance leakage |
| Q9 | Engineered features | 85 columns (60 row-level + 16 time-series + 6 cross-sectional PIT + 3 pre-event momentum); strict Fluff/Filler exclusion | Covers 7 information dimensions and stays far below Stretch ~405 |
| Q10 | SignalType slices | IC + baseline decile for Total/CEO/CFO/Analysts/Executives; complex models still use Total | Speaker ensemble left for future work |
| Q11 | Look-ahead automation | Per-day streaming, `np.allclose(rtol=1e-9, atol=1e-12)`; 8 assertion classes | Per-event is too slow; per-month is not strict enough |
| Q12 | Cadence comparability | Net by stock after aggregation -> fixed gross=200% / net=0% before computing Sharpe | Direct comparison requires the same capital convention |
| Q13 | Market-cap bucket | `adj_close_{T-1} x shares_{T-1}`; same-day universe cross-sectional percentiles | Missing shares only affects bucket analysis, not the main strategy |
| Q14 | Capacity quantification | 20d ADDV / name count / top-10 concentration / `%ADV consumed` AUM grid | Recommended cadence needs quantitative support |

---

## 8. Risks & Limitations (Fill in Numbers After Experiments)

- Data coverage gaps (red-zone list in `results/audit/universe_coverage_gaps.csv`)
- ETF approximation sampling bias for SP1500/RU3K universes
- Single-source yfinance price risk, especially for delisted RU3K samples
- Bias from approximate PIT GICS sector classification in sector-relative features
- 2026Q2 partial censoring (`right_censored_target` sample count)
- Shared hparams across universes: avoids leakage at the cost of per-universe optimality

---

## 9. Future Work

- `Presentation / Question / Answer` slices: study prepared remarks vs Q&A signal differences
- Speaker-slice ensemble: combine CEO / CFO / Analysts with Total
- Beta-neutral / vol-targeting position sizing, conditional on access to a PIT factor model
- After-the-fact exposure neutralization using PIT covariance
- Join IBES consensus after obtaining institutional data access to validate the surprise dimension
- Multi-universe ensemble (SP500 signal + RU3K signal weighted blend)
- Adaptive cross-cadence portfolio: daily when event density is high, weekly when sparse

---

## 10. Experiment Results (Fill After Runs Finish)

### 10.1 Per-universe result summary
- SP500: (fill IC / decile spread / walk-forward Sharpe)
- SP1500: (fill)
- RU3K: (fill)

### 10.2 Deployment recommendation
- Recommended cadence + rationale based on joint evidence from post-cost Sharpe / alpha decay / turnover / capacity
- Recommended model based on Ridge / LGBM / XGB comparison
- Recommended position sizing rule
- Estimated supportable AUM

### 10.3 Robustness conclusions
- Subperiod stability
- Marginal effect of sector neutralization
- Market-cap bucket heterogeneity
- Sector-classification sensitivity
- Equal-weight vs ATC-score-weighted
- 5 required OFAT sensitivities
- R8 beta window sampling
