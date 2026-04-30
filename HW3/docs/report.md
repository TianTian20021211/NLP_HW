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
  - SP500 month-end component uses the **latest SP500 daily PIT snapshot on or before each month-end date**, not the union of all tickers that appeared in SP500 on any trading day that month. This prevents a stock added and removed mid-month from remaining in the month-end snapshot.
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
| PIT percentiles | `compute_pit_percentiles(base)` | PASS: 6 columns — exact historical percentiles inside `(SECTOR, SignalType)` with `availability_date < call_entry_date`; values are `NaN` or in [0, 1]; optional Numba fast path uses the same Fenwick-tree logic |
| Momentum (vectorized) | `compute_momentum_features(base)` | PASS: 3 columns — 21d pre-event return / sector-relative / 5d idiosyncratic residual; sector returns use median; rolling beta is shifted to end at T-5; per-ticker `merge_asof` with 10-day tolerance |
| Stretch tier | `build_features(df, tier="stretch")` | PASS: 405 `AspectTheme_*` columns correctly aligned via `_row_id` join; zero row duplication |
| Exclusion list | Assertion check | PASS: `QTR_YEAR`, `INGESTDATEUTC`, `Return_*d` absent from feature columns |
| Full pipeline | `python -m features.engineer --tier enhanced --output results/features_enhanced.parquet` | PASS: 2,738,206 rows x 91 columns in 85.1s; output row order matches `signals.parquet`; sector one-hot valid; no forbidden target/date columns |

**Deviation from plan — sector returns use median.** Some yfinance price series for micro-cap tickers contain extreme daily returns (e.g., +475% single-day), which caused equal-weight sector average returns to explode (21d sector returns exceeding +4,000,000%). Sector daily return aggregation was switched from mean to median, which is robust to these outliers while preserving the central tendency of sector movements. Stock-level n-day returns use the price-ratio method (`adj_close / adj_close.shift(n) - 1`) and are unaffected.

**Deviation from plan — vectorized momentum.** The plan described a per-row loop for momentum features. Implementation uses a fully vectorized design: pre-compute daily returns, sector returns, and rolling beta for all tickers; then join with events via `merge_asof`. This is ~10× faster than the per-row approach.

**Correctness fixes after review.** The initial Phase 2 artifact used calendar-day AMC dates, self-inclusive PIT percentiles, same-day QoQ chains, and beta estimates ending at T-1. The corrected implementation rolls weekends forward, uses strict prior availability for QoQ and PIT features, shifts beta to T-5, and restores output row order after sorted computations.

**Pandas 3.0 workaround.** `pd.merge_asof(by=)` is broken in pandas 3.0.2. Merging is done per-ticker without the `by` parameter, which adds negligible overhead given ~400–17K tickers.

**Full-dataset runtime.** The corrected full Enhanced rebuild took 85.1s: timestamps 3.0s, row features 15.7s, time-series 3.4s, strict PIT percentiles 54.4s, momentum 6.8s. Strict PIT percentiles are now the dominant Phase 2 cost.

**Post-audit PIT acceleration.** After the full Phase 3 audit exposed PIT percentiles as a single-core bottleneck, `_strict_historical_percentile()` was given an optional `numba.njit(cache=True)` fast path. The compiled implementation keeps the same strict Fenwick-tree algorithm and falls back to the original Python implementation when Numba is unavailable. The Numba path was validated against a naive O(n^2) strict-history percentile implementation on randomized timestamp/value fixtures; the full-run timing table above should be refreshed after the next complete rebuild.

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
| 8 | PIT percentile Python hot loop | Optional Numba-compiled Fenwick core with exact Python fallback |

These fixes eliminated the O(n²) and O(n×m) patterns that caused the full run to hang while preserving the strict no-look-ahead rules.

### 2.8 Reproducibility entry point — `run_all.py`

The full pipeline is orchestrated by `python run_all.py`, which runs Phase 1 → 5 sequentially for the selected tier(s). `run_all.py` no longer skips a phase just because one representative artifact exists; this avoids stale or missing tier/model/universe outputs. Network-heavy Phase 1 loaders still use their own resumable manifests and freshness checks. Key options:
- `--tier enhanced|stretch|both` — select feature tier(s); default `enhanced`
- `--from-phase N` / `--stop-at-phase N` — partial reruns for development
- `--force` — accepted for backward compatibility; the orchestrator already runs selected phases by default
- `--dry-run` — preview without executing

Phase 1 price/shares loaders are self-resuming through their manifests, so interrupted downloads can resume without re-fetching fresh successful tickers. Later phases are rerun by the orchestrator so outputs stay complete across tiers, models, cadences, and universes.

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
- **Cross-universe hyperparameter discipline**: the same `feature tier x model x horizon` freezes one hparam set and shares it across all three universes. Hparam files are stored at ``results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json`` with a manifest at ``results/hparams/hparams_manifest.json`` that records tuning metadata (tier, horizon, CV IC, git hash). **Never** tune separately to improve one universe, especially RU3K; that would overfit the test set and is explicitly prohibited by audit item 10 in section 3.

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
- **Availability column**: uses ``availability_date`` (unified operational rule). For events before ProntoNLP's 2023-07-06 launch date, ``availability_date = call_entry_date + 2 business days`` (simulating the operational processing delay that would have existed if the NLP system had been running historically). For events on or after 2023-07-06, ``availability_date = max(call_entry_date, ingest_entry_date)`` (reflecting the actual ProntoNLP ingest timestamp). This is the strategy-facing availability timestamp used for fold construction, tuning sample filtering, G11 label purge, forward-return entry timing, portfolio signal dates, and rebalance eligibility. The pre-2023 ``+2 business days`` assumption is an operational simulation; the raw ``call_entry_date`` (derived from ``MOSTIMPORTANTDATEUTC``, the actual earnings call publication time) is preserved in the feature output and can be used via ``--availability-col call_entry_date`` for backward compatibility.
- **Inner loop must use `TimeSeriesSplit(n_splits=5)`**; **never use `KFold` / `StratifiedKFold`** because random CV can use 2018 data to predict 2015, leaking inside the inner loop
- **Share one frozen hparam set across universes**: reuse the same `feature tier x model x horizon` hparams across all three universes. Per-universe tuning equals test-set overfitting. Hparam files are stored in tier/horizon-specific directories: ``results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json``.
- **Write frozen hparams to tier/horizon-specific paths**: the main walk-forward reads ``frozen_hparams_{model}.json`` from the appropriate ``{tier}/h{horizon}d`` subdirectory and does not tune again. A manifest at ``results/hparams/hparams_manifest.json`` records tuning metadata.

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

All 10 assertion classes are fully wired:

- **Items 1-6, 10**: Automated in ``features/audit.py`` via ``run_all_audits()``; evidence in ``feature_parity_summary.json``.
- **Item 7 (Fold boundary + label purge)**: ``assert_fold_boundaries()`` is called per fold in ``backtest/model.py _run_one_fold()``, checking ``max(train_feature_date) < test_start`` and ``max(train_target_available_date_h) < test_start``. Fold metadata with per-horizon target dates is persisted to ``results/audit/fold_manifest.parquet``, which also records ``model`` and ``tier``.
- **Item 8 (fit() call-stack monitoring)**: ``monitor_fit_calls()`` context manager monkey-patches ``StandardScaler.fit / SimpleImputer.fit / LassoCV.fit / model.fit`` inside every fold. After the context exits, ``assert_fit_callstack()`` validates that all fit calls used training-fold data. The full log (390 entries across 3 enhanced models) is persisted to ``results/audit/fit_audit_log_*.jsonl``.
- **Item 9 (Trade execution log validation)**: ``validate_trade_log()`` runs after each portfolio simulation in ``backtest/portfolio.py``, checking date ordering and skip-reason consistency. Violations are persisted to ``results/audit/trade_execution_violations.parquet`` with summary counts by violation type logged at WARNING level.
- ``validation_summary.json`` records per-check status (pass/fail/pending) with evidence file paths.

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
- Full-data audit uses all rows in `data/cache/signals.parquet`; `--full-dates N` samples N availability dates for streaming replay. The default final run is `--full-dates 15`.
- Full-regression optimization: after computing the full batch feature matrix, `features.audit` switches the no-momentum full regression to a target-only streaming path. Row-level features are reused because they are row-local; QoQ deltas, 4Q trend, and strict PIT percentiles are recomputed only for sampled-date rows from the row-level history. The strict PIT query still enforces `history_date < cutoff_date`, so future rows cannot enter a sampled-date result.
- Validation after the target-only audit optimization passed on the small subset, an equivalence check against the old full-prefix streaming path on sampled small-subset dates, and the final full-data command with `--full-dates 15` (`110,592` rows compared, zero strict mismatches). In the measured run, the full no-momentum batch took 39.9s and the target-only streaming comparison took about 3.6s for all 15 sampled dates.

### 4.5 Corporate-action / delisting handling
- **Entry**: planned entry uses `next_valid_close_on_or_after(planned_entry_date)`; if there is still no valid quote after 3 consecutive trading days, mark `skip_no_entry_quote` and do not open the position
- **Exit**: after a position is opened, planned exit uses `next_valid_close_on_or_after(planned_exit_date)`; if there is still no quote after 3 consecutive trading days, do not delete the whole trade. First mark `right_censored_no_exit_quote` and exclude it from ordinary horizon-return statistics, while summarizing it separately in the delisting / no-exit audit table. If the data source provides a final delisting/merger transaction price, exit at that price and mark `delisting_exit_used`
- **Never** use backfill, fake forward-fill, or price=0 for entry/exit. All states go into `trade_execution_log.parquet` and are reported by universe x year
- **Trade log columns**: ``trade_execution_log.parquet`` contains ``planned_entry_date`` / ``actual_entry_date`` / ``entry_price`` / ``planned_exit_date`` / ``actual_exit_date`` / ``exit_price`` / ``skip_reason`` (``None`` for normal trades, ``skip_no_entry_quote`` for missing entry, ``right_censored_no_exit_quote`` for missing exit). After each portfolio run, ``validate_trade_log()`` checks date ordering and skip-reason consistency; any violations are persisted to ``trade_execution_violations.parquet``.
- **Daily P&L gap accounting**: When computing daily returns inside the portfolio simulation, every held ticker is classified each day into one of five categories:
  1. Normal return (valid next-day quote) — included in headline P&L
  2. One-day forward-filled quote gap — the price is forward-filled for return continuity; the daily return is 0% and the position remains in headline P&L; the cumulative return is correctly captured when the price reappears
  3. Two-day forward-filled quote gap — same logic extended for two consecutive missing trading days
  4. Long gap recovered (gap > 2 trading days, but a valid quote resumes within 30 trading days) — censored from headline fully-observable returns; reported in a supplemental column
  5. Possible delisting or data unavailable (gap > 2 trading days, no quote within 30 trading days) — censored from headline returns; classified separately
- **Gap reporting columns** added to the daily returns parquet: ``ffill_1d_weight``, ``ffill_2d_weight``, ``long_gap_recovered_weight``, ``possible_delisting_or_unavailable_weight`` plus position counts for each category
- **Gap summary statistics** in the portfolio summary JSON report the share of position-days in each gap category
- A supplemental **30-trading-day recovery audit** runs after the simulation: for each position with a gap longer than 2 days, the code scans forward 30 trading days for a valid quote. Recovered positions are classified separately in the gap accounting output but do not enter headline Sharpe calculations
- **Gap accounting parquet** (``gap_accounting_*.parquet``) persists per-gap-event details including gap date, ticker, weight, and recovery status

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
  - **Daily return gap handling**: For quotes missing during daily P&L computation (as opposed to entry/exit), bounded forward-fill is used for continuity:
    - One- or two-day price gaps are forward-filled (price held constant, net zero return for that day). The cumulative return is correctly captured when the price reappears. The affected weight and position count are reported in the daily returns parquet.
    - Gaps longer than two trading days censor the position from headline fully-observable portfolio returns. A supplemental 30-trading-day recovery audit classifies the gap as either `long_quote_gap_recovered` (if a valid quote resumes) or `possible_delisting_or_data_unavailable` (if evidence remains absent). Recovered-gap returns are reported in supplemental columns only and do not enter headline Sharpe.
- **Do not** fill with a second data source: multi-source stitching can introduce silent bias and is more dangerous than "single source + transparent missingness"

### 5.4 G15. Historical shares-outstanding coverage
- Market-cap buckets depend on PIT `shares_outstanding`
- Tickers missing historical share series are excluded only from market-cap bucket / `%ADV consumed` supplemental analysis and **do not affect main strategy returns**
- Coverage must be reported by universe x year; low-coverage buckets are qualitative reference only
- **Coverage denominator**: the denominator is the union of PIT members on snapshots within the calendar year only (not all of history). Years with no PIT snapshots (e.g., RU3K) get explicit zero-member rows. This prevents early-year coverage from being understated by including future members.
- **Confirmed after Phase 1**: yfinance's `Ticker.get_shares_full` returns no data before ~2015. `results/audit/marketcap_capacity_coverage.csv` shows per-year membership (521–537 tickers for SP500, not a flat 817 all-historical count). SP500 / SP1500 coverage is 0.0–0.6% for 2010–2014 (well below the 70% floor) and 91.9–99.8% for 2015–2026 (above floor). RU3K has no PIT snapshots and is reported as explicit empty rows. **Market-cap buckets and `%ADV-consumed` AUM analysis are therefore quantitative only for 2015 onward**; the 2010–2014 subperiod is qualitative-robustness-only and is flagged in red in the PDF

### 5.5 ETF approximation for SP1500/RU3K PIT constituents
- iShares ETF actual holdings are not identical to official FTSE Russell / S&P constituents; ETF sampling can skip some small-cap names
- This is a data-source transparency issue and **not look-ahead bias**
- Disclose it only in the PDF methodology section
- **Phase 1 status**: iShares does not expose a historical-snapshot API, so `data/load_universes.py` only persists the iShares snapshot for *today*. Until enough monthly snapshots accumulate going forward, the SP1500 PIT is effectively the SP500 monthly slice (390 month-end gaps logged for `IJH` + `IJR`) and the RU3K PIT is empty (195 month-end gaps). All gaps are written to `results/audit/universe_coverage_gaps.csv`; the loader **never** falls back to the current snapshot for past dates. Practical consequence: the SP1500 and RU3K backtests for 2010–2025 are tagged `coverage_constrained` and reported alongside SP500 with explicit coverage disclosure, rather than being silently filled with survivorship-biased data

### 5.6 G16. Unified operational availability-date rule

- **ProntoNLP / ATC's NLP processing system went live on 2023-07-06**. Earnings call transcripts existed publicly before that date (``MOSTIMPORTANTDATEUTC`` dates go back to 2010), but the structured ATC signals were backfilled at the time the system launched.
- ``INGESTDATEUTC`` records when ATC actually processed the call (all >= 2023-07-06), not when the earnings call itself was published. ``MOSTIMPORTANTDATEUTC`` is the actual call date.
- Therefore, a strict ``availability_date = max(call_entry_date, ingest_entry_date)`` labels every pre-2023-07-06 event as unavailable (because all historical ``INGESTDATEUTC`` values are >= 2023-07-06), making the entire 2010-2019 period unavailable for tuning.
- **Unified operational rule**: for calls before 2023-07-06, ``availability_date = call_entry_date + 2 business days`` (simulating the operational processing delay if the NLP system had existed). For calls on or after 2023-07-06, ``availability_date = max(call_entry_date, ingest_entry_date)`` (reflecting the real ingest-based availability).
- This unified ``availability_date`` is used everywhere: feature history, PIT percentiles, forward-return entry, fold construction, tuning sample filtering, label purge, portfolio signal dates, and rebalance eligibility. The raw ``call_entry_date`` and ``ingest_entry_date`` fields are preserved in the feature output for audit and transparency.
- **Impact assessment**: The ``+2 business days`` pre-launch rule is an operational assumption, not an observed ProntoNLP vendor timestamp. It is a practical compromise that avoids making all pre-2023 events unavailable while respecting that a same-day entry is unrealistic for signals that were backfilled. The exact effect on early-period IC and Sharpe depends on the specific call dates and weekend/holiday patterns, but is expected to be small since most earnings calls occur on business days.
- Code location: ``features/engineer.py`` ``compute_timestamps()`` implements the rule. ``backtest/model.py`` CLI ``--availability-col`` defaults to ``availability_date``. ``backtest/splits.py`` ``compute_forward_returns()`` accepts an ``entry_date_col`` parameter (default ``"availability_date"``). ``backtest/portfolio.py`` ``--date-col`` defaults to ``availability_date``. ``run_all.py`` passes ``--date-col availability_date`` in Phase 5 portfolio calls.

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

### 10.0 Experiment infrastructure status — 2026-04-30
All Phase 5 modules are implemented and validated end-to-end on SP500 + SP1500 enhanced features (full pipeline: 14m32s).

| Module | File | Lines | Status |
|---|---|---|---|
| 5.1 IC analysis | `backtest/single_feature_ic.py` | 422 | Validated — 14 features × 5 horizons × 5 SignalTypes; Newey-West t-stats (manual Bartlett kernel); sector-split IC |
| 5.2 Quintile/decile | `backtest/quintile.py` | 422 | Validated — decile baseline + quintile; monthly cross-sectional bucketing; cumulative equity / drawdown / rolling Sharpe |
| 5.3 Walk-forward preds | `backtest/model.py` | 679 | Shared with Phase 4; OOS predictions for Ridge/LightGBM/XGBoost |
| 5.4 Portfolio sim | `backtest/portfolio.py` | 952 | Validated — PIT-filtered OOS portfolios; daily P&L accounting for all cadences; model-tagged outputs; trade log + universe coverage artifacts |
| 5.5 Robustness | `backtest/robustness.py` | 680 | Validated — 8 check categories (subperiod / sector-neutral / mcap / weighting / bootstrap / OFAT quantile / OFAT cost / R8 beta) |

**Key robustness findings (SP500).**
- **Signal decay**: ATCClassifierScore IC drops from 0.062 (pre-2020) to −0.012 (2023–2026)
- **Sector neutralization reduces Sharpe**: raw > sector-neutral for most horizons
- **Score-weighting helps at h=3–10d**: improves Sharpe vs equal-weight; hurts at h=1d and h=20d
- **Top-50 cutoff optimal**: Sharpe 0.84 (h=5d) vs 0.20 for top-5; top-100 is negative
- **R8 CFO warning**: `pre_event_idio_resid_5d` IC negative for all CFO SignalTypes → remove from final model per transparency requirement

Full run: `python run_all.py --from-phase 5`.

### 10.0.1 Post-review correctness and speed fixes — 2026-04-30

After reviewing Phase 0–5 against `docs/requirement.md`, the following issues were fixed in code:

- Replaced per-ticker/stale universe joins with a shared global PIT snapshot filter in `backtest/universe.py`; empty RU3K now keeps 0 rows instead of all rows
- Applied PIT filtering inside portfolio construction, so OOS prediction rows and portfolio weights cannot include non-members on the rebalance date
- Changed weekly/monthly portfolio accounting from rebalance-only rows to true daily P&L rows while retaining rebalance markers
- Added model-tagged portfolio outputs to prevent Ridge/LightGBM/XGBoost overwrites
- Persisted `results/audit/trade_execution_log*.parquet` and `results/audit/universe_coverage_by_date*.csv`
- Changed forward-return entry matching from a 10-calendar-day tolerance to a max 3-business-day entry gap
- Fixed sector-neutral robustness so sector-neutral and raw comparisons both use `SignalType == Total`
- Made `lookahead_checklist_onepager.md` report pending Phase 4/5 items instead of saying `ALL PASSED`
- Batched `data/load_shares.py` failed-ticker log writes to avoid repeated CSV read/write overhead

Targeted validation:

- `python -m compileall data features backtest reports run_all.py` passed
- Empty RU3K smoke: IC and quintile filters kept `0 / 1000` rows
- LightGBM weekly SP500 portfolio smoke: `1582` daily rows, `295` rebalance rows, median row gap `1.0` day, `4788` weight rows, `0` out-of-universe weights
- Trade log validation: `0` violations; universe coverage rows `295`, minimum coverage ratio `0.9091`

### 10.0.2 Round-2 review fixes — 2026-04-30

A second pass surfaced one numerical bug and two large safe speed wins. All applied:

- **M6 — Block bootstrap annualization** (`backtest/robustness.py`). `block_bootstrap` was annualizing with `*252 / sqrt(252)`, but the call site at the bottom of `run_robustness` feeds *monthly* L/S returns from `_build_equity_curves`. The reported Sharpe 95% CIs were therefore overstated by `sqrt(252/12) ≈ 4.58×`. Added a `periods_per_year` parameter (default 252) and pass `periods_per_year=12` from the monthly call site. Smoke test: with monthly synthetic returns the new CI lies in roughly `[-0.3, 1.5]` instead of the inflated `[0.02, 9.2]` range that the old code would have produced.
- **S1 — Cache forward returns** (`backtest/splits.py` + 4 call sites). Phase 5a/5b/5c/5d each invoked `compute_forward_returns` independently, scanning every per-ticker price parquet from scratch. Added `get_forward_returns_cached(features_path, df, ...)` which initially wrote a parquet cache keyed on features mtime. Later upgraded to `joblib.Memory` with an explicit `ForwardReturnsCacheSignature` dataclass capturing features parquet metadata (path, size, mtime), price manifest metadata (path, size, mtime, SHA-256 content hash), source-code hash of `compute_forward_returns`, horizons, `entry_date_col`, and a cache schema version. The cached function uses `@memory.cache(ignore=['df'])` so the cache key is the dependency signature only, not the full DataFrame. A human-readable manifest is written to `results/cache/forward_returns_manifest.json`. Eliminates 3+ duplicate passes per `run_all.py --from-phase 5`.
- **S2 — Vectorize and correct `assign_market_cap_buckets`** (`backtest/robustness.py`). Old version reloaded each ticker's parquet for every month (~100k parquet reads on full SP500), assigned buckets via per-row `df.loc[idx, ...]`, and formed thresholds from event-month tickers rather than the same-day PIT universe. Rewrote to: (a) pre-load price + shares history for PIT universe members once via `_load_ticker_history`; (b) use `np.searchsorted` for strict T-1 price and strict T-1 shares relative to each event date; (c) compute mega / large / mid / small cutoffs from covered members in the latest PIT universe snapshot on or before that event date; (d) use `np.select` for vectorized event-row bucket assignment. Missing data remains `"unknown"` and coverage is now reported as covered PIT members / PIT members for the event date.

Each fix passed `python -m compileall backtest features data run_all.py`. The market-cap bucket rewrite is intentionally not numerically equivalent to the previous approximation: it now uses same-day PIT universe thresholds and strict event-date T-1 price/shares lookups. The Phase 5d full re-run will produce refreshed timing and market-cap bucket tables for the next report update.

### 10.0.3 Round-3 review fixes — 2026-04-30

A third pass surfaced four issues across audit wiring, plan compliance, and code hygiene. All applied:

- **A1 — Wire fit() call-stack monitoring** (`backtest/model.py`). `features/audit.py` defined `monitor_fit_calls()` and `assert_fit_callstack()` for intercepting every `fit()` call during walk-forward (Plan §3.2 assertion 3), but `backtest/model.py` never imported or used them. Fixed by wrapping the impute/scale → LassoCV → model.fit section of `_run_one_fold()` inside `monitor_fit_calls()`, extracting the log, and calling `assert_fit_callstack()` with the fold's `[train_start, train_end]` boundaries. Violations are recorded in `FoldResult.fit_violations` and persisted to `fit_audit_log.jsonl`.

- **A2 — Add 3-day entry/exit quote tolerance** (`backtest/portfolio.py`). Plan §4.5 requires checking "3 consecutive trading days" for a valid quote before marking `skip_no_entry_quote` or `right_censored_no_exit_quote`. The portfolio simulator previously checked only the exact rebalance date for entry and the exact next-rebalance date for exit. Added `_next_valid_quote()` helper that scans up to 3 calendar days forward in the trading calendar; both entry and exit now use this helper. Actual entry/exit dates and prices are recorded in `trade_execution_log.parquet`.

- **A3 — Remove hard-coded `tolerance_days=65`** (`backtest/single_feature_ic.py`, `backtest/quintile.py`). Both modules hard-coded `tolerance_days=65` for universe PIT filtering, overriding the adaptive tolerance in `backtest/universe.py` (7 days for daily SP500 snapshots, 65 for monthly SP1500/RU3K). Changed the default to `None` so the shared module auto-detects the correct tolerance from snapshot frequency.

- **A4 — Replace deprecated `datetime.utcnow()`** (`data/config.py`, `data/load_prices.py`, `data/load_shares.py`). `datetime.utcnow()` is deprecated since Python 3.12. Added `utc_now_iso()` helper to `data/config.py` using `datetime.now(dt.UTC)` with fallback; updated all 5 call sites.

Validation: `python -m compileall backtest features data` passes; `from backtest.model import ...` and `from backtest.portfolio import ...` import cleanly.

### 10.0.4 Round-4 review fix — 2026-04-30

A fourth pass addressed the most complex review item — missing daily returns and delisting treatment in portfolio P&L (Bug #8).

- **B8 — Bounded forward-fill and recovery audit for missing daily returns** (`backtest/portfolio.py`). The daily P&L loop previously used ``dropna()`` before summing returns, silently ignoring held names with missing next-day quotes. Replaced with a per-day classification system:
  - Added ``_audit_long_gap_recovery()`` module-level function for the 30-trading-day recovery audit.
  - During the daily loop, each held ticker is classified: normal (valid next-day quote), ``ffill_1d`` (gap < 2 days, forward-filled with 0% return), ``ffill_2d`` (gap = 2 days), or long gap (>2 days, censored from headline P&L).
  - Long-gap positions are recorded and audited post-loop via the 30-trading-day recovery scan.
  - Eight gap accounting columns added to ``daily_returns`` parquet: weights (``ffill_1d_weight``, ``ffill_2d_weight``, ``long_gap_recovered_weight``, ``possible_delisting_or_unavailable_weight``) and position counts (``n_*`` equivalents).
  - Four gap summary statistics added to portfolio summary JSON (share of position-days in each category).
  - ``gap_accounting_*.parquet`` persisted with per-gap-event details.
- **Documentation**: ``ideas/plan.md`` Phase 5.4 updated; ``docs/report.md`` Sections 4.5 and 5.3 updated; ``docs/debugging.md`` Section 5 "Fix Applied" added.

Validation: ``python -m compileall backtest features data`` passes. Portfolio smoke test on SP500 weekly Ridge produces daily returns with all 15 columns including gap accounting columns. Summary JSON includes all four gap share statistics.

### 10.0.5 Round-5 review fix — 2026-04-30 (Phase 5 expanded loops)

A fifth pass expanded Phase 5's ``run_all.py`` orchestration to cover all requested dimensions:

- **Tier loop**: Phase 5 now iterates over all tiers in ``--tier``, so ``--tier both`` runs both Enhanced and Stretch experiments for every sub-phase (previously only ``args.tiers[0]`` was used, so Stretch Phase 5 was silently skipped).
- **Universe loop**: IC (5a), quintile (5b), and robustness (5d) now iterate over every universe in ``--universes`` (default: ``sp500 sp1500 ru3k``). Portfolio simulation (5c) loops over universes × models × cadences.
- **OOS prediction discovery**: The portfolio glob pattern changed from ``oos_pred_*_{tier}_h5d.parquet`` to ``oos_pred_*_{tier}_h*.parquet``, discovering predictions across all horizons.
- **RU3K coverage-constrained handling**: Added ``--skip-empty-universe`` flag. By default, empty/missing universe PIT files (e.g., RU3K) produce explicit ``{module}_{universe}_coverage_constrained.json`` sentinel artifacts instead of silently skipping.
- **Dry-run expansion**: ``python run_all.py --dry-run --tier both`` now enumerates all sub-tasks with their tiers, universes, models, and cadences.

See ``docs/debugging.md`` Section 2 for full details.

### 10.0.6 Round-6 review fix — 2026-04-30 (default parameter + cache stability)

A sixth pass surfaced two issues:

- **C1 — `run_walk_forward()` default `availability_col` mismatch** (`backtest/model.py`). The function signature defaulted to ``availability_col="call_entry_date"``, but the CLI and plan specify ``"availability_date"`` (the unified operational availability date) as the main pipeline default. If ``run_walk_forward()`` were called programmatically without passing this argument, it would silently use the wrong date column for fold construction, training-set filtering, and G11 label purge. Changed the function-signature default to ``"availability_date"`` to match the CLI and plan.

- **C2 — Deterministic source-code hash for cache signature** (`backtest/splits.py`). ``_build_cache_signature()`` used Python's built-in ``hash()`` to fingerprint the source code of ``compute_forward_returns``. Since Python's string ``hash()`` is randomised via ``PYTHONHASHSEED`` across interpreter restarts, the ``source_hash`` field in ``ForwardReturnsCacheSignature`` could differ between runs even when the source code was unchanged, causing unnecessary forward-return cache recomputation. Replaced with ``int(hashlib.sha256(source.encode()).hexdigest()[:16], 16)``, matching the deterministic SHA-256 approach already used for the price-manifest hash in the same function.

Validation: ``python -m compileall backtest/model.py backtest/splits.py`` passes.

### 10.0.7 Round-7 review fix — 2026-04-30 (PIT membership, model sample, audit, and portfolio execution)

A seventh pass reviewed the code against the intended Phase 0-5 methodology rather than only checking syntax. This surfaced several material issues that affected reproducibility and the interpretation of existing Phase 4/5 artifacts. All fixes below were applied in code. **Important consequence:** previously generated Phase 4/5 outputs in `results/` should be treated as stale and regenerated before final conclusions are reported.

- **D1 — Correct SP500 PIT effective dates** (`data/load_universes.py`). The SP500 Wikipedia reverse-replay logic previously recorded the pre-change state at `change_date - 1` and then forward-filled it, which could keep removed names active after the effective date and delay added names until a later snapshot. The corrected implementation records the post-change membership at the actual effective date, then undoes all additions/removals for that same date before continuing backward. A synthetic fixture with `A -> B` on `2020-01-15` now returns `A` on `2020-01-14` and `B` from `2020-01-15` onward.

- **D2 — Filter predictive model samples by PIT universe and `SignalType`** (`backtest/model.py`). The walk-forward model path previously trained on the full 2.74M-row feature table and was filtered only later during portfolio construction. This mixed universes and signal slices inside model fitting. The CLI now defaults to `--universe sp500 --signal-type Total`; `filter_model_sample()` applies PIT universe membership before tuning and walk-forward. `_orig_df_index` is preserved so OOS predictions can still be joined to the original feature parquet in portfolio simulation. On the current enhanced artifact, the SP500 + `Total` model sample is `32,136` rows.

- **D3 — Per-horizon frozen hparams and universe-aware Phase 4 orchestration** (`backtest/model.py`, `run_all.py`). Tuning now runs once per requested horizon and writes `results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json`. `run_all.py` tunes on the first populated universe, then runs walk-forward separately for every requested populated universe while reusing the shared frozen hparams. This preserves the cross-universe hparam discipline while making Phase 4 reproducible from source.

- **D4 — Prevent portfolio output overwrites across prediction horizons** (`backtest/model.py`, `run_all.py`, `backtest/portfolio.py`). OOS prediction filenames now include model, tier, universe, signal slice, and prediction horizon, e.g. `oos_pred_ridge_enhanced_sp500_total_h5d.parquet`. Phase 5 portfolio tags include `predh{horizon}d`, so h1/h3/h5/h10/h20 prediction runs no longer overwrite the same `{model, cadence, lookback}` output filenames.

- **D5 — Strengthen fit-call audit evidence** (`features/audit.py`, `backtest/model.py`). Fit-call monitoring now patches `Ridge.fit()` directly. Training matrices are kept as pandas DataFrames with a `DatetimeIndex` through imputation, scaling, feature selection, and model fitting, so the fit log records real min/max training dates instead of just NumPy array shapes. `FoldResult` stores the fit-call log and `write_fit_audit_log()` persists it.

- **D6 — Apply actual delayed entry dates to daily P&L** (`backtest/portfolio.py`). `_next_valid_quote()` may fill an intended trade 1-3 trading days after the planned rebalance date. The previous simulator logged the delayed `actual_entry_date` but let the position contribute to daily P&L immediately on the planned date. The simulator now tracks `current_entry_dates` and includes a position in gross/P&L only once `actual_entry_date <= current_date`.

Validation completed after this round:

- `python -m compileall data features backtest reports run_all.py` passed.
- Main module import smoke test passed.
- Synthetic SP500 add/remove fixture passed.
- `filter_model_sample()` kept `32,136` SP500 `Total` rows with unique `_orig_df_index`.
- Synthetic walk-forward smoke wrote `oos_pred_ridge_enhanced_sp500_total_h1d.parquet`, preserved original feature indices, and produced dated fit logs for `SimpleImputer`, `StandardScaler`, and `Ridge`.
- Portfolio smoke confirmed that delayed-entry positions are excluded from gross exposure and P&L before `actual_entry_date`.

Required rerun before using final numbers:

```bash
python -m data.load_universes --only sp500
python run_all.py --from-phase 4 --stop-at-phase 5 --tier enhanced
```

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
