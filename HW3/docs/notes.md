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
- **Fallback discipline**: when iShares historical holdings are not available for the full range, the earliest available snapshot is used as a static universe for all prior dates, with an explicit survivorship-bias note logged to `results/audit/universe_coverage_gaps.csv`. Per requirement §6.3, "document the survivorship-bias caveat explicitly in your research PDF and apply your model on a 'current-membership' universe — your reported alpha will be an upper bound."
- **PDF transparency statement** (in methodology): using ETF holdings as approximate PIT constituents for SP1500/RU3K may introduce small sampling bias because they are not official FTSE Russell / S&P constituent files; this is a data-source transparency issue and **does not create look-ahead bias**

### 2.5 Phase 1 smoke-test log - 2026-04-29
All Phase 1 smoke tests passed after the fixes below.

| Area | Smoke command / check | Final result |
|---|---|---|
| Signals | temporary-output run of `data.load_signals.run(data/sample.csv, chunksize=4)` plus full rebuild `python -m data.load_signals --chunksize 100000` | PASS: 2,740,437 raw rows read; 2,231 `delete` rows dropped; 2,738,206 rows written; main cache 448 columns; slim cache 43 columns |
| SP500 PIT | `python -m data.load_universes --only sp500` plus `members_at()` boundary assertions | PASS: `data/cache/universes/sp500_pit.parquet` has 2,289,593 rows; future dates raise; pre-history dates return an empty set |
| SP1500 PIT | `python -m data.load_universes --only sp1500` | PASS: SP500 component written; missing IJH/IJR history uses earliest snapshot as static universe (survivorship bias documented in coverage gaps) |
| RU3K PIT | `python -m data.load_universes --only ru3k` | PASS: no historical IWV snapshots available; using today's snapshot as static universe (survivorship bias documented in coverage gaps) |
| Prices | `python -m data.load_prices --only AAPL --start 2024-01-01 --end 2024-01-10 --sleep 0` and smoke batch `python -m data.load_prices --limit 20 --no-parallel` | PASS: `AAPL.parquet` has 6 rows with `date, adj_close, volume`; 20-ticker smoke fetches 10 new tickers (8 success + 2 empty for delisted ABMD/ACAS) with zero errors and no yfinance log spam |
| Shares | `python -m data.load_shares --only AAPL --start 2024-01-01 --end 2024-01-10 --sleep 0` | PASS: `AAPL.parquet` has 4 rows with positive shares; manifest status is `success`; coverage report written |
| Final verification | `python -m compileall data` and a consolidated Python assertion script over all Phase 1 artifacts | PASS |

### 2.6 Phase 1 full-universe final run - 2026-04-29

After the smoke tests, the same loaders ran end-to-end across the full ticker universe; results are consistent with the smoke logs and form the baseline that Phase 2 builds on.

| Area | Final-run output | Numbers |
|---|---|---|
| Signals | `data/cache/signals.parquet`, `data/cache/signals_slim.parquet` | 2,738,206 rows, 8 SignalType slices (`Total / Executives / Presentation / Answer / Question / Analysts / CEO / CFO`); main 448 cols (29 ID + 12 EventScore + 1 ATCClassifierScore + 1 `call_hour_utc` + 405 non-Fluff/Filler `AspectTheme_*`); slim 43 cols; zero `call_hour_utc = -1` rows; 17,636 unique BESTTICKERs (193 nulls) |
| SP500 PIT | `sp500_pit.parquet` | 2,289,593 rows, 817 unique tickers across history, 4,520 daily dates from 2009-01-01 to 2026-04-29 |
| SP1500 PIT | `sp1500_pit.parquet` | 289,185 rows, ~1,500 tickers, 196 month-ends 2010-01-31 → 2026-04-30. IJH + IJR use today's snapshot as static universe for all historical months (survivorship bias), combined with SP500 rolling monthly PIT. Three coverage gaps logged (one per component: IJH, IJR, IWV). |
| RU3K PIT | `ru3k_pit.parquet` | 506,268 rows, 2,583 tickers, 196 month-ends 2010-01-31 → 2026-04-30. Today's IWV snapshot used as static universe for all historical months (survivorship bias — reported alpha is an upper bound). One coverage gap logged. |
| Prices | `data/cache/prices/{ticker}.parquet` | 817 universe tickers attempted: 667 success / 150 empty / 0 error; 667 parquet files on disk; `failed_tickers.csv` contains 150 `empty` rows; AAPL/A series cover 2009-01-02 → 2026-04-29 with 4,357 daily rows; ABBV starts 2013-01-02 (post-IPO) |
| Shares | `data/cache/shares/{ticker}.parquet` | 817 attempted: 780 success / 37 empty / 0 error; AAPL series has 336 rows from 2015-10-28 — Yahoo's `get_shares_full` history begins ~2015 for most tickers |
| Marketcap coverage | `results/audit/marketcap_capacity_coverage.csv` | 34 `(universe, year)` rows for SP500 + SP1500 over 2010–2026. **2010–2014 fail the 70% floor** (coverage 0.0–0.9% — the yfinance shares endpoint has no pre-2015 history); **2015–2026 are above floor** (77–85%). RU3K rows are pending a rerun with the new static-universe PIT. Practical consequence: market-cap buckets are quantitative for 2015+ only; the 2010–2014 subperiod is qualitative-robustness-only and is flagged in red in the PDF |

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
- **Performance: shares loader serial per-ticker fetch** (`data/load_shares.py`): the original implementation iterated tickers one-by-one in a serial loop, with no prefilter to skip already-cached tickers. Unlike `yf.download()` (which supports up to 50 tickers per HTTP request), `yf.Ticker().get_shares_full()` has no batch endpoint — each ticker is one API call. Fixed with two changes: (1) **upfront prefilter** — scan all tickers against the manifest first, split into `cached` (skip) and `stale` (fetch) groups, reporting counts before any network call; (2) **`ThreadPoolExecutor` with 3 workers** — parallelise per-ticker calls over the stale group. Each worker still retains the 0.6s + jitter per-request sleep and exponential backoff (max 4 retries, cap 60s), so at most 3 concurrent HTTP requests are in flight at any time. Serial mode available via `--no-parallel`. Manifest and failed-log writes use `threading.Lock` for thread safety.

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
- **Entry**: planned entry uses `next_valid_close_on_or_after(planned_entry_date)`; if there is still no valid quote after 5 consecutive trading days, mark `skip_no_entry_quote` and do not open the position
- **Exit**: after a position is opened, planned exit uses `next_valid_close_on_or_after(planned_exit_date)`; if there is still no quote after 5 consecutive trading days, do not delete the whole trade. First mark `right_censored_no_exit_quote` and exclude it from ordinary horizon-return statistics, while summarizing it separately in the delisting / no-exit audit table. If the data source provides a final delisting/merger transaction price, exit at that price and mark `delisting_exit_used`
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
- **Phase 1 status (updated 2026-04-30)**: iShares does not expose a historical-snapshot API. Today's IJH, IJR, and IWV snapshots are used as a static universe for all historical months, with explicit survivorship-bias documentation in `results/audit/universe_coverage_gaps.csv`. Per requirement §6.3, this is acceptable when documented: "If you cannot obtain one, document the survivorship-bias caveat explicitly ... your reported alpha will be an upper bound." SP500 uses daily PIT from Wikipedia historical changes and is unaffected.

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
| Q1 | Universe membership | SP500 Wikipedia daily PIT; SP1500 IJH+IJR+SP500 monthly (static-universe fallback for IJH/IJR pre-2026); RU3K IWV monthly (static-universe fallback for all pre-2026 months). Survivorship bias documented in `universe_coverage_gaps.csv` per §6.3. | Meets sections 3.6 / 6.3 |
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
| 5.4 Portfolio sim | `backtest/portfolio.py`, `backtest/portfolio_batch.py` | 952+ | Validated — PIT-filtered OOS portfolios; daily P&L accounting for all cadences; model-tagged outputs; trade log + universe coverage artifacts; batch runner reuses price/calendar caches across Phase 5 jobs |
| 5.5 Robustness | `backtest/robustness.py` | 680+ | Validated — subperiod / sector-neutral / mcap / weighting / bootstrap / OFAT quantile / OFAT cost / OFAT cadence-lookback / weekly timing / label-purge gap / R8 beta-window checks |

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
- Changed forward-return entry matching from a 10-calendar-day tolerance to a max 5-business-day entry gap, aligned with portfolio entry/exit quote lookup
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

- **A2 — Add 5-trading-day entry/exit quote tolerance** (`backtest/portfolio.py`). Plan §4.5 now uses the same 5-business-day roll-forward standard as forward returns before marking `skip_no_entry_quote` or `right_censored_no_exit_quote`. `_next_valid_quote()` scans forward in the price-derived trading calendar; both entry and exit use this helper. Actual entry/exit dates and prices are recorded in `trade_execution_log.parquet`.

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

- **D6 — Apply actual delayed entry dates to daily P&L** (`backtest/portfolio.py`). `_next_valid_quote()` may fill an intended trade 1-5 trading days after the planned rebalance date. The previous simulator logged the delayed `actual_entry_date` but let the position contribute to daily P&L immediately on the planned date. The simulator now tracks `current_entry_dates` and includes a position in gross/P&L only once `actual_entry_date <= current_date`.

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

### 10.0.8 Rewrite unit-test hardening plan — 2026-04-30

Before the planned codespace reconstruction, the unit-test strategy in
`ideas/rewrite.md` was revised to make the refactor safer. No runtime code was
changed in this documentation pass.

| ID | Change | Rationale |
|---|---|---|
| RUT-01 | Persistent rewrite baselines plus `manifest.json` | `/tmp` alone is too easy to lose during a multi-step refactor. The manifest records git SHA, commands, package versions, input metadata, hashes, and creation time so golden comparisons are reproducible. |
| RUT-02 | `tmp_path`-only file tests | Unit tests for manifests and failed logs must not mutate real cache, results, or baseline artifacts. |
| RUT-03 | Explicit failed-log edge cases | The shared helper tests now cover same-ticker success/failure ordering so price and shares loaders keep their intentionally different semantics. |
| RUT-04 | Pandas-compatible Spearman NaN tests | Existing Spearman uses pandas pairwise NaN dropping, so the tests now verify that behavior instead of expecting blanket NaN propagation. |
| RUT-05 | Observed-bucket equity schema tests | Equity-curve tests now distinguish full-bucket fixtures from sparse qcut fixtures, avoiding a false requirement for columns that current code does not create. |
| RUT-06 | Portfolio-stat edge cases | Added coverage for one-row equity curves and zero-volatility Sharpe to preserve current summary behavior. |
| RUT-07 | Locked update wording | Failed-log update tests are now described as locked read-apply-write, not crash-atomic, unless the implementation later adds temp-file replacement. |
| RUT-08 | Pytest availability gate | The rewrite checklist now starts by checking `python -m pytest --version`; the current environment needs pytest installed before these tests can run. |

### 10.0.9 Code restructuring completed — 2026-04-30

The behavior-preserving refactor described in `ideas/rewrite/` was executed
across 14 modules (A1–C7). Two new shared leaf modules were created
(`data/_utils.py`, `backtest/_stats.py`) with 47 unit tests. All data loaders,
backtest modules, and feature engineering code now import from these shared
modules instead of maintaining local copies of the same functions. Six large
functions were split into focused helpers:
`compute_forward_returns`, `PortfolioSimulator.run`,
`compute_momentum_features`, `_run_one_fold`/`run_walk_forward`,
`run_all_robustness`, and `run_streaming_vs_batch_test`. CLI `main()` functions
were extracted into helpers. Minor deviations from the plan are documented in
`ideas/plan.md` Phase 5.11 and in the individual `ideas/rewrite/` files.

One bug was found and fixed: `_process_rebalance` in `backtest/portfolio.py` had
an incorrect `extend` call on a function that appends directly to the trade
records list and returns `None`.

Verification: `python -m compileall data features backtest reports run_all.py`
clean; 47/47 unit tests pass; `python run_all.py --dry-run` enumerates all
sub-tasks correctly.

**RU3K / SP1500 static-universe fallback (2026-04-30):** `expand_to_month_grid`
in `data/load_universes.py` was updated to use the earliest available iShares
snapshot as a static universe for all earlier months, with a single
survivorship-bias coverage gap logged per ETF component. This enables the full
three-universe pipeline (SP500 / SP1500 / RU3K) while transparently documenting
the survivorship-bias caveat per requirement §6.3.

### 10.0.10 Static logic-review fixes — 2026-05-01

After the full-scale implementation pass, the code was reviewed directly
against `docs/requirement.md` without attempting to rerun the full pipeline.
Seven logic gaps were fixed:

- **Pre-event momentum no longer includes call-day prices.** The event anchor in
  `features/engineer.py` now normalizes `MOSTIMPORTANTDATEUTC` to the call
  calendar date and selects the last trading day strictly before that date.
  This enforces the documented T-1 rule for BMO and AMC calls.
- **OOS prediction rows are no longer selected using future returns.**
  `backtest/model.py` now predicts every test event. Rows with missing
  `forward_return_*` are excluded only from OOS IC/MSE calculations, not from
  the prediction parquet consumed by portfolio simulation.
- **Forward returns and portfolio execution are volume-aware.**
  `backtest/splits.py` and `backtest/portfolio.py` now require positive volume
  on price bars used for entry, exit, and daily P&L. A positive adjusted close
  alone is no longer treated as enough evidence of tradability.
- **IC reporting now includes explicit year and sector splits.**
  `backtest/single_feature_ic.py` writes `ic_yearly_{universe}.parquet` and
  `ic_sector_split_{universe}.parquet` for all short-list features, horizons,
  and SignalTypes.
- **Quintile/decile bucketing is now T-0 rather than full-month ex-post.**
  `backtest/quintile.py` buckets by `availability_date`, so later events in the
  same month cannot influence the rank of earlier events.
- **Chart and audit discovery now match the actual artifact names.**
  `reports/charts.py` discovers model tags such as
  `ridge_enhanced_predh5d`; `backtest/model.py` writes compatibility
  `fit_audit_log_{model}_{tier}.jsonl` files; `features/audit.py` writes
  `validation_summary.json`.
- **Universe fallback wording now matches implementation.**
  `data/load_universes.py` now documents that the iShares static-snapshot
  fallback is allowed only with explicit survivorship-bias records in
  `universe_coverage_gaps.csv`.

Required rerun after these fixes:

```bash
python run_all.py --from-phase 2 --stop-at-phase 7 --tier enhanced
```

### 10.0.11 Final static logic-review fixes - 2026-05-01

A final code-only review against `docs/requirement.md` was completed while the
full-scale run was in progress. The goal was not to prove the full pipeline can
finish, but to remove remaining logic gaps before interpreting the artifacts.

- **Daily portfolio PnL now handles short quote gaps correctly.**
  `backtest/portfolio.py` now carries each position's last valid price through
  1- and 2-trading-day quote gaps, records the bounded forward-fill exposure,
  and books the cumulative return when the quote resumes. Gaps longer than two
  trading days remain censored from headline PnL and are audited with the
  existing 30-trading-day recovery scan.
- **Audit verification now fails on failed evidence, not only missing files.**
  `run_all.py` Phase 6 now rejects failed or pending entries in
  `validation_summary.json`, non-empty generic or scenario-specific
  `trade_execution_violations*.parquet` files, unreadable or missing violation
  parquet files, missing `fit_audit_log_*.jsonl`, fit-audit violations, and
  malformed fit-audit log lines.
- **Forward-return isolation is checked on generated feature output.**
  `features/audit.py` rebuilds a no-momentum feature sample and checks generated
  non-identifier columns for `Return_*`, `forward_return_*`, and
  `target_available_date_*` leakage.
- **Sector IC now matches the requirement.**
  `backtest/single_feature_ic.py` computes monthly cross-sectional IC within
  each sector, then summarizes the sector-month IC series.
- **Decile summaries now include long-only and short-only legs.**
  `backtest/quintile.py` records all three required legs for the
  `ATCClassifierScore` decile baseline.
- **Robustness drawdowns now use cumulative equity.**
  `backtest/robustness.py` uses `cum_long_short` for max-drawdown calculations
  in subperiod, OFAT cutoff, and weighting-scheme outputs.
- **Raw data loading and feature hygiene were tightened.**
  `data/load_signals.py` can extract the CSV from the raw zip, and
  `backtest/splits.py` excludes `_in_universe` from model features.
- **Chart generation no longer trusts stale PNGs.**
  `reports/charts.py` regenerates every requested chart when Phase 7 runs.

Validation:

```bash
python -m compileall data/load_signals.py features/audit.py backtest/splits.py backtest/single_feature_ic.py backtest/quintile.py backtest/robustness.py backtest/portfolio.py reports/charts.py run_all.py
python -m pytest -q
```

Both checks passed locally; the unit suite reports `47 passed`.

### 10.0.12 Phase 4 hparam manifest serialization fix - 2026-05-01

During Phase 4, LightGBM tuning completed and wrote
`results/hparams/enhanced/h1d/frozen_hparams_lightgbm.json`, but the pipeline
then failed while appending to `results/hparams/hparams_manifest.json`:
`TypeError: Object of type int64 is not JSON serializable`.

Root cause: random-search hyperparameters sampled via `numpy.random.choice`
were `np.int64` / `np.float64` values. The per-model frozen hparam writer
converted top-level numpy scalars before writing the model JSON, but the
manifest append path still received the original unsanitized params dict.

Fix: `backtest/splits.py` now uses a shared recursive `_json_safe()` helper for
frozen hparam files, manifest appends, and manifest rebuilds. Random-search
param combinations are normalized at sampling time as well, so later logging
and result dictionaries use JSON-native Python scalars.

Validation: targeted hparam-write smoke tests with `np.int64`, `np.float64`,
`np.bool_`, arrays, and nested dict/list values pass; `python -m compileall
backtest/splits.py` passes. The interrupted Phase 4 run should be resumed or
rerun so the manifest contains the completed LightGBM entry.

### 10.0.13 Final pipeline synchronization pass - 2026-05-01

This pass records the final code changes made after the project logic review
and before writing the final PDF. The changes are intended to make the code,
`docs/requirement.md`, this report, and `ideas/plan.md` use the same
definitions.

- **Forward-return and portfolio execution quote lookup now use one 5-trading-day standard.**
  `backtest/splits.py` defines `MAX_ENTRY_GAP_BDAYS = 5` and bumps the
  forward-return cache schema version to 3. `backtest/portfolio.py` defines
  `MAX_QUOTE_FORWARD_DAYS = 5`; entry and exit quote lookup both use the same
  helper default. Older references to the shorter prior quote window were
  updated in the report and plan.
- **Forward-return cache signatures force a price-manifest sync.**
  Before building `ForwardReturnsCacheSignature`, `backtest/splits.py` now
  fingerprints current price parquet names, sizes, and mtimes into
  `data/cache/prices/_manifest.json`. This prevents stale price manifests from
  causing a forward-return cache hit after local price files changed.
- **Phase 2 PIT percentile calculation groups once per bucket.**
  `features/engineer.py` now groups by `(SECTOR, SignalType)` once and computes
  all available PIT percentile columns inside that group pass. The strict
  `history_date < cutoff_date` rule and optional Numba Fenwick core are
  unchanged; this removes repeated groupby overhead.
- **Phase 5 portfolio simulation has a batch runner and in-process market cache.**
  New `backtest/portfolio_batch.py` accepts a JSON job list and runs all jobs in
  one Python process. `PortfolioSimulator` caches the loaded close matrix,
  trading calendar, rebalance dates, and universe coverage by ticker set, date
  range, cadence, lookback, and weekly timing. `run_all.py` writes
  `results/cache/portfolio_jobs_{tier}.json` and runs the batch module once per
  tier. The dry-run check showed `45` enhanced/SP500 portfolio jobs collapsed
  into one batch command.
- **`run_all.py` no longer treats one representative artifact as proof that a full phase is complete.**
  Only Phase 1 raw-loader artifacts may be skipped by `_skip()`. Phase 2 and
  later are rerun by the orchestrator so missing tier/model/universe/cadence
  outputs cannot be hidden by one stale file. The Phase 1 universe artifact
  check now requires all three PIT universe files.
- **Phase 6 audit verification now scans scenario-specific evidence.**
  The prior risk was that generic files such as
  `trade_execution_violations.parquet` were "last run wins": a later clean
  scenario could overwrite an earlier scenario with violations. Phase 6 now
  checks every `trade_execution_violations*.parquet`, requires at least one
  `fit_audit_log_*.jsonl`, and fails on malformed fit logs, fit violations,
  failed/pending validation-summary checks, or non-empty trade violations.
- **Robustness coverage now matches the plan dimensions.**
  `backtest/robustness.py` writes `robustness_ofat_lookback.parquet` for
  daily `{1,3}`, weekly `{3,5,10}`, and monthly `{15,21,30}` portfolio
  reruns; `robustness_weekly_timing.parquet` for Monday versus Friday weekly
  timing; `robustness_label_purge_gap.parquet` for `{0,5,21}` business-day
  purge-gap sample impact; and `robustness_r8_beta_window.parquet` after
  recomputing R8 idiosyncratic residuals with beta windows `{40,60,90}`.
- **Hyperparameter tuning trial loop parallelized.**
  `backtest/splits.py` `tune_lightgbm()` and `tune_xgboost()` previously ran 30
  random trials per CV fold in a serial inner loop. Each trial set `n_jobs=1`
  (correct to avoid nested OpenMP), but this left all other CPU cores idle.
  Extracted `_fit_lgbm_trial` / `_fit_xgb_trial` module-level helpers and
  replaced the inner loop with `joblib.Parallel(n_jobs=-1)` using the default
  `loky` backend. Each CV fold's 30 trials now run concurrently across all
  available cores. Ridge tuning (6 alphas × 5 folds, sub-100ms per fit) is left
  serial to avoid parallelism overhead dominating the work.
- **Weekly timing is now explicit in the portfolio interface.**
  `backtest/portfolio.py` adds `--weekly-day {monday,friday}`. Monday remains
  the default production timing; Friday is used for robustness sensitivity and
  is included in output suffixes when non-default.

Validation after this synchronization pass:

```bash
python -m compileall backtest features run_all.py
python -m pytest -q
python run_all.py --dry-run --from-phase 5 --stop-at-phase 5 --tier enhanced --universes sp500 --skip-empty-universe
python run_all.py --from-phase 6 --stop-at-phase 6 --tier enhanced --universes sp500 --skip-empty-universe
```

All commands passed locally; pytest reports `47 passed`.

### 10.0.18 Phase 5 performance fixes — 2026-05-01

Seven performance recommendations from `docs/review/phase5.md` (items 1-6, 8) were applied across 6 backtest modules. Item 7 (vectorize daily PnL loop) was deferred to §10.0.19.

- **Shared parquet cache (`backtest/splits.py`)**: Added `_cached_read_features` with `@lru_cache(maxsize=1)` so the features parquet is loaded into memory once and shared across IC, quintile, and robustness modules, eliminating redundant multi-GB reads.
- **Combined IC worker (`backtest/single_feature_ic.py`)**: Merged three separate `ProcessPoolExecutor` workers (monthly IC, yearly IC, sector IC) into one `_ic_full_worker` that computes all three in one pass, reducing worker submissions by ~67%.
- **Pre-indexed signals (`backtest/portfolio.py`)**: Signals DataFrame is indexed by `date` once in `run()`, then rebalance lookups use `pd.DatetimeIndex.loc[]` for O(log N) range scans instead of per-rebalance boolean masks.
- **`groupby().last()` dedup (`backtest/single_feature_ic.py`)**: Replaced `sort_values().drop_duplicates(keep="last")` with `groupby("BESTTICKER").last()` under an `is_monotonic_increasing` assertion, reducing per-group dedup from O(n log n) to O(n).
- **`scipy.stats.spearmanr` (`backtest/_stats.py`)**: Replaced `pd.Series.corr(method="spearman")` with `scipy.stats.spearmanr(a, b, nan_policy="omit")` for 3-5x faster rank correlation without intermediate Series objects.
- **Pre-loaded price table (`backtest/portfolio.py`)**: `compute_capacity_metrics` accepts an optional `price_table` argument, avoiding per-ticker parquet file loads when the caller already has prices loaded (e.g., from the portfolio simulator's internal cache).
- **Pre-allocated bootstrap arrays (`backtest/robustness.py`)**: `block_bootstrap` uses `np.empty(n_boot)` and `np.concatenate()` instead of Python list accumulation followed by `np.array()`.

Validation: `python -m compileall backtest/` passes; `python -m pytest tests/ -v` reports 47/47 passed.

### 10.0.19 Vectorize daily PnL loop — 2026-05-01

Item 7 from `docs/review/phase5.md` and the full plan in `docs/review/faster.md`
were implemented in 6 steps (Steps 0–5). The daily PnL loop
(`_compute_daily_pnl` in `backtest/portfolio.py`) was the hottest path in the
project: called once per trading day (~2500 times) with a Python `for` loop
iterating every active ticker (~320K ticker-day iterations for SP500).

**Step 0 — Extract `_advance_daily_state` free function + 12 unit tests.**

The core daily-PnL logic was extracted into `_advance_daily_state()`, a pure
function over plain Python dicts with no pandas, no `SimulationState`, and no
`close_matrix` slicing. `tests/test_portfolio_pnl.py` was created with 12 unit
tests covering: empty weights, valid both days, 1d/2d/3d quote gaps, censored
ticker skip, `setdefault` initialization, missing-without-last-price edge case,
mixed tickers, cumulative gap recovery, zero-weight, and short positions.

**Step 1 — Build flat price matrix once (upstream of the simulation loop).**

After `_load_calendar_prices_and_coverage` returns `close_matrix`, the simulator
builds: `_ticker_to_idx: dict[str, int]`, `_idx_to_ticker: list[str]`,
`_day_index: dict[pd.Timestamp, int]`, and `_prices_2d: np.ndarray` (shape
`(n_tickers, n_days)` float64, `close_matrix.values.T`). Two integer-indexed
lookups replace the per-day `.loc` + `.reindex` DataFrame operations.

**Step 2 — Add numpy array fields to `SimulationState`.**

The `SimulationState` dataclass gained five new fields: `last_prices_arr`,
`missing_streaks_arr`, `is_censored_arr`, `weights_arr`, and `entry_dates_arr`
(all `np.ndarray | None`, default `None`). These are initialized in `run()` to
size `n_tickers` and populated in `_process_rebalance` at each rebalance.

**Step 3 — `_advance_daily_state_numpy` with 4 boolean-mask passes.**

A new pure-numpy kernel replaces the per-ticker Python loop with four
boolean-mask passes over the active ticker subset:

1. **setdefault**: `np.isnan(last_prices) & np.isfinite(p_today) & (p_today > 0)`
2. **Valid PnL**: `np.isfinite(p_next) & (p_next > 0) & ...` → `np.dot(weights, ret)`
3. **Missing streaks**: increment counters, classify into 1d / 2d / long_gap
4. **Gap accounting**: `np.abs(weights[mask]).sum()` for weight columns

Censored tickers are handled in a separate sub-pass before the active-PnL
passes. `_compute_daily_pnl` auto-selects the vectorized path when
`state.last_prices_arr is not None`, falling back to the dict kernel for
tests and bootstrapping.

**Step 4 — Dual-implementation equivalence tests.**

Ten additional tests (tests 13–22) run identical scenarios through both
`_advance_daily_state` (dict) and `_advance_daily_state_numpy` (numpy),
asserting identical `pnl`, `gross_exposure`, `net_exposure`, `n_positions`,
all gap-weight columns, gap counts, `long_gap_tickers` membership, and
mutated state array values.

**Step 5 — Numba `@njit(cache=True)` single fused loop.**

`_advance_daily_state_numba` is an `@njit`-compiled single loop over all
tickers that fuses all four passes into one register-level iteration with
zero temporary boolean arrays. `_advance_daily_state_numba_wrapper` allocates
a pre-sized `long_gap_out` output array, calls the numba kernel, and builds
the result dict from the returned scalars. `_compute_daily_pnl_vectorized`
auto-selects numba when available, falling back to pure numpy. Seven numba-
specific equivalence tests (`TestNumbaEquivalence`) gate the compiled path.

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

Validation: `python -m compileall backtest/` — clean. `python -m pytest tests/ -v` — 76/76 passed (47 existing + 29 new). No regressions in any existing test suite.


### 10.1 Per-universe result summary
- SP500: (fill IC / decile spread / walk-forward Sharpe)
- SP1500: (fill)
- RU3K: (fill)

### 10.2 Deployment recommendation
- Recommended cadence + rationale based on joint evidence from post-cost Sharpe / alpha decay / turnover / capacity
- Recommended model based on Ridge / LGBM / XGB comparison
- Recommended position sizing rule
- Estimated supportable AUM

### 10.0.14 Post-review code fixes — 2026-05-01

After a systematic code review across all 9 phases (`docs/review/phase_0.md` through `phase_8.md`), the following changes were applied:

**Phase 0** (`data/config.py`, `run_all.py`):
- Fixed `RAW_SIGNAL_ZIP` path to match `Earnings_ATC_until_2026-04-21.csv.zip`
- Added `set_global_seed()` call in `run_all.py` `main()`

**Phase 1** (`data/_utils.py`, `data/load_signals.py`, `data/load_shares.py`):
- Normalized ticker format (dot → dash) in signals loader for consistency with universe PIT
- Added random jitter to `exponential_backoff` to prevent thundering-herd retries under `ThreadPoolExecutor`
- Made manifest writes crash-atomic via `tempfile` + `Path.replace()`
- Batched failed-log updates in shares loader (every 50 tickers) to avoid O(n) per-ticker DataFrame copies
- Changed `parse_call_hour` to use nullable `Int8` instead of -1 sentinel for NaN timestamps
- Added delete-row count deviation warning when dropped rows differ >20% from expected ~2,231

**Phase 2** (`features/engineer.py`):
- Applied `EXCLUDED_FEATURE_PATTERN` regex (`^Return_\d+d$`) in `build_features()` to drop any `Return_*d` columns before output
- Referenced `MAGNITUDE_WEIGHT` constant in `_aspect_features()` and `_theme_features()` instead of hardcoded `3, 2, 1`
- Removed dead `IDENTIFIER_COLS` constant

**Phase 3** (`features/audit.py`, `backtest/portfolio.py`, `backtest/model.py`):
- Added `delisting_exit_used` skip-reason: long-gap unrecovered positions in portfolio now get this reason; `validate_trade_log()` validates matching price evidence
- Deleted dead `assert_feature_parity()` function (canonical path is `run_streaming_vs_batch_test()`)
- Added expected `availability_date` assertions to timestamp boundary fixtures for all 7 single-day and cross-day fixtures
- Removed unused `fold_test_end` and `horizon_days` parameters from `assert_fold_boundaries()` and its call site
- Added 6-row synthetic fixture for `assert_rebalance_eligibility()` covering boundary dates, NaN, and duplicate tickers

**Phase 4** (`backtest/splits.py`, `backtest/model.py`):
- Changed test-fold quarter assignment in `generate_folds()` to use `availability_date` instead of `call_entry_date`
- Added clarifying comment in `_prepare_fold_sample()` documenting that right-censored test events are retained for OOS signal generation
- Added explicit `right_censored_{h}d` boolean columns after forward-return computation; updated `write_sample_size_audit()` and `get_feature_cols()` exclusion

**Phase 5** (`backtest/quintile.py`, `backtest/robustness.py`, `backtest/_stats.py`, `backtest/single_feature_ic.py`):
- **C1 (CRITICAL)**: Fixed Sharpe annualization in quintile/decile analysis — `ann_factor=252.0` → `ann_factor=252.0/horizon`. The equity curve uses h-day forward returns, so annualization must be `252/h`, not `252`. Sharpe was overstated by `sqrt(h)`.
- **C2 (CRITICAL)**: Fixed Sharpe annualization in `run_subperiod_quintile()` — hardcoded `*252` / `*sqrt(252)` → `*(252/horizon)` / `*sqrt(252/horizon)`
- **H1**: Fixed rolling Sharpe window from 12 → 252 periods (12 periods on daily-indexed equity curve ≈ 2.5 weeks, not 12 months)
- **H2**: Fixed block bootstrap `block_size=3` → `block_size=1` (monthly returns with block_size=1 ≈ 21 trading day blocks per plan spec)
- **M2**: Changed OFAT cost "no portfolio found" log level from `INFO` → `WARNING`
- **L3**: Added `pd.isna()` guard before `int(row["year"])` in yearly IC worker

**Phase 6** (`run_all.py`, `features/audit.py`, `backtest/splits.py`, `backtest/portfolio.py`):
- **C1**: Phase 6 now re-generates `lookahead_checklist_onepager.md` items 7-9 status from PEND → PASS/FAIL based on actual artifact inspection
- **C2**: Appended Phase 4/5 entries (fold boundary, fit monitoring, trade execution log) to `validation_summary.json`; fixed per-check evidence paths via `evidence_map`
- **C3**: Refactored `write_sample_size_audit()` to accept `universe` and `signal_type`, break down by `universe × signal_type × quarter`, and add `low_sample_flag` (True when `n_events < 100`)
- **M3**: Changed checklist item 10 status from hardcoded `True` → `"ENFORCED AT BUILD TIME"` with DESIGN icon
- Wrote empty scenario-specific `trade_execution_violations_{suffix}.parquet` sentinels

**Phase 7** (`reports/pdf.py`, `reports/charts.py`):
- **CRITICAL**: Added `base_url=str(REPORTS_DIR)` to weasyprint `HTML()` call — fixes all embedded chart images in the PDF
- Added pre-flight `_check_figure_files()` that scans `report.md` for `![](...)` references and warns on missing files
- Removed dead `_FEATURES_14` list
- Added quintile-directory fallback for baseline equity data in `_read_portfolio_daily()`
- Clarified double-cadence fallback filename pattern with comment

**Phase 8** (`run_all.py`, `README.md`):
- Complete README rewrite with reproduction command, dependency list, data sources, expected runtime, and artifact paths table
- Extended `--dry-run` to all phases (0-8) with informative per-phase summaries
- Consistent empty-universe sentinel files in Phase 4 (matching Phase 5)
- Removed dead `_skip()` calls from Phases 2-5
- Used `REPORT_PDF` from `data/config.py` instead of hardcoded path
- Added `_manifest_coverage_sufficient()` helper and coverage check before skipping price/shares loaders
- Added tuning fallback warning when all universes are empty

Validation: `python -m compileall data/ features/ backtest/ reports/ run_all.py` passes; `python -m pytest tests/ -v` reports 47/47 passed; `python run_all.py --dry-run --tier enhanced --universes sp500 --skip-empty-universe` enumerates all phases correctly.

### 10.0.15 Phase 3 performance fixes — 2026-05-01

Four performance recommendations from `docs/review/phase3.md` were applied to `features/audit.py`:

1. **Remove `gc.collect()` from per-date streaming loop**: The per-date body of `_compare_one_streaming_date` no longer calls `gc.collect()` after deleting local DataFrames — pandas DataFrames have no reference cycles, so Python's reference counting frees them immediately. A single `gc.collect()` remains after the streaming loop in `run_streaming_vs_batch_test`.
2. **Swap loop nesting in `_compute_target_pit_percentiles`**: The outer loop now iterates over groups and the inner loop over columns, so each group's history is fetched once and the Fenwick tree is built once per group instead of once per column per group.
3. **Pre-group batch DataFrame in feature parity check**: The batch DataFrame is pre-grouped by `availability_date` into a dict before the per-date loop, replacing repeated O(n) linear scans with O(1) dict lookups.
4. **Drop redundant `sorted()` in `_sample_availability_dates`**: `pd.DatetimeIndex.unique()` already returns sorted values.

Validation: `python -m compileall features/` passes. `python -m pytest tests/ -v` — 47/47 passed.

### 10.0.15.1 Bug fix — 2026-05-02: missing id columns in no-momentum artifact batch

- **Bug**: `_build_artifact_features_for_audit` loaded batch rows with only
  `["availability_date"] + strict_cols + xsectional_cols`. The no-momentum
  streaming target builder (`_build_streaming_targets_no_momentum`) expects
  `BESTTICKER`, `SECTOR`, `SignalType`, and `call_entry_date` in the batch
  DataFrame to recompute QoQ deltas and PIT percentiles. Missing columns
  caused `KeyError: "['BESTTICKER', 'SignalType'] not in index"` at line 660
  in `_compute_target_timeseries_features`.
- **Fix**: Added these id columns to `compare_cols` in
  `_build_artifact_features_for_audit` using `dict.fromkeys` to preserve
  `strict_cols + xsectional_cols` order without duplication.

### 10.0.16 Phase 4 performance fixes — 2026-05-01

Performance recommendations 1-6 from `docs/review/phase4.md` were applied to `backtest/splits.py` and `backtest/model.py`:

1. **Extract `pd.to_datetime` from repeated calls in `_prepare_fold_sample`**: The `df[availability_col].iloc[train_idx]` column was converted to datetime four times (sorting, DatetimeIndex construction, min, max). Now converted once at the top and reused for all four uses.
2. **Avoid full DataFrame copy in `generate_folds`**: Removed `df.reset_index(drop=True)` which copied the entire feature DataFrame (millions of rows x hundreds of columns). Only the two date columns are extracted via `.to_numpy()`.
3. **Parallelize `tune_ridge` over alphas**: Extracted `_fit_ridge_alpha` module-level helper. The 6-alpha grid search per fold now uses `Parallel(n_jobs=-1)` matching LightGBM/XGBoost tuning. Each Ridge uses `n_jobs=1` to avoid nested parallelism.
4. **Remove redundant `.copy()` in `filter_model_sample`**: Removed the initial `df.copy()` that created a deep copy before filtering. The `_orig_df_index` column assignment is safe on the original DataFrame.
5. **Remove redundant `.copy()` after `.sort_values()`**: In `_compute_ticker_forward_return_matches`, `sort_values` already returns a new DataFrame, so the trailing `.copy()` was removed (called ~2600 times during forward-returns computation).
6. **Add dtype guard in `purge_train_for_horizon`**: Skip `pd.to_datetime` when `target_available_date_{h}d` is already datetime64[ns]; use `.to_numpy()` directly.

Items 7 and 8 from the review (ticker-level and horizon-level parallelism) were skipped as higher effort.

Validation: `python -m compileall backtest/` passes. `python -m pytest tests/ -v` — 47/47 passed.

### 10.0.17 Phase 2 performance fixes from code review — 2026-05-01

Applied 5 performance recommendations from `docs/review/phase2.md` to `features/engineer.py`:

1. **Eliminate duplicate `_group_aspect_theme_cols` call**: `_aspect_features` and `_theme_features` each called `_group_aspect_theme_cols(df)` independently. Now computed once in `compute_row_features` and passed as a parameter to both functions.
2. **Share sorted intermediate between `compute_qoq_deltas` and `compute_4q_trend`**: Both functions independently sorted by the same 5 keys, added `_orig_idx`/`_sort_tiebreaker`, and sorted. Extracted `_prepare_timeseries_base()` helper that performs the common sort + annotation, returning `(work, keys, block_cols)`.
3. **Stream groupby in `_merge_momentum_metrics`**: Replaced `dict(list(events.groupby(ticker_col)))` with a streaming `for tkr, ev in events.groupby(ticker_col)` loop, halving the per-ticker memory footprint.
4. **Restructure Fenwick tree to avoid redundant setup per column**: Split `_strict_historical_percentile` into `_prepare_percentile_dates` (shared date conversion + per-group sorting) and `_percentile_column` (per-column Fenwick tree walk using pre-sorted indices, with per-column value-validity filtering). `_strict_historical_percentile` remains as a backward-compatible wrapper.
5. **Use `pd.get_dummies` for sector one-hot**: Replaced 11 `df["SECTOR"] == sector` string comparisons with `pd.get_dummies` + `reindex` to guarantee all 11 GICS sector columns exist.

Validation: `python -m compileall features/` passes. `python -m pytest tests/ -v` — 46/47 passed (1 pre-existing failure in `test_spearman_pairwise_nan_compatibility`).

### 10.0.20 IC dedup sort-order fix — 2026-05-01

- **Bug**: `backtest/single_feature_ic.py` `_compute_cross_sectional_ic` and `_compute_sector_monthly_ic` assumed input was sorted by `call_entry_date` and asserted `is_monotonic_increasing` before using `groupby("BESTTICKER").last()` to keep the latest event per ticker. The features parquet (`results/features_enhanced.parquet`) has 40,484 rows out of chronological order because `features/engineer.py` does not sort by `call_entry_date` after computing time-series and PIT features. The sp500 universe happened to dodge the assertion (its unsorted year-month groups were below `min_samples` after SignalType/NaN filtering); the sp1500 universe hit a group large enough to trigger the assertion, causing `run_all.py --from-phase 5` to fail at Phase 5a for sp1500.
- **Fix**: replaced `assert is_monotonic_increasing` + `groupby("BESTTICKER").last()` with `sort_values("call_entry_date").groupby("BESTTICKER").last()` in both functions (lines 119, 160). This produces identical dedup results regardless of input row order.

### 10.0.21 Memory + cache fixes — 2026-05-01

Three issues surfaced by a Phase 5 OOM failure and a review of the Phase 1 cache-resumption logic.

**M1 — Phase 5.1 OOM: cap workers and drop columns before ProcessPoolExecutor fork.** `backtest/single_feature_ic.py` `run_single_feature_ic`. The full features DataFrame (~2.7M rows × ~100 columns, ~3–5 GB) was shared with 28 forked worker processes via a module-level `_IC_GLOBAL_DF`. Each worker's filtering/slicing triggered copy-on-write page duplication, exhausting 25 GB RAM and triggering 2 GB swap. Fixed by: (1) capping default `_n_jobs` at 2 (`min(os.cpu_count() or 4, 2)`) and making `run_all.py` pass `--n-jobs 2` for IC/quintile; (2) selecting only the ~24 columns actually needed by the workers and clearing `_cached_read_features` before fork; (3) defaulting Phase 5 outer fan-out to `--max-workers 1`.

**M2 — Phase 1.4 `is_cached_fresh` did not cache "empty" results.** `data/load_shares.py` `is_cached_fresh` (lines 258–289). The function checked parquet file existence before manifest status, so `empty` tickers (delisted/invalid, no parquet file) were re-fetched from yfinance on every invocation. Rewrote to match `data/load_prices.py`'s `is_cached_fresh`: check manifest entry first, then dispatch by status — `success` cached 14 days, `empty` cached 90 days, `error` never cached.

**M3 — `_manifest_coverage_sufficient` never returned `True`.** `run_all.py` `_manifest_coverage_sufficient` (line 156). The manifest has keys `updated_at` (string) and `tickers` (dict); iterating `manifest.values()` and calling `.get("status")` on the string raised `AttributeError`, caught by bare `except`, always returning `False`. Phase 1.3/1.4 could **never** be skipped even with a fully-populated cache (2756 success entries). Fixed by iterating `manifest.get("tickers", {}).values()`.

### 10.0.22 Portfolio batch parallelization — 2026-05-01

Phase 5.4 portfolio simulation batch runner (`backtest/portfolio_batch.py`) was serial — all 135 jobs processed in a single `for` loop on one core (~67 minutes). Parallelized with `ProcessPoolExecutor`.

**`backtest/portfolio_batch.py`:**
- New `_run_single_job(job)` module-level function serves as the subprocess worker entry point. Each worker independently loads signals, instantiates `PortfolioSimulator`, runs the simulation, and persists results.
- `run_jobs()` gains `max_workers` (default `min(os.cpu_count(), 2, len(jobs))`) and `parallel` parameters. Parallel path uses `ProcessPoolExecutor` with `as_completed` collection; sequential path (single job or `--no-parallel`) retains the original in-process simulator-sharing cache for same-universe jobs.
- Added `--max-workers` and `--no-parallel` CLI arguments.

**`run_all.py`:**
- Phase 5c portfolio batch command now passes `--max-workers` from `args.max_workers` (default 1).

**`max_workers` constraints:**
1. **Memory**: each subprocess loads independent copies of price matrix (~32 MB), features parquet (50–500 MB depending on tier), and signal data. Memory scales roughly linearly with worker count.
2. **Disk I/O**: all workers read the same 2583 price parquet files; contention becomes the bottleneck beyond ~8–12 workers.
3. **CPU**: inner loop is numpy/numba compiled (GIL-released), so true multi-core scaling works up to physical core count.
4. **Output safety**: suffix-specific portfolio outputs are process-safe; shared convenience copies (`trade_execution_log.parquet`, etc.) may have last-writer-wins semantics but are not authoritative.

### 10.0.23 Phase 5d robustness performance optimizations — 2026-05-01

The original `backtest/robustness.py` `run_all_robustness()` ran 12 analysis modules with redundant data passes, sequential inner loops of 350 feature×horizon×SignalType combinations, a duplicate portfolio simulation of `weekly_5d_monday`, and a per-ticker Python `searchsorted` loop in market-cap bucket assignment (~1M calls for SP500). Six optimizations targeting both computational reuse and unavoidable-work acceleration were applied across `backtest/robustness.py` and `run_all.py`:

**P0-1 — Pre-split df by SignalType.** The full features DataFrame is split into a `signal_dfs` dict (`{SignalType: filtered_df}`) once in `run_all_robustness()`. All 11 downstream section functions receive pre-filtered data, eliminating ~700 redundant O(N) boolean-mask filter passes over the full ~2.7M-row DataFrame.

**P0-2 — Merge OFAT lookback and weekly timing into one combined section.** `_run_portfolio_combined_section` replaces the two separate section functions. A single `PortfolioSimulator` instance runs all 11 unique cadence/lookback/timing combos; `weekly_5d_monday` is computed once and shared between the OFAT and weekly-timing output tables. Saves one full portfolio backtest and one price-table load.

**P0-3 — Reuse existing R8 column for baseline beta window.** The `beta_window=60` case now reads `df["pre_event_idio_resid_5d"]` (computed during Phase 2 feature engineering) instead of recomputing `compute_momentum_features`. Results for `beta_window=40` and `beta_window=90` are cached as `results/cache/momentum_beta_{window}.parquet` and loaded on subsequent runs. Eliminates 1–2 of 3 expensive momentum-feature recomputations.

**P0-4 — Memory-safe section-level parallelism.** Full robustness sections default to `_n_jobs=1`, and `run_all.py` Phase 5d passes `--n-jobs 1` by default via `--phase5-robustness-workers`. This avoids running market-cap matrices, beta-window momentum recomputation, bootstrap, and portfolio reruns concurrently inside the same robustness process.

**P1-1 — Parallelize inner 350-combo loops in subperiod IC and quintile.** `run_subperiod_ic` and `run_subperiod_quintile` can dispatch the `14 features × 5 horizons × 5 SignalTypes = 350` independent combinations via `ThreadPoolExecutor`, defaulting to 2 workers. Module-level global `_ROBUSTNESS_DFS` and worker functions (`_subperiod_ic_worker`, `_subperiod_quintile_worker`) follow the same pattern used by `backtest/quintile.py`. `ThreadPoolExecutor` is used instead of `ProcessPoolExecutor` to avoid fork-safety issues inside the section-level thread pool.

**P1-2 — Vectorize market-cap bucket inner loop.** `assign_market_cap_buckets` now builds aligned `(n_tickers × n_dates)` float64 price and shares matrices after pre-loading per-ticker histories. Matrices are forward-filled with `pd.DataFrame.ffill(axis=1)` so any date position yields the last valid value on or before that date. The per-date inner loop replaces the per-ticker `for tkr in members: np.searchsorted(px_dates[tkr], ...)` pattern (~500 tickers × ~2000 unique dates ≈ 1M Python-level calls) with two vectorised indexing operations (`px_mat[:, date_pos]`, `sh_mat[:, date_pos]`) per date, yielding T-1 values for all tickers at once. Cross-sectional quantile cutoffs use `np.quantile`; per-row bucket assignment uses `np.select`. Total matrix memory is ~32 MB for SP500.

### 10.0.24 Turnover liquidation fix — 2026-05-03

Turnover was found to be nearly identical across all models within the same (universe, cadence) group (variation <0.3%), making it useless for model comparison. Root cause: `_process_rebalance()` was liquidating the portfolio (`current_weights = {}`) whenever no eligible signals existed in the lookback window, producing artificial turnover = 2.0 on each liquidation. Since the "Total" earnings-call signal is sparse (~8 stocks/day for SP500), this happened on ~70% of trading days.

**Fix** (`backtest/portfolio.py:1338-1340`): when `agg_weights_df` is empty, `_process_rebalance()` now returns early, keeping existing positions intact. Positions persist until new signals arrive and replace them, or until gap accounting phases them out. Turnover now reflects actual trading-driven portfolio changes rather than forced liquidations.

**Impact**: annualized turnover will drop significantly (no more artificial 2.0 liquidations on empty-signal days). Models may still show similar turnover since all operate on the same sparse signal grid, but turnover will now be driven by the rate of position replacement rather than signal availability.

### 10.3 Robustness conclusions
- Subperiod stability
- Marginal effect of sector neutralization
- Market-cap bucket heterogeneity
- Sector-classification sensitivity
- Equal-weight vs ATC-score-weighted
- 5 required OFAT sensitivities
- R8 beta window sampling
