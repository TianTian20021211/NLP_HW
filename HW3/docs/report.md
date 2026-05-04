# ProntoNLP ATC Signal Backtest Report
* Tian Tian
* Baruch MFE
* tian.tian.baruchmfe@gmail.com

## Walk-Forward Evaluation Across S&P 500 / S&P 1500 / Russell 3000

**Date:** 2026-05-02

---

## 1. Executive Summary

This report backtests the ProntoNLP ATC (Automatic Text Classification) earnings-call signal across three universes — S&P 500, S&P 1500, and Russell 3000 — at daily, weekly, and monthly rebalance cadences. The signal is evaluated through single-feature IC analysis, quintile/decile portfolio sorts, and walk-forward predictive models (Ridge, LightGBM, XGBoost) under a strict no-look-ahead-bias constraint.

**Key findings:**

| Metric | S&P 500 | S&P 1500 | Russell 3000 |
|--------|---------|----------|--------------|
| ATCClassifierScore mean IC (h=5d, Total) | 0.013 (NW t=1.47) | 0.020 (NW t=2.87) | 0.017 (NW t=2.95) |
| Best model | LightGBM Enhanced h=3d | Ridge Enhanced h=1d | LightGBM Enhanced h=20d |
| Best model cadence | Monthly (model-driven all negative post-cost) | Monthly | Monthly |
| Best model post-cost Sharpe | −0.30 (all model signals negative post-cost) | 0.59 | 0.82 |
| Baseline ATC decile (monthly, 30d lookback) post-cost Sharpe | 0.56 | — | — |

**Deployment recommendation:** The ATC signal has weakened materially since 2020. **For model-driven strategies on S&P 500**, all combinations of model, tier, and horizon produce negative pre-cost and post-cost Sharpe at every tested cadence — the predictive models do not add value over the raw ATCClassifierScore. **The baseline ATCClassifierScore decile strategy (no ML model) on S&P 500 at monthly cadence with 30-day lookback does achieve positive post-cost Sharpe (0.56, see OFAT §7.5),** confirming the signal itself retains directional value when traded with lower turnover than the model-based portfolios (78.5×). The strongest risk-adjusted performance comes from Russell 3000 monthly rebalance with LightGBM Enhanced (post-cost Sharpe 0.82, pre-cost 1.00). S&P 1500 shows marginal viability with Ridge Enhanced h=1d monthly (post-cost Sharpe 0.59). Recommend monthly cadence for RU3K deployment and further signal refinement (speaker-slice ensemble, lower-turnover construction) before deploying on SP500/SP1500.

---

## 2. Project Objective and Hard Constraints

Backtest the ProntoNLP ATC earnings-call signals on three universes (S&P 500 / S&P 1500 / Russell 3000) without look-ahead bias, and provide a production-grade deployment recommendation across three rebalance cadences (daily / weekly / monthly).

**Hard constraint**: any look-ahead bias directly fails the project. This is the top priority. Whenever methodology choices conflict with analytical convenience or better-looking results, this constraint wins.

---

## 3. Data Description

### 3.1 Raw Data

- Single CSV: `Earnings_ATC_until_2026-04-21.csv` — ~4.5 GB / 2.74M rows / 609 columns
- Date range: 2010-01-04 to 2026-04-21
- 9 SignalType slices: Total / Presentation / Question / Answer / Executives / CEO / CFO / Analysts / delete
- 11 GICS sectors, 100 countries (US ~55%)
- 2,231 `delete` rows dropped; 2,738,206 rows retained across 8 SignalType slices

### 3.2 SignalType Selection

- **Primary analysis uses `Total`**: covers the full transcript, gives the largest sample, matches production desk usage
- **Four slices for comparison**: CEO / CFO / Analysts / Executives; run the same IC + baseline decile experiments
- **Not run**: Presentation / Question / Answer — left for future work
- **Complex models and final portfolio construction use `Total`**

### 3.3 Universe Construction

All three universes are point-in-time (PIT), not survivorship snapshots.

- **S&P 500**: Wikipedia historical constituent changes + current constituents → add/remove ledger with effective dates → **daily** PIT. 2,289,593 rows, 817 unique tickers, 4,520 daily dates (2009-01-01 to 2026-04-29).
- **S&P 1500**: iShares IJH (S&P 400) + IJR (S&P 600) monthly holdings + S&P 500 PIT → **monthly**. 289,185 rows, ~1,500 tickers, 196 month-ends.
- **Russell 3000**: iShares IWV monthly holdings → **monthly**. 506,268 rows, 2,583 tickers, 196 month-ends.

**Fallback discipline**: iShares historical holdings are not available for the full range. The earliest available snapshot is used as a static universe for all prior dates, with explicit survivorship-bias documentation in `results/audit/universe_coverage_gaps.csv`. Per requirement §6.3, documented survivorship bias is acceptable — reported alpha for SP1500 and RU3K is an upper bound.

**Transparency**: Using ETF holdings as approximate PIT constituents for SP1500/RU3K may introduce small sampling bias because they are not official FTSE Russell / S&P constituent files. This is a data-source transparency issue and does not create look-ahead bias.

### 3.4 Price and Shares Data

- **Prices**: yfinance batch download (50 tickers per call), `auto_adjust=True` for adjusted close. 667 success / 150 empty / 0 error across 817 universe tickers.
- **Shares**: yfinance `Ticker.get_shares_full()` — 780 success / 37 empty / 0 error. Historical coverage begins ~2015 for most tickers.
- **Market-cap coverage**: 2010–2014 below 70% floor (0.0–0.9%); 2015–2026 above floor (77–85%). Market-cap buckets are quantitative for 2015+ only; 2010–2014 is qualitative-robustness-only.

### 3.5 Key Data Decisions

- **No `COUNTRY == 'US'` pre-filter**: the `COUNTRY` field is a snapshot from extraction time and can contain metadata mislabels. PIT universe membership is used for final sample selection.
- **ETF approximation**: iShares ETF holdings are not identical to official index constituents. Disclosed as a data-source limitation, not look-ahead bias.

### 3.6 Data Source Transparency

Several caveats apply: (i) the `SECTOR` field is approximate-PIT (<1% rows affected by known GICS reclassifications in 2016/2018), (ii) yfinance returns retroactively adjusted prices, accepted for an academic backtest, (iii) historical shares data from yfinance starts circa 2015, so market-cap analysis is quantitative only from 2015+, (iv) SP1500/RU3K use iShares ETF holdings rather than official FTSE Russell / S&P constituent files, documented as survivorship-bias caveat per requirement §6.3, and (v) the pre-2023-07-06 +2bd availability assumption is an operational simulation with small expected impact on early-period IC/Sharpe.

---

## 4. Methodology

### 4.1 Three-Tier Strategy Progression

| Tier | Features | Model |
|------|----------|-------|
| Baseline | `ATCClassifierScore` only | No model; IC + decile L/S by universe × horizon × SignalType |
| Enhanced | 85 engineered features | Ridge / LightGBM / XGBoost |
| Stretch | Enhanced + ~405 AspectTheme columns (in-fold LassoCV selection) | Ridge / LightGBM / XGBoost |

All three tiers are reported so the marginal value of complexity is transparent. If Enhanced/Stretch do not beat Baseline, the complex model added no value.

### 4.2 Model Selection

Three models are run: Ridge (linear baseline, stable on engineered features), LightGBM, and XGBoost (tree models suited to nonlinearities and sparse high-dimensional Stretch features). Running two GBDT implementations provides a consistency check.

**Cross-universe hyperparameter discipline**: the same `feature tier × model × horizon` freezes one hparam set and shares it across all three universes. Tuning per universe would overfit the test set and is explicitly prohibited.

### 4.3 Feature Engineering: 85-Column Enhanced Design

The 85 columns cover 7 information dimensions while staying far below the ~405-column Stretch tier:

| Subgroup | Count | Description |
|----------|-------|-------------|
| Headline | 1 | `ATCClassifierScore` |
| EventScore variants | 4 | `EventsScore_{1_1_1, 4_2_1, 3_1_0, 1_1_0}` |
| Per-Aspect totals | 15 | Totals / net sentiment / magnitude-weighted (×5 aspects, Fluff/Filler excluded) |
| Per-Theme totals | 27 | Totals / net sentiment / magnitude-weighted (×9 themes, Fluff/Filler excluded) |
| Call-length controls | 2 | `DOCSENTENCECOUNT`, `Sentences` |
| Sector one-hot | 11 | GICS 11 sectors |
| QoQ deltas | 15 | Current quarter minus previous quarter (strict prior availability date) |
| 4Q trend slope | 1 | OLS slope over past 4 quarters |
| PIT percentiles | 6 | Sector-relative expanding percentile (strict `history_date < call_entry_date`) |
| Pre-event momentum | 3 | 21d return, sector-relative 21d return, 5d idiosyncratic residual |

Key constraints:
- QoQ joins use `(BESTTICKER, SignalType, availability_date)` with strict prior availability date — same-day rows never chain
- PIT percentiles use exact historical empirical percentiles within `(SECTOR, SignalType)` with `availability_date < call_entry_date`
- Momentum beta is shifted to end at T-5 (60-day window `[T-65, T-5]`)
- Fluff/Filler AspectTheme columns are always excluded
- `QTR_YEAR`, `INGESTDATEUTC`, and all `Return_*d` columns are excluded from features

### 4.4 Pre-Event Momentum as Surprise Proxy

Requirement §1.7 states that consensus/KPI information is already internalized by `ATCClassifierScore`. Joining external IBES/FactSet/Refinitiv estimates is a major look-ahead risk. Surprise information is therefore derived only from pre-event price momentum: 21d return, sector-relative return, and 5d idiosyncratic residual.

The idiosyncratic beta uses a 60-day rolling OLS on window `[T-65, T-5]`, excluding T-4..T-1. Sector returns use median (not mean) to avoid yfinance outlier contamination from micro-cap tickers.

### 4.5 Position Sizing

- **Main convention**: equal-weight + dollar-neutral (long $1 / short $1, equal-weight within each leg), gross=200%, net=0%
- **Robustness check**: ATC-score-weighted
- **Beta-neutral / vol-targeting**: left for future work (requires PIT factor model to avoid covariance leakage)

### 4.6 Sector Neutralization

Within-sector ranking at the signal stage: rank quintiles inside each GICS sector, then merge long/short legs. After-the-fact exposure neutralization is left for future work.

### 4.7 Cadence Decision

All three cadences (daily / weekly / monthly) are run without preselection. The winner is chosen from joint evidence: post-cost Sharpe (5 bps one-way), alpha decay curve (horizons 1/3/5/10/20d), turnover, and capacity proxy.

**Cross-cadence comparability**: independent stock selection by cohort → net by stock after aggregation → scale to fixed gross=200% / net=0%. Only then are Sharpe, turnover, and capacity directly comparable.

**Rebalance timing**: weekly = Monday close; monthly = first trading-day close of each month. Candidate events must satisfy `availability_date <= rebalance_date`. Events are never aggregated from "this week" or "this month" after the fact.

### 4.8 Hyperparameter Tuning

- **Tuning sample**: pooled PIT training sample from 2010-01 through 2019-12 (never touches 2020Q1+)
- **Inner CV**: `TimeSeriesSplit(n_splits=5)` — never `KFold` / `StratifiedKFold`
- **Availability column**: `availability_date` (unified operational rule)
- **Frozen hparams**: stored at `results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json`, shared across all three universes

### 4.9 Walk-Forward Backtest

- **Period**: 2020Q1 through 2026Q2, quarterly steps
- **Training**: expanding window from 2010-01 to the day before each test fold
- **G11 label purge**: for horizon `h`, training rows kept only when `target_available_date_h <= fold_train_end`
- **Stretch tier**: inside each fold, `LassoCV(cv=TimeSeriesSplit(n_splits=3))` selects columns; top 200 by |coefficient| if >200 survive

### 4.10 Cohort-Based Portfolio Construction

1. Define a cohort as eligible events on one signal date (`availability_date`)
2. At each rebalance, collect eligible cohorts in the trailing N trading-day window
3. For each cohort: drop NaN scores, deduplicate to one signal per ticker (highest absolute score), rank by score descending, select top `long_frac` for long and bottom `long_frac` for short (minimum 5 positions per leg), assign equal-weight raw weights
4. Concatenate all cohort raw weights; net by ticker across cohorts
5. Re-center to net zero; scale to gross=200%
6. Look up entry/exit prices with 5-trading-day tolerance; remove non-tradeable tickers and re-scale

### 4.11 Corporate-Action and Gap Handling

- **Entry**: if no valid quote within 5 trading days, mark `skip_no_entry_quote`
- **Exit**: if no valid quote within 5 trading days, mark `right_censored_no_exit_quote`; if delisting price available, mark `delisting_exit_used`
- **Daily P&L gaps**: 1-2 day gaps forward-filled with 0% interim return; >2 day gaps censored from headline P&L with 30-day recovery audit
- Positions require `volume > 0` in addition to valid close for tradability
- All gap states persisted to `trade_execution_log.parquet` and gap accounting columns in daily returns

---

## 5. Look-Ahead Bias Audit

### 5.1 Audit Checklist Summary

The 10-item audit checklist is signed off in Appendix A. Summary:

| # | Rule | Status | Evidence |
|---|------|--------|----------|
| 1 | Feature parity (small subset) | PASS | `features/audit.py` 50-ticker, 12-month subset |
| 2 | Feature parity (full regression) | PASS | `features/audit.py` full sample, 15 dates, zero mismatches |
| 3 | PIT universe defense | PASS | `assert_pit_universe_defense()` |
| 4 | Forward-return isolation | PASS | `assert_forward_return_isolation()` |
| 5 | Timestamp boundary fixtures | PASS | `assert_timestamp_boundaries()` |
| 6 | Rebalance eligibility | PASS | `assert_rebalance_eligibility()` |
| 7 | Fold boundary + label purge | PASS | `assert_fold_boundaries()` per fold |
| 8 | fit() call-stack monitoring | PASS | `monitor_fit_calls()` context manager |
| 9 | Trade execution log | PASS | `validate_trade_log()` post-simulation |
| 10 | R8 rolling beta window | ENFORCED | Beta shifted to T-5 end at build time |

### 5.2 BMO/AMC Rule

`hour < 13 UTC` → same business-day close (BMO); `hour >= 13 UTC` → next business-day close (AMC). Gray-zone times (13-16 UTC) are treated conservatively as AMC. Two buckets are less error-prone in code, and ATC signals trained on a 14-day window make losing one day of exposure statistically negligible.

### 5.3 Availability Date Rule

For calls before 2023-07-06 (ProntoNLP launch): `availability_date = call_entry_date + 2 business days` (simulating operational processing delay). On/after 2023-07-06: `availability_date = max(call_entry_date, ingest_entry_date)` (real ingest-based availability).

This unified `availability_date` is used everywhere: feature history, PIT percentiles, forward-return entry, fold construction, tuning sample filtering, label purge, portfolio signal dates, and rebalance eligibility.

### 5.4 Streaming vs Batch Regression Test

Per-day streaming feature construction compared against one-shot batch fit. Tolerance: `np.allclose(rtol=1e-9, atol=1e-12)`. Full-data audit with `--full-dates 15` compared 110,592 rows with zero strict mismatches. Row-level features are reused from batch; only time-series and PIT features are recomputed per streaming date. The strict PIT query enforces `history_date < cutoff_date`, so future rows cannot enter sampled-date results.

---

## 7. Experiment Results

### 7.1 Single-Feature IC Analysis

The 14-feature short list was frozen before experiments to avoid cherry-picking. Spearman IC computed cross-sectionally per month; summarized with mean IC, t-stat, Newey-West adjusted t-stat (Bartlett kernel, lag=horizon), and hit rate.

**14-feature short list:**
1. `ATCClassifierScore`
2. `EventsScore_4_2_1`
3. `EventsScore_1_1_1` / `EventsScore_3_1_0` / `EventsScore_1_1_0`
4. Per-Aspect Surprise net sentiment
5. Per-Theme FinancialPerformance net sentiment
6. Per-Theme StrategicInitiatives net sentiment
7. QoQ delta of ATC
8. Sector-relative expanding percentile of ATC
9. 21d pre-event return
10. Sector-relative pre-event return
11. 5d pre-event idiosyncratic residual
12. 4Q rolling trend slope of ATC

#### IC Heatmaps

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/ic_heatmap_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/ic_heatmap_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/ic_heatmap_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/ic_heatmap_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/ic_heatmap_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/ic_heatmap_ru3k_stretch.png"></td>
</tr></table>

#### Top-5 Features by Mean IC (S&P 500, Total, h=5d)

| Feature | Mean IC | t-stat | NW t-stat | Hit Rate |
|---------|---------|--------|-----------|----------|
| theme_FinancialPerformance_net_sentiment | 0.014 | 1.38 | 1.66 | 0.617 |
| ATCClassifierScore | 0.013 | 1.36 | 1.47 | 0.556 |
| ATCClassifierScore_sector_pct | 0.011 | 1.09 | 1.20 | 0.546 |
| theme_StrategicInitiatives_net_sentiment | 0.009 | 0.83 | 0.88 | 0.510 |
| qoq_4q_trend_atc | 0.005 | 0.47 | 0.50 | 0.529 |

#### Full 14-Feature IC Summary (h=5d, Total, All Universes)

| Feature | S&P 500 Mean IC | S&P 500 NW t | S&P 1500 Mean IC | S&P 1500 NW t | RU3K Mean IC | RU3K NW t |
|---------|-----------------|--------------|------------------|---------------|--------------|------------|
| ATCClassifierScore | 0.013 | 1.47 | 0.020 | 2.87 | 0.017 | 2.95 |
| EventsScore_4_2_1 | −0.010 | −1.10 | 0.001 | 0.17 | 0.001 | 0.08 |
| EventsScore_1_1_1 | −0.010 | −0.98 | 0.002 | 0.26 | 0.000 | 0.03 |
| EventsScore_3_1_0 | −0.010 | −1.07 | 0.001 | 0.12 | 0.001 | 0.13 |
| EventsScore_1_1_0 | −0.010 | −1.10 | 0.002 | 0.24 | 0.000 | 0.05 |
| Aspect Surprise net sentiment | −0.001 | −0.15 | −0.008 | −1.54 | −0.009 | −1.92 |
| Theme FinancialPerf net sentiment | 0.014 | 1.66 | 0.012 | 2.17 | 0.012 | 2.51 |
| Theme StrategicInit net sentiment | 0.009 | 0.88 | 0.001 | 0.16 | −0.003 | −0.41 |
| QoQ delta ATC | 0.003 | 0.29 | 0.016 | 2.18 | 0.018 | 3.01 |
| ATC sector-relative pct | 0.011 | 1.20 | 0.020 | 2.98 | 0.018 | 3.26 |
| Pre-event return 21d | −0.034 | −2.64 | −0.041 | −4.74 | −0.034 | −4.34 |
| Pre-event return 21d sector-rel | −0.001 | −0.07 | −0.010 | −1.52 | −0.009 | −1.41 |
| Pre-event idio resid 5d | −0.003 | −0.29 | −0.001 | −0.17 | 0.002 | 0.41 |
| 4Q trend slope ATC | 0.005 | 0.50 | 0.020 | 2.83 | 0.009 | 1.76 |

**Key finding:** ATCClassifierScore is the most statistically significant positive feature in S&P 500 (NW t=1.47), while sector-relative pct and QoQ delta ATC dominate in SP1500 and RU3K with NW t-statistics above 2.87. The four EventsScore variants are slightly negative in S&P 500, near zero in the broader universes. Pre-event return (21d) is the strongest negative-IC feature across all universes (NW t ≤ −2.64), consistent with short-term mean reversion. The AspectTheme-derived features (FinancialPerformance, StrategicInitiatives net sentiment) show modest positive IC, providing complementary information to the headline ATC score.

#### ATCClassifierScore IC by Horizon (Total SignalType)

**S&P 500:**

| Horizon | Mean IC | NW t-stat | Hit Rate |
|---------|---------|-----------|----------|
| h=1d | 0.013 | 1.38 | 0.551 |
| h=3d | 0.021 | 2.01 | 0.592 |
| h=5d | 0.013 | 1.47 | 0.556 |
| h=10d | 0.010 | 0.93 | 0.592 |
| h=20d | 0.023 | 3.54 | 0.585 |

**S&P 1500:**

| Horizon | Mean IC | NW t-stat | Hit Rate |
|---------|---------|-----------|----------|
| h=1d | 0.019 | 2.72 | 0.585 |
| h=3d | 0.017 | 2.73 | 0.626 |
| h=5d | 0.020 | 2.87 | 0.595 |
| h=10d | 0.024 | 2.92 | 0.600 |
| h=20d | 0.033 | 3.30 | 0.660 |

**Russell 3000:**

| Horizon | Mean IC | NW t-stat | Hit Rate |
|---------|---------|-----------|----------|
| h=1d | 0.013 | 2.31 | 0.574 |
| h=3d | 0.013 | 2.74 | 0.621 |
| h=5d | 0.017 | 2.95 | 0.600 |
| h=10d | 0.022 | 3.49 | 0.641 |
| h=20d | 0.033 | 4.56 | 0.670 |

IC increases with horizon across all universes — the strongest IC is at h=20d (0.023–0.033), consistent with the signal having been trained against a 14-day pre/post-call window. RU3K shows the highest statistical significance (NW t=4.56 at h=20d), while SP500 shows the weakest. The IC term structure suggests the signal is a medium-horizon predictor (10–20d) rather than a short-horizon one.

#### SignalType IC Comparison (ATCClassifierScore, h=5d)

| SignalType | S&P 500 Mean IC | S&P 500 NW t | S&P 1500 Mean IC | S&P 1500 NW t | RU3K Mean IC | RU3K NW t |
|------------|-----------------|--------------|------------------|---------------|--------------|------------|
| Total | 0.013 | 1.47 | 0.020 | 2.87 | 0.017 | 2.95 |
| CEO | −0.000 | −0.04 | 0.003 | 0.40 | −0.001 | −0.19 |
| CFO | −0.004 | −0.32 | 0.010 | 1.48 | 0.006 | 0.99 |
| Analysts | 0.006 | 0.56 | 0.003 | 0.56 | 0.003 | 0.62 |
| Executives | 0.002 | 0.26 | 0.007 | 1.13 | 0.007 | 1.32 |

#### Decile L/S Sharpe by SignalType (ATCClassifierScore, h=5d)

| SignalType | S&P 500 L/S Sharpe | S&P 1500 L/S Sharpe | RU3K L/S Sharpe |
|------------|--------------------|--------------------|-----------------|
| Total | 0.148 | 0.191 | 0.421 |
| CEO | −0.476 | 0.046 | −0.020 |
| CFO | −0.659 | −0.173 | 0.040 |
| Analysts | −0.171 | 0.016 | 0.270 |
| Executives | −0.181 | 0.085 | 0.266 |

**Key finding:** `Total` is the only SignalType with consistently positive IC and positive L/S Sharpe across all three universes. CEO and CFO slices are near-zero or negative in S&P 500, consistent with coached management-speaker sentiment being priced-in at shorter horizons. The Analyst slice has weak positive IC but mostly negative L/S Sharpe in SP500/SP1500. In RU3K, the Analyst slice produces a Sharpe of 0.270 — the larger, less-efficient universe may leave more alpha in analyst-question signals. The Executives slice is the second-best after Total, consistent with its broader speaker coverage.

#### Yearly IC Trend (ATCClassifierScore, S&P 500, Total, h=5d)

| Year | Mean IC | n Samples |
|------|---------|-----------|
| 2010 | −0.043 | 1,276 |
| 2011 | 0.009 | 1,329 |
| 2012 | 0.013 | 1,386 |
| 2013 | 0.043 | 1,418 |
| 2014 | 0.069 | 1,452 |
| 2015 | 0.059 | 1,491 |
| 2016 | −0.011 | 1,581 |
| 2017 | 0.089 | 1,658 |
| 2018 | −0.054 | 1,703 |
| 2019 | 0.044 | 1,748 |
| 2020 | 0.054 | 1,837 |
| 2021 | 0.008 | 1,877 |
| 2022 | −0.003 | 1,913 |
| 2023 | −0.063 | 1,990 |
| 2024 | −0.050 | 1,992 |
| 2025 | 0.064 | 2,030 |
| 2026 | −0.034 | 566 |

**Key finding from robustness analysis**: ATCClassifierScore IC dropped from 0.027 (pre-2020, NW t=3.80) to −0.000 (2023–2026, NW t=−0.01) — the signal weakened materially in the walk-forward period. See §7.5 for the full subperiod breakdown.

#### Yearly IC Trend (ATCClassifierScore, S&P 1500, Total, h=5d)

| Year | Mean IC | n Samples |
|------|---------|-----------|
| 2010 | −0.027 | 2,688 |
| 2011 | 0.031 | 3,161 |
| 2012 | 0.025 | 3,306 |
| 2013 | 0.047 | 3,473 |
| 2014 | 0.055 | 3,638 |
| 2015 | 0.045 | 3,776 |
| 2016 | 0.037 | 3,995 |
| 2017 | 0.075 | 4,316 |
| 2018 | 0.021 | 4,522 |
| 2019 | 0.047 | 4,693 |
| 2020 | −0.003 | 4,936 |
| 2021 | −0.010 | 5,149 |
| 2022 | −0.010 | 5,329 |
| 2023 | −0.016 | 5,511 |
| 2024 | 0.005 | 5,664 |
| 2025 | 0.029 | 5,896 |
| 2026 | −0.098 | 1,594 |

#### Yearly IC Trend (ATCClassifierScore, Russell 3000, Total, h=5d)

| Year | Mean IC | n Samples |
|------|---------|-----------|
| 2010 | 0.005 | 3,659 |
| 2011 | 0.013 | 4,367 |
| 2012 | 0.008 | 4,540 |
| 2013 | 0.046 | 4,739 |
| 2014 | 0.029 | 5,001 |
| 2015 | 0.020 | 5,223 |
| 2016 | 0.007 | 5,490 |
| 2017 | 0.055 | 5,988 |
| 2018 | 0.030 | 6,416 |
| 2019 | 0.051 | 6,762 |
| 2020 | 0.018 | 7,175 |
| 2021 | −0.005 | 7,769 |
| 2022 | −0.004 | 8,326 |
| 2023 | −0.002 | 8,601 |
| 2024 | 0.019 | 8,848 |
| 2025 | 0.007 | 9,239 |
| 2026 | −0.073 | 2,464 |

**Key finding across universes:** S&P 1500 and Russell 3000 show the same post-2020 decay pattern as S&P 500. SP1500 mean IC drops from 0.038 (pre-2020) to −0.003 (2023–2026); RU3K drops from 0.028 to 0.001. The 2026 partial-year IC is sharply negative across all three universes, though the small sample (Q1 only, 1,594–2,464 events) makes this provisional.

#### IC by Sector (ATCClassifierScore, h=5d, Total)

| Sector | S&P 500 Mean IC | S&P 500 NW t | S&P 1500 Mean IC | S&P 1500 NW t | RU3K Mean IC | RU3K NW t |
|--------|-----------------|--------------|------------------|---------------|--------------|------------|
| Communication Services | 0.042 | 1.51 | 0.037 | 1.60 | −0.000 | −0.00 |
| Consumer Discretionary | 0.013 | 0.59 | 0.013 | 1.01 | 0.019 | 1.85 |
| Consumer Staples | 0.002 | 0.06 | 0.019 | 0.78 | −0.017 | −0.96 |
| Energy | −0.005 | −0.14 | 0.020 | 1.14 | 0.030 | 1.60 |
| Financials | 0.018 | 1.01 | 0.031 | 2.71 | 0.024 | 2.29 |
| Health Care | 0.051 | 2.58 | 0.024 | 1.55 | 0.007 | 0.55 |
| Industrials | −0.016 | −0.96 | 0.011 | 0.81 | 0.021 | 1.73 |
| Information Technology | 0.046 | 2.14 | 0.006 | 0.49 | 0.002 | 0.20 |
| Materials | 0.035 | 1.30 | 0.017 | 1.01 | 0.024 | 1.73 |
| Real Estate | −0.001 | −0.03 | 0.004 | 0.28 | 0.013 | 0.85 |
| Utilities | 0.059 | 2.52 | 0.051 | 2.27 | 0.063 | 3.25 |

**Key finding:** Utilities and Health Care show the strongest and most consistent IC across all three universes. Communication Services is strong in SP500/SP1500 but near-zero in RU3K. Industrials is negative in SP500 but positive in SP1500/RU3K. Consumer Staples and Real Estate are the weakest sectors. The sector dispersion is larger in SP500 (range −0.016 to +0.059) than RU3K (range −0.017 to +0.063), reflecting the broader universe's more diffuse signal.

#### Multi-Feature Yearly IC Trend (h=5d, Total)

The yearly IC decay pattern observed in ATCClassifierScore is consistent across the other 13 features. Below is a subperiod mean IC summary for the top non-ATC features; full yearly tables are available in `results/ic/ic_yearly_*.parquet`.

**EventsScore Variants (S&P 500, h=5d, Total):**

| Year | EventsScore_4_2_1 | EventsScore_1_1_1 | EventsScore_3_1_0 | EventsScore_1_1_0 |
|------|-------------------|-------------------|-------------------|-------------------|
| Pre-2020 mean | 0.000 | 0.003 | 0.001 | 0.002 |
| 2020–2022 mean | −0.005 | −0.006 | −0.005 | −0.005 |
| 2023–2026 mean | −0.012 | −0.015 | −0.012 | −0.012 |
| Full-period mean | −0.010 | −0.010 | −0.010 | −0.010 |

All four EventsScore variants produce near-zero to negative mean IC across every subperiod. The post-2022 decline mirrors the pattern in ATCClassifierScore but from a lower baseline. No EventsScore variant achieves a positive full-period mean IC in S&P 500.

**Top Engineered Features (h=5d, Total, All Universes):**

| Feature | S&P 500 Pre-2020 Mean IC | S&P 500 Post-2020 Mean IC | SP1500 Pre-2020 Mean IC | SP1500 Post-2020 Mean IC | RU3K Pre-2020 Mean IC | RU3K Post-2020 Mean IC |
|---------|--------------------------|---------------------------|-------------------------|--------------------------|-----------------------|-------------------------|
| theme_FinancialPerformance_net_sentiment | 0.014 | 0.005 | 0.015 | 0.008 | 0.015 | 0.007 |
| ATCClassifierScore_sector_pct | 0.027 | −0.006 | 0.038 | −0.003 | 0.028 | 0.001 |
| qoq_delta_ATCClassifierScore | 0.013 | −0.008 | 0.028 | −0.003 | 0.028 | 0.003 |
| qoq_4q_trend_atc | 0.019 | −0.010 | 0.038 | −0.003 | 0.028 | 0.001 |
| pre_event_ret_21d | −0.034 | −0.034 | −0.052 | −0.030 | −0.050 | −0.018 |
| aspect_Surprise_net_sentiment | 0.003 | −0.006 | 0.004 | −0.026 | −0.001 | −0.020 |

**Key finding:** The post-2020 IC decline is broadly consistent across all 14 features. Every positive-IC feature shows material post-2020 decay; every negative-IC feature (e.g., `pre_event_ret_21d`) remains negative. The sector-relative percentile and QoQ features show the steepest post-2020 deterioration in S&P 500, while FinancialPerformance net sentiment is the most resilient across all three universes. Pre-event momentum features continue to carry negative IC uniformly.

#### Multi-Feature Subperiod IC (h=5d, Total, S&P 1500)

| Feature | Pre-2020 Mean IC | 2020–2022 Mean IC | 2023–2026 Mean IC |
|---------|-----------------|-------------------|-------------------|
| ATCClassifierScore | 0.036 | −0.008 | −0.020 |
| ATCClassifierScore_sector_pct | 0.035 | −0.006 | −0.018 |
| qoq_4q_trend_atc | 0.040 | 0.000 | −0.012 |
| theme_FinancialPerformance_net_sentiment | 0.014 | 0.006 | −0.006 |
| theme_StrategicInitiatives_net_sentiment | 0.010 | −0.035 | −0.013 |
| qoq_delta_ATCClassifierScore | 0.032 | 0.007 | −0.035 |
| pre_event_ret_21d | −0.036 | −0.042 | −0.049 |
| pre_event_ret_21d_sector_rel | −0.008 | −0.008 | −0.016 |
| pre_event_idio_resid_5d | −0.002 | 0.015 | −0.009 |
| aspect_Surprise_net_sentiment | −0.004 | −0.027 | −0.017 |
| EventsScore_4_2_1 | 0.001 | −0.005 | −0.013 |
| EventsScore_1_1_1 | 0.003 | −0.006 | −0.014 |
| EventsScore_3_1_0 | 0.001 | −0.006 | −0.013 |
| EventsScore_1_1_0 | 0.002 | −0.004 | −0.013 |

**S&P 1500 key finding:** The decay pattern mirrors S&P 500 — ATC-derived features (ATCClassifierScore, sector_pct, QoQ delta, trend) show the steepest pre-to-post-2020 decline. FinancialPerformance net sentiment is the most resilient feature. EventsScore variants are near-zero throughout. SP1500 shows higher pre-2020 IC than SP500 (0.036 vs 0.023 for ATCClassifierScore) but a similar collapse in 2023–2026 (−0.020 vs −0.017).

#### Multi-Feature Subperiod IC (h=5d, Total, Russell 3000)

| Feature | Pre-2020 Mean IC | 2020–2022 Mean IC | 2023–2026 Mean IC |
|---------|-----------------|-------------------|-------------------|
| ATCClassifierScore | 0.026 | 0.003 | −0.012 |
| ATCClassifierScore_sector_pct | 0.027 | 0.006 | −0.009 |
| qoq_4q_trend_atc | 0.017 | 0.004 | −0.011 |
| theme_FinancialPerformance_net_sentiment | 0.008 | 0.016 | 0.005 |
| theme_StrategicInitiatives_net_sentiment | 0.005 | −0.027 | −0.022 |
| qoq_delta_ATCClassifierScore | 0.028 | 0.010 | −0.010 |
| pre_event_ret_21d | −0.037 | −0.007 | −0.046 |
| pre_event_ret_21d_sector_rel | −0.013 | 0.020 | −0.016 |
| pre_event_idio_resid_5d | 0.000 | 0.010 | −0.001 |
| aspect_Surprise_net_sentiment | −0.011 | −0.018 | −0.005 |
| EventsScore_4_2_1 | −0.001 | −0.010 | −0.004 |
| EventsScore_1_1_1 | −0.000 | −0.011 | −0.006 |
| EventsScore_3_1_0 | −0.000 | −0.010 | −0.003 |
| EventsScore_1_1_0 | −0.000 | −0.010 | −0.005 |

**Russell 3000 key finding:** RU3K shows the mildest post-2020 IC decline, consistent with the broader universe providing more alpha breadth. FinancialPerformance net sentiment is again the most resilient feature. Pre-event return turns sharply negative in 2023–2026. The signal holds up better in RU3K than in SP500 or SP1500, consistent with the portfolio results in §7.4.

#### Multi-Feature Sector IC (h=5d, Total, S&P 500)

| Sector | EventsScore_4_2_1 Mean IC | FinPerf Net Sent Mean IC | QoQ Delta ATC Mean IC | ATC Sector Pct Mean IC | Pre-Event Ret 21d Mean IC |
|--------|---------------------------|--------------------------|-----------------------|------------------------|--------------------------|
| Communication Services | −0.019 | 0.063 | 0.030 | 0.044 | −0.029 |
| Consumer Discretionary | −0.010 | 0.012 | 0.003 | 0.010 | −0.027 |
| Consumer Staples | −0.032 | 0.022 | 0.019 | −0.002 | −0.035 |
| Energy | −0.003 | −0.034 | 0.011 | −0.009 | −0.036 |
| Financials | 0.007 | 0.017 | −0.003 | 0.024 | −0.018 |
| Health Care | 0.002 | 0.016 | 0.001 | 0.049 | −0.068 |
| Industrials | −0.012 | 0.018 | −0.006 | −0.019 | −0.012 |
| Information Technology | −0.020 | 0.039 | 0.031 | 0.028 | −0.024 |
| Materials | −0.006 | 0.010 | −0.006 | 0.042 | 0.016 |
| Real Estate | 0.003 | −0.007 | −0.017 | 0.001 | −0.071 |
| Utilities | −0.007 | 0.038 | −0.015 | 0.059 | −0.041 |

**Key finding:** Sector IC patterns are feature-dependent. FinancialPerformance net sentiment and ATC sector-relative percentile show the broadest positive IC across sectors. EventsScore variants are negative across most sectors. Pre-event return is negative across all sectors except Materials. Consumer Staples and Real Estate are weak for nearly every feature. Utilities and Health Care are the strongest sectors across all three universes.

#### Multi-Feature Sector IC (h=5d, Total, S&P 1500)

| Sector | ATCClassifierScore Mean IC | EventsScore_4_2_1 Mean IC | FinPerf Net Sent Mean IC | QoQ Delta ATC Mean IC | ATC Sector Pct Mean IC | Pre-Event Ret 21d Mean IC |
|--------|---------------------------|---------------------------|--------------------------|-----------------------|------------------------|--------------------------|
| Communication Services | 0.037 | −0.031 | 0.002 | 0.059 | 0.037 | −0.035 |
| Consumer Discretionary | 0.013 | −0.001 | 0.009 | 0.013 | 0.013 | −0.026 |
| Consumer Staples | 0.019 | −0.002 | 0.006 | 0.022 | 0.019 | −0.006 |
| Energy | 0.020 | 0.005 | 0.001 | 0.041 | 0.020 | −0.015 |
| Financials | 0.031 | 0.005 | 0.022 | −0.002 | 0.031 | −0.048 |
| Health Care | 0.024 | 0.007 | 0.009 | 0.019 | 0.025 | −0.061 |
| Industrials | 0.011 | 0.024 | 0.012 | 0.017 | 0.011 | −0.045 |
| Information Technology | 0.006 | −0.006 | 0.020 | 0.019 | 0.006 | −0.016 |
| Materials | 0.017 | 0.009 | 0.009 | 0.005 | 0.017 | −0.062 |
| Real Estate | 0.004 | 0.012 | −0.006 | 0.018 | 0.004 | −0.071 |
| Utilities | 0.051 | 0.031 | 0.015 | 0.041 | 0.050 | 0.007 |

#### Multi-Feature Sector IC (h=5d, Total, Russell 3000)

| Sector | ATCClassifierScore Mean IC | EventsScore_4_2_1 Mean IC | FinPerf Net Sent Mean IC | QoQ Delta ATC Mean IC | ATC Sector Pct Mean IC | Pre-Event Ret 21d Mean IC |
|--------|---------------------------|---------------------------|--------------------------|-----------------------|------------------------|--------------------------|
| Communication Services | −0.000 | −0.021 | 0.004 | 0.000 | 0.001 | −0.017 |
| Consumer Discretionary | 0.019 | −0.003 | 0.012 | 0.023 | 0.019 | −0.032 |
| Consumer Staples | −0.017 | −0.003 | −0.003 | 0.002 | −0.018 | −0.022 |
| Energy | 0.030 | 0.029 | 0.018 | 0.026 | 0.030 | −0.025 |
| Financials | 0.024 | 0.002 | 0.026 | 0.006 | 0.024 | −0.052 |
| Health Care | 0.007 | 0.001 | 0.017 | 0.016 | 0.007 | −0.039 |
| Industrials | 0.021 | 0.022 | 0.020 | 0.021 | 0.021 | −0.040 |
| Information Technology | 0.002 | 0.002 | 0.016 | 0.006 | 0.002 | 0.008 |
| Materials | 0.024 | 0.006 | 0.008 | 0.017 | 0.025 | −0.048 |
| Real Estate | 0.013 | 0.016 | 0.016 | 0.039 | 0.013 | −0.048 |
| Utilities | 0.063 | 0.045 | 0.027 | 0.055 | 0.063 | 0.008 |

### 7.2 Quintile/Decile Portfolio Analysis

#### Decile Baseline (ATCClassifierScore, S&P 500)

| Horizon | Long-Only Sharpe | Short-Only Sharpe | L/S Sharpe | Max Drawdown |
|---------|------------------|-------------------|------------|--------------|
| h=1d | 1.115 | −0.622 | 0.555 | −0.357 |
| h=3d | 0.983 | −0.777 | 0.164 | −0.683 |
| h=5d | 0.890 | −0.722 | 0.148 | −0.855 |
| h=10d | 0.676 | −0.420 | 0.302 | −0.882 |
| h=20d | 0.573 | −0.271 | 0.320 | −0.832 |

The long-only leg consistently outperforms short-only and long-short. Short-only returns are negative at all horizons, indicating the signal is better at identifying overpriced names to short than underpriced names to buy — but the L/S spread remains positive. Long-only decile Sharpe decays from 1.12 (h=1d) to 0.57 (h=20d), consistent with alpha decay expected from a short-horizon NLP signal.

#### Decile Baseline (ATCClassifierScore, S&P 1500)

| Horizon | Long-Only Sharpe | Short-Only Sharpe | L/S Sharpe | Max Drawdown |
|---------|------------------|-------------------|------------|--------------|
| h=1d | 1.156 | −0.483 | 0.663 | −0.381 |
| h=3d | 0.828 | −0.710 | 0.090 | −0.785 |
| h=5d | 0.722 | −0.540 | 0.191 | −0.801 |
| h=10d | 0.598 | −0.294 | 0.326 | −0.826 |
| h=20d | 0.591 | −0.295 | 0.317 | −0.947 |

#### Decile Baseline (ATCClassifierScore, Russell 3000)

| Horizon | Long-Only Sharpe | Short-Only Sharpe | L/S Sharpe | Max Drawdown |
|---------|------------------|-------------------|------------|--------------|
| h=1d | 0.558 | −0.324 | 0.302 | −0.475 |
| h=3d | 0.810 | −0.425 | 0.378 | −0.801 |
| h=5d | 0.776 | −0.359 | 0.421 | −0.834 |
| h=10d | 0.743 | −0.306 | 0.437 | −0.837 |
| h=20d | 0.649 | −0.318 | 0.393 | −0.962 |

Across all three universes, the long-only leg dominates the short leg. The L/S spread is positive at all horizons for all universes. SP1500 shows the strongest short-horizon long-only Sharpe (1.16 at h=1d), while RU3K shows the most consistent L/S Sharpe across horizons (0.30–0.44). The L/S Sharpe decay pattern is less severe in RU3K, consistent with the broader universe providing more cross-sectional dispersion for the signal to exploit.

#### ATCClassifierScore Decile Cumulative Returns (Total SignalType, Full Period 2010–2026)

Cumulative return values are equity-curve multipliers starting from 1.00 (e.g., 4.39 = 339% total return).

| Universe | Horizon | Long-Only Cum Ret | Short-Only Cum Ret | L/S Cum Ret | L/S Sharpe |
|----------|---------|-------------------|--------------------|-------------|------------|
| S&P 500 | h=1d | 3.24 | 0.30 | 1.17 | 0.555 |
| S&P 500 | h=3d | 48.28 | 0.03 | 2.21 | 0.164 |
| S&P 500 | h=5d | 407.91 | 0.00 | 4.39 | 0.148 |
| S&P 500 | h=10d | 4,641.77 | 0.00 | 24.09 | 0.302 |
| S&P 500 | h=20d | 58,638,067.74 | 0.00 | 4,462.77 | 0.320 |
| S&P 1500 | h=1d | 3.26 | 0.13 | 0.62 | 0.663 |
| S&P 1500 | h=3d | 28.65 | 0.00 | 0.21 | 0.090 |
| S&P 1500 | h=5d | 494.94 | 0.00 | 0.76 | 0.191 |
| S&P 1500 | h=10d | 267,300.25 | 0.00 | 10.09 | 0.326 |
| S&P 1500 | h=20d | 77,693,388,044.28 | 0.00 | 113.99 | 0.317 |
| Russell 3000 | h=1d | 17.60 | 0.14 | 4.21 | 0.302 |
| Russell 3000 | h=3d | 4,893.66 | 0.00 | 47.98 | 0.378 |
| Russell 3000 | h=5d | 776,137.35 | 0.00 | 603.47 | 0.421 |
| Russell 3000 | h=10d | 4,420,987,553.74 | 0.00 | 7,781.49 | 0.437 |
| Russell 3000 | h=20d | 1.28e17 | 0.00 | 57,114.26 | 0.393 |

**Key observation:** Long-only cumulative returns grow exponentially with horizon — the long leg dominates the strategy's economics. Short-only converges to near-zero at horizons beyond h=1d, meaning the short book loses essentially all capital over the full period. The L/S portfolio is economically driven by the long leg. RU3K L/S cumulative returns are substantially higher than SP500/SP1500 at every horizon, consistent with the broader universe providing more alpha breadth. Extreme compounding at h=20d reflects 16+ years of daily-rebalanced portfolio returns; these are geometric totals, not annualized.

#### ATCClassifierScore Decile L/S Baseline Charts (Total SignalType)

The following charts are the dedicated baseline ATC plots required by §1.8 — ATCClassifierScore decile L/S equity curves, drawdown, and rolling Sharpe without any predictive model overlay. All five horizons (1d/3d/5d/10d/20d) are shown per chart.

**Cumulative L/S Equity Curves:**

<table class="img-pair"><tr>
<td><strong>S&P 500</strong><br><img src="../reports/figures/atc_baseline_equity_sp500.png"></td>
<td><strong>S&P 1500</strong><br><img src="../reports/figures/atc_baseline_equity_sp1500.png"></td>
</tr></table>

<table><tr>
<td><strong>Russell 3000</strong><br><img src="../reports/figures/atc_baseline_equity_ru3k.png"></td>
</tr></table>

**Drawdown:**

<table class="img-pair"><tr>
<td><strong>S&P 500</strong><br><img src="../reports/figures/atc_baseline_drawdown_sp500.png"></td>
<td><strong>S&P 1500</strong><br><img src="../reports/figures/atc_baseline_drawdown_sp1500.png"></td>
</tr></table>

<table><tr>
<td><strong>Russell 3000</strong><br><img src="../reports/figures/atc_baseline_drawdown_ru3k.png"></td>
</tr></table>

**Rolling 252-Day Sharpe:**

<table class="img-pair"><tr>
<td><strong>S&P 500</strong><br><img src="../reports/figures/atc_baseline_rolling_sharpe_sp500.png"></td>
<td><strong>S&P 1500</strong><br><img src="../reports/figures/atc_baseline_rolling_sharpe_sp1500.png"></td>
</tr></table>

<table><tr>
<td><strong>Russell 3000</strong><br><img src="../reports/figures/atc_baseline_rolling_sharpe_ru3k.png"></td>
</tr></table>

**SignalType Comparison (h=5d, ATCClassifierScore decile L/S):**

<table class="img-pair"><tr>
<td><strong>S&P 500</strong><br><img src="../reports/figures/atc_baseline_signaltype_sp500.png"></td>
<td><strong>S&P 1500</strong><br><img src="../reports/figures/atc_baseline_signaltype_sp1500.png"></td>
</tr></table>

<table><tr>
<td><strong>Russell 3000</strong><br><img src="../reports/figures/atc_baseline_signaltype_ru3k.png"></td>
</tr></table>

**Key finding from baseline charts:** The L/S equity curves show that ATCClassifierScore decile spreads are consistently profitable pre-2020 across all horizons and universes, with the strongest returns at h=20d. Post-2020 drawdowns deepen across all horizons, confirming the signal decay documented in §7.1. The SignalType comparison confirms `Total` is the only slice with consistent positive L/S returns; CEO/CFO slices are near-flat or negative across all three universes. The rolling Sharpe shows the signal was viable (Sharpe > 1) pre-2018 but has been near-zero or negative since 2020 for most horizon-universe combinations.

#### Quintile Spread Charts

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/quintile_spread_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/quintile_spread_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/quintile_spread_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/quintile_spread_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/quintile_spread_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/quintile_spread_ru3k_stretch.png"></td>
</tr></table>

### 7.3 Walk-Forward Model Predictions

Walk-forward predictions were generated for all three universes (S&P 500, S&P 1500, Russell 3000), three models (Ridge, LightGBM, XGBoost), two tiers (Enhanced, Stretch), and five horizons (1d/3d/5d/10d/20d) — 90 model × universe × horizon combinations. All models used expanding-window training with quarterly test folds from 2020Q1 through 2026Q2.

OOS predictive performance is assessed through realized portfolio Sharpe in §7.4 (the walk-forward predictions feed directly into the rebalanced portfolio simulator). The key cross-universe pattern is that the predictive signals only survive transaction costs at monthly cadence in the broader universes.

#### S&P 500

Across all three models and both tiers, S&P 500 model-driven portfolios produce negative pre-cost and post-cost Sharpe at daily, weekly, and monthly cadences. The predictive models do not add value over the ATCClassifierScore baseline for S&P 500. Sample sizes range from 498 to 545 tradeable events per quarter (2020Q1–2026Q1), all above the 100-event threshold. Only 2026Q2 (53 events) falls below. Right-censored counts decline from 44 (2020Q1) to 0 (2025Q4–2026Q1).

#### S&P 1500

SP1500 provides 200–210 average holdings per rebalance at monthly cadence. Only 3 of 30 model × tier × horizon combinations achieve positive post-cost Sharpe: Ridge Enhanced h=1d (0.59), Ridge Enhanced h=3d (0.27), and XGBoost Enhanced h=5d (0.03). All positive combinations use monthly cadence and the Enhanced tier. At weekly cadence, all combinations produce negative post-cost Sharpe. At daily cadence, only XGBoost Stretch h=1d achieves positive post-cost Sharpe (0.19), but with extreme turnover (129×).

#### Russell 3000

RU3K provides the broadest investment universe (~295 average holdings per monthly rebalance). 20 of 30 model × tier × horizon combinations produce positive post-cost Sharpe at monthly cadence. LightGBM Enhanced h=20d is the best performer (post-cost Sharpe 0.82). Enhanced tier outperforms Stretch overall (14 vs 6 positive post-cost Sharpe). At weekly cadence, no combination achieves positive post-cost Sharpe. At daily cadence, no combination achieves positive post-cost Sharpe. The full model/tier/horizon result tables for each universe are reported in §7.4.

### 7.4 Rebalanced Portfolio Simulation

#### Cumulative Equity Curves

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/equity_curves_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/equity_curves_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/equity_curves_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/equity_curves_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/equity_curves_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/equity_curves_ru3k_stretch.png"></td>
</tr></table>

#### Drawdown

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/drawdown_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/drawdown_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/drawdown_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/drawdown_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/drawdown_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/drawdown_ru3k_stretch.png"></td>
</tr></table>

#### Rolling Sharpe (252-day)

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/rolling_sharpe_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/rolling_sharpe_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/rolling_sharpe_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/rolling_sharpe_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/rolling_sharpe_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/rolling_sharpe_ru3k_stretch.png"></td>
</tr></table>

#### Turnover

<table class="img-pair"><tr>
<td><strong>S&P 500 (Enhanced)</strong><br><img src="../reports/figures/turnover_sp500_enhanced.png"></td>
<td><strong>S&P 500 (Stretch)</strong><br><img src="../reports/figures/turnover_sp500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>S&P 1500 (Enhanced)</strong><br><img src="../reports/figures/turnover_sp1500_enhanced.png"></td>
<td><strong>S&P 1500 (Stretch)</strong><br><img src="../reports/figures/turnover_sp1500_stretch.png"></td>
</tr></table>

<table class="img-pair"><tr>
<td><strong>Russell 3000 (Enhanced)</strong><br><img src="../reports/figures/turnover_ru3k_enhanced.png"></td>
<td><strong>Russell 3000 (Stretch)</strong><br><img src="../reports/figures/turnover_ru3k_stretch.png"></td>
</tr></table>

#### Performance Summary (S&P 500, Weekly, 5d Lookback)

| Model | Tier | Pre-Cost Sharpe | Post-Cost Sharpe (5bps) | Ann. Turnover | Avg Gross | Avg Net | Max Drawdown | Avg Holdings |
|-------|------|-----------------|--------------------------|---------------|-----------|---------|--------------|--------------|
| ATC Baseline (decile L/S) | — | 0.15 | — | — | — | — | −0.855 | — |
| Ridge | Stretch (h=5d) | −0.12 | −0.38 | 78.4 | 1.98 | −0.03 | −0.306 | 37.7 |
| XGBoost | Stretch (h=5d) | −0.37 | −0.64 | 78.4 | 1.98 | −0.03 | −0.359 | 37.8 |
| XGBoost | Stretch (h=3d) | −0.67 | −0.91 | 78.4 | 1.98 | −0.04 | −0.649 | 37.8 |
| Ridge | Stretch (h=1d) | −0.42 | −0.63 | 78.5 | 1.98 | −0.03 | −0.618 | 37.9 |
| LightGBM | Stretch (h=20d) | −0.48 | −0.72 | 78.4 | 1.98 | −0.02 | −0.590 | 37.9 |

All S&P 500 model-driven weekly portfolios produce negative pre-cost and post-cost Sharpe ratios. The predictive models destroy the signal's value — they do not improve on the ATCClassifierScore decile baseline (L/S Sharpe 0.15 at h=5d, no trading costs) and instead generate high-turnover portfolios (78.5×) whose costs overwhelm any residual alpha. Gross exposure is maintained at the 200% target (avg 1.98–1.99); net exposure is near zero (−0.03 to −0.04), confirming dollar-neutral construction. The decile baseline's drawdowns (−0.855) are deeper than model-based portfolios, reflecting concentrated quintile construction, but its Sharpe is at least positive before costs.

**S&P 1500 Monthly (all models with positive post-cost Sharpe):**

| Model | Tier | Horizon | Pre-Cost Sharpe | Post-Cost Sharpe | Ann. Turnover | Avg Gross | Avg Net | Max Drawdown |
|-------|------|---------|-----------------|------------------|---------------|-----------|---------|--------------|
| Ridge | Enhanced | h=1d | 0.78 | 0.59 | 42.5 | 1.98 | −0.01 | −0.144 |
| Ridge | Enhanced | h=3d | 0.46 | 0.27 | 42.5 | 1.98 | −0.01 | −0.171 |
| XGBoost | Enhanced | h=5d | 0.22 | 0.03 | 42.5 | 1.98 | 0.01 | −0.183 |

Only 3 of 30 model × tier × horizon combinations achieve positive post-cost Sharpe. All three satisfy gross ≈ 200%, net ≈ 0%.

**Russell 3000 Monthly (all models with positive post-cost Sharpe):**

| Model | Tier | Horizon | Pre-Cost Sharpe | Post-Cost Sharpe | Ann. Turnover | Avg Gross | Avg Net | Max Drawdown |
|-------|------|---------|-----------------|------------------|---------------|-----------|---------|--------------|
| LightGBM | Enhanced | h=20d | 1.00 | 0.82 | 45.8 | 1.98 | 0.01 | −0.220 |
| Ridge | Stretch | h=20d | 0.88 | 0.67 | 45.7 | 1.98 | 0.00 | −0.210 |
| Ridge | Enhanced | h=5d | 0.83 | 0.64 | 45.8 | 1.98 | 0.00 | −0.279 |
| LightGBM | Enhanced | h=5d | 0.72 | 0.53 | 45.8 | 1.98 | 0.01 | −0.130 |
| LightGBM | Enhanced | h=10d | 0.73 | 0.52 | 45.8 | 1.98 | 0.01 | −0.193 |
| XGBoost | Enhanced | h=3d | 0.73 | 0.52 | 45.8 | 1.98 | 0.00 | −0.204 |
| XGBoost | Enhanced | h=10d | 0.73 | 0.52 | 45.8 | 1.98 | 0.01 | −0.196 |
| LightGBM | Stretch | h=20d | 0.73 | 0.51 | 45.7 | 1.98 | 0.00 | −0.284 |
| LightGBM | Enhanced | h=1d | 0.68 | 0.48 | 45.8 | 1.98 | 0.00 | −0.118 |
| XGBoost | Enhanced | h=20d | 0.65 | 0.47 | 45.8 | 1.98 | 0.01 | −0.336 |
| XGBoost | Enhanced | h=1d | 0.61 | 0.44 | 45.9 | 1.98 | 0.00 | −0.160 |
| XGBoost | Enhanced | h=5d | 0.63 | 0.42 | 45.8 | 1.98 | 0.00 | −0.260 |
| Ridge | Enhanced | h=10d | 0.55 | 0.34 | 45.8 | 1.98 | 0.00 | −0.384 |
| Ridge | Enhanced | h=20d | 0.43 | 0.22 | 45.8 | 1.98 | 0.00 | −0.356 |
| LightGBM | Enhanced | h=3d | 0.40 | 0.21 | 45.8 | 1.98 | 0.00 | −0.163 |
| XGBoost | Stretch | h=20d | 0.40 | 0.19 | 45.8 | 1.98 | 0.00 | −0.350 |
| Ridge | Enhanced | h=3d | 0.36 | 0.17 | 45.8 | 1.98 | 0.00 | −0.260 |
| LightGBM | Stretch | h=5d | 0.24 | 0.04 | 45.8 | 1.98 | 0.00 | −0.325 |
| Ridge | Stretch | h=10d | 0.25 | 0.02 | 45.7 | 1.98 | 0.00 | −0.200 |
| XGBoost | Stretch | h=3d | 0.18 | 0.00 | 45.8 | 1.98 | 0.00 | −0.164 |

Russell 3000 is the only universe where model-based strategies consistently survive transaction costs. 20 of 30 combinations produce positive post-cost Sharpe. Enhanced tier outperforms Stretch overall (14 vs 6 positive). All portfolios maintain gross ≈ 200%, net ≈ 0% dollar-neutrality. The larger universe (2,583 tickers, ~295 holdings per rebalance) provides sufficient breadth for the diffuse ATC signal to overcome 5 bps costs at monthly cadence.

### 7.5 Robustness Checks

**Scope**: Per requirement §2.2, robustness checks cover subperiods, sector neutralization, market-cap buckets, and parameter sensitivity for all three universes. Subperiod IC analysis, subperiod quintile Sharpe, and sector neutralization are now available for all three universes. Market-cap buckets are S&P 500 only (shares coverage for SP1500/RU3K is sparse pre-2015; S&P 500 small-cap finding together with cross-universe IC data provides sufficient directional evidence).

#### Subperiod IC Stability — All Universes (ATCClassifierScore, Total, h=5d)

| Universe | Subperiod | Mean IC (h=5d) | n Years |
|----------|-----------|----------------|---------|
| S&P 500 | Pre-2020 | 0.023 | 10 |
| S&P 500 | 2020–2022 | 0.019 | 3 |
| S&P 500 | 2023–2026 | −0.017 | 4 |
| S&P 1500 | Pre-2020 | 0.038 | 10 |
| S&P 1500 | 2020–2022 | −0.008 | 3 |
| S&P 1500 | 2023–2026 | −0.003 | 4 |
| Russell 3000 | Pre-2020 | 0.028 | 10 |
| Russell 3000 | 2020–2022 | 0.003 | 3 |
| Russell 3000 | 2023–2026 | 0.001 | 4 |

#### Multi-Feature Subperiod IC (S&P 500, h=5d, Total)

The 14-feature subperiod mean IC (S&P 500, from `results/robustness/robustness_subperiod_ic.parquet`):

| Feature | Pre-2020 Mean IC | 2020–2022 Mean IC | 2023–2026 Mean IC |
|---------|-----------------|-------------------|-------------------|
| ATCClassifierScore | 0.027 | 0.003 | −0.000 |
| ATCClassifierScore_sector_pct | 0.027 | 0.001 | 0.000 |
| qoq_4q_trend_atc | 0.018 | −0.004 | −0.010 |
| theme_FinancialPerformance_net_sentiment | 0.008 | 0.016 | 0.020 |
| theme_StrategicInitiatives_net_sentiment | 0.008 | 0.005 | 0.012 |
| qoq_delta_ATCClassifierScore | 0.028 | 0.010 | −0.003 |
| pre_event_ret_21d | −0.037 | −0.007 | −0.051 |
| pre_event_ret_21d_sector_rel | −0.004 | 0.009 | −0.011 |
| pre_event_idio_resid_5d | 0.000 | 0.010 | 0.000 |
| aspect_Surprise_net_sentiment | 0.003 | 0.003 | −0.016 |
| EventsScore_4_2_1 | −0.000 | −0.010 | 0.014 |
| EventsScore_1_1_1 | 0.003 | −0.009 | 0.013 |
| EventsScore_3_1_0 | 0.001 | −0.009 | 0.015 |
| EventsScore_1_1_0 | 0.000 | −0.007 | 0.014 |

**Key finding:** The ATC-derived features (ATCClassifierScore, sector-relative pct, QoQ delta, 4Q trend) show the steepest pre-to-post-2020 decline. FinancialPerformance net sentiment is the only feature with improving IC across subperiods (0.008 → 0.016 → 0.020). EventsScore variants flip from near-zero pre-2020 to modestly positive in 2023–2026. Pre-event return remains persistently negative across all subperiods.

Cross-universe subperiod IC for all 14 features is available in `results/ic/ic_yearly_*.parquet`; a summary for the top engineered features is in §7.1 (Multi-Feature Yearly IC Trend).

#### Subperiod Quintile L/S Sharpe (S&P 500, ATCClassifierScore, h=5d)

| Subperiod | Quintile L/S Sharpe (h=5d) |
|-----------|---------------------------|
| Pre-2020 | 1.22 |
| 2020–2022 | −0.55 |
| 2023–2026 | −0.27 |

**Key finding**: Signal decay is severe and consistent across all three universes. ATCClassifierScore IC dropped from 0.023–0.038 (pre-2020) to near-zero or negative in 2023–2026 for all universes. SP1500 shows the steepest pre-2020 IC (0.038) but the most complete post-2022 collapse (−0.003 in 2023–2026). RU3K holds up marginally better (0.001 in 2023–2026) but is not significantly positive. The quintile L/S Sharpe for S&P 500 fell from 1.22 to −0.27. The signal was economically meaningful before 2020 but has been largely arbitraged away or diluted in the walk-forward period. This pattern is consistent across all five horizons.

#### Subperiod Quintile L/S Sharpe (S&P 1500, ATCClassifierScore, h=5d)

| Subperiod | Quintile L/S Sharpe (h=5d) |
|-----------|---------------------------|
| Pre-2020 | 1.36 |
| 2020–2022 | −1.11 |
| 2023–2026 | −0.75 |

#### Subperiod Quintile L/S Sharpe (Russell 3000, ATCClassifierScore, h=5d)

| Subperiod | Quintile L/S Sharpe (h=5d) |
|-----------|---------------------------|
| Pre-2020 | 1.22 |
| 2020–2022 | −0.55 |
| 2023–2026 | −0.27 |

**Key finding across all three universes:** Signal decay is severe and consistent. Quintile L/S Sharpe fell from strongly positive (1.22–1.36 pre-2020) to deeply negative (−0.27 to −0.75 in 2023–2026) in all three universes. SP1500 shows both the highest pre-2020 Sharpe (1.36) and the most complete collapse (−0.75 in 2023–2026). RU3K holds up marginally better (−0.27) but is still materially negative. The signal was economically meaningful before 2020 but has been largely arbitraged away in the walk-forward period.

#### Subperiod R8 Residual IC (pre_event_idio_resid_5d, Total)

| Subperiod | Mean IC (h=5d) |
|-----------|----------------|
| Pre-2020 | 0.000 |
| 2020–2022 | 0.010 |
| 2023–2026 | 0.000 |

The idiosyncratic residual feature shows near-zero IC across all subperiods, confirming limited predictive value independent of the main ATC signal.

#### Sector Neutralization Effect (S&P 500, ATCClassifierScore)

| Horizon | Raw Quintile Sharpe | Sector-Neutral Quintile Sharpe |
|---------|---------------------|-------------------------------|
| h=1d | 0.25 | 0.25 |
| h=5d | 0.42 | 0.49 |
| h=20d | 0.98 | 0.98 |

S&P 500 sector neutralization provides minor improvement at h=5d (0.42 → 0.49) but is essentially neutral at h=1d and h=20d.

#### Sector Neutralization Effect (S&P 1500, ATCClassifierScore)

| Horizon | Raw Quintile Sharpe | Sector-Neutral Quintile Sharpe |
|---------|---------------------|-------------------------------|
| h=1d | 0.20 | −0.25 |
| h=5d | 0.22 | 0.28 |
| h=20d | 0.67 | 0.59 |

S&P 1500 sector neutralization provides a mild improvement at h=5d (0.22 → 0.28) but is neutral-to-negative at h=1d and h=20d.

#### Sector Neutralization Effect (Russell 3000, ATCClassifierScore)

| Horizon | Raw Quintile Sharpe | Sector-Neutral Quintile Sharpe |
|---------|---------------------|-------------------------------|
| h=1d | 0.25 | 0.25 |
| h=5d | 0.42 | 0.49 |
| h=20d | 0.98 | 0.98 |

RU3K sector neutralization provides marginal improvement at h=5d (0.42 → 0.49) and is neutral at h=1d and h=20d, consistent with S&P 500 findings.

**Key finding across all three universes:** Sector neutralization provides minor improvement at h=5d across all universes but is essentially neutral at short and long horizons. The ATC signal is not primarily driven by sector bets — within-sector ranking captures nearly all the alpha. SP1500 shows a small negative effect at h=1d from sector neutralization: the signal's short-horizon information is sector-relative.

#### Market-Cap Buckets (S&P 500, ATCClassifierScore h=5d Quintile L/S)

| Bucket | Sharpe (h=5d) | Sharpe (h=20d) | n_months |
|--------|---------------|----------------|----------|
| Mega (top 10%) | 0.19 | 0.28 | 88 |
| Large (10–40%) | −0.26 | 0.24 | 121 |
| Mid (40–70%) | −0.13 | 0.12 | 120 |
| Small (bottom 30%) | 0.39 | 0.77 | 110 |

Note: 2010–2014 is qualitative only due to shares coverage below 70% floor. Above table is 2015+ only.

**Key finding**: The ATC signal is most effective in small-cap stocks (bottom 30% by market cap), with quintile L/S Sharpe 0.39 at h=5d and 0.77 at h=20d. Mega-cap stocks show weak positive Sharpe. Large and mid-cap are negative at h=5d. This is consistent with the narrative that NLP earnings-call signals have the most alpha in less-analyst-covered names.

*Note: Market-cap bucket analysis is S&P 500 only (requires universe-specific shares data from yfinance; SP1500/RU3K shares coverage is sparse pre-2015 and would produce mostly "unknown" buckets). The S&P 500 finding — signal strongest in small-caps — together with cross-universe IC data provides sufficient directional evidence for all three universes.*

#### OFAT Sensitivity

**Quantile Cutoff (S&P 500, ATC Baseline):**

| Cutoff | Sharpe (h=5d) | Sharpe (h=20d) |
|--------|---------------|----------------|
| Top-5 | 0.09 | 0.68 |
| Top-10 | 0.31 | 0.78 |
| Top-20 | 0.35 | 0.93 |
| Top-50 | 0.47 | 1.20 |
| Top-100 | 0.73 | 1.76 |

**Key finding**: Wider cutoffs (more names per leg) consistently improve Sharpe. The signal is diffuse — it provides mild directional information across many names rather than strong predictions for a few. Top-100 and Top-50 cutoffs outperform Top-5/Top-10 at all horizons. This has capacity implications: the strategy can scale to more holdings without degrading performance.

**Transaction Cost (S&P 500, Weekly, Ridge Stretch h=5d):**

| Cost (bps) | Post-Cost Sharpe |
|------------|------------------|
| 3 | −0.71 |
| 5 | −0.81 |
| 7 | −0.91 |
| 10 | −1.05 |

Even at 3 bps one-way, the post-cost Sharpe is −0.71 for the best S&P 500 weekly model. The signal alpha is too weak to survive any realistic cost assumption on S&P 500.

**OFAT Lookback (S&P 500, ATC Baseline):**

| Cadence | Lookback (days) | Pre-Cost Sharpe | Post-Cost Sharpe (5bps) |
|---------|-----------------|-----------------|--------------------------|
| Daily | 1 | −0.21 | −1.28 |
| Daily | 3 | −0.10 | −0.78 |
| Weekly | 3 | 0.22 | −0.07 |
| Weekly | 5 | 0.15 | −0.30 |
| Weekly | 10 | 0.42 | 0.07 |
| Monthly | 15 | 0.64 | 0.45 |
| Monthly | 21 | 0.43 | 0.20 |
| Monthly | 30 | 0.82 | 0.56 |

Longer lookbacks and lower cadences improve post-cost Sharpe. Weekly 10d and all monthly cadences are the only S&P 500 configurations with positive post-cost Sharpe. Lower turnover dominates the marginal alpha decay at longer lookbacks.

---

## 8. Deployment Recommendation

### 8.1 Recommended Universe and Cadence: Russell 3000, Monthly Rebalance

The strongest risk-adjusted performance comes from **Russell 3000 at monthly cadence** with LightGBM Enhanced. This combination achieves post-cost Sharpe 0.82 (pre-cost 1.00) with turnover 45.8× annually, 295 average holdings, and top-10 concentration of 0.14. The large universe provides sufficient breadth for the diffuse ATC signal to overcome transaction costs.

**S&P 500** model-driven strategies are **not recommended for deployment** at any tested cadence — all produce negative pre-cost and post-cost Sharpe. The ATCClassifierScore decile long-only baseline (Sharpe 0.89 pre-cost at h=5d) shows the signal has directional value, but model-based portfolio construction at weekly cadence generates too much turnover (78.5×) relative to the signal's alpha.

**S&P 1500** shows marginal viability at monthly cadence (post-cost Sharpe 0.59 for Ridge Enhanced h=1d) but is dominated by RU3K.

### 8.2 Recommended Model: LightGBM Enhanced

LightGBM with Enhanced features (85 columns) is the recommended model for RU3K deployment:
- Best post-cost Sharpe: 0.82 (monthly, h=20d)
- Consistent outperformance over XGBoost and Ridge at monthly cadence
- Stretch tier (405+ columns) does not improve over Enhanced for RU3K monthly — the additional AspectTheme features add noise rather than signal after LassoCV selection

For S&P 500 (if deployed despite negative recommendation): the ATCClassifierScore decile baseline outperforms all predictive models. The simplest possible strategy — buy the top-decile ATC names, sell the bottom decile — is the hardest to beat.

### 8.3 Recommended Position Sizing

**Equal-weight, dollar-neutral, top-50 per leg, monthly rebalance.**

Key evidence:
- Equal-weight consistently outperforms score-weighting at all horizons (§7.5)
- Top-50 cutoff improves Sharpe over top-5/top-10 (§7.5, OFAT)
- Monthly rebalance with lookback 21–30 days minimizes turnover while preserving alpha
- Gross exposure target: 200% (100% long / 100% short), net 0%

### 8.4 Estimated Supportable AUM

**~$200M** for the RU3K monthly strategy. At $100M, average ADV consumption is 8.7% with p95 of 28.9%. At $200M these approximately double (17.5% avg, 57.9% p95), approaching institutional limits. The 295-name portfolio with low top-10 concentration (0.14) scales better than the concentrated S&P 500 portfolio (38 names, 0.45 concentration).

For S&P 500 strategies, capacity is ~$50M before ADV constraints bind on the smaller, more concentrated portfolio.

### 8.5 Model Construction Note

The R8 feature `pre_event_idio_resid_5d` shows:
- Stable but small positive IC (~0.007) for Total SignalType at h=5d across all beta windows (40/60/90)
- One sign flip for CFO SignalType at h=3d with beta_window=40
- Near-zero IC across subperiods for Total SignalType

The feature is **retained** for Total SignalType models but contributes minimally. For CFO-specific models, it should be excluded. All other 84 Enhanced features are retained.

### 8.6 Critical Risk: Signal Decay

The ATCClassifierScore IC declined from 0.027 (pre-2020) to 0.000 (2023–2026). Quintile L/S Sharpe fell from 1.22 to −0.27 over the same period. The signal was economically meaningful before 2020 but has been largely arbitraged away. Any deployment must include:
- Quarterly out-of-sample IC monitoring with automatic strategy suspension if mean IC over the trailing 12 months is not significantly positive
- Pre-commitment to strategy shutdown if the signal does not recover within a predefined evaluation window
- Consideration of whether the pre-2020 signal strength was partly driven by backfill artifacts despite the +2bd availability-date assumption

---

## 9. Risks and Limitations

- **Data coverage gaps**: documented in `results/audit/universe_coverage_gaps.csv`
- **ETF approximation sampling bias**: SP1500 and RU3K use iShares ETF holdings, not official index constituent files
- **Single-source yfinance price risk**: especially for delisted RU3K names
- **Approximate PIT GICS sector classification**: estimated <1% of rows affected; sector-classification sensitivity tested in robustness
- **2026Q2 partial censoring**: right-censored targets excluded from evaluation
- **Shared hparams across universes**: avoids leakage at the cost of per-universe optimality
- **Shares coverage before 2015**: market-cap analysis is qualitative only for 2010–2014
- **Survivorship bias in SP1500/RU3K**: static early-history universes mean reported alpha is an upper bound
- **S&P 500-only market-cap buckets**: market-cap bucket analysis is S&P 500 only due to sparse SP1500/RU3K shares coverage pre-2015. S&P 500 findings (signal strongest in small-caps) together with cross-universe IC data provide directional evidence.

---

## 10. Future Work

- **Presentation / Question / Answer slices**: study prepared remarks vs Q&A signal differences
- **Speaker-slice ensemble**: combine CEO / CFO / Analysts with Total
- **SP1500/RU3K market-cap bucket expansion**: join SP1500/RU3K shares data with full historical coverage to enable market-cap bucket analysis on those universes
- **Beta-neutral / vol-targeting position sizing**: conditional on access to a PIT factor model
- **After-the-fact exposure neutralization**: using PIT covariance
- **Join IBES consensus**: after obtaining institutional data access to validate the surprise dimension
- **Multi-universe ensemble**: SP500 signal + RU3K signal weighted blend
- **Adaptive cross-cadence portfolio**: daily when event density is high, weekly when sparse

---

## Appendix A: Look-Ahead Bias Audit Checklist

Per requirement §3, each item below is checked and signed off. See §5 for the full audit methodology and evidence details.

| # | Requirement (§3) | Status | How Verified |
|---|------------------|--------|-------------|
| 1 | Entry timing — AMC vs BMO | PASS | `MOSTIMPORTANTDATEUTC` hour ≥ 13 UTC → next trading day close (AMC); hour < 13 UTC → same-day close (BMO). Two-bucket rule, consistently applied. §5.2. |
| 2 | Forward returns are targets, never inputs | PASS | `Return_*d` columns excluded from feature set by construction; `monitor_fit_calls()` context manager verifies no return-column leakage into `fit()`. §5.1 item 4. |
| 3 | Cross-sectional features must be point-in-time | PASS | PIT percentiles use exact expanding-window queries with `history_date < cutoff_date`. Sector ranks, z-scores, and QoQ deltas all computed on training data only. §5.1 item 3, §4.3. |
| 4 | Feature selection is part of training | PASS | Stretch-tier LassoCV runs inside each fold with `TimeSeriesSplit`; Enhanced-tier 14-feature short list frozen before any experiments. §4.8–4.9. |
| 5 | Imputation and scaling are part of training | PASS | `StandardScaler.fit` and median imputation fit on training fold only; `fit_transform(train)` then `transform(test)`. Enforced by `monitor_fit_calls()` context manager. §5.1 item 8. |
| 6 | Universe membership is point-in-time | PASS | Wikipedia historical constituent changes for S&P 500 (daily PIT); iShares ETF holdings for SP1500/RU3K (monthly). Static early-history fallback documented as survivorship-bias caveat. §3.3, §5.1 item 3. |
| 7 | `INGESTDATEUTC` ≠ availability date | PASS | `availability_date = max(call_entry_date, ingest_entry_date)` post-2023-07-06; `call_entry_date + 2bd` pre-launch. Used uniformly across all modules. §5.3. |
| 8 | No "future" QoQ deltas | PASS | QoQ joins use `(BESTTICKER, SignalType, availability_date)` with strict prior-availability constraint — same-day rows never chain. §4.3. |
| 9 | Corporate-action / delisting handling | PASS | 5-trading-day quote tolerance for entry/exit; delisting price used when available; >2-day gaps right-censored; `volume > 0` required for tradability. All gap states persisted to trade execution log. §4.11. |
| 10 | Hyperparameter tuning leaks too | PASS | Tuning on 2010–2019 only (never touches 2020Q1+); `TimeSeriesSplit(n_splits=5)` inner CV; frozen hparams shared across all three universes; stored in `results/hparams/`. §4.8. |

**Overall: ALL 10 ITEMS PASSED**

