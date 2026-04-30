# HW3 — Backtesting the ProntoNLP Earnings-Call ATC Signal

Backtest framework for the ATC signal across S&P 500 / S&P 1500 / Russell 3000
at daily / weekly / monthly rebalance cadences. The hard constraint is a
clean, look-ahead-free pipeline (see `docs/requirement.md` §3 and `ideas/plan.md`).

## Repo layout

```text
HW3/
├── data/                        # raw input + per-stage parquet caches (caches gitignored)
│   ├── load_signals.py          # Phase 1.1 - ATC CSV -> Parquet
│   ├── load_universes.py        # Phase 1.2 - PIT membership for SP500 / SP1500 / RU3K
│   ├── load_prices.py           # Phase 1.3 - yfinance adj_close + volume per ticker
│   ├── load_shares.py           # Phase 1.4 - historical shares outstanding
│   ├── cache/                   # generated artifacts (parquet, manifest, etc.)
│   └── universe_raw/            # raw universe snapshots (Wikipedia, iShares CSVs)
├── features/
│   ├── engineer.py              # Phase 2 - feature engineering
│   └── audit.py                 # Phase 3 - look-ahead audit assertions
├── backtest/
│   ├── splits.py                # Phase 4 - walk-forward splits
│   ├── model.py                 # Phase 4/5 - Ridge / LightGBM / XGBoost wrapper
│   ├── single_feature_ic.py     # Phase 5.1 - IC analysis
│   ├── quintile.py              # Phase 5.2 - quintile / decile portfolios
│   └── portfolio.py             # Phase 5.4 - rebalanced portfolio simulation
├── reports/
│   ├── charts.py                # Phase 7 - matplotlib charts
│   └── pdf.py                   # Phase 7 - reportlab writer
├── results/                     # outputs (gitignored)
│   └── audit/                   # mandated audit artifacts
├── docs/
│   ├── requirement.md           # course requirements
│   └── report.md                # research write-up draft
└── ideas/
    ├── understanding.md         # personal notes on data + task
    └── plan.md                  # execution plan / checklist
```

## Inputs

- `data/Earnings_ATC_until_2026-04-21.csv` (~4.47 GB, ~2.74M rows). Loaded
  in 100k-row chunks by `data/load_signals.py`; never read whole into memory.
- yfinance for prices and historical shares outstanding.
- Wikipedia + iShares (IJH / IJR / IWV) monthly holdings for PIT universe membership.

## Configuration

A single project config module fixes the global random seed and the cache
locations used by every later step:

```python
# data/config.py
SEED = 42
CACHE_DIR        = ".../HW3/data/cache"
UNIVERSE_RAW_DIR = ".../HW3/data/universe_raw"
RESULTS_DIR      = ".../HW3/results"
AUDIT_DIR        = ".../HW3/results/audit"
```

Import `from data.config import SEED` everywhere and seed numpy / random / torch
from that single source — this is the one place to change it.

## Reproduction (one command)

> Filled in at Phase 8. Will be either `make all` or `python run_all.py` and
> will run: raw zip -> universes -> prices -> shares -> features -> audit
> tests -> walk-forward -> experiments -> charts -> PDF, from a clean clone.

Until Phase 8 is wired up, run each loader directly:

```bash
python -m data.load_signals
python -m data.load_universes
python -m data.load_prices
python -m data.load_shares
```

## Look-ahead audit

`features/audit.py` runs the eight assertion classes from `ideas/plan.md` §3.2.
Any failure turns CI red. The signed one-page checklist lives at
`results/audit/lookahead_checklist_onepager.md` and is appended to the final PDF.
