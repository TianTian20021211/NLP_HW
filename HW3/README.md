# ProntoNLP Earnings-Call ATC Signal Backtest

Reproducible backtest of the ProntoNLP ATC signal across SP500/SP1500/RU3K at
daily/weekly/monthly rebalance cadences.

## Quick start

1. Place `Earnings_ATC_until_2026-04-21.csv.zip` in the project root.
2. Install dependencies:

   ```bash
   pip install pandas numpy scipy scikit-learn lightgbm xgboost yfinance \
               matplotlib weasyprint markdown pygments pyarrow numba joblib
   ```
3. Run the full pipeline:

   ```bash
   python run_all.py --tier both
   ```
4. Output: `reports/final_report.pdf`

## Phased execution

```bash
python run_all.py --from-phase 1 --stop-at-phase 1   # data loading only
python run_all.py --from-phase 4 --stop-at-phase 5   # walk-forward + experiments
python run_all.py --tier enhanced                     # enhanced tier only (85 cols)
python run_all.py --tier both                         # enhanced + stretch tiers
python run_all.py --dry-run --tier both               # preview without executing
```

## Data sources

- Signal data: `Earnings_ATC_until_2026-04-21.csv.zip` (instructor-provided)
- Price data: yfinance (auto-fetched and cached to `data/cache/prices/`)
- Universe membership:
  - SP500: Wikipedia historical constituent changes
  - SP1500: iShares IJH + IJR monthly holdings + SP500
  - RU3K: iShares IWV monthly holdings

## Expected runtime

~30-60 minutes for enhanced tier, ~60-90 minutes for both tiers (all 3
universes) on a machine with 8+ cores and 16 GB RAM. Price fetching dominates
the first run; subsequent runs use cached data and complete in ~15-30 minutes.

## Key artifacts

| Path | Description |
|------|-------------|
| `data/cache/signals.parquet` | Cleaned signal data |
| `data/cache/prices/` | Per-ticker price parquet files |
| `results/features_enhanced.parquet` | 85 engineered features |
| `results/ic/` | IC analysis outputs |
| `results/quintile/` | Quintile/decile portfolio outputs |
| `results/portfolio/` | Rebalanced portfolio simulation outputs |
| `results/robustness/` | Robustness check outputs |
| `results/audit/` | Look-ahead audit artifacts |
| `reports/figures/` | Chart PNGs |
| `reports/final_report.pdf` | Final research PDF |

## Configuration

A single project config module fixes the global random seed and the cache
locations used by every later step:

```python
# data/config.py
SEED = 42
```

Import `from data.config import SEED` everywhere and seed numpy / random / torch
from that single source.

## Look-ahead audit

`features/audit.py` runs look-ahead audit assertions. Any failure turns CI red.
The signed one-page checklist lives at `results/audit/lookahead_checklist_onepager.md`
and is appended to the final PDF.
