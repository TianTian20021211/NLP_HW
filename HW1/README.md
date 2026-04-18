# HW1 — Earnings-Call NLP Signals

End-to-end pipeline that turns 131 SP500 earnings-call transcripts into (a) structured JSON of wins / risks / guidance and (b) a back-tested long-only trading signal on forward excess returns.

## Directory layout

```
HW1/
├── src/                      pipeline code (all the real logic)
├── notebook/
│   ├── extraction.ipynb      Part 1 — transcripts → call JSONs (Stages A-D)
│   └── evaluation.ipynb      Part 2 — call JSONs → features, labels, backtest (E0-E3)
├── data/
│   ├── transcripts/          131 input files (committed)
│   ├── external/LM_dict.csv  Loughran-McDonald dictionary (user-provided; gitignored)
│   └── cache/                every derived artifact (gitignored)
├── report.md                 final writeup
├── report.pdf                rendered writeup (from `python -m src.report_pdf`)
├── requirements.yml          minimal conda spec (top-level deps for this repo only)
└── README.md
```

## Setup

Create and activate the conda environment (one-time):

```bash
conda env create -f requirements.yml   # skip if env `nlp` already exists
conda activate nlp
```

`requirements.yml` lists only **direct** dependencies for running notebooks and `src/` (conda then installs everything else). It is not a full `conda env export` lockfile, so the solver may pick newer transitive versions over time. For LLM extraction you also need a local [Ollama](https://ollama.com) daemon serving `gemma3:4b` and `llama3.1:8b`.

### One-time: Loughran-McDonald dictionary

Part 2's `ext_lm_llm` variant needs the LM Master Dictionary. Download the CSV from [the Notre Dame SRAF page](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and save it as:

```
data/external/LM_dict.csv
```

The file is gitignored (license-restricted redistribution).

## Run order

**Part 1 (if call JSONs aren't cached yet)**: open `notebook/extraction.ipynb`, set `PHASE = "P3"` at the top, Run-All. This produces 131 `data/cache/calls/*.json` records plus their unit/sentiment/extraction sidecars.

**Part 2**: open `notebook/evaluation.ipynb`, set `PHASE = "E3"`, Run-All. It will

1. Fetch daily Close for 14 tickers + SPY from yfinance (cached; ~7 s cold / <1 s warm).
2. Score every unit with the LM dictionary (~0.5 s).
3. Rebuild `calls_lm/` from the LM sentiment cache (~12 s).
4. Build `feature_table.parquet` and `labels.parquet`.
5. Train + backtest **24 cells** (3 extraction variants × 2 horizons × 4 models) and upsert the row in `summary.parquet`.
6. **Stages A1-A6** — qualitative + extra-credit deliverables on top of the backtest:
   - per-ticker cross-quarter narratives for NVDA / INTC / FDX (`analysis/narratives.json`)
   - corpus-wide reactive-vs-proactive risk table (`analysis/reactive_proactive.csv`)
   - per-call gemma3:4b vs llama3.1:8b guidance agreement (`analysis/llm_agreement_*.csv`)
   - cross-sectional long-short top-3/bottom-3 backtest (`analysis/cross_sectional_quarterly.csv`)
   - momentum-only baseline alone-vs-combined comparison
   - 5 figures rendered to `data/cache/plots/`: `equity_curves.png`, `raw_vs_spy.png`, `ic_grid.png`, `llm_agreement.png`, `cross_sectional.png`

**Part 3 (PDF deliverable)**: render the writeup to PDF.

```bash
conda run -n nlp python -m src.report_pdf  # → report.pdf (~540 KiB, 14 pages)
```

Total end-to-end runtime (warm caches) is ~40 s; cold run (with yfinance fetch + LM-sentiment scoring + calls_lm build) is ~75 s. Stages A1-A6 add ~5 s.

## What each PHASE does (Part 2)

| PHASE | What it runs                                                                   | Writes to disk                       |
|-------|--------------------------------------------------------------------------------|--------------------------------------|
| `E0`  | 4 tickers × 2 quarters, `ext_finbert_llm` only, +5d only, forced τ=0.0         | 4 backtest cells (rule/logreg/xgb/ridge) |
| `E1`  | Full corpus, `ext_finbert_llm`, +5d and +21d                                   | 8 cells                              |
| `E2`  | E1 + `ext_lm_llm`                                                              | 16 cells                             |
| `E3`  | E2 + `ext_finbert_only` ablation                                               | 24 cells                             |

## Where the artifacts live

Under `data/cache/` (all gitignored):

| Path                                             | Written by                         |
|--------------------------------------------------|------------------------------------|
| `units/<ticker>_<quarter>.jsonl`                 | `src/parser.py` (+ `.key` sidecar) |
| `sentiment/<unit_id>.json`                       | `src/sentiment_finbert.py`         |
| `sentiment_lm/<unit_id>.json`                    | `src/lexicon_lm.py`                |
| `extractions/<model>/<unit_id>.json`             | `src/extract_llm.py`               |
| `extractions/<model>/qa_groups/<call>__<role>.json` | `src/extract_qa_group.py`       |
| `calls/<ticker>_<quarter>.json`                  | `src/aggregate.py` variant=finbert |
| `calls_lm/<ticker>_<quarter>.json`               | `src/aggregate.py` variant=lm      |
| `prices/<ticker>.parquet`                        | `src/prices.py`                    |
| `features/feature_table.parquet`                 | `src/features.py`                  |
| `labels/labels.parquet`                          | `src/labels.py`                    |
| `models/<extraction>__<model>__<horizon>.pkl`    | `src/models.py`                    |
| `models/preds_<extraction>__<model>__<horizon>.parquet` | `src/backtest.py`            |
| `backtest/<extraction>__<model>__<horizon>.json` | `src/backtest.py`                  |
| `backtest/equity_<...>.csv`                      | `src/backtest.py`                  |
| `backtest/summary.parquet`                       | `src/backtest.py` upsert           |
| `qc/<report_name>.json`                          | `src/qc.py`                        |
| `plots/<name>.png`                               | `src/plots.py`                     |
| `analysis/narratives.json`                       | `src/analysis.py`                  |
| `analysis/reactive_proactive.csv`                | `src/analysis.py`                  |
| `analysis/llm_agreement_per_call.csv`            | `src/analysis.py`                  |
| `analysis/llm_agreement_per_ticker.csv`          | `src/analysis.py`                  |
| `analysis/cross_sectional_quarterly.csv`         | `src/analysis.py`                  |
| `analysis/analysis_summary.json`                 | `src/analysis.py` bundle           |

## Cache invalidation

Every cache entry carries a `_cache_key` field (or a sibling `.key` sidecar for parquet / jsonl) derived from the AST fingerprint of the source file(s) that produced it (see `src/cache_keys.py`). Editing any `src/*.py` file changes the fingerprint; on the next run `prune_stale(stage)` removes the now-stale files and the stage rebuilds.

That's why `notebook/evaluation.ipynb` opens every stage cell with a `prune_stale(...)` call — it's the mechanism that keeps "re-run on edit" correct without explicit cache-busting commands.
