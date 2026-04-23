# BPClassifier: Boilerplate vs. Substantive Sentence Classifier

This project classifies every sentence in an earnings-call transcript as either **Boilerplate** or **Substantive**. It includes the full reproducible pipeline requested in `requirements.md`: sentence extraction, multi-source gold labeling, train/validation/test splitting, handcrafted features, a classifier zoo, recall-constrained thresholding, a saved best model, and a Streamlit GUI that tags transcripts inline.

## Current Requirement Status

| Requirement | Status | Implementation |
|---|---|---|
| End-to-end notebook/script | Complete | `gold_standard.py`, `notebooks/project.ipynb`, `src/project_notebook.py` |
| Sentence extraction | Complete | `src/sentence_extraction.py`; output `cache/sentence_pool.parquet` |
| Gold-label creation with multiple sources | Complete with audit caveat | `src/gold_labeling.py`, `data/docs/labeling_rubric.md`, `cache/gold_labeled.parquet` |
| Hand audit of disagreements | Documented caveat | `data/docs/gold_disagreement_audit.md`; 10 recommended corrections were recorded but not folded into the frozen model run |
| Frozen held-out split | Complete | `src/train_split.py`; transcript-level 60/20/20 split |
| 20-30 handcrafted features | Complete | 30 regex/numeric/style features in `src/handcrafted_features.py` |
| At least five classifier families | Complete | Rules, LogReg, HistGB, FastText, FinBERT, SetFit, two ensembles |
| Recall-constrained threshold tuning | Complete | 5-fold OOF threshold search in `src/model_zoo.py` |
| Saved best model | Complete | `cache/zoo_winner/` plus component artifacts in `cache/` |
| GUI with inline tagging and stats | Complete | `gui/app.py`, `src/gui_inference.py` |
| Final report PDF | Complete after this cleanup | `report.md` and `report.pdf` |

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the GUI with the provided saved winner:

```bash
streamlit run gui/app.py
```

If the default port is busy:

```bash
streamlit run gui/app.py --server.port 8502
```

The GUI loads `cache/zoo_winner/winner_manifest.json` and `cache/zoo_winner/winner_threshold.txt`. The saved winner is `G_ensemble_mean` at threshold `0.417`, using the mean of `P(substantive)` from:

- `C_histgb_enriched`
- `B_logreg_enriched`
- `D_fasttext`
- `A_rules`

## Project Layout

```text
.
├── gold_standard.py              # CLI for sentence pool, gold sample, Ollama judges, Haiku arbitration
├── gui/app.py                    # Streamlit app
├── notebooks/project.ipynb       # End-to-end notebook wrapper
├── src/
│   ├── sentence_extraction.py    # Transcript parsing, dedupe, paragraph split, sentence tokenization
│   ├── gold_labeling.py          # Sampling, Ollama labeling, majority vote, Haiku arbitration
│   ├── train_split.py            # Transcript-level stratified train/val/test split
│   ├── handcrafted_features.py   # 30 handcrafted features
│   ├── model_zoo.py              # Model training, OOF thresholds, leaderboard, saved winner
│   ├── gui_inference.py          # Loads saved winner for Streamlit inference
│   └── project_notebook.py       # Notebook-friendly pipeline helpers
├── data/
│   ├── transcripts/              # Raw earnings-call transcripts
│   ├── docs/                     # Labeling rubric and disagreement audit
│   └── external/GUI_screenshot.png
├── cache/                        # Gold labels, leaderboard, and saved model artifacts
├── tests/                        # Unit tests
├── report.md
└── report.pdf
```

## Pipeline Logic

### 1. Sentence Extraction

`src/sentence_extraction.py` reads all transcript `.txt` files from `data/transcripts`, removes duplicate lines within each transcript, splits paragraphs at blank lines and transcript headers, tokenizes sentences with NLTK, normalizes whitespace, and drops sentences shorter than 40 characters.

Current cached extraction:

| Metric | Value |
|---|---:|
| Candidate sentences | 55,485 |
| Minimum sentence length | 40 characters |
| Cache | `cache/sentence_pool.parquet` |

### 2. Gold Standard

`src/gold_labeling.py` samples 3,000 sentences from the sentence pool using seed 42 and source-file stratification. Three local Ollama judges label each sentence with the shared rubric in `data/docs/labeling_rubric.md`:

- `llama3.1:8b`
- `qwen3:8b-q4_K_M`
- `gemma2:9b`

Unanimous rows use the local majority. Any 2-1 disagreement is sent to Claude Haiku 4.5 for arbitration. The ambiguity rule is recall-oriented: genuinely ambiguous cases default to substantive.

Current cached gold summary:

| Metric | Value |
|---|---:|
| Gold rows | 3,000 |
| Boilerplate | 945 |
| Substantive | 2,055 |
| Three-judge unanimous agreement | 63.7% |
| Disagreement rows arbitrated by Haiku | 1,089 |
| Pairwise agreement A/B/C | 0.654 / 0.788 / 0.832 |

Audit caveat: `data/docs/gold_disagreement_audit.md` records a manual audit of 40 disagreement rows. It recommends 10 boilerplate-to-substantive corrections. Those corrections were not applied to the frozen `cache/gold_labeled.parquet` before the final model run, so the final leaderboard and GUI model are evaluated against the frozen table.

### 3. Split

`src/train_split.py` keeps valid `gold_final` rows and assigns entire transcripts to train, validation, or test. The split is 60/20/20 by transcript count and stratified by each transcript's majority label.

| Split | Sentences | Boilerplate | Substantive | Transcripts |
|---|---:|---:|---:|---:|
| Train | 1,826 | 565 | 1,261 | 79 |
| Validation | 631 | 210 | 421 | 26 |
| Test | 543 | 170 | 373 | 26 |

### 4. Features

`src/handcrafted_features.py` builds 30 handcrafted features:

- Boilerplate cues: operator/moderator patterns, analyst-firm intros, greetings, welcomes, call-flow phrases, next-question cues, webcast/replay/press-release references, safe-harbor language, SEC/Regulation FD references, generic thanks, generic praise.
- Substantive cues: forward-looking modal combinations, dollars, percentages, digit density, million/billion/thousand, basis points, quarter/sequential/year-over-year language, financial terms, enumerated points, numbered steps, corporate actions, restructuring/headcount terms.
- Shape/style cues: character length, short-answer flag, question mark, exclamation count, punctuation density, uppercase-run ratio.

Rules use these features directly. LogReg and HistGB concatenate these features with frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings. Text-native models use the same split and labels.

### 5. Classifier Zoo and Thresholding

`src/model_zoo.py` trains and evaluates eight rows:

- `A_rules`
- `B_logreg_enriched`
- `C_histgb_enriched`
- `D_fasttext`
- `E_finbert`
- `F_setfit`
- `G_ensemble_mean`
- `H_ensemble_rankmean`

Thresholds are selected on 5-fold out-of-fold predictions over train+validation. A threshold is eligible only if substantive recall is at least 0.96; among eligible thresholds, the code maximizes macro-F1. The saved winner additionally enforces the stricter test-recall floor from the grading rubric.

Leaderboard from `cache/zoo_leaderboard.json`:

| Family | Macro-F1 | Accuracy | Boilerplate F1 | Substantive F1 | Substantive recall | Threshold | Infer sent/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| E_finbert | 0.867 | 0.890 | 0.811 | 0.922 | 0.949 | 0.148 | 551 |
| F_setfit | 0.841 | 0.867 | 0.776 | 0.906 | 0.928 | 0.028 | 3,319 |
| G_ensemble_mean | 0.816 | 0.856 | 0.729 | 0.902 | 0.965 | 0.417 | 31,091 |
| C_histgb_enriched | 0.802 | 0.843 | 0.712 | 0.893 | 0.946 | 0.337 | 186,631 |
| H_ensemble_rankmean | 0.799 | 0.842 | 0.705 | 0.892 | 0.949 | 0.288 | 31,091 |
| B_logreg_enriched | 0.750 | 0.818 | 0.621 | 0.880 | 0.973 | 0.016 | 67,255 |
| D_fasttext | 0.747 | 0.814 | 0.616 | 0.877 | 0.968 | 0.103 | 84,309 |
| A_rules | 0.506 | 0.689 | 0.207 | 0.806 | 0.944 | 0.276 | 13,207,820 |

FinBERT has the highest macro-F1 but misses the 0.96 substantive-recall floor on test. The deployed winner is therefore `G_ensemble_mean`, which reaches test substantive recall 0.965.

Winner test confusion matrix, rows as gold labels and columns as predictions:

| Gold \\ Predicted | Boilerplate | Substantive |
|---|---:|---:|
| Boilerplate | 105 | 65 |
| Substantive | 13 | 360 |

## Reproducing the Project

Run the full gold pipeline. This requires Ollama running locally with the three judge models and `ANTHROPIC_API_KEY` set for Haiku arbitration:

```bash
python gold_standard.py --mode full --use-api-check --random-seed 42
```

Run the classifier zoo and save the leaderboard/winner:

```bash
python - <<'PY'
from src.model_config import ZooRunConfig
from src.project_notebook import run_classifier_zoo_report

cfg = ZooRunConfig.for_8gb_gpu(show_progress=True)
run_classifier_zoo_report(cfg=cfg)
PY
```

Run tests:

```bash
pytest -q
```

Latest verification:

```text
33 passed in 1.39s
```

## GUI Behavior

The GUI supports:

- Uploading a `.txt` transcript.
- Selecting a sample transcript from `data/transcripts`.
- Pasting transcript text.
- Inline sentence rendering in original order.
- Red background for predicted Boilerplate sentences.
- Plain rendering for predicted Substantive sentences.
- Statistics panel with class counts, percentages, total classified sentences, model family, threshold, short dropped sentence count, and duplicate-line count.

The screenshot required by the report is `data/external/GUI_screenshot.png`.

## Final Submission Contents

The final zip should keep:

- Source code under `src/`, `gui/`, and `gold_standard.py`
- `notebooks/project.ipynb`
- `data/transcripts/`
- `data/docs/`
- `data/external/GUI_screenshot.png`
- Required cache artifacts for the frozen gold table, leaderboard, and saved winner
- `requirements.txt`
- `README.md`
- `report.pdf`

Generated Python caches, pytest caches, notebook checkpoints, empty training directories, and non-winner transformer checkpoints are not needed for final submission.
