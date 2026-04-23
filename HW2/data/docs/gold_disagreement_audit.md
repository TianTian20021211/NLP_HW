# Gold Disagreement Hand Audit

This audit records a manual review of a stratified sample from the rows where the
three LLM judges did not unanimously agree. It is intended to support the
Gold-standard methodology section of the report.

## Audit Date and Scope

- Audit date: 2026-04-22
- Source table: `cache/gold_labeled.parquet`
- Source rows: 3,000 gold-labeled sentences
- Disagreement rows audited: 40
- Sampling frame: rows with `discord != "3-0"`
- Disagreement population: 1,089 rows, all `2-1` vote splits
- Sampling method: deterministic stratified sample by `(j1_label, j2_label, j3_label, gold_final)` using random seed 42; all tiny strata were fully included, larger strata contributed 6 or 7 rows.

The full row-level audit record is in:

- `data/docs/gold_disagreement_audit.tsv`

## Audit Rule

The audit used `data/docs/labeling_rubric.md` as the governing rubric.

Rules applied during audit:

- Material numbers, named products, named programs, business strategy, market commentary, segment/customer commentary, guidance timing, or concrete Q&A content were treated as `substantive`.
- Pure structure, generic confidence language, generic caveats, vague transition sentences, and sentence fragments with no material content were treated as `boilerplate`.
- For genuinely hard borderline cases, the rubric's recall-oriented ambiguity rule was applied: default to `substantive`.

## Summary

| Metric | Value |
|---|---:|
| Audited disagreement rows | 40 |
| Frozen `gold_final` agreed with audit | 30 |
| Recommended corrections | 10 |
| Correction rate in audited disagreement sample | 25.0% |
| Correction direction | 10 boilerplate -> substantive; 0 substantive -> boilerplate |

The most common correction pattern was a sentence labeled `boilerplate` by Haiku
that nevertheless contained business strategy, market/growth commentary,
guidance timing, named initiatives, or concrete temporal/financial content.
Because the assignment places a high priority on substantive recall, these were
marked as `substantive` in the audit record.

## Recommended Corrections

These rows have `audit_label != gold_final` in the audit TSV:

| audit_id | sentence_id | gold_final | audit_label | reason |
|---|---|---|---|---|
| A08 | `NVDA_Q4-2025.txt#0000203` | boilerplate | substantive | Strategic/product-market claim about the future of software. |
| A17 | `BLK_Q1-2025.txt#0000314` | boilerplate | substantive | Business footprint growth statement. |
| A18 | `C_Q1-2024.txt#0000296` | boilerplate | substantive | Strategy around private capital asset class. |
| A21 | `GS_Q1-2026.txt#0000241` | boilerplate | substantive | Balance sheet growth reference. |
| A22 | `NKE_Q2-2025.txt#0000027` | boilerplate | substantive | Operational/consumer-channel commentary from retail visits. |
| A30 | `AVGO_Q4-2025.txt#0000201` | boilerplate | substantive | Guidance timing statement. |
| A31 | `C_Q3-2024.txt#0000480` | boilerplate | substantive | Concrete timing clarification across 2025 and 1Q 2026. |
| A32 | `FDX_Q3-2025.txt#0000492` | boilerplate | substantive | Named operating program (`DRIVE`) used as work-process commentary. |
| A33 | `GS_Q1-2025.txt#0000372` | boilerplate | substantive | Global client engagement commentary. |
| A34 | `NVDA_Q4-2026.txt#0000220` | boilerplate | substantive | Revenue timing/measurement commentary. |

## Frozen Label Handling

The existing frozen gold table was not rewritten during this audit pass. Current
model artifacts, thresholds, leaderboard metrics, and the GUI winner were trained
and evaluated against `cache/gold_labeled.parquet`. Rewriting that table would
require rerunning the classifier zoo and regenerating the leaderboard.

For the report, describe this as a hand-audit of stratified disagreement rows
with recommended corrections recorded in `audit_label`. If the final submission
is rebuilt, apply the rows where `audit_label != gold_final` before retraining.

## Ambiguity Notes

Several reviewed rows were intentionally kept even though they are borderline:

- Sentences that only set up an answer, such as "you really got 3 things going
  on", stayed `boilerplate` because they contain structure but no material claim.
- Generic confidence/caveat language stayed `boilerplate` unless tied to a
  concrete business metric, market, product, or guidance point.
- Short but domain-specific fragments, such as named funds, named products, or
  market/pipeline commentary, stayed or were corrected to `substantive` under
  the recall-oriented ambiguity rule.
