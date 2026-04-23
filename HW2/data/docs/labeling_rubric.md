# Labeling rubric: Boilerplate vs. Substantive (earnings-call sentences)

Use this rubric for all judges (Ollama models and Haiku). Output **only** JSON: `{"label":"boilerplate"}` or `{"label":"substantive"}`.

## Boilerplate (scripted / housekeeping / generic)

**Anchors**

- Operator instructions and call logistics: “Please note today’s call is being recorded”, “[Operator Instructions]”, “Our first question will come from …”.
- Safe-harbor / disclaimer blocks: forward-looking statements, “subject to risks and uncertainties”, “non-GAAP measures”, “reconciliation … in our SEC filings”.
- Generic thanks / welcomes without material numbers: “Thank you for joining us today”, “Good morning, everyone, and welcome”.
- Repeated analyst name-only intros with no question yet (if the line is only naming the firm/person and handing the floor).

**Boundary**

- If a sentence is **mostly** disclaimer or logistics but contains **specific** guidance or KPIs, prefer **substantive** (mixed material wins).

## Substantive (material business / financial / strategy content)

**Anchors**

- Numeric results, guidance, margins, growth rates, segment commentary with specifics.
- Strategy, product, customer wins, M&A, regulatory updates with concrete facts.
- Q&A that references **specific** numbers, segments, or forward outlook beyond boilerplate.

**Boundary**

- Short answers that still carry **material** meaning (e.g., a numeric range, a clear yes/no on guidance) are substantive even if brief, **unless** the sentence is below the pipeline’s minimum length filter (then it is not in the labeling domain).

## Hard ambiguity rule (for ties / parsing failures)

If genuinely ambiguous after applying the anchors, default to **substantive** (recall-oriented), and note that this rule biases toward substantive frequency.

## Edge cases called out in the assignment

- **Analyst name intros**: boilerplate if purely housekeeping; substantive once the question contains material inquiry.
- **Generic thanks**: usually boilerplate unless bundled with material disclosure in the same sentence.
- **One-word answers**: substantive if materially dispositive; otherwise boilerplate only if purely procedural (“Next question.”).
- **Mixed sentences**: if any material number/guidance/segment fact appears, choose **substantive**.
