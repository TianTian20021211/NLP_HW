"""Transcript text -> list of `Unit` dicts (Plan B rule-based segmenter).

The input format is a plain `.txt` file with repeating header markers:

  `Presentation Operator Message`
  `Presenter Speech`
  `Question and Answer Operator Message`
  `Question`
  `Answer`

Each marker is followed by a role line and a body paragraph. This module
walks the markers, groups presenter speeches by speaker, and pairs Q&A
blocks into `(question, answer*)` tuples. See `plan/plan.md` section 1.4.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import roles
from .cache_keys import read_sidecar, stage_key, write_sidecar
from .io_paths import error, units_path


SECTION_HEADERS = (
    "Presentation Operator Message",
    "Presenter Speech",
    "Question and Answer Operator Message",
    "Question",
    "Answer",
)

_HEADER_RE = re.compile(
    r"^(Presentation Operator Message|Presenter Speech|"
    r"Question and Answer Operator Message|Question|Answer)\s*$",
    re.MULTILINE,
)

_TITLE_META_RE = re.compile(
    r"^(?P<company>.+?),\s*(?:Q(?P<q>\d)\s*)?(?P<y>\d{4}).*?Earnings\s+Call"
    r".*?(?P<date>[A-Z][a-z]+\.?\s+\d{1,2},\s*\d{4})",
    re.I,
)

_ROLE_LINE_KEYWORDS = (
    "Executives",
    "Analysts",
    "Analyst",
    "Operator",
    "Company Representatives",
    "Company Representative",
)

_ROLE_LINE_RE = re.compile(
    r"^(Executives|Analysts?|Operator|Company\s+Representatives?)"
    r"(?:\s*-\s*.+)?\s*$"
)

_IR_INTRO_PHRASE_RE = re.compile(
    r"^(?:Thanks?[,.]\s"
    r"|Our\s+next\s+question\b"
    r"|Turning\s+(?:back|to)\b"
    r"|Thank\s+you[.,]?\s+(?:Turning|That\s+concludes)"
    r"|We\s+received\b)"
    r"|\bnext\s+question\s+is\s+from\b",
    re.I,
)

_IR_CLOSING_PHRASE_RE = re.compile(
    r"^(?:Thank\s+you[.,]?\s+)?That\s+concludes\b", re.I
)


@dataclass
class Unit:
    unit_id: str
    ticker: str
    quarter: str
    call_date: str | None
    kind: str
    speaker_role: str
    speaker_title: str
    speaker_name: str | None
    question_text: str | None
    text: str
    n_chars: int
    order: int
    q_role: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Block:
    kind: str
    role_line: str
    body: str


def parse_transcript(text: str, ticker: str, quarter: str) -> list[dict]:
    """Full parse: header -> blocks -> roster -> units.

    Returns a list of plain dicts (JSONL-ready). All errors during header
    parsing fall back to filename meta; block-level issues are logged as
    `warnings` on the offending unit so the pipeline does not halt.
    """
    call_date = _parse_header(text, quarter)
    blocks = _split_blocks(text)
    blocks = _dedupe_blocks(blocks)
    blocks = _promote_ir_intros(blocks)
    roster = _build_call_roster(blocks)
    units = _blocks_to_units(
        blocks, ticker=ticker, quarter=quarter, call_date=call_date, roster=roster
    )
    return [asdict(u) for u in units]


def parse_file(path: Path, ticker: str, quarter: str) -> list[dict]:
    """Read `path` from disk and dispatch to `parse_transcript`."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return parse_transcript(raw, ticker=ticker, quarter=quarter)


def write_units(units: list[dict], ticker: str, quarter: str) -> Path:
    """Persist units to `data/cache/units/<TICKER>_Q<q>-<yyyy>.jsonl`.

    Also writes a ``<stem>.jsonl.key`` sidecar that records the current
    parser source fingerprint so :func:`cache_keys.prune_stale` and any
    future cache-hit shortcut can detect stale units.
    """
    out = units_path(ticker, quarter)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    write_sidecar(out, "units")
    return out


def read_units(ticker: str, quarter: str) -> list[dict]:
    """Load units cached by `write_units`. Errors if missing or stale.

    A sidecar whose key no longer matches the parser source fingerprint
    signals stale units (e.g. the notebook was run with an older parser
    and the user has since modified ``parser.py``). Failing loud prevents
    downstream stages from silently aggregating mixed-version data.
    """
    p = units_path(ticker, quarter)
    if not p.is_file():
        error(f"units cache missing: {p}")
    if read_sidecar(p) != stage_key("units"):
        error(f"units cache stale: {p}")
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _parse_header(text: str, quarter: str) -> str | None:
    """Extract the call_date from the first ~3 lines.

    Returns None when the regex fails (caller keeps going with filename meta).
    `quarter` is passed for sanity-checking but we do not fail if it mismatches.
    """
    head = "\n".join(text.splitlines()[:3])
    m = _TITLE_META_RE.search(head)
    if not m:
        return None
    return _normalize_date(m.group("date"))


def _normalize_date(raw: str) -> str | None:
    """Convert `"Apr 30, 2024"` / `"April 30, 2024"` to `"2024-04-30"`."""
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", raw)
    if not m:
        return None
    mon = months.get(m.group(1).lower()[:4]) or months.get(m.group(1).lower()[:3])
    if mon is None:
        return None
    return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"


def _split_blocks(text: str) -> list[Block]:
    """Walk section-header matches, slicing out `(kind, role_line, body)`.

    The first line of each slice after the header is treated as the role
    line (e.g. `Executives - Chair, President & CEO`); the rest is body.
    Operator-message sections are skipped (not useful for extraction).
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return []
    blocks: list[Block] = []
    for i, m in enumerate(matches):
        kind = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip("\n")
        if kind in ("Presentation Operator Message", "Question and Answer Operator Message"):
            continue
        blocks.extend(_split_chunk_by_speakers(kind, chunk))
    return blocks


_LOOKS_LIKE_TITLE_RE = re.compile(
    r"\b(CEO|CFO|CTO|COO|Chief|Chair|Chairman|Chairwoman|President|Vice\s+President|"
    r"VP|EVP|SVP|Executive|Director|Treasurer|Investor\s+Relations|Head\s+of|Analyst|Founder|Officer)\b",
    re.I,
)


def _split_chunk_by_speakers(kind: str, chunk: str) -> list[Block]:
    """Split a section chunk on role-line boundaries.

    A single `Presenter Speech` often contains IR -> CEO -> CFO in one
    block, each introduced by a line like `Executives - ...`. We locate
    those role lines and carve the chunk into one `Block` per speaker.
    """
    lines = chunk.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _is_role_line(ln)]
    if not idxs:
        role_line, body = _split_role_and_body(chunk)
        if not role_line and not body:
            return []
        return [Block(kind=kind, role_line=role_line, body=body)]
    out: list[Block] = []
    for j, start in enumerate(idxs):
        end = idxs[j + 1] if j + 1 < len(idxs) else len(lines)
        role_line = lines[start].strip()
        body_lines = list(lines[start + 1 : end])
        if role_line in _ROLE_LINE_KEYWORDS and body_lines:
            first_nonempty = next(
                (k for k, ln in enumerate(body_lines) if ln.strip()), None
            )
            if (
                first_nonempty is not None
                and " - " not in role_line
                and _LOOKS_LIKE_TITLE_RE.search(body_lines[first_nonempty])
                and len(body_lines[first_nonempty].strip()) < 120
            ):
                role_line = f"{role_line} - {body_lines[first_nonempty].strip()}"
                body_lines = body_lines[first_nonempty + 1 :]
        body = "\n".join(body_lines).strip()
        if not body and not role_line:
            continue
        out.append(Block(kind=kind, role_line=role_line, body=body))
    return out


def _is_role_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(_ROLE_LINE_RE.match(s))


def _split_role_and_body(chunk: str) -> tuple[str, str]:
    """Separate the first non-empty line (role line) from the rest (body).

    Some transcripts use a multi-line role line ("Executives\nTitle"); we
    join the first block of non-empty lines up to the first blank line
    as the role line when the first line alone does not contain a title.
    """
    lines = [ln.rstrip() for ln in chunk.splitlines()]
    first_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first_idx is None:
        return "", ""
    role = lines[first_idx].strip()
    body_start = first_idx + 1
    if (
        " - " not in role
        and "-" not in role
        and role in _ROLE_LINE_KEYWORDS
        and body_start < len(lines)
        and lines[body_start].strip()
    ):
        role = f"{role} - {lines[body_start].strip()}"
        body_start += 1
    body = "\n".join(lines[body_start:]).strip()
    return role, body


def _dedupe_blocks(blocks: list[Block]) -> list[Block]:
    """Drop adjacent duplicates sharing the same role line + body."""
    out: list[Block] = []
    for b in blocks:
        if out and out[-1].role_line == b.role_line and out[-1].body == b.body:
            continue
        out.append(b)
    return out


def _promote_ir_intros(blocks: list[Block]) -> list[Block]:
    """Normalize IR-host preamble/closing blocks inside a Q&A region.

    Some transcripts (notably PLTR) mark an entire Q&A exchange with a
    single ``Answer`` section header and rely on role lines to separate
    speakers. The IR host (Ana, Simona, ...) appears with the bare role
    line ``Executives`` and her body is either a shareholder-question
    preamble ("Our next question is from ...") or the closing ("Thank
    you. That concludes..."). We:

    1. Drop closing remarks (never useful downstream).
    2. Promote intro remarks from ``Answer`` to ``Question`` so the
       following executive block pairs naturally.
    3. When an IR intro is immediately followed by an analyst-tagged
       ``Question`` block, fold the intro text into that analyst block
       so the analyst's own question carries the full context and we do
       not emit a spurious unpaired IR unit.

    The rule is deliberately narrow (bare ``Executives`` role line +
    anchored IR-intro phrases) to avoid touching transcripts like NVDA
    where the CEO also appears with a bare ``Executives`` role line but
    whose bodies never match IR-intro phrasing.
    """
    qa_kinds = ("Question", "Answer")

    step1: list[Block] = []
    for b in blocks:
        role_bare = b.role_line.strip().lower() == "executives"
        in_qa = b.kind in qa_kinds
        body_head = b.body.lstrip()
        if in_qa and role_bare and _IR_CLOSING_PHRASE_RE.match(body_head):
            continue
        if in_qa and role_bare and _IR_INTRO_PHRASE_RE.search(body_head):
            step1.append(Block(kind="Question", role_line=b.role_line, body=b.body))
        else:
            step1.append(b)

    out: list[Block] = []
    i = 0
    while i < len(step1):
        cur = step1[i]
        nxt = step1[i + 1] if i + 1 < len(step1) else None
        cur_is_ir_q = (
            cur.kind == "Question"
            and cur.role_line.strip().lower() == "executives"
        )
        nxt_is_analyst_q = (
            nxt is not None
            and nxt.kind == "Question"
            and nxt.role_line.strip().lower().startswith("analyst")
        )
        if cur_is_ir_q and nxt_is_analyst_q:
            merged_body = f"{cur.body.strip()}\n\n{nxt.body.strip()}".strip()
            out.append(Block(kind="Question", role_line=nxt.role_line, body=merged_body))
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _build_call_roster(blocks: list[Block]) -> dict[str, roles.RosterEntry]:
    """Scan the first few Presenter Speeches for an IR opener roster line."""
    roster: dict[str, roles.RosterEntry] = {}
    for b in blocks[:4]:
        if b.kind != "Presenter Speech":
            continue
        roster = roles.build_roster(b.body)
        if roster:
            break
    return roster


def _extract_name_from_role_line(role_line: str) -> tuple[str | None, str, str | None]:
    """Return `(speaker_name_or_None, raw_title_portion, category_or_None)`.

    Role lines look like `"Executives - Chair, President & CEO"` or
    `"Analysts - MD & Semiconductor Analyst"`; the first segment before
    ` - ` is the *category* (Executives / Analysts / Operator / ...), not
    a personal name. We return it separately so callers can force the
    correct canonical role when the title itself is ambiguous.
    """
    if " - " in role_line:
        head, tail = role_line.split(" - ", 1)
        return None, tail.strip(), head.strip() or None
    return None, role_line.strip(), role_line.strip() or None


def _role_from_category(category: str | None) -> str | None:
    """Map a role-line category (`Executives`/`Analysts`/`Operator`) to a role.

    Returns `Other_Exec` for the generic `Executives` category -- the
    speaker is clearly an executive even if the title is missing, which
    is more useful than `Unknown` for the downstream by-role aggregation.
    """
    if not category:
        return None
    c = category.strip().lower()
    if c.startswith("analyst"):
        return "Analyst"
    if c == "operator":
        return "Operator"
    if c.startswith("executive") or c.startswith("company representative"):
        return "Other_Exec"
    return None


def _blocks_to_units(
    blocks: list[Block],
    ticker: str,
    quarter: str,
    call_date: str | None,
    roster: dict[str, roles.RosterEntry],
) -> list[Unit]:
    """Fold `Presenter Speech` + paired `Question`/`Answer` into units.

    Presenter blocks each become one unit. Q/A blocks are walked in order:
    the most recent `Question` attaches to every following `Answer` until
    the next `Question`. Orphans (Answer w/o Q, Q w/o A) are emitted with
    a warning tag per Plan 1.4 rule 5.
    """
    units: list[Unit] = []
    order = 0
    pres_idx = 0
    pending_q: Block | None = None
    pending_q_role: str | None = None
    q_has_answer: bool = False

    for b in blocks:
        if b.kind == "Presenter Speech":
            if pending_q is not None and not q_has_answer:
                units.append(
                    _emit_qa_unit(
                        order=_next_order(units),
                        ticker=ticker,
                        quarter=quarter,
                        call_date=call_date,
                        question=pending_q,
                        q_role=pending_q_role or "Analyst",
                        answer=None,
                        roster=roster,
                    )
                )
                pending_q = None
                pending_q_role = None
                q_has_answer = False

            pres_idx += 1
            name, title, category = _extract_name_from_role_line(b.role_line)
            forced = _role_from_category(category)
            role, resolved_name = roles.lookup_role(title, roster, name)
            if forced and role in ("Unknown", "Other_Exec"):
                role = forced
            uid = f"{ticker}_{quarter}__pres__{pres_idx:02d}"
            units.append(
                Unit(
                    unit_id=uid,
                    ticker=ticker,
                    quarter=quarter,
                    call_date=call_date,
                    kind="presenter",
                    speaker_role=role,
                    speaker_title=title,
                    speaker_name=resolved_name,
                    question_text=None,
                    text=b.body,
                    n_chars=len(b.body),
                    order=_next_order(units),
                )
            )
        elif b.kind == "Question":
            if pending_q is not None and not q_has_answer:
                units.append(
                    _emit_qa_unit(
                        order=_next_order(units),
                        ticker=ticker,
                        quarter=quarter,
                        call_date=call_date,
                        question=pending_q,
                        q_role=pending_q_role or "Analyst",
                        answer=None,
                        roster=roster,
                        warn="orphan_question",
                    )
                )
            pending_q = b
            _, q_title, q_cat = _extract_name_from_role_line(b.role_line)
            forced = _role_from_category(q_cat)
            base = roles.classify_title(q_title)
            pending_q_role = forced or (base if base != "Unknown" else "Analyst")
            q_has_answer = False
        elif b.kind == "Answer":
            units.append(
                _emit_qa_unit(
                    order=_next_order(units),
                    ticker=ticker,
                    quarter=quarter,
                    call_date=call_date,
                    question=pending_q,
                    q_role=pending_q_role or "Analyst",
                    answer=b,
                    roster=roster,
                    warn=None if pending_q is not None else "orphan_answer",
                )
            )
            q_has_answer = True
        order += 1

    if pending_q is not None and not q_has_answer:
        units.append(
            _emit_qa_unit(
                order=_next_order(units),
                ticker=ticker,
                quarter=quarter,
                call_date=call_date,
                question=pending_q,
                q_role=pending_q_role or "Analyst",
                answer=None,
                roster=roster,
                warn="orphan_question",
            )
        )
    return units


def _next_order(units: list[Unit]) -> int:
    return (units[-1].order + 1) if units else 0


def _emit_qa_unit(
    order: int,
    ticker: str,
    quarter: str,
    call_date: str | None,
    question: Block | None,
    q_role: str,
    answer: Block | None,
    roster: dict[str, roles.RosterEntry],
    warn: str | None = None,
) -> Unit:
    """Build one QA unit. Requires at least one of question/answer present."""
    qa_count = sum(1 for _ in _iter_existing(question, answer))
    if qa_count == 0:
        error("QA unit requires at least one of question/answer")
    qid_seed = (answer.body if answer is not None else (question.body if question is not None else ""))
    uid_suffix = f"{order:03d}"
    uid = f"{ticker}_{quarter}__qa__{uid_suffix}"

    if answer is not None:
        a_name, a_title, a_cat = _extract_name_from_role_line(answer.role_line)
        forced = _role_from_category(a_cat)
        a_role, resolved = roles.lookup_role(a_title, roster, a_name)
        if forced and a_role in ("Unknown", "Other_Exec"):
            a_role = forced
        text = answer.body
    else:
        a_title = ""
        resolved = None
        a_role = "Unknown"
        text = ""

    q_text = question.body if question is not None else ""
    u = Unit(
        unit_id=uid,
        ticker=ticker,
        quarter=quarter,
        call_date=call_date,
        kind="qa",
        speaker_role=a_role,
        speaker_title=a_title,
        speaker_name=resolved,
        question_text=q_text,
        text=text,
        n_chars=len(text),
        order=order,
        q_role=q_role,
    )
    if warn:
        u.warnings.append(warn)
    _ = qid_seed
    return u


def _iter_existing(*blocks: Block | None) -> Iterable[Block]:
    for b in blocks:
        if b is not None:
            yield b
