"""Title string -> canonical role, plus per-call roster extraction.

Canonical roles: `CEO`, `CFO`, `CTO`, `IR`, `Other_Exec`, `Analyst`,
`Operator`, `Unknown`. `COO` is folded into `Other_Exec` per Plan 1.4.

Two layers of evidence are used, in order:
  1. The role line above a speaker block (e.g. `Executives - Chair, President & CEO`).
  2. The IR opener roster (`Joining me today are Alex Karp, CEO; ...`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CANONICAL_ROLES = (
    "CEO",
    "CFO",
    "CTO",
    "IR",
    "Other_Exec",
    "Analyst",
    "Operator",
    "Unknown",
)


_TITLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bchief\s+executive\b", re.I), "CEO"),
    (re.compile(r"\bceo\b", re.I), "CEO"),
    (re.compile(r"\bchief\s+financial\b", re.I), "CFO"),
    (re.compile(r"\bcfo\b", re.I), "CFO"),
    (re.compile(r"\btreasurer\b", re.I), "CFO"),
    (re.compile(r"\bchief\s+technology\b", re.I), "CTO"),
    (re.compile(r"\bcto\b", re.I), "CTO"),
    (re.compile(r"\bchief\s+operating\b", re.I), "Other_Exec"),
    (re.compile(r"\bcoo\b", re.I), "Other_Exec"),
    (re.compile(r"\binvestor\s+relations\b", re.I), "IR"),
    (re.compile(r"\bhead\s+of\s+ir\b", re.I), "IR"),
    (re.compile(r"\banalyst\b", re.I), "Analyst"),
    (re.compile(r"\boperator\b", re.I), "Operator"),
)


def classify_title(title: str | None) -> str:
    """Return the canonical role for a raw title string.

    Priority order follows `_TITLE_PATTERNS`. `Chair(man)` alone maps to
    `Other_Exec` but co-occurrence with CEO/President is handled by the
    CEO pattern firing first. Empty / None -> `Unknown`.
    """
    if not title:
        return "Unknown"
    s = title.strip()
    if not s:
        return "Unknown"
    for pat, role in _TITLE_PATTERNS:
        if pat.search(s):
            return role
    if (
        re.search(r"\bchair(man|woman|person)?\b", s, re.I)
        or re.search(
            r"\b(president|executive\s+vp|evp|senior\s+vp|svp|vice\s+president|director)\b",
            s,
            re.I,
        )
        or re.search(r"\bchief\s+\w+\s+officer\b", s, re.I)
        or re.search(r"\b(officer|founder|head\s+of)\b", s, re.I)
    ):
        return "Other_Exec"
    return "Unknown"


@dataclass
class RosterEntry:
    name: str
    title: str
    role: str


_JOIN_PAT = re.compile(
    r"Joining\s+(?:me|us)\s+(?:on\s+)?today(?:'s\s+call)?\s+(?:are|is)\s+(.+?)(?:\.|$)",
    re.I | re.DOTALL,
)
_ALT_INTRO_PAT = re.compile(
    r"(?:Participants|Joining\s+me\s+today|Also\s+joining\s+me|With\s+me\s+today)[^.]*?(?:are|is|include[s]?)\s+(.+?)(?:\.|$)",
    re.I | re.DOTALL,
)


def build_roster(ir_opener_text: str) -> dict[str, RosterEntry]:
    """Parse IR opener text into `{normalized_name -> RosterEntry}`.

    Handles segments like:
      `Joining me today are Alex Karp, Chief Executive Officer;
       Shyam Sankar, Chief Technology Officer; ...`
    Tolerates `and` before the last entry and commas inside titles
    (splits on `;` first, falls back to a `,<Name>,<Title>` split).
    """
    m = _JOIN_PAT.search(ir_opener_text) or _ALT_INTRO_PAT.search(ir_opener_text)
    if not m:
        return {}
    blob = m.group(1)
    blob = re.sub(r"\s+and\s+", "; ", blob, count=1, flags=re.I)
    entries = [e.strip(" ;,.") for e in blob.split(";") if e.strip(" ;,.")]
    if len(entries) <= 1:
        entries = _fallback_split(blob)
    roster: dict[str, RosterEntry] = {}
    for e in entries:
        if "," not in e:
            continue
        name, title = e.split(",", 1)
        name = name.strip().strip(".")
        title = title.strip().rstrip(".")
        if not name or not title:
            continue
        roster[_norm_name(name)] = RosterEntry(name=name, title=title, role=classify_title(title))
    return roster


def _fallback_split(blob: str) -> list[str]:
    """When no `;` separators, split on `<Title>, <Name>, <Title>` boundaries.

    Heuristic: treat `,` followed by a Capitalized token sequence ending
    before the next `,` as a name boundary. Used for the minority of
    transcripts whose IR opener uses commas rather than semicolons.
    """
    parts = [p.strip() for p in re.split(r",\s*(?=[A-Z][a-z]+\s+[A-Z])", blob) if p.strip()]
    return parts


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def lookup_role(
    role_line: str | None,
    roster: dict[str, RosterEntry] | None = None,
    speaker_name: str | None = None,
) -> tuple[str, str | None]:
    """Return `(role, resolved_name)` for a speaker block.

    Tries the role line's own title first, then falls back to the roster
    if a `speaker_name` is provided (or can be extracted from `role_line`).
    """
    role = classify_title(role_line)
    name: str | None = speaker_name
    if role != "Unknown":
        return role, name
    if roster and speaker_name:
        entry = roster.get(_norm_name(speaker_name))
        if entry is not None:
            return entry.role, entry.name
    return "Unknown", name
