from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

SHORT_WORD_MAX = 5

FEATURE_COLUMNS: tuple[str, ...] = tuple(f"h{i:02d}" for i in range(1, 31))

_RE_HOST = re.compile(
    r"(?i)(^|\s)(operator|moderator|coordinator)\s*:|\b(operator|moderator|coordinator)\s*:",
)
_RE_FIRM = re.compile(
    r"(?i)\b(from|at|with)\s+(?-i:[A-Z])[a-zA-Z0-9&\-\.]*\b",
)
_RE_GREET_TIME = re.compile(r"(?i)\bgood\s+(morning|afternoon|evening)\b")
_RE_WELCOME = re.compile(r"(?i)(thank\s+you\s+for\s+joining|welcome\s+(everyone|you\s+all))\b")
_RE_FLOW = re.compile(
    r"(?i)(\bturn\s+it\s+over\b|\bopen\s+(the\s+)?call\b|\bbegin\s+(the\s+)?question\b)",
)
_RE_QA = re.compile(r"(?i)\b(next\s+question|our\s+next\s+caller|moving\s+on)\b")
_RE_MATERIAL = re.compile(r"(?i)\b(replay|webcast|press\s+release|8[\-\s]?K)\b")
_RE_SAFE = re.compile(r"(?i)\b(safe\s+harbor|forward[\-\s]?looking)\b")
_RE_GAAP = re.compile(r"(?i)\b(GAAP|non[\-\s]?GAAP)\b")
_RE_REG = re.compile(
    r"(?i)\b(SEC|Regulation\s+FD|Private\s+Securities\s+Litigation\s+Reform)\b",
)
_RE_FORWARD_WORDS = re.compile(r"(?i)\b(beliefs?|expectations?|projections?)\b")
_RE_MODALS = re.compile(r"(?i)\b(may|could|might|will)\b")
_RE_MONEY = re.compile(r"\$|\bUSD\b")
_RE_PCT = re.compile(r"\d+(?:\.\d+)?%")
_RE_MAG = re.compile(r"(?i)\b(million|billion|thousand)\b")
_RE_BPS = re.compile(r"(?i)(\bbasis\s+points\b|\bbps\b)")
_RE_TS = re.compile(
    r"(?i)(\bYoY\b|year\s+over\s+year|\bsequential\b|\bQ[1-4]\b)",
)
_RE_FIN = re.compile(r"(?i)\b(margin|EPS|EBITDA|revenue|guidance|outlook)\b")
_RE_THANKS = re.compile(r"(?i)\b(thank\s+you|thanks)\b")
_RE_PRAISE = re.compile(r"(?i)\b(congratulations|great\s+quarter|nice\s+job)\b")
_RE_ENUM = re.compile(r"(?i)\b(first|second|third|finally)\s*,")
_RE_STEP = re.compile(r"(?i)\b(step|phase)\s*\d+\b")
_RE_CORP = re.compile(r"(?i)\b(buybacks?|repurchases?|dividends?|M&A|acquisitions?)\b")
_RE_HR = re.compile(r"(?i)\b(layoffs?|restructuring|headcount)\b")
_RE_DIGIT = re.compile(r"\d")


def _caps_run_ratio(t: str) -> float:
    if not t:
        return 0.0
    run = 0
    run_chars = 0
    for ch in t:
        if ch.isupper():
            run += 1
        else:
            if run >= 2:
                run_chars += run
            run = 0
    if run >= 2:
        run_chars += run
    return run_chars / max(len(t), 1)


def _bool_match(pat: re.Pattern[str], t: str) -> float:
    return 1.0 if pat.search(t) else 0.0


def _weak_modal_forward(t: str) -> float:
    return 1.0 if (_RE_FORWARD_WORDS.search(t) and _RE_MODALS.search(t)) else 0.0


def handcrafted_feature_row(text: object) -> np.ndarray:
    """Return shape ``(30,)`` float handcrafted features for one sentence (plan feature IDs 1–30)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        t = ""
    else:
        t = str(text)
    words = t.split()
    n_words = len(words)
    n_chars = len(t)
    digit_count = len(_RE_DIGIT.findall(t))
    punct_cs = sum(1 for ch in t if ch in ",;")

    row = np.array(
        [
            _bool_match(_RE_HOST, t),
            _bool_match(_RE_FIRM, t),
            _bool_match(_RE_GREET_TIME, t),
            _bool_match(_RE_WELCOME, t),
            _bool_match(_RE_FLOW, t),
            _bool_match(_RE_QA, t),
            _bool_match(_RE_MATERIAL, t),
            _bool_match(_RE_SAFE, t),
            _bool_match(_RE_GAAP, t),
            _bool_match(_RE_REG, t),
            _weak_modal_forward(t),
            _bool_match(_RE_MONEY, t),
            _bool_match(_RE_PCT, t),
            digit_count / max(n_chars, 1),
            _bool_match(_RE_MAG, t),
            _bool_match(_RE_BPS, t),
            _bool_match(_RE_TS, t),
            _bool_match(_RE_FIN, t),
            _bool_match(_RE_THANKS, t),
            _bool_match(_RE_PRAISE, t),
            float(n_chars),
            1.0 if (0 < n_words <= SHORT_WORD_MAX) else 0.0,
            1.0 if "?" in t else 0.0,
            float(t.count("!")),
            punct_cs / max(n_chars, 1),
            _caps_run_ratio(t),
            _bool_match(_RE_ENUM, t),
            _bool_match(_RE_STEP, t),
            _bool_match(_RE_CORP, t),
            _bool_match(_RE_HR, t),
        ],
        dtype=np.float64,
    )
    assert row.shape[0] == 30
    return row


def handcrafted_features_matrix(texts: Sequence[object] | pd.Series) -> np.ndarray:
    """Shape ``(n_sentences, 30)`` for parallel training/inference."""
    col = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    n = len(col)
    mat = np.zeros((n, 30), dtype=np.float64)
    for i, raw in enumerate(col.tolist()):
        mat[i] = handcrafted_feature_row(raw)
    return mat


def handcrafted_features_frame(texts: Sequence[object] | pd.Series) -> pd.DataFrame:
    """Labeled columns ``h01``–``h30`` aligned with ``FEATURE_COLUMNS``."""
    mat = handcrafted_features_matrix(texts)
    return pd.DataFrame(mat, columns=list(FEATURE_COLUMNS))
