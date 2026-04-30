"""Project-wide constants and paths.

Single source of truth for the random seed, cache locations, and the
input CSV. Every later step imports from here so we change paths in
one place.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
PRICE_CACHE_DIR: Path = CACHE_DIR / "prices"
SHARES_CACHE_DIR: Path = CACHE_DIR / "shares"
UNIVERSE_CACHE_DIR: Path = CACHE_DIR / "universes"
UNIVERSE_RAW_DIR: Path = DATA_DIR / "universe_raw"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
AUDIT_DIR: Path = RESULTS_DIR / "audit"

RAW_SIGNAL_CSV: Path = DATA_DIR / "Earnings_ATC_until_2026-04-21.csv"
RAW_SIGNAL_ZIP: Path = DATA_DIR / "data.zip"

SIGNALS_PARQUET: Path = CACHE_DIR / "signals.parquet"
SIGNALS_SLIM_PARQUET: Path = CACHE_DIR / "signals_slim.parquet"

PRICE_MANIFEST: Path = PRICE_CACHE_DIR / "_manifest.json"
PRICE_FAILED_TICKERS: Path = PRICE_CACHE_DIR / "failed_tickers.csv"

UNIVERSE_NAMES: tuple[str, ...] = ("sp500", "sp1500", "ru3k")


def ensure_dirs() -> None:
    for p in (
        CACHE_DIR,
        PRICE_CACHE_DIR,
        SHARES_CACHE_DIR,
        UNIVERSE_CACHE_DIR,
        UNIVERSE_RAW_DIR,
        RESULTS_DIR,
        AUDIT_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
