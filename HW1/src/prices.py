"""yfinance Close-price fetcher for Part 2 backtest.

Every ticker (+ SPY) is persisted to ``data/cache/prices/<TICKER>.parquet`` with
a minimal ``[Date, Close]`` schema and is treated as immutable: yfinance
nightly patches history which would introduce silent drift if we re-fetched
every session. Set ``REFRESH_PRICES=True`` in the notebook to force a
re-download.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .io_paths import PRICES, error, price_path


PRICE_START_DEFAULT = "2023-09-01"
CORPUS_TICKERS = (
    "AMD",
    "AVGO",
    "BLK",
    "C",
    "FAST",
    "FDX",
    "GS",
    "INTC",
    "JNJ",
    "JPM",
    "NKE",
    "NVDA",
    "PLTR",
    "WFC",
)
BENCHMARK_TICKER = "SPY"


def fetch_prices(
    ticker: str,
    start: str = PRICE_START_DEFAULT,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily Close for ``ticker`` and persist as parquet.

    Returns a ``DataFrame`` indexed ``0..N-1`` with columns ``[Date, Close]``.
    Cache is considered definitive unless ``refresh=True`` — yfinance nightly
    patches historical Close prices, so any re-fetch would silently shift the
    pipeline's label computations.

    ``end`` is passed straight to yfinance (exclusive upper bound). ``None``
    means "up to today's trading day".
    """
    path = price_path(ticker)
    if path.is_file() and not refresh:
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if raw is None or raw.empty:
        error(f"yfinance returned nothing for {ticker} ({start} → {end or 'today'})")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        error(f"yfinance payload for {ticker} missing 'Close' column: {list(raw.columns)}")
    df = raw[["Close"]].reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df["Close"] = df["Close"].astype("float64")
    df = df.sort_values("Date").reset_index(drop=True)
    PRICES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_prices(ticker: str) -> pd.DataFrame:
    """Load a previously-fetched prices parquet; fail loud if absent."""
    path = price_path(ticker)
    if not path.is_file():
        error(f"prices cache missing for {ticker}: {path} (run fetch_prices first)")
    return pd.read_parquet(path)


def fetch_corpus_prices(
    tickers: Iterable[str] = CORPUS_TICKERS,
    start: str = PRICE_START_DEFAULT,
    end: str | None = None,
    refresh: bool = False,
    include_benchmark: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch every corpus ticker + ``SPY`` benchmark, printing a sanity line each.

    Returns a ``{ticker: DataFrame}`` map. Identical behavior whether the
    cache is cold or warm, because the sanity print comes from the loaded
    DataFrame regardless of fetch path.
    """
    out: dict[str, pd.DataFrame] = {}
    all_tickers: list[str] = list(tickers)
    if include_benchmark and BENCHMARK_TICKER not in all_tickers:
        all_tickers.append(BENCHMARK_TICKER)
    for tic in all_tickers:
        df = fetch_prices(tic, start=start, end=end, refresh=refresh)
        if df.empty:
            error(f"empty prices frame for {tic}")
        dmin = df["Date"].min().date().isoformat()
        dmax = df["Date"].max().date().isoformat()
        print(f"  prices[{tic:>5s}] rows={len(df):>4d}  {dmin} → {dmax}")
        out[tic] = df
    return out
