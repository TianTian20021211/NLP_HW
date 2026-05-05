"""Phase 1.1 - ATC signal CSV -> Parquet caches.

Plan rules (`ideas/plan.md`):
- Read `Earnings_ATC_until_2026-04-21.csv` in 100k-row chunks (4.5 GB CSV
  cannot be loaded whole).
- Drop `SignalType == 'delete'` (~2,231 rows).
- Do NOT pre-filter on `COUNTRY == 'US'`; keep the full cleaned signal set.
- Parse `MOSTIMPORTANTDATEUTC`, extract the hour, persist `call_hour_utc`.
- Main cache keeps every non-`Fluff` / non-`Filler` `AspectTheme_*` column
  for both Enhanced and Stretch tiers.
- Slim cache keeps only identifiers + headline fields + the EventScore
  family + `ATCClassifierScore` (for fast feature-prototyping).

Outputs: `data/cache/signals.parquet`, `data/cache/signals_slim.parquet`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data._utils import setup_logger
from data.config import (
    RAW_SIGNAL_CSV,
    RAW_SIGNAL_ZIP,
    SIGNALS_PARQUET,
    SIGNALS_SLIM_PARQUET,
    ensure_dirs,
    set_global_seed,
)
from data.progress import progress


CHUNKSIZE = 100_000
DROP_SIGNAL_TYPE = "delete"

ID_COLS = [
    "SignalType",
    "DocDate",
    "Ticker",
    "DocID",
    "CORPUS",
    "KEYDEVID",
    "HEADLINE",
    "DOCTYPE",
    "MOSTIMPORTANTDATEUTC",
    "TRANSCRIPTCREATIONDATEUTC",
    "COMPANYNAME",
    "COMPANYID",
    "SIMPLEINDUSTRYDESCRIPTION",
    "SECTOR",
    "COUNTRY",
    "GVKEY",
    "CIK",
    "IID",
    "ISIN",
    "BESTTICKER",
    "EXCHANGE",
    "EX_NAME",
    "EX_CODE",
    "EX_MIC",
    "QTR_YEAR",
    "DOCSECTIONCOUNT",
    "DOCSENTENCECOUNT",
    "INGESTDATEUTC",
    "Sentences",
]

EVENT_SCORE_COLS = [
    "EventPos_1_1_1", "EventNeg_1_1_1", "EventsScore_1_1_1",
    "EventPos_4_2_1", "EventNeg_4_2_1", "EventsScore_4_2_1",
    "EventPos_3_1_0", "EventNeg_3_1_0", "EventsScore_3_1_0",
    "EventPos_1_1_0", "EventNeg_1_1_0", "EventsScore_1_1_0",
]

HEADLINE_NUMERIC_COLS = ["ATCClassifierScore"]

EXCLUDED_ASPECTS = ("Fluff", "Filler")
INTEGER_ID_COLS = {"DOCSECTIONCOUNT", "DOCSENTENCECOUNT", "Sentences"}
STRING_ID_COLS = [c for c in ID_COLS if c not in INTEGER_ID_COLS]
INTEGER_EVENT_COLS = [c for c in EVENT_SCORE_COLS if not c.startswith("EventsScore_")]
FLOAT_EVENT_COLS = [c for c in EVENT_SCORE_COLS if c.startswith("EventsScore_")]
FLOAT_COLS = FLOAT_EVENT_COLS + HEADLINE_NUMERIC_COLS

log = setup_logger("load_signals")


def ensure_signal_csv(
    csv_path: Path = RAW_SIGNAL_CSV, zip_path: Path | None = None
) -> Path:
    """Return an extracted signal CSV path, extracting the raw zip if needed."""
    if csv_path.exists():
        return csv_path

    zip_path = zip_path or RAW_SIGNAL_ZIP
    if zip_path.exists():
        log.info("signal CSV missing; extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            member = next(
                (m for m in zf.namelist() if Path(m).name == csv_path.name),
                None,
            )
            if member is None:
                member = next((m for m in zf.namelist() if m.lower().endswith(".csv")), None)
            if member is None:
                raise FileNotFoundError(
                    f"no CSV member found inside {RAW_SIGNAL_ZIP}"
                )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, csv_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        return csv_path

    raise FileNotFoundError(f"signal CSV not found at {csv_path}")


def read_header(path: Path) -> list[str]:
    """Read just the header row to discover column names without loading data."""
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def classify_columns(all_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split `AspectTheme_*` columns into kept (non-Fluff/Filler) vs dropped.

    Column shape: ``AspectTheme_{Aspect}_{Theme} - {Magnitude} - {Sentiment}``.
    The aspect token is the second underscore-segment.
    """
    keep: list[str] = []
    drop: list[str] = []
    for col in all_columns:
        if not col.startswith("AspectTheme_"):
            continue
        rest = col[len("AspectTheme_"):]
        aspect = rest.split("_", 1)[0]
        if aspect in EXCLUDED_ASPECTS:
            drop.append(col)
        else:
            keep.append(col)
    return keep, drop


def parse_call_hour(series: pd.Series) -> pd.Series:
    """Parse MOSTIMPORTANTDATEUTC -> hour (UTC). NaN stays as NaN (nullable Int8).

    Downstream code must check for pd.NA before using the value, rather than
    relying on a -1 sentinel.
    """
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0 has no format="mixed"
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    hours = parsed.dt.hour.astype("Int8")
    return hours


def coerce_chunk_types(chunk: pd.DataFrame, aspect_cols: list[str]) -> pd.DataFrame:
    """Apply stable dtypes so every streamed Parquet chunk has one schema."""
    for col in STRING_ID_COLS:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype("string")

    for col in INTEGER_ID_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

    for col in INTEGER_EVENT_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")

    for col in FLOAT_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")

    # Batch-convert all aspect columns at once (up to ~400 cols) to avoid
    # per-column Python->C boundary crossing overhead.
    mask = [c for c in aspect_cols if c in chunk.columns]
    if mask:
        chunk[mask] = (
            chunk[mask]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype("int32")
        )

    return chunk


def estimate_chunk_count(
    csv_path: Path,
    chunksize: int,
    sample_rows: int = 10_000,
) -> int | None:
    """Estimate chunk count without scanning the full multi-GB CSV."""
    if chunksize <= 0:
        return None

    file_size = csv_path.stat().st_size
    with csv_path.open("rb") as f:
        header = f.readline()
        bytes_read = 0
        rows_read = 0
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            bytes_read += len(line)
            rows_read += 1

    if rows_read == 0 or bytes_read == 0:
        return None

    avg_row_bytes = bytes_read / rows_read
    estimated_rows = max(1, int((file_size - len(header)) / avg_row_bytes))
    return math.ceil(estimated_rows / chunksize)


def stream_chunks(
    csv_path: Path,
    main_cols: list[str],
    slim_cols: list[str],
    chunksize: int = CHUNKSIZE,
) -> dict:
    """Stream the raw CSV chunk-by-chunk into two Parquet writers."""
    main_writer: pq.ParquetWriter | None = None
    slim_writer: pq.ParquetWriter | None = None
    stats = {
        "rows_read": 0,
        "rows_dropped_delete": 0,
        "rows_written": 0,
        "chunks": 0,
    }

    usecols = sorted(set(main_cols))
    int_aspect_cols = [c for c in usecols if c.startswith("AspectTheme_")]
    string_id_cols = [c for c in STRING_ID_COLS if c in usecols]
    dtype_hints = {c: "string" for c in string_id_cols}

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=usecols,
        dtype=dtype_hints,
        low_memory=False,
    )

    total_chunks = estimate_chunk_count(csv_path, chunksize)
    chunk_bar = progress(
        reader,
        total=total_chunks,
        desc="signals chunks",
        unit="chunk",
    )

    t0 = time.time()
    try:
        for i, chunk in enumerate(chunk_bar):
            stats["chunks"] += 1
            before = len(chunk)
            stats["rows_read"] += before

            chunk = chunk.loc[chunk["SignalType"] != DROP_SIGNAL_TYPE]
            stats["rows_dropped_delete"] += before - len(chunk)
            if chunk.empty:
                chunk_bar.set_postfix({
                    "rows_read": f"{stats['rows_read']:,}",
                    "rows_written": f"{stats['rows_written']:,}",
                })
                continue

            # Normalize ticker format: dots -> dashes for consistency with
            # universe PIT data and price cache keys.
            chunk["BESTTICKER"] = chunk["BESTTICKER"].str.replace(".", "-", regex=False)
            chunk["Ticker"] = chunk["Ticker"].str.replace(".", "-", regex=False)

            chunk = coerce_chunk_types(chunk, int_aspect_cols)
            call_hour = parse_call_hour(chunk["MOSTIMPORTANTDATEUTC"])

            main_df = chunk[main_cols].copy()
            main_df["call_hour_utc"] = call_hour.values

            main_table = pa.Table.from_pandas(main_df, preserve_index=False)
            slim_columns = [c for c in slim_cols if c in main_table.column_names]
            slim_columns.append("call_hour_utc")
            slim_table = main_table.select(slim_columns)

            if main_writer is None:
                main_writer = pq.ParquetWriter(
                    str(SIGNALS_PARQUET),
                    main_table.schema,
                    compression="zstd",
                )
                slim_writer = pq.ParquetWriter(
                    str(SIGNALS_SLIM_PARQUET),
                    slim_table.schema,
                    compression="zstd",
                )

            main_writer.write_table(main_table)
            slim_writer.write_table(slim_table)
            stats["rows_written"] += len(chunk)
            chunk_bar.set_postfix({
                "rows_read": f"{stats['rows_read']:,}",
                "rows_written": f"{stats['rows_written']:,}",
            })

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                log.info(
                    "chunk=%d rows_read=%s rows_written=%s elapsed=%.1fs",
                    i + 1,
                    f"{stats['rows_read']:,}",
                    f"{stats['rows_written']:,}",
                    elapsed,
                )
    finally:
        chunk_bar.close()
        if main_writer is not None:
            main_writer.close()
        if slim_writer is not None:
            slim_writer.close()
    return stats


def run(
    csv_path: Path = RAW_SIGNAL_CSV,
    zip_path: Path | None = None,
    chunksize: int = CHUNKSIZE,
) -> dict:
    set_global_seed()
    ensure_dirs()
    csv_path = ensure_signal_csv(csv_path, zip_path=zip_path)

    header = read_header(csv_path)
    keep_aspect, drop_aspect = classify_columns(header)
    log.info(
        "header: total=%d aspect_kept=%d aspect_dropped=%d",
        len(header),
        len(keep_aspect),
        len(drop_aspect),
    )

    id_cols = [c for c in ID_COLS if c in header]
    event_cols = [c for c in EVENT_SCORE_COLS if c in header]
    headline_cols = [c for c in HEADLINE_NUMERIC_COLS if c in header]

    main_cols = id_cols + event_cols + headline_cols + keep_aspect
    slim_cols = id_cols + event_cols + headline_cols

    log.info(
        "main cache: %d cols (id=%d, event=%d, headline=%d, aspect=%d) -> %s",
        len(main_cols) + 1,
        len(id_cols),
        len(event_cols),
        len(headline_cols),
        len(keep_aspect),
        SIGNALS_PARQUET,
    )
    log.info(
        "slim cache: %d cols -> %s",
        len(slim_cols) + 1,
        SIGNALS_SLIM_PARQUET,
    )

    stats = stream_chunks(csv_path, main_cols, slim_cols, chunksize=chunksize)
    log.info(
        "done: rows_read=%s rows_dropped_delete=%s rows_written=%s chunks=%d",
        f"{stats['rows_read']:,}",
        f"{stats['rows_dropped_delete']:,}",
        f"{stats['rows_written']:,}",
        stats["chunks"],
    )

    # Sanity check: documented delete count is ~2,231; warn on >20% deviation.
    expected_delete = 2_231
    actual_delete = stats["rows_dropped_delete"]
    if actual_delete > 0:
        deviation = abs(actual_delete - expected_delete) / expected_delete
        if deviation > 0.20:
            log.warning(
                "delete rows %s deviates >20%% from documented %s — "
                "CSV format may have changed",
                f"{actual_delete:,}",
                f"{expected_delete:,}",
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="ATC signal CSV -> Parquet")
    parser.add_argument("--csv", type=Path, default=RAW_SIGNAL_CSV)
    parser.add_argument("--zip", type=Path, default=None,
                        help="Path to the source ZIP file (extracted if CSV is missing)")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    args = parser.parse_args()
    run(csv_path=args.csv, zip_path=args.zip, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
