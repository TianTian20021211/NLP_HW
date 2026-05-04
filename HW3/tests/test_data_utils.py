"""Unit tests for data/_utils.py (Layer 1, TDD)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pandas as pd
import pytest

from data._utils import (
    FAILED_LOG_COLUMNS,
    US_TICKER_RE,
    FetchResult,
    apply_failed_log_result,
    apply_failed_log_results,
    exponential_backoff,
    load_failed_log,
    load_manifest,
    save_failed_log,
    save_manifest,
    setup_logger,
    suppress_yfinance_logging,
    update_failed_log_result,
    update_failed_log_results,
)


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

class TestFetchResult:
    def test_fetch_result_constructor(self):
        r = FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01")
        assert r.ticker == "AAPL"
        assert r.status == "success"
        assert r.rows == 100
        assert r.first_date == "2020-01-01"
        assert r.last_date == "2024-01-01"
        assert r.error is None

        r2 = FetchResult("AAPL", "error", 0, None, None, error="timeout")
        assert r2.error == "timeout"

        r3 = FetchResult("AAPL", "empty", 0, None, None)
        assert r3.error is None


# ---------------------------------------------------------------------------
# US_TICKER_RE
# ---------------------------------------------------------------------------

class TestUsTickerRe:
    def test_us_ticker_re_valid(self):
        for ticker in ["AAPL", "BRK.B", "A", "ABC-D"]:
            assert US_TICKER_RE.match(ticker), f"{ticker!r} should be valid"

    def test_us_ticker_re_invalid(self):
        for ticker in ["", "12345", "abc", "TOOLONG"]:
            assert not US_TICKER_RE.match(ticker), f"{ticker!r} should be invalid"


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------

class TestSetupLogger:
    def test_setup_logger_returns_logger(self):
        log = setup_logger("test_logger_a1")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_logger_a1"

        debug_log = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert isinstance(debug_log, logging.Logger)
        assert debug_log.name == "test_logger_debug"


# ---------------------------------------------------------------------------
# suppress_yfinance_logging
# ---------------------------------------------------------------------------

class TestSuppressYfinanceLogging:
    def test_suppress_yfinance_logging(self):
        suppress_yfinance_logging()
        assert logging.getLogger("yfinance").level == logging.CRITICAL
        assert logging.getLogger("peewee").level == logging.CRITICAL


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

class TestManifest:
    def test_load_manifest_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        manifest = load_manifest(path)
        assert manifest == {"updated_at": None, "tickers": {}}

    def test_load_manifest_valid_file(self, tmp_path):
        path = tmp_path / "manifest.json"
        original = {"updated_at": None, "tickers": {"AAPL": {"status": "success"}}}
        save_manifest(original, path)
        loaded = load_manifest(path)
        assert loaded["tickers"] == original["tickers"]
        assert "updated_at" in loaded
        assert loaded["updated_at"] is not None

    def test_save_manifest_stamps_updated_at(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest = {"updated_at": None, "tickers": {}}
        save_manifest(manifest, path)
        loaded = json.loads(path.read_text())
        assert loaded["updated_at"] is not None
        assert loaded["updated_at"].endswith("Z")

    def test_save_manifest_thread_safety(self, tmp_path):
        path = tmp_path / "manifest.json"
        lock = threading.Lock()
        n = 10

        def _write(i):
            m = {"tickers": {f"T{i}": {"v": i}}}
            save_manifest(m, path, lock=lock)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert path.exists()
        data = json.loads(path.read_text())
        assert "updated_at" in data


# ---------------------------------------------------------------------------
# exponential_backoff
# ---------------------------------------------------------------------------

class TestExponentialBackoff:
    def test_exponential_backoff(self):
        # With jitter, attempt=0 returns [base, base*1.5)
        val = exponential_backoff(0, 1.0, 120.0)
        assert 1.0 <= val < 1.5
        # attempt=1: [base*2, base*3)
        val = exponential_backoff(1, 1.0, 120.0)
        assert 2.0 <= val < 3.0
        val = exponential_backoff(2, 1.0, 120.0)
        assert 4.0 <= val < 6.0
        val = exponential_backoff(3, 1.0, 120.0)
        assert 8.0 <= val < 12.0
        val = exponential_backoff(4, 1.0, 120.0)
        assert 16.0 <= val < 24.0
        val = exponential_backoff(5, 1.0, 120.0)
        assert 32.0 <= val < 48.0
        # Max cap still respected
        assert exponential_backoff(10, 1.0, 120.0) == 120.0
        # Different base with jitter
        val = exponential_backoff(0, 0.6, 60.0)
        assert 0.6 <= val < 0.9
        assert exponential_backoff(10, 0.6, 60.0) == 60.0


# ---------------------------------------------------------------------------
# Failed-log helpers
# ---------------------------------------------------------------------------

class TestFailedLogIO:
    def test_load_failed_log_missing(self, tmp_path):
        path = tmp_path / "nonexistent.csv"
        df = load_failed_log(path)
        assert list(df.columns) == FAILED_LOG_COLUMNS
        assert df.empty

    def test_load_failed_log_roundtrip(self, tmp_path):
        path = tmp_path / "failed.csv"
        original = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        save_failed_log(original, path)
        loaded = load_failed_log(path)
        assert list(loaded.columns) == FAILED_LOG_COLUMNS
        assert len(loaded) == 1
        assert loaded["ticker"].iloc[0] == "AAPL"
        assert loaded["status"].iloc[0] == "error"


class TestApplyFailedLogResult:
    """apply_failed_log_result — shares semantics: one latest row per ticker."""

    def test_apply_failed_log_result_success_clears_ticker(self):
        existing = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
            {"ticker": "MSFT", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        result = FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01")
        new = apply_failed_log_result(existing, result)
        assert "AAPL" not in new["ticker"].values
        assert "MSFT" in new["ticker"].values

    def test_apply_failed_log_result_empty_appends(self):
        existing = pd.DataFrame(columns=FAILED_LOG_COLUMNS)
        result = FetchResult("AAPL", "empty", 0, None, None)
        new = apply_failed_log_result(existing, result)
        assert len(new) == 1
        assert new["ticker"].iloc[0] == "AAPL"
        assert new["status"].iloc[0] == "empty"

    def test_apply_failed_log_result_failure_replaces_previous_failure(self):
        existing = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        result = FetchResult("AAPL", "error", 0, None, None, error="rate_limit")
        new = apply_failed_log_result(existing, result)
        assert len(new) == 1
        assert new["ticker"].iloc[0] == "AAPL"
        assert new["error"].iloc[0] == "rate_limit"

    def test_apply_failed_log_result_failure_then_success_clears(self):
        existing = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        result = FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01")
        new = apply_failed_log_result(existing, result)
        assert new.empty

    def test_apply_failed_log_result_success_then_failure_records_failure(self):
        existing = pd.DataFrame(columns=FAILED_LOG_COLUMNS)
        result_success = FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01")
        existing = apply_failed_log_result(existing, result_success)
        assert existing.empty
        result_failure = FetchResult("AAPL", "error", 0, None, None, error="timeout")
        existing = apply_failed_log_result(existing, result_failure)
        assert len(existing) == 1
        assert existing["ticker"].iloc[0] == "AAPL"
        assert existing["status"].iloc[0] == "error"


class TestApplyFailedLogResults:
    """apply_failed_log_results — price semantics: append batch misses, clear successes."""

    def test_apply_failed_log_results_success_clears(self):
        existing = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
            {"ticker": "MSFT", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        results = [
            FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01"),
            FetchResult("GOOG", "success", 200, "2020-01-01", "2024-01-01"),
        ]
        new = apply_failed_log_results(existing, results)
        assert "AAPL" not in new["ticker"].values
        assert "MSFT" in new["ticker"].values
        assert len(new) == 1

    def test_apply_failed_log_results_empty_appends_no_dedup(self):
        existing = pd.DataFrame(columns=FAILED_LOG_COLUMNS)
        results = [
            FetchResult("AAPL", "empty", 0, None, None),
            FetchResult("AAPL", "empty", 0, None, None),
        ]
        new = apply_failed_log_results(existing, results)
        assert len(new) == 2
        assert all(new["ticker"] == "AAPL")

    def test_apply_failed_log_results_same_ticker_success_and_failure(self):
        existing = pd.DataFrame([
            {"ticker": "AAPL", "status": "error", "rows": "0",
             "first_date": "", "last_date": "", "error": "timeout"},
        ])
        results = [
            FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01"),
            FetchResult("AAPL", "error", 0, None, None, error="rate_limit"),
        ]
        new = apply_failed_log_results(existing, results)
        assert len(new) == 1
        assert new["ticker"].iloc[0] == "AAPL"
        assert new["status"].iloc[0] == "error"
        assert new["error"].iloc[0] == "rate_limit"


class TestUpdateFailedLog:
    """update_failed_log_result / update_failed_log_results — file roundtrip."""

    def test_update_failed_log_result_locked_read_apply_write(self, tmp_path):
        path = tmp_path / "failed.csv"
        lock = threading.Lock()

        result = FetchResult("AAPL", "error", 0, None, None, error="timeout")
        update_failed_log_result(path, result, lock=lock)
        assert path.exists()
        df = pd.read_csv(path, dtype=str)
        assert len(df) == 1
        assert df["ticker"].iloc[0] == "AAPL"
        assert df["error"].iloc[0] == "timeout"

        success = FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01")
        update_failed_log_result(path, success, lock=lock)
        df = pd.read_csv(path, dtype=str)
        assert len(df) == 0

    def test_update_failed_log_results_locked_batch_read_apply_write(self, tmp_path):
        path = tmp_path / "failed.csv"
        lock = threading.Lock()

        results = [
            FetchResult("AAPL", "error", 0, None, None, error="timeout"),
            FetchResult("MSFT", "empty", 0, None, None),
        ]
        update_failed_log_results(path, results, lock=lock)
        assert path.exists()
        df = pd.read_csv(path, dtype=str)
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAPL", "MSFT"}

        results2 = [
            FetchResult("AAPL", "success", 100, "2020-01-01", "2024-01-01"),
        ]
        update_failed_log_results(path, results2, lock=lock)
        df = pd.read_csv(path, dtype=str)
        assert len(df) == 1
        assert df["ticker"].iloc[0] == "MSFT"
