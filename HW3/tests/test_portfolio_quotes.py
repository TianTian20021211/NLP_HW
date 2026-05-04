"""Unit tests for _next_valid_quote — 5-trading-day forward scan for valid
close+volume data, used in portfolio entry/exit lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio import _next_valid_quote


def _make_calendar(dates: list[str]) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    cal = [pd.Timestamp(d) for d in dates]
    cal_pos = {d: i for i, d in enumerate(cal)}
    return cal, cal_pos


class TestNextValidQuote:
    """Scans up to *max_forward_days* trading days from *planned_date*."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix(index, columns, values):
        """Build a close_matrix DataFrame from lists."""
        return pd.DataFrame(values, index=pd.DatetimeIndex(index), columns=columns)

    # ------------------------------------------------------------------
    # happy path
    # ------------------------------------------------------------------

    def test_exact_match(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03", "2020-03-04"])
        cm = self._matrix(
            ["2020-03-02", "2020-03-03"],
            ["AAPL"],
            [[100.0], [101.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-02")
        assert price == 100.0

    def test_next_day_when_planned_missing(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03", "2020-03-04"])
        cm = self._matrix(
            ["2020-03-03"],
            ["AAPL"],
            [[101.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-03")
        assert price == 101.0

    def test_scan_forward_multiple_days(self):
        cal, cal_pos = _make_calendar([
            "2020-03-02", "2020-03-03", "2020-03-04",
            "2020-03-05", "2020-03-06", "2020-03-09",
        ])
        cm = self._matrix(
            ["2020-03-05"],
            ["AAPL"],
            [[104.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-05")
        assert price == 104.0

    def test_custom_max_forward(self):
        cal, cal_pos = _make_calendar([
            "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05",
        ])
        cm = self._matrix(
            ["2020-03-05"],
            ["AAPL"],
            [[104.0]],
        )
        # With max_forward_days=2, positions 0,1,2 are scanned
        # Position 3 (2020-03-05) is beyond 2 forward days
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
            max_forward_days=2,
        )
        assert pd.isna(actual_date)
        assert np.isnan(price)

    # ------------------------------------------------------------------
    # no valid quote
    # ------------------------------------------------------------------

    def test_no_valid_quote_in_window(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03", "2020-03-04"])
        cm = self._matrix(
            ["2020-03-09"],  # not in the window
            ["AAPL"],
            [[105.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert pd.isna(actual_date)
        assert np.isnan(price)

    def test_planned_date_not_in_calendar(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03"])
        cm = self._matrix(
            ["2020-03-02"],
            ["AAPL"],
            [[100.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-02-28"), cal, cal_pos,
        )
        assert pd.isna(actual_date)
        assert np.isnan(price)

    def test_ticker_not_in_matrix(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03"])
        cm = self._matrix(
            ["2020-03-02"],
            ["MSFT"],
            [[200.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert pd.isna(actual_date)
        assert np.isnan(price)

    def test_nan_price_skipped(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03", "2020-03-04"])
        cm = self._matrix(
            ["2020-03-02", "2020-03-03"],
            ["AAPL"],
            [[np.nan], [101.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-03")
        assert price == 101.0

    # ------------------------------------------------------------------
    # boundary
    # ------------------------------------------------------------------

    def test_at_end_of_calendar(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03"])
        cm = self._matrix(
            ["2020-03-03"],
            ["AAPL"],
            [[101.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-03"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-03")
        assert price == 101.0

    def test_beyond_end_of_calendar(self):
        cal, cal_pos = _make_calendar(["2020-03-02"])
        cm = self._matrix(
            ["2020-03-02"],
            ["AAPL"],
            [[100.0]],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
            max_forward_days=5,
        )
        # Reaches end of calendar after scanning all positions
        assert actual_date == pd.Timestamp("2020-03-02")
        assert price == 100.0

    def test_scan_stops_at_calendar_end(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03"])
        cm = self._matrix(
            [],  # empty matrix
            [],
            [],
        )
        actual_date, price = _next_valid_quote(
            cm, "AAPL", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert pd.isna(actual_date)
        assert np.isnan(price)

    def test_multiple_tickers_selects_correct_one(self):
        cal, cal_pos = _make_calendar(["2020-03-02", "2020-03-03"])
        cm = self._matrix(
            ["2020-03-02", "2020-03-03"],
            ["AAPL", "MSFT", "GOOG"],
            [
                [100.0, 200.0, np.nan],
                [101.0, np.nan, 300.0],
            ],
        )
        actual_date, price = _next_valid_quote(
            cm, "GOOG", pd.Timestamp("2020-03-02"), cal, cal_pos,
        )
        assert actual_date == pd.Timestamp("2020-03-03")
        assert price == 300.0
