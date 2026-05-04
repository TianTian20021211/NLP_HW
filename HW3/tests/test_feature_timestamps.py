"""Unit tests for timestamp logic: entry_rule and compute_timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineer import entry_rule, compute_timestamps


# ---------------------------------------------------------------------------
# entry_rule
# ---------------------------------------------------------------------------

class TestEntryRule:
    """BMO (< 13 UTC) → same business day; AMC (>= 13 UTC) → next business day."""

    def test_bmo_same_business_day(self):
        ts = pd.Series(["2020-03-02 12:59:00+00:00"])  # Monday BMO
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-02")
        assert result.iloc[0] == expected

    def test_amc_next_business_day(self):
        ts = pd.Series(["2020-03-02 13:00:00+00:00"])  # Monday AMC
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-03")
        assert result.iloc[0] == expected

    def test_amc_after_market_close(self):
        ts = pd.Series(["2020-03-02 16:00:00+00:00"])  # >= 16 UTC, definite AMC
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-03")
        assert result.iloc[0] == expected

    def test_amc_late_evening(self):
        ts = pd.Series(["2020-03-02 22:30:00+00:00"])  # AMC, cross-day
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-03")
        assert result.iloc[0] == expected

    def test_amc_cross_day_bmo_next(self):
        ts = pd.Series(["2020-03-15 22:30:00+00:00"])  # Sunday AMC
        # day = Sunday → +1D = Monday → busday_offset(Mon, roll=forward) = Monday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-16")
        assert result.iloc[0] == expected

    def test_friday_amc_monday(self):
        ts = pd.Series(["2020-03-06 16:00:00+00:00"])  # Friday AMC → Monday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-09")
        assert result.iloc[0] == expected

    def test_friday_bmo_same_day(self):
        ts = pd.Series(["2020-03-06 12:00:00+00:00"])  # Friday BMO → Friday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-06")
        assert result.iloc[0] == expected

    def test_saturday_bmo_monday(self):
        ts = pd.Series(["2020-03-07 12:00:00+00:00"])  # Saturday BMO → Monday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-09")
        assert result.iloc[0] == expected

    def test_saturday_amc_monday(self):
        ts = pd.Series(["2020-03-07 16:00:00+00:00"])  # Saturday AMC → Monday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-09")
        assert result.iloc[0] == expected

    def test_sunday_amc_monday(self):
        ts = pd.Series(["2020-03-08 14:00:00+00:00"])  # Sunday AMC → Monday
        result = entry_rule(ts)
        expected = pd.Timestamp("2020-03-09")
        assert result.iloc[0] == expected

    def test_nat_input(self):
        ts = pd.Series([pd.NaT])
        result = entry_rule(ts)
        assert pd.isna(result.iloc[0])

    def test_all_nat(self):
        ts = pd.Series([pd.NaT, pd.NaT])
        result = entry_rule(ts)
        assert result.isna().all()

    def test_series_output_type(self):
        ts = pd.Series(["2020-01-15 12:00:00+00:00"])
        result = entry_rule(ts)
        assert isinstance(result, pd.Series)
        assert result.dtype == "datetime64[ns]"

    def test_bmo_just_before_cutoff(self):
        ts = pd.Series(["2020-03-02 12:59:00+00:00"])
        result = entry_rule(ts)
        assert result.iloc[0] == pd.Timestamp("2020-03-02")

    def test_amc_at_cutoff(self):
        ts = pd.Series(["2020-03-02 13:00:00+00:00"])
        result = entry_rule(ts)
        assert result.iloc[0] == pd.Timestamp("2020-03-03")


# ---------------------------------------------------------------------------
# compute_timestamps
# ---------------------------------------------------------------------------

class TestComputeTimestamps:
    """Unified availability_date rule: pre-launch call+2bd, post-launch max."""

    PRE_CUTOFF = "2020-01-15"       # before 2023-07-06
    POST_CUTOFF = "2024-01-15"      # on/after 2023-07-06

    def _make_df(self, call_utc, ingest_utc):
        return pd.DataFrame({
            "MOSTIMPORTANTDATEUTC": [call_utc],
            "INGESTDATEUTC": [ingest_utc],
        })

    def test_pre_launch_availability_call_plus_2bd(self):
        df = self._make_df("2020-01-13 12:00:00+00:00", pd.NaT)
        result = compute_timestamps(df)
        # Monday BMO → call_entry = Monday; +2bd = Wednesday
        assert result["availability_date"].iloc[0] == pd.Timestamp("2020-01-15")

    def test_pre_launch_availability_friday_amc(self):
        df = self._make_df("2020-01-10 16:00:00+00:00", pd.NaT)
        result = compute_timestamps(df)
        # Friday AMC → call_entry = Monday; +2bd = Wednesday
        assert result["availability_date"].iloc[0] == pd.Timestamp("2020-01-15")

    def test_post_launch_call_equals_ingest(self):
        df = self._make_df(
            f"{self.POST_CUTOFF} 12:00:00+00:00",
            f"{self.POST_CUTOFF} 12:00:00+00:00",
        )
        result = compute_timestamps(df)
        assert result["availability_date"].iloc[0] == pd.Timestamp(self.POST_CUTOFF)

    def test_post_launch_ingest_later(self):
        df = self._make_df(
            f"{self.POST_CUTOFF} 12:00:00+00:00",
            f"{self.POST_CUTOFF}T15:00:00+00:00",  # AMC → next day
        )
        result = compute_timestamps(df)
        assert result["availability_date"].iloc[0] == pd.Timestamp("2024-01-16")

    def test_post_launch_call_later(self):
        df = self._make_df(
            f"{self.POST_CUTOFF} 15:00:00+00:00",  # AMC → next day
            f"{self.POST_CUTOFF} 12:00:00+00:00",
        )
        result = compute_timestamps(df)
        assert result["availability_date"].iloc[0] == pd.Timestamp("2024-01-16")

    def test_post_launch_ingest_nat_falls_back_to_call(self):
        df = self._make_df(f"{self.POST_CUTOFF} 12:00:00+00:00", pd.NaT)
        result = compute_timestamps(df)
        # pandas max skips NaT, so max(NaT, call) = call
        assert result["availability_date"].iloc[0] == pd.Timestamp(self.POST_CUTOFF)

    def test_call_nat_everywhere(self):
        df = self._make_df(pd.NaT, "2020-06-01 12:00:00+00:00")
        result = compute_timestamps(df)
        assert pd.isna(result["call_entry_date"].iloc[0])
        assert pd.isna(result["availability_date"].iloc[0])

    def test_output_columns(self):
        df = self._make_df("2020-06-01 12:00:00+00:00", pd.NaT)
        result = compute_timestamps(df)
        for col in ["call_entry_date", "ingest_entry_date", "availability_date"]:
            assert col in result.columns

    def test_cutoff_boundary_before(self):
        df = self._make_df("2023-07-05 12:00:00+00:00", pd.NaT)
        result = compute_timestamps(df)
        # Still pre-launch: call_entry + 2bd
        call_entry = result["call_entry_date"].iloc[0]
        expected_avail = pd.Timestamp("2023-07-07")  # Wed +2bd = Fri
        assert result["availability_date"].iloc[0] == expected_avail

    def test_cutoff_boundary_on(self):
        df = self._make_df(
            "2023-07-06 12:00:00+00:00",
            "2023-07-06 12:00:00+00:00",
        )
        result = compute_timestamps(df)
        # Post-launch (>= cutoff): max(call, ingest)
        assert result["availability_date"].iloc[0] == pd.Timestamp("2023-07-06")

    def test_availability_never_before_call_entry(self):
        # Pre-launch: avail = call + 2bd >= call; post-launch: max >= call
        df = pd.DataFrame({
            "MOSTIMPORTANTDATEUTC": [
                "2020-06-01 12:00:00+00:00",
                "2024-06-01 12:00:00+00:00",
                "2024-06-01 15:00:00+00:00",
            ],
            "INGESTDATEUTC": [
                pd.NaT,
                "2024-06-01 12:00:00+00:00",
                "2024-06-02 12:00:00+00:00",
            ],
        })
        result = compute_timestamps(df)
        assert (result["availability_date"] >= result["call_entry_date"]).all()
