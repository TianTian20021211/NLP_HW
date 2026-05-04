"""Unit tests for backtest/_stats.py (Layer 1, TDD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest._stats import (
    bucket_returns,
    build_equity_curves,
    dedup_latest_per_ticker,
    make_median_imputer,
    max_drawdown_from_equity,
    portfolio_stats,
    spearman,
)


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------

class TestSpearman:
    def test_spearman_perfect_positive(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert spearman(a, b) == pytest.approx(1.0)

    def test_spearman_perfect_negative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert spearman(a, b) == pytest.approx(-1.0)

    def test_spearman_with_ties(self):
        a = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 4.0, 4.0, 3.0, 1.0])
        result = spearman(a, b)
        assert not np.isnan(result)

    def test_spearman_pairwise_nan_compatibility(self):
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, np.nan, 1.0])
        # After pairwise NaN drop: a=[1,3,5], b=[5,3,1] -> Spearman = -1.0
        assert spearman(a, b) == pytest.approx(-1.0)

    def test_spearman_insufficient_pairs_nan(self):
        a = np.array([1.0, np.nan])
        b = np.array([np.nan, 2.0])
        result = spearman(a, b)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# make_median_imputer
# ---------------------------------------------------------------------------

class TestMakeMedianImputer:
    def test_make_median_imputer_returns_simple_imputer(self):
        from sklearn.impute import SimpleImputer

        imp = make_median_imputer()
        assert isinstance(imp, SimpleImputer)
        assert imp.strategy == "median"


# ---------------------------------------------------------------------------
# dedup_latest_per_ticker
# ---------------------------------------------------------------------------

class TestDedupLatestPerTicker:
    def test_dedup_latest_per_ticker(self):
        # Basic dedup
        df = pd.DataFrame({
            "call_entry_date": pd.to_datetime(
                ["2020-01-03", "2020-01-01", "2020-01-02"],
            ),
            "BESTTICKER": ["AAPL", "AAPL", "AAPL"],
            "value": [30, 10, 20],
        })
        result = dedup_latest_per_ticker(df)
        assert len(result) == 1
        assert result["value"].iloc[0] == 30

        # Custom column names
        df2 = pd.DataFrame({
            "date": pd.to_datetime(["2020-02-01", "2020-02-02"]),
            "ticker": ["MSFT", "MSFT"],
            "value": [10, 20],
        })
        result2 = dedup_latest_per_ticker(
            df2, date_col="date", ticker_col="ticker",
        )
        assert len(result2) == 1
        assert result2["value"].iloc[0] == 20

    def test_dedup_latest_per_ticker_empty(self):
        df = pd.DataFrame(columns=["call_entry_date", "BESTTICKER", "value"])
        result = dedup_latest_per_ticker(df)
        assert result.empty


# ---------------------------------------------------------------------------
# bucket_returns
# ---------------------------------------------------------------------------

class TestBucketReturns:
    def test_bucket_returns_empty_input(self):
        df = pd.DataFrame(
            columns=["feature", "return", "year_month",
                     "call_entry_date", "BESTTICKER"],
        )
        result = bucket_returns(df, "feature", "return", n_buckets=5)
        assert list(result.columns) == ["year_month", "bucket", "ret", "n"]
        assert result.empty

    def test_bucket_returns_basic(self):
        months = pd.period_range("2020-01", periods=2, freq="M")
        rows = []
        for i, m in enumerate(months):
            # 18 unique tickers per month, features 0..17, returns 0.0..0.17
            for j in range(18):
                rows.append({
                    "year_month": m,
                    "call_entry_date": pd.Timestamp(
                        f"2020-{i+1:02d}-{(j % 28) + 1:02d}",
                    ),
                    "BESTTICKER": f"T{i:02d}_{j:02d}",
                    "feature": float(j),
                    "return": float(j * 0.01),
                })
        df = pd.DataFrame(rows)
        result = bucket_returns(df, "feature", "return", n_buckets=3)
        assert list(result.columns) == ["year_month", "bucket", "ret", "n"]
        assert len(result) == 6  # 2 months x 3 buckets
        assert (result["n"] >= 5).all()

        # qcut with 18 values and q=3:
        #   bucket 0: [0..5],  bucket 1: [6..11],  bucket 2: [12..17]
        b0 = result[result["bucket"] == 0]["ret"].iloc[0]
        b1 = result[result["bucket"] == 1]["ret"].iloc[0]
        b2 = result[result["bucket"] == 2]["ret"].iloc[0]
        # mean return per bucket: 0.025, 0.085, 0.145
        assert b0 == pytest.approx(0.025, abs=0.001)
        assert b1 == pytest.approx(0.085, abs=0.001)
        assert b2 == pytest.approx(0.145, abs=0.001)

    def test_bucket_returns_min_samples(self):
        rows = []
        # Month 1: only 5 tickers (< 3*5=15, should be skipped)
        for j in range(5):
            rows.append({
                "year_month": pd.Period("2020-01", freq="M"),
                "call_entry_date": pd.Timestamp(f"2020-01-{j+1:02d}"),
                "BESTTICKER": f"T{j:02d}",
                "feature": float(j),
                "return": float(j) * 0.01,
            })
        # Month 2: 18 tickers (>= 15, should be included)
        for j in range(18):
            rows.append({
                "year_month": pd.Period("2020-02", freq="M"),
                "call_entry_date": pd.Timestamp(f"2020-02-{j+1:02d}"),
                "BESTTICKER": f"U{j:02d}",
                "feature": float(j % 6),
                "return": float(j) * 0.01,
            })
        df = pd.DataFrame(rows)
        result = bucket_returns(
            df, "feature", "return", n_buckets=3, min_samples_per_bucket=5,
        )
        assert not result.empty
        assert result["year_month"].nunique() == 1
        assert result["year_month"].iloc[0] == pd.Period("2020-02", freq="M")

    def test_bucket_returns_qcut_duplicate_edges(self):
        rows = []
        for j in range(30):
            rows.append({
                "year_month": pd.Period("2020-01", freq="M"),
                "call_entry_date": pd.Timestamp(f"2020-01-{(j % 25) + 1:02d}"),
                "BESTTICKER": f"T{(j % 25):02d}",
                "feature": 1.0 if j < 15 else 2.0,
                "return": 0.01 * j,
            })
        df = pd.DataFrame(rows)
        result = bucket_returns(df, "feature", "return", n_buckets=5)
        assert not result.empty
        assert result["bucket"].nunique() < 5


# ---------------------------------------------------------------------------
# build_equity_curves
# ---------------------------------------------------------------------------

class TestBuildEquityCurves:
    def test_build_equity_curves_full_bucket_columns(self):
        data = {
            "year_month": pd.period_range(
                "2020-01", periods=6, freq="M",
            ).repeat(5),
            "bucket": list(range(5)) * 6,
            "ret": [0.01, 0.02, 0.03, 0.04, 0.05] * 6,
            "n": [10] * 30,
        }
        bucket_ret = pd.DataFrame(data)
        eq = build_equity_curves(bucket_ret, n_buckets=5)
        for i in range(5):
            assert f"bucket_{i}" in eq.columns
            assert f"cum_bucket_{i}" in eq.columns
        assert "long_only" in eq.columns
        assert "short_only" in eq.columns
        assert "long_short" in eq.columns
        assert "cum_long_only" in eq.columns
        assert "cum_short_only" in eq.columns
        assert "cum_long_short" in eq.columns

    def test_build_equity_curves_sparse_bucket_columns(self):
        # Only buckets 0, 2, 4 observed (qcut dropped 1 and 3)
        data = {
            "year_month": pd.period_range(
                "2020-01", periods=6, freq="M",
            ).repeat(3),
            "bucket": [0, 2, 4] * 6,
            "ret": [0.01, 0.03, 0.05] * 6,
            "n": [10] * 18,
        }
        bucket_ret = pd.DataFrame(data)
        eq = build_equity_curves(bucket_ret, n_buckets=5)
        for i in [0, 2, 4]:
            assert f"bucket_{i}" in eq.columns
            assert f"cum_bucket_{i}" in eq.columns
        for i in [1, 3]:
            assert f"bucket_{i}" not in eq.columns
        # top=bucket_4, bot=bucket_0 both present
        assert "long_only" in eq.columns
        assert "short_only" in eq.columns
        assert "long_short" in eq.columns

    def test_build_equity_curves_cumulative(self):
        data = {
            "year_month": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03"], freq="M",
            ),
            "bucket": [0, 0, 0],
            "ret": [0.01, 0.02, -0.01],
            "n": [10, 10, 10],
        }
        bucket_ret = pd.DataFrame(data)
        eq = build_equity_curves(bucket_ret, n_buckets=3)
        expected = pd.Series(
            [1.01, 1.0302, 1.019898], name="cum_bucket_0",
        )
        pd.testing.assert_series_equal(
            eq["cum_bucket_0"], expected, check_names=False, check_index=False,
        )

    def test_build_equity_curves_long_short(self):
        data = {
            "year_month": pd.period_range(
                "2020-01", periods=3, freq="M",
            ).repeat(5),
            "bucket": list(range(5)) * 3,
            "ret": [0.01, 0.02, 0.03, 0.04, 0.05] * 3,
            "n": [10] * 15,
        }
        bucket_ret = pd.DataFrame(data)
        eq = build_equity_curves(bucket_ret, n_buckets=5)
        pd.testing.assert_series_equal(
            eq["long_only"] + eq["short_only"],
            eq["long_short"],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# max_drawdown_from_equity
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_max_drawdown_from_equity_basic(self):
        cum = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 85.0])
        # Peak = 110, trough = 80, dd = (80-110)/110 = -0.2727...
        assert max_drawdown_from_equity(cum) == pytest.approx(
            -0.2727272727, rel=1e-5,
        )

    def test_max_drawdown_from_equity_empty(self):
        cum = pd.Series([], dtype=float)
        result = max_drawdown_from_equity(cum)
        assert np.isnan(result)

    def test_max_drawdown_from_equity_no_drawdown(self):
        cum = pd.Series([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown_from_equity(cum) == 0.0


# ---------------------------------------------------------------------------
# portfolio_stats
# ---------------------------------------------------------------------------

class TestPortfolioStats:
    def test_portfolio_stats_long_short(self):
        eq = pd.DataFrame({
            "year_month": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03"], freq="M",
            ),
            "long_only": [0.01, 0.02, -0.01],
            "short_only": [0.005, 0.01, 0.015],
            "long_short": [0.015, 0.03, 0.005],
            "cum_long_only": [1.01, 1.0302, 1.019898],
            "cum_short_only": [1.005, 1.01505, 1.03027575],
            "cum_long_short": [1.015, 1.04545, 1.05067725],
        })
        stats = portfolio_stats(eq, leg="long_short")
        expected_keys = [
            "ann_return", "ann_vol", "sharpe",
            "rolling_sharpe_12m", "max_drawdown", "n_months",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
        assert stats["n_months"] == 3

    def test_portfolio_stats_count_key(self):
        eq = pd.DataFrame({
            "year_month": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03"], freq="M",
            ),
            "long_short": [0.01, 0.02, -0.01],
            "cum_long_short": [1.01, 1.0302, 1.019898],
        })
        stats = portfolio_stats(eq, leg="long_short")
        assert "n_months" in stats
        assert stats["n_months"] == 3

        # Custom count_key
        stats2 = portfolio_stats(eq, leg="long_short", count_key="n_periods")
        assert "n_periods" in stats2
        assert "n_months" not in stats2

    def test_portfolio_stats_empty_equity(self):
        eq = pd.DataFrame()
        assert portfolio_stats(eq, leg="long_short") == {}

    def test_portfolio_stats_missing_leg(self):
        eq = pd.DataFrame({
            "year_month": pd.PeriodIndex(
                ["2020-01", "2020-02"], freq="M",
            ),
            "long_only": [0.01, 0.02],
            "cum_long_only": [1.01, 1.0302],
        })
        assert portfolio_stats(eq, leg="long_short") == {}

    def test_portfolio_stats_one_row_equity(self):
        eq = pd.DataFrame({
            "year_month": pd.PeriodIndex(["2020-01"], freq="M"),
            "long_short": [0.01],
            "cum_long_short": [1.01],
        })
        assert portfolio_stats(eq, leg="long_short") == {}

    def test_portfolio_stats_zero_vol_sharpe_nan(self):
        eq = pd.DataFrame({
            "year_month": pd.PeriodIndex(
                ["2020-01", "2020-02", "2020-03"], freq="M",
            ),
            "long_short": [0.01, 0.01, 0.01],
            "cum_long_short": [1.01, 1.0201, 1.030301],
        })
        stats = portfolio_stats(eq, leg="long_short")
        assert np.isnan(stats["sharpe"])
        assert "ann_return" in stats
        assert "ann_vol" in stats
        assert "max_drawdown" in stats
        assert "n_months" in stats
        assert stats["n_months"] == 3
