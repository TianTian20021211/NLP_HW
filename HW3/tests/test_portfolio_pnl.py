"""Unit tests for the daily PnL state machine.

Tests 1-12 verify the gap-state semantics via _advance_daily_state (dict-based).
Tests 13-22 verify equivalence between dict-based and numpy vectorized
implementations (_advance_daily_state_numpy).
"""

from __future__ import annotations

import numpy as np
import pytest

from backtest.portfolio import (
    _NUMBA_AVAILABLE,
    _advance_daily_state,
    _advance_daily_state_numba_wrapper,
    _advance_daily_state_numpy,
)


# ---------------------------------------------------------------------------
# Test 1: empty weights produce zero record
# ---------------------------------------------------------------------------

def test_empty_weights():
    last_prices: dict[str, float] = {}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=[],
        weights={},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={},
        p_next={},
    )

    assert result["pnl"] == 0.0
    assert result["gross_exposure"] == 0.0
    assert result["net_exposure"] == 0.0
    assert result["n_positions"] == 0
    assert result["ffill_1d_weight"] == 0.0
    assert result["ffill_2d_weight"] == 0.0
    assert result["n_ffill_1d"] == 0
    assert result["n_ffill_2d"] == 0
    assert result["long_gap_tickers"] == {}


# ---------------------------------------------------------------------------
# Test 2: single ticker, valid both days, correct PnL
# ---------------------------------------------------------------------------

def test_single_ticker_valid_both_days():
    last_prices: dict[str, float] = {}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    # pnl = 0.5 * (110/100 - 1) = 0.05
    assert result["pnl"] == pytest.approx(0.05)
    assert result["gross_exposure"] == 0.5
    assert result["net_exposure"] == 0.5
    assert result["n_positions"] == 1
    assert result["ffill_1d_weight"] == 0.0
    assert result["ffill_2d_weight"] == 0.0
    assert result["n_ffill_1d"] == 0
    assert result["n_ffill_2d"] == 0
    assert result["long_gap_tickers"] == {}
    assert last_prices["AAPL"] == 110.0
    assert missing_streaks.get("AAPL", 0) == 0


# ---------------------------------------------------------------------------
# Test 3: single ticker, missing next price (1-day gap)
# ---------------------------------------------------------------------------

def test_single_ticker_missing_next_1d_gap():
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    assert result["pnl"] == 0.0
    assert result["ffill_1d_weight"] == 0.5
    assert result["n_ffill_1d"] == 1
    assert result["ffill_2d_weight"] == 0.0
    assert result["n_ffill_2d"] == 0
    assert result["long_gap_tickers"] == {}
    assert missing_streaks["AAPL"] == 1
    assert "AAPL" not in censored_tickers


# ---------------------------------------------------------------------------
# Test 4: ticker with two consecutive missing days (2-day gap)
# ---------------------------------------------------------------------------

def test_two_consecutive_missing_2d_gap():
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {"AAPL": 1}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    assert result["pnl"] == 0.0
    assert result["ffill_1d_weight"] == 0.0
    assert result["n_ffill_2d"] == 1
    assert result["ffill_2d_weight"] == 0.5
    assert result["long_gap_tickers"] == {}
    assert missing_streaks["AAPL"] == 2
    assert "AAPL" not in censored_tickers


# ---------------------------------------------------------------------------
# Test 5: ticker with three consecutive missing days (censored)
# ---------------------------------------------------------------------------

def test_three_consecutive_missing_censored():
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {"AAPL": 2}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    assert result["pnl"] == 0.0
    assert result["ffill_1d_weight"] == 0.0
    assert result["ffill_2d_weight"] == 0.0
    assert result["long_gap_tickers"] == {"AAPL": 0.5}
    assert "AAPL" in censored_tickers
    assert missing_streaks["AAPL"] == 3


# ---------------------------------------------------------------------------
# Test 6: already-censored ticker is skipped
# ---------------------------------------------------------------------------

def test_censored_ticker_skipped():
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = {"AAPL"}

    # Case A: censored ticker with valid next_px → no PnL contribution
    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    assert result["pnl"] == 0.0
    assert result["n_positions"] == 1  # still counted as a position

    # Case B: censored ticker with NaN next_px → long_gap flagged
    result2 = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={},
        censored_tickers={"AAPL"},
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    assert result2["pnl"] == 0.0
    assert result2["long_gap_tickers"] == {"AAPL": 0.5}


# ---------------------------------------------------------------------------
# Test 7: setdefault initializes last_prices on first valid today price
# ---------------------------------------------------------------------------

def test_setdefault_initializes_last_prices():
    last_prices: dict[str, float] = {}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    assert result["pnl"] == pytest.approx(0.05)
    # last_prices should have been setdefault'd to 100, then updated to 110
    assert last_prices["AAPL"] == 110.0


# ---------------------------------------------------------------------------
# Test 8: ticker without last_prices skips streak on missing next
# ---------------------------------------------------------------------------

def test_missing_without_last_price_skips_streak():
    last_prices: dict[str, float] = {}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    # today_px is NaN AND ticker not in last_prices → streak NOT incremented
    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    assert result["pnl"] == 0.0
    assert result["ffill_1d_weight"] == 0.0
    assert result["ffill_2d_weight"] == 0.0
    assert "AAPL" not in missing_streaks
    assert "AAPL" not in censored_tickers


# ---------------------------------------------------------------------------
# Test 9: multiple tickers, mixed valid/gap
# ---------------------------------------------------------------------------

def test_multiple_tickers_mixed():
    last_prices: dict[str, float] = {"A": 100.0, "B": 50.0, "C": 200.0}
    missing_streaks: dict[str, int] = {"C": 1}  # C already has 1-day gap
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["A", "B", "C"],
        weights={"A": 0.4, "B": 0.3, "C": 0.3},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"A": 105.0, "B": 52.0, "C": float("nan")},
        p_next={"A": 110.0, "B": float("nan"), "C": float("nan")},
    )

    # A: valid → pnl = 0.4 * (110/100 - 1) = 0.04
    assert result["pnl"] == pytest.approx(0.04)
    # B: 1d gap → ffill_1d_weight += 0.3
    assert result["ffill_1d_weight"] == 0.3
    assert result["n_ffill_1d"] == 1
    # C: pre-streak=1, missing again → streak=2 → ffill_2d
    assert result["ffill_2d_weight"] == 0.3
    assert result["n_ffill_2d"] == 1
    assert result["gross_exposure"] == 1.0  # 0.4 + 0.3 + 0.3
    assert result["net_exposure"] == 1.0
    assert result["n_positions"] == 3
    assert missing_streaks["B"] == 1
    assert missing_streaks["C"] == 2
    assert "B" not in censored_tickers
    assert "C" not in censored_tickers


# ---------------------------------------------------------------------------
# Test 10: gap recovery captures cumulative return
# ---------------------------------------------------------------------------

def test_gap_recovery_cumulative_return():
    """Two-day simulation: day 1 gap, day 2 recovery captures cumulative return."""
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    # Day 1: valid today, missing next → streak=1, no PnL
    result1 = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    assert result1["pnl"] == 0.0
    assert result1["ffill_1d_weight"] == 0.5
    assert missing_streaks["AAPL"] == 1
    # last_prices still holds the original base
    assert last_prices["AAPL"] == 100.0

    # Day 2: quote reappears at 105 → cumulative return booked
    result2 = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.5},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 105.0},
        p_next={"AAPL": 105.0},
    )

    # PnL = 0.5 * (105/100 - 1) = 0.025
    assert result2["pnl"] == pytest.approx(0.025)
    assert result2["ffill_1d_weight"] == 0.0
    assert missing_streaks["AAPL"] == 0
    assert last_prices["AAPL"] == 105.0


# ---------------------------------------------------------------------------
# Edge case: ticker with zero weight contributes nothing
# ---------------------------------------------------------------------------

def test_zero_weight_ticker():
    last_prices: dict[str, float] = {"AAPL": 100.0}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": 0.0},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    # PnL = 0.0 * (110/100 - 1) = 0.0
    assert result["pnl"] == 0.0
    # gross = |0.0| = 0.0
    assert result["gross_exposure"] == 0.0
    assert result["net_exposure"] == 0.0


# ---------------------------------------------------------------------------
# Edge case: short position (negative weight)
# ---------------------------------------------------------------------------

def test_short_position():
    last_prices: dict[str, float] = {}
    missing_streaks: dict[str, int] = {}
    censored_tickers: set[str] = set()

    result = _advance_daily_state(
        tickers=["AAPL"],
        weights={"AAPL": -0.3},
        last_prices=last_prices,
        missing_streaks=missing_streaks,
        censored_tickers=censored_tickers,
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 90.0},
    )

    # pnl = -0.3 * (90/100 - 1) = -0.3 * (-0.1) = 0.03
    assert result["pnl"] == pytest.approx(0.03)
    assert result["gross_exposure"] == 0.3
    assert result["net_exposure"] == -0.3


# ======================================================================
# Dual-implementation equivalence tests (Step 4)
# ======================================================================

TICKERS = ["AAPL", "MSFT", "GOOG"]
N = len(TICKERS)


def _make_arrays(
    weights: dict[str, float],
    last_prices: dict[str, float],
    missing_streaks: dict[str, int],
    censored_tickers: set[str],
    p_today: dict[str, float],
    p_next: dict[str, float],
    entry_dates: dict[str, int] | None = None,
):
    """Convert dict-based state to numpy arrays.

    Returns (arrays_dict, idx_to_ticker) where arrays_dict has keys matching
    _advance_daily_state_numpy parameters.
    """
    idx_to_ticker = list(TICKERS)
    ticker_to_idx = {t: i for i, t in enumerate(TICKERS)}

    weights_arr = np.zeros(N, dtype=np.float64)
    last_prices_arr = np.full(N, np.nan, dtype=np.float64)
    missing_streaks_arr = np.zeros(N, dtype=np.int32)
    is_censored_arr = np.zeros(N, dtype=bool)
    entry_dates_arr = np.full(N, -1, dtype=np.int64)

    for tkr, w in weights.items():
        idx = ticker_to_idx[tkr]
        weights_arr[idx] = w
    for tkr, lp in last_prices.items():
        idx = ticker_to_idx[tkr]
        last_prices_arr[idx] = lp
    for tkr, s in missing_streaks.items():
        idx = ticker_to_idx[tkr]
        missing_streaks_arr[idx] = s
    for tkr in censored_tickers:
        idx = ticker_to_idx[tkr]
        is_censored_arr[idx] = True
    if entry_dates:
        for tkr, ed in entry_dates.items():
            idx = ticker_to_idx[tkr]
            entry_dates_arr[idx] = ed
    else:
        # Default: all weighted tickers entered at day 0
        for tkr in weights:
            idx = ticker_to_idx[tkr]
            entry_dates_arr[idx] = 0

    # Build prices_2d (n_tickers × 2) for days 0 and 1
    prices_2d = np.full((N, 2), np.nan, dtype=np.float64)
    for tkr, pt in p_today.items():
        idx = ticker_to_idx[tkr]
        prices_2d[idx, 0] = pt
    for tkr, pn in p_next.items():
        idx = ticker_to_idx[tkr]
        prices_2d[idx, 1] = pn

    return {
        "weights_arr": weights_arr,
        "last_prices_arr": last_prices_arr,
        "missing_streaks_arr": missing_streaks_arr,
        "is_censored_arr": is_censored_arr,
        "entry_dates_arr": entry_dates_arr,
        "prices_2d": prices_2d,
        "d": 0,
        "d_next": 1,
        "idx_to_ticker": idx_to_ticker,
    }, idx_to_ticker, ticker_to_idx


def _dict_strip(result: dict) -> dict:
    """Return result dict with only the comparison-relevant keys."""
    return {
        "pnl": result["pnl"],
        "gross_exposure": result["gross_exposure"],
        "net_exposure": result["net_exposure"],
        "n_positions": result["n_positions"],
        "ffill_1d_weight": result["ffill_1d_weight"],
        "ffill_2d_weight": result["ffill_2d_weight"],
        "n_ffill_1d": result["n_ffill_1d"],
        "n_ffill_2d": result["n_ffill_2d"],
        "long_gap_tickers": result["long_gap_tickers"],
    }


# -------------------------------------------------------------------
# Test 13: empty weights → both implementations produce zeros
# -------------------------------------------------------------------

def test_dual_empty_weights():
    kwargs, _idx, _t2i = _make_arrays({}, {}, {}, set(), {}, {})

    dict_result = _advance_daily_state(
        tickers=[], weights={},
        last_prices={}, missing_streaks={}, censored_tickers=set(),
        p_today={}, p_next={},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)


# -------------------------------------------------------------------
# Test 14: single ticker, valid both days → identical PnL
# -------------------------------------------------------------------

def test_dual_valid_both_days():
    kwargs, idx_to_ticker, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={},
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    lp: dict[str, float] = {}
    ms: dict[str, int] = {}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    assert lp["AAPL"] == pytest.approx(kwargs["last_prices_arr"][ticker_to_idx["AAPL"]])
    assert ms.get("AAPL", 0) == kwargs["missing_streaks_arr"][ticker_to_idx["AAPL"]]


# -------------------------------------------------------------------
# Test 15: missing next price (1-day gap) → identical
# -------------------------------------------------------------------

def test_dual_missing_next_1d():
    kwargs, idx_to_ticker, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    lp = {"AAPL": 100.0}
    ms: dict[str, int] = {}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    assert ms["AAPL"] == kwargs["missing_streaks_arr"][ticker_to_idx["AAPL"]]
    assert ct == set()


# -------------------------------------------------------------------
# Test 16: two consecutive missing (2-day gap) → identical
# -------------------------------------------------------------------

def test_dual_two_day_gap():
    kwargs, idx_to_ticker, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={"AAPL": 1},
        censored_tickers=set(),
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    lp = {"AAPL": 100.0}
    ms = {"AAPL": 1}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": float("nan")}, p_next={"AAPL": float("nan")},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    assert ms["AAPL"] == kwargs["missing_streaks_arr"][ticker_to_idx["AAPL"]]
    assert "AAPL" not in ct


# -------------------------------------------------------------------
# Test 17: three consecutive missing (censored) → identical
# -------------------------------------------------------------------

def test_dual_three_day_censored():
    kwargs, idx_to_ticker, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={"AAPL": 2},
        censored_tickers=set(),
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    lp = {"AAPL": 100.0}
    ms = {"AAPL": 2}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": float("nan")}, p_next={"AAPL": float("nan")},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    assert "AAPL" in ct
    assert bool(kwargs["is_censored_arr"][ticker_to_idx["AAPL"]]) is True


# -------------------------------------------------------------------
# Test 18: censored ticker skipped → identical
# -------------------------------------------------------------------

def test_dual_censored_skipped():
    # Case A: censored with valid next → no PnL
    kwargs_a, _idx, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={},
        censored_tickers={"AAPL"},
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    lp_a = {"AAPL": 100.0}
    ms_a: dict[str, int] = {}
    ct_a: set[str] = {"AAPL"}
    dict_result_a = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp_a, missing_streaks=ms_a, censored_tickers=ct_a,
        p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
    )

    numpy_result_a = _advance_daily_state_numpy(**kwargs_a)
    assert _dict_strip(dict_result_a) == _dict_strip(numpy_result_a)

    # Case B: censored with NaN next → long_gap
    kwargs_b, _idx, _t2i = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={},
        censored_tickers={"AAPL"},
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    lp_b = {"AAPL": 100.0}
    ms_b: dict[str, int] = {}
    ct_b: set[str] = {"AAPL"}
    dict_result_b = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp_b, missing_streaks=ms_b, censored_tickers=ct_b,
        p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
    )

    numpy_result_b = _advance_daily_state_numpy(**kwargs_b)
    assert _dict_strip(dict_result_b) == _dict_strip(numpy_result_b)


# -------------------------------------------------------------------
# Test 19: setdefault initialization → identical
# -------------------------------------------------------------------

def test_dual_setdefault():
    kwargs, _idx, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={},
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": 100.0},
        p_next={"AAPL": 110.0},
    )

    lp: dict[str, float] = {}
    ms: dict[str, int] = {}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    assert lp["AAPL"] == pytest.approx(kwargs["last_prices_arr"][ticker_to_idx["AAPL"]])


# -------------------------------------------------------------------
# Test 20: no last_prices → streak skipped (both impls)
# -------------------------------------------------------------------

def test_dual_missing_no_last_price():
    kwargs, _idx, _t2i = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={},
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": float("nan")},
        p_next={"AAPL": float("nan")},
    )

    lp: dict[str, float] = {}
    ms: dict[str, int] = {}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": float("nan")}, p_next={"AAPL": float("nan")},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)


# -------------------------------------------------------------------
# Test 21: multiple tickers mixed → identical
# -------------------------------------------------------------------

def test_dual_multiple_mixed():
    kwargs, _idx, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3},
        last_prices={"AAPL": 100.0, "MSFT": 50.0, "GOOG": 200.0},
        missing_streaks={"GOOG": 1},
        censored_tickers=set(),
        p_today={"AAPL": 105.0, "MSFT": 52.0, "GOOG": float("nan")},
        p_next={"AAPL": 110.0, "MSFT": float("nan"), "GOOG": float("nan")},
    )

    lp = {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 200.0}
    ms = {"GOOG": 1}
    ct: set[str] = set()
    dict_result = _advance_daily_state(
        tickers=["AAPL", "MSFT", "GOOG"],
        weights={"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3},
        last_prices=lp, missing_streaks=ms, censored_tickers=ct,
        p_today={"AAPL": 105.0, "MSFT": 52.0, "GOOG": float("nan")},
        p_next={"AAPL": 110.0, "MSFT": float("nan"), "GOOG": float("nan")},
    )

    numpy_result = _advance_daily_state_numpy(**kwargs)

    assert _dict_strip(dict_result) == _dict_strip(numpy_result)
    for tkr in TICKERS:
        assert ms.get(tkr, 0) == int(kwargs["missing_streaks_arr"][ticker_to_idx[tkr]])


# -------------------------------------------------------------------
# Test 22: gap recovery → identical cumulative return
# -------------------------------------------------------------------

def test_dual_gap_recovery():
    # Day 1: gap
    kwargs1, _idx, ticker_to_idx = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={"AAPL": 100.0},
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": 100.0},
        p_next={"AAPL": float("nan")},
    )

    lp1 = {"AAPL": 100.0}
    ms1: dict[str, int] = {}
    ct1: set[str] = set()
    dict_result1 = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp1, missing_streaks=ms1, censored_tickers=ct1,
        p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
    )

    numpy_result1 = _advance_daily_state_numpy(**kwargs1)
    assert _dict_strip(dict_result1) == _dict_strip(numpy_result1)

    # Day 2: recovery (use the mutated arrays from day 1)
    kwargs2, _idx, _t2i = _make_arrays(
        weights={"AAPL": 0.5},
        last_prices={},  # will use mutated arrays
        missing_streaks={},
        censored_tickers=set(),
        p_today={"AAPL": 105.0},
        p_next={"AAPL": 105.0},
    )
    # Carry over mutated state from day 1 numpy run
    kwargs2["last_prices_arr"] = kwargs1["last_prices_arr"]
    kwargs2["missing_streaks_arr"] = kwargs1["missing_streaks_arr"]
    kwargs2["is_censored_arr"] = kwargs1["is_censored_arr"]

    dict_result2 = _advance_daily_state(
        tickers=["AAPL"], weights={"AAPL": 0.5},
        last_prices=lp1, missing_streaks=ms1, censored_tickers=ct1,
        p_today={"AAPL": 105.0}, p_next={"AAPL": 105.0},
    )

    numpy_result2 = _advance_daily_state_numpy(**kwargs2)
    assert _dict_strip(dict_result2) == _dict_strip(numpy_result2)
    assert dict_result2["pnl"] == pytest.approx(0.025)
    assert numpy_result2["pnl"] == pytest.approx(0.025)


# ======================================================================
# Numba-specific equivalence tests (Step 5)
# ======================================================================

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestNumbaEquivalence:
    """Run each scenario through the numba kernel and compare to dict baseline."""

    def test_numba_empty_weights(self):
        kwargs, _idx, _t2i = _make_arrays({}, {}, {}, set(), {}, {})

        dict_result = _advance_daily_state(
            tickers=[], weights={},
            last_prices={}, missing_streaks={}, censored_tickers=set(),
            p_today={}, p_next={},
        )

        numba_result = _advance_daily_state_numba_wrapper(**kwargs)
        assert _dict_strip(dict_result) == _dict_strip(numba_result)

    def test_numba_valid_both_days(self):
        kwargs, _idx, ticker_to_idx = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={}, missing_streaks={},
            censored_tickers=set(),
            p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
        )

        lp: dict[str, float] = {}
        ms: dict[str, int] = {}
        ct: set[str] = set()
        dict_result = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp, missing_streaks=ms, censored_tickers=ct,
            p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
        )

        numba_result = _advance_daily_state_numba_wrapper(**kwargs)
        assert _dict_strip(dict_result) == _dict_strip(numba_result)
        assert lp["AAPL"] == pytest.approx(kwargs["last_prices_arr"][ticker_to_idx["AAPL"]])

    def test_numba_missing_1d(self):
        kwargs, _idx, ticker_to_idx = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={"AAPL": 100.0},
            missing_streaks={}, censored_tickers=set(),
            p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
        )

        lp = {"AAPL": 100.0}
        ms: dict[str, int] = {}
        ct: set[str] = set()
        dict_result = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp, missing_streaks=ms, censored_tickers=ct,
            p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
        )

        numba_result = _advance_daily_state_numba_wrapper(**kwargs)
        assert _dict_strip(dict_result) == _dict_strip(numba_result)
        assert ms["AAPL"] == int(kwargs["missing_streaks_arr"][ticker_to_idx["AAPL"]])

    def test_numba_three_day_censored(self):
        kwargs, _idx, ticker_to_idx = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={"AAPL": 100.0},
            missing_streaks={"AAPL": 2}, censored_tickers=set(),
            p_today={"AAPL": float("nan")}, p_next={"AAPL": float("nan")},
        )

        lp = {"AAPL": 100.0}
        ms = {"AAPL": 2}
        ct: set[str] = set()
        dict_result = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp, missing_streaks=ms, censored_tickers=ct,
            p_today={"AAPL": float("nan")}, p_next={"AAPL": float("nan")},
        )

        numba_result = _advance_daily_state_numba_wrapper(**kwargs)
        assert _dict_strip(dict_result) == _dict_strip(numba_result)
        assert "AAPL" in ct
        assert bool(kwargs["is_censored_arr"][ticker_to_idx["AAPL"]]) is True

    def test_numba_censored_skipped(self):
        kwargs_a, _idx, _t2i = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={"AAPL": 100.0},
            missing_streaks={}, censored_tickers={"AAPL"},
            p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
        )

        lp_a = {"AAPL": 100.0}
        ms_a: dict[str, int] = {}
        ct_a: set[str] = {"AAPL"}
        dict_result_a = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp_a, missing_streaks=ms_a, censored_tickers=ct_a,
            p_today={"AAPL": 100.0}, p_next={"AAPL": 110.0},
        )

        numba_result_a = _advance_daily_state_numba_wrapper(**kwargs_a)
        assert _dict_strip(dict_result_a) == _dict_strip(numba_result_a)

    def test_numba_multiple_mixed(self):
        kwargs, _idx, ticker_to_idx = _make_arrays(
            weights={"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3},
            last_prices={"AAPL": 100.0, "MSFT": 50.0, "GOOG": 200.0},
            missing_streaks={"GOOG": 1}, censored_tickers=set(),
            p_today={"AAPL": 105.0, "MSFT": 52.0, "GOOG": float("nan")},
            p_next={"AAPL": 110.0, "MSFT": float("nan"), "GOOG": float("nan")},
        )

        lp = {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 200.0}
        ms = {"GOOG": 1}
        ct: set[str] = set()
        dict_result = _advance_daily_state(
            tickers=["AAPL", "MSFT", "GOOG"],
            weights={"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3},
            last_prices=lp, missing_streaks=ms, censored_tickers=ct,
            p_today={"AAPL": 105.0, "MSFT": 52.0, "GOOG": float("nan")},
            p_next={"AAPL": 110.0, "MSFT": float("nan"), "GOOG": float("nan")},
        )

        numba_result = _advance_daily_state_numba_wrapper(**kwargs)
        assert _dict_strip(dict_result) == _dict_strip(numba_result)
        for tkr in TICKERS:
            assert ms.get(tkr, 0) == int(kwargs["missing_streaks_arr"][ticker_to_idx[tkr]])

    def test_numba_gap_recovery(self):
        kwargs1, _idx, _t2i = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={"AAPL": 100.0},
            missing_streaks={}, censored_tickers=set(),
            p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
        )

        lp1 = {"AAPL": 100.0}
        ms1: dict[str, int] = {}
        ct1: set[str] = set()
        dict_result1 = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp1, missing_streaks=ms1, censored_tickers=ct1,
            p_today={"AAPL": 100.0}, p_next={"AAPL": float("nan")},
        )

        numba_result1 = _advance_daily_state_numba_wrapper(**kwargs1)
        assert _dict_strip(dict_result1) == _dict_strip(numba_result1)

        kwargs2, _idx, _t2i = _make_arrays(
            weights={"AAPL": 0.5}, last_prices={}, missing_streaks={},
            censored_tickers=set(),
            p_today={"AAPL": 105.0}, p_next={"AAPL": 105.0},
        )
        kwargs2["last_prices_arr"] = kwargs1["last_prices_arr"]
        kwargs2["missing_streaks_arr"] = kwargs1["missing_streaks_arr"]
        kwargs2["is_censored_arr"] = kwargs1["is_censored_arr"]

        dict_result2 = _advance_daily_state(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            last_prices=lp1, missing_streaks=ms1, censored_tickers=ct1,
            p_today={"AAPL": 105.0}, p_next={"AAPL": 105.0},
        )

        numba_result2 = _advance_daily_state_numba_wrapper(**kwargs2)
        assert _dict_strip(dict_result2) == _dict_strip(numba_result2)
        assert dict_result2["pnl"] == pytest.approx(0.025)
        assert numba_result2["pnl"] == pytest.approx(0.025)
