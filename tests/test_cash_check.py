"""Pure helpers of the cash check; the detectors themselves are never asserted."""

import pytest

from src.features.cash_check import in_window, pick_frame_indices, summarize


def test_pick_frame_indices_spreads_evenly():
    assert pick_frame_indices(5, 12) == [0, 1, 2, 3, 4]
    assert pick_frame_indices(100, 3) == [0, 49, 99]


def test_in_window_is_inclusive():
    assert in_window([0.0, 1.0, 2.0, 5.0, 8.0], 5, half=3) == [2, 3, 4]


def test_summarize_counts_frames_over_thresholds():
    row = summarize("X-1", 22, person=[0.9, 0.4, 0.6, 0.1], cash=[0.0, 0.2, 0.05, 0.0])
    assert row.cashier_frac == pytest.approx(0.5)
    assert row.cash_frac == pytest.approx(0.25)
    assert (row.cashier_conf, row.cash_conf) == (0.9, 0.2)
    assert row.cashier_found and row.cash_found


def test_summarize_empty_window_finds_nothing():
    row = summarize("X-1", 22, [], [])
    assert not row.cashier_found and not row.cash_found
