"""Window labelling, leakage guard, threshold and metrics against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from src.models.windows import (
    NEGATIVE,
    POSITIVE,
    UNLABELED,
    choose_threshold,
    group_of,
    labels_by_video,
    logit_shift,
    lovo_folds,
    make_windows,
    shift_threshold,
    timecode,
    video_metrics,
    video_score,
)


def _label_at(w: pd.DataFrame, start: float) -> int:
    return int(w.loc[w["start"] == start, "label"].iloc[0])


def test_window_grid():
    w = make_windows(10.0)
    assert len(w) == 17  # starts 0 .. 8 step 0.5
    assert w["end"].iloc[-1] == 10.0
    assert len(make_windows(1.9)) == 0


def test_clean_video_is_negative_everywhere():
    assert set(make_windows(20.0)["label"]) == {NEGATIVE}


def test_zones_around_one_label():
    # N = 20: positive = start in (17, 24) -> 17.5 .. 23.5; negative = end < 17 -> start <= 14.5.
    w = make_windows(40.0, [20])
    assert _label_at(w, 14.5) == NEGATIVE
    assert _label_at(w, 15.0) == UNLABELED  # ends exactly at N-3: not "before"
    assert _label_at(w, 17.0) == UNLABELED  # [17, 19] only touches the zone at 19
    assert _label_at(w, 17.5) == POSITIVE
    assert _label_at(w, 23.5) == POSITIVE
    assert _label_at(w, 24.0) == UNLABELED  # after N+4: lasting state, no label
    assert _label_at(w, 38.0) == UNLABELED


def test_zones_with_two_labels():
    # N = 10 and 30: positive starts 7.5..13.5 and 27.5..33.5; negatives only before the first label (start <= 4.5).
    w = make_windows(50.0, [10, 30])
    pos = w.loc[w["label"] == POSITIVE, "start"]
    assert list(pos) == [*np.arange(7.5, 14.0, 0.5), *np.arange(27.5, 34.0, 0.5)]
    neg = w.loc[w["label"] == NEGATIVE, "start"]
    assert neg.max() == 4.5 and len(neg) == 10
    assert _label_at(w, 24.0) == UNLABELED  # ends before 30-3 but sits after the first label
    assert _label_at(w, 14.0) == UNLABELED  # after N1+4
    assert _label_at(w, 27.0) == UNLABELED  # [27, 29] only touches 29 = N2-1


def test_labels_by_video():
    df = pd.DataFrame(
        {"video_id": ["X-1", "X-3", "X-4", "X-4"], "second": pd.array([22, None, 57, 68], dtype="Int64")}
    )
    assert labels_by_video(df) == {"X-1": [22], "X-3": [], "X-4": [57, 68]}


def test_lovo_never_shares_a_group_and_merges_duplicates():
    ids = ["X-1", "X-4", "X-47", "X-4", "X-5", "X-48", "X-2"]
    folds = list(lovo_folds(ids))
    assert len(folds) == 4  # X-1, X-2, X-4(+X-47), X-5(+X-48)
    for train, val in folds:
        assert not set(train) & set(val)
        assert len(train) + len(val) == len(ids)
    val_sets = [set(val) for _, val in folds]
    assert {1, 2, 3} in val_sets and {4, 5} in val_sets


def test_threshold_recall_first_then_fewest_fp():
    scores = [0.9, 0.4, 0.7, 0.5, 0.3, 0.8]
    is_violation = [True, True, True, False, False, False]
    thr = choose_threshold(scores, is_violation)
    assert thr == 0.4  # lowest violation score keeps recall at 1.0
    m = video_metrics(scores, is_violation, thr, [None] * 6, [[]] * 6)
    assert m["video_recall"] == 1.0
    assert m["fp_rate"] == pytest.approx(2 / 3)  # 0.5 and 0.8 flagged


def test_threshold_needs_a_violation_video():
    with pytest.raises(ValueError):
        choose_threshold([0.1, 0.2], [False, False])


def test_timecode_onset_and_peak():
    starts = [0.0, 0.5, 1.0, 1.5, 2.0]
    scores = [0.1, 0.6, 0.9, 0.7, 0.2]
    assert timecode(starts, scores, 0.5, "onset") == 2.5  # window [0.5, 2.5] ends at 2.5
    assert timecode(starts, scores, 0.5, "peak") == 2.0  # start 1.0 + half a window
    assert timecode(starts, scores, 0.95) is None


def test_metrics_by_hand():
    scores = [0.9, 0.2, 0.8, 0.6]
    is_violation = [True, True, False, False]
    timecodes = [22.5, None, None, None]
    m = video_metrics(scores, is_violation, 0.5, timecodes, [[22], [10], [], []])
    assert m["video_recall"] == 0.5  # second violation video scored below the threshold
    assert m["fp_rate"] == 1.0
    assert m["median_abs_dt"] == 0.5


def test_median_abs_dt_uses_nearest_label():
    m = video_metrics([0.9, 0.9, 0.9], [True] * 3, 0.5, [58.0, 12.0, 30.0], [[57, 68], [10], [20]])
    assert m["median_abs_dt"] == 2.0  # deltas 1, 2, 10


def test_windows_of_one_video_stay_on_one_side_of_every_fold():
    labels = {"X-1": [22], "X-4": [57], "X-47": [57], "X-5": [10], "X-48": [10], "X-3": []}
    rows = pd.concat(
        [make_windows(30.0, ns).assign(video_id=v) for v, ns in labels.items()], ignore_index=True
    )
    ids = rows["video_id"]  # one entry per window, ids repeat
    folds = list(lovo_folds(ids))
    assert len(folds) == 4  # X-1, X-4(+X-47), X-5(+X-48), X-3
    for train, val in folds:
        assert not {group_of(v) for v in ids.iloc[train]} & {group_of(v) for v in ids.iloc[val]}
        assert len(train) + len(val) == len(rows)
        # merged duplicates travel together, checked by name rather than via group_of
        for a, b in [("X-4", "X-47"), ("X-5", "X-48")]:
            assert (a in set(ids.iloc[val])) == (b in set(ids.iloc[val]))


def test_video_score_is_worst_window():
    assert video_score([0.1, 0.9, 0.4]) == 0.9
    assert video_score([]) == 0.0


def test_logit_shift_is_zero_for_equal_scores_and_signed_otherwise():
    assert logit_shift([0.2, 0.9], [0.2, 0.9]).tolist() == [0.0, 0.0]
    shift = logit_shift([0.5, 0.0], [0.5, 0.5])  # score 0 is clipped, not -inf
    assert shift[0] == 0.0 and shift[1] < -10


def test_shift_threshold_moves_down_on_the_logit_scale_and_never_up():
    assert shift_threshold(0.5, -1.0) == pytest.approx(1 / (1 + np.exp(1.0)))
    assert shift_threshold(0.2, 0.7) == pytest.approx(0.2)
    assert shift_threshold(0.2, 0.0) == pytest.approx(0.2)
