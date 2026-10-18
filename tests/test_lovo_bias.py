"""LOVO-bias diagnostics (scripts/lovo_bias.py): fold builders, pair AUC and the kNN bank. Pure functions only."""

import numpy as np

from scripts.lovo_bias import (
    balanced_folds,
    bootstrap_auc_interval,
    centre_on_start,
    drop_folds,
    fp_at_full_recall,
    knn_scores,
    layout_folds,
    max_minus_median,
    nearest_twin,
    normal_bank,
    pair_auc,
    pair_folds,
    random_stand_in,
    recall_at_fp,
    top_mean,
)
from src.models.windows import NEGATIVE, POSITIVE, UNLABELED

IDS = [f"X-{i}" for i in range(1, 11)]
VIOLATION = np.array([True] * 6 + [False] * 4)  # X-1..X-6 violate, X-7..X-10 are clean


def test_balanced_folds_hold_out_one_clip_and_keep_class_counts_equal():
    folds = list(balanced_folds(IDS, VIOLATION, seed=0))
    assert len(folds) == len(IDS)
    for train, val in folds:
        assert len(val) == 1
        assert val[0] not in train
        # held-out violation: 5 violation + 3 clean left; held-out clean: 5 violation + 3 clean left
        assert VIOLATION[train].sum() == 5
        assert (~VIOLATION[train]).sum() == 3


def test_balanced_folds_drop_the_opposite_class_and_depend_on_the_seed():
    a = [tuple(t) for t, _ in balanced_folds(IDS, VIOLATION, seed=0)]
    assert a == [tuple(t) for t, _ in balanced_folds(IDS, VIOLATION, seed=0)]
    b = [tuple(t) for t, _ in balanced_folds(IDS, VIOLATION, seed=1)]
    assert a != b


def test_pair_folds_cover_every_violation_clean_pair_once_and_never_leak_the_pair():
    folds = list(pair_folds(IDS, VIOLATION))
    assert len(folds) == 6 * 4
    assert len({(int(pos), int(neg)) for _, pos, neg in folds}) == 24
    for train, pos, neg in folds:
        assert VIOLATION[pos] and not VIOLATION[neg]
        assert pos not in train and neg not in train
        assert len(train) == len(IDS) - 2


def test_pair_auc_is_the_share_of_pairs_where_the_violation_scores_higher():
    # pairs: (0.9 > 0.2) (0.9 > 0.7) (0.5 > 0.2) (0.5 < 0.7) -> 3 of 4
    assert pair_auc([0.9, 0.5], [0.2, 0.7]) == 0.75
    # a tie counts half
    assert pair_auc([0.4], [0.4]) == 0.5


def test_knn_scores_are_the_mean_distance_to_the_k_nearest_bank_windows():
    bank = np.array([[0.0], [1.0], [10.0]])
    queries = np.array([[0.4], [9.0]])
    assert np.allclose(knn_scores(bank, queries, k=1), [0.4, 1.0])
    # k=2: (0.4 + 0.6) / 2 for the first query, (1 + 8) / 2 for the second (nearest are 10 and 1)
    assert np.allclose(knn_scores(bank, queries, k=2), [0.5, 4.5])


def test_normal_bank_takes_negative_windows_of_training_clips_only():
    feats = [np.array([[1.0], [2.0], [3.0]]), np.array([[10.0], [20.0]]), np.array([[100.0]])]
    labels = [np.array([NEGATIVE, POSITIVE, UNLABELED]), np.array([NEGATIVE, NEGATIVE]), np.array([NEGATIVE])]
    bank = normal_bank(feats, labels, train_idx=np.array([0, 1]))
    assert sorted(bank.ravel().tolist()) == [
        1.0,
        10.0,
        20.0,
    ]  # clip 2 is held out, clip 0 keeps only its negative


def test_top_mean_averages_the_n_highest_scores_and_tolerates_short_clips():
    assert top_mean([0.1, 0.9, 0.5, 0.7], 2) == 0.8
    assert top_mean([0.4], 5) == 0.4
    assert top_mean([], 5) == 0.0


def test_balanced_folds_with_layouts_drop_the_opposite_class_from_the_held_out_clips_layout():
    # layout a: X-1..X-3 violate, X-7 and X-8 are clean; layout b: X-4..X-6 violate, X-9 and X-10 are clean
    layouts = np.array(["a", "a", "a", "b", "b", "b", "a", "a", "b", "b"])
    for train, val in balanced_folds(IDS, VIOLATION, seed=0, layouts=layouts):
        mine = layouts == layouts[val[0]]
        in_train = np.zeros(len(IDS), dtype=bool)
        in_train[train] = True
        # held-out layout: one violation clip and one clean clip less, whichever kind was held out
        assert (in_train & mine & VIOLATION).sum() == (mine & VIOLATION).sum() - 1
        assert (in_train & mine & ~VIOLATION).sum() == (mine & ~VIOLATION).sum() - 1
        # the other layout is left whole
        assert (in_train & ~mine).sum() == (~mine).sum()


def test_fp_at_full_recall_counts_clean_clips_scoring_at_or_above_the_weakest_violation():
    violation = np.array([True, True, False, False, False])
    # threshold = lowest violation score 0.4; clean clips at 0.6 and 0.5 are flagged, 0.2 is not
    assert fp_at_full_recall([0.9, 0.4, 0.6, 0.2, 0.5], violation) == 2
    # a clean clip tied with the threshold is flagged too
    assert fp_at_full_recall([0.9, 0.4, 0.4, 0.2, 0.1], violation) == 1
    # every violation scores above every clean clip: no false positive
    assert fp_at_full_recall([0.9, 0.8, 0.3, 0.2, 0.1], violation) == 0


def test_nearest_twin_is_the_most_similar_other_clip_by_cosine():
    mean = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]])
    # clip 4 points away from everything, its closest direction is clip 2 or 3 (orthogonal, cosine 0), never itself
    twin = nearest_twin(mean)
    assert twin[:4].tolist() == [1, 0, 3, 2]
    assert twin[4] != 4


def test_random_stand_in_has_the_twins_class_and_is_neither_the_clip_nor_its_twin():
    twin = np.array([6, 7, 8, 9, 0, 1, 2, 3, 4, 5])  # every clip is paired with another one
    for seed in range(5):
        stand_in = random_stand_in(twin, VIOLATION, seed)
        for i, j in enumerate(stand_in):
            assert VIOLATION[j] == VIOLATION[twin[i]]
            assert j not in (i, twin[i])


def test_drop_folds_hold_out_one_clip_and_also_remove_the_dropped_one_from_training():
    dropped = np.array([6, 6, 6, 6, 6, 6, 0, 0, 0, 0])
    folds = list(drop_folds(IDS, dropped))
    assert len(folds) == len(IDS)
    for train, val in folds:
        assert val[0] not in train and dropped[val[0]] not in train
        assert len(train) == len(IDS) - 2


def test_pair_folds_can_also_remove_a_dropped_clip_per_member_of_the_pair():
    dropped = np.array([6, 6, 6, 6, 6, 6, 0, 0, 0, 0])
    for train, pos, neg in pair_folds(IDS, VIOLATION, dropped):
        assert not {pos, neg, dropped[pos], dropped[neg]} & set(train.tolist())


def test_centre_on_start_subtracts_the_mean_of_the_first_windows_from_every_window():
    feats = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    assert np.allclose(centre_on_start(feats, 2), [[-1.0, -1.0], [1.0, 1.0], [3.0, 3.0]])
    # a reference longer than the clip uses the whole clip; an empty clip stays empty
    assert np.allclose(centre_on_start(feats, 10), feats - 3.0)
    assert centre_on_start(np.zeros((0, 2)), 4).shape == (0, 2)


def test_max_minus_median_is_the_peak_logit_above_the_clips_own_typical_logit():
    # logits 0, 0, ln 9: the median is 0, so the score is the peak logit ln 9
    assert np.isclose(max_minus_median([0.5, 0.5, 0.9]), np.log(9))
    # a clip that scores high everywhere has no peak above its own level
    assert np.isclose(max_minus_median([0.9, 0.9, 0.9]), 0.0)
    assert max_minus_median([]) == 0.0


def test_recall_at_fp_counts_violations_that_beat_the_clean_clip_just_past_the_allowed_fp():
    scores = np.array([0.9, 0.7, 0.3, 0.8, 0.6, 0.5, 0.1])
    violation = np.array([True, True, True, False, False, False, False])
    # clean scores high to low: 0.8, 0.6, 0.5, 0.1
    assert recall_at_fp(scores, violation, 0) == 1 / 3  # must beat 0.8: only 0.9
    assert recall_at_fp(scores, violation, 1) == 2 / 3  # must beat 0.6: 0.9 and 0.7
    assert recall_at_fp(scores, violation, 4) == 1.0  # every clean clip may be flagged


def test_layout_folds_hold_out_one_whole_layout_and_train_on_the_rest():
    layouts = ["a", "a", "b", "b", "b", "a"]
    folds = list(layout_folds(layouts))
    assert [val.tolist() for _, val in folds] == [[0, 1, 5], [2, 3, 4]]
    for train, val in folds:
        assert not set(train) & set(val)
        assert len(train) + len(val) == len(layouts)


def test_bootstrap_auc_interval_collapses_when_the_scores_are_decisive_and_straddles_chance_when_not():
    bad = np.array([True] * 5 + [False] * 5)
    assert bootstrap_auc_interval(np.arange(10.0)[::-1], bad) == (
        1.0,
        1.0,
    )  # violations all score above clean clips
    assert bootstrap_auc_interval(np.zeros(10), bad) == (0.5, 0.5)  # all tied
    low, high = bootstrap_auc_interval(np.array([0, 3, 1, 4, 2, 1, 4, 0, 3, 2.0]), bad)
    assert low < 0.5 < high  # a weak ordering cannot be told from chance with ten clips
