"""LOVO-bias diagnostics (scripts/lovo_bias.py): fold builders, pair AUC and the kNN bank. Pure functions only."""

import numpy as np

from scripts.lovo_bias import balanced_folds, knn_scores, normal_bank, pair_auc, pair_folds, top_mean
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
