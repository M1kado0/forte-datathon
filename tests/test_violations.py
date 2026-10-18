"""Which violations are reported, and when (#34): type order, timecode by scenario, the heuristics' vote."""

from dataclasses import replace

import numpy as np
import pytest

from src.heuristics import Hit
from src.models.violations import (
    FALLBACK_TYPE,
    Knn,
    Settings,
    Vote,
    decide,
    evidence_kind,
    knn_type,
)

OUTSIDER, OBJECT, LEFT = "outsider_in_cash_room", "foreign_object_at_desk", "client_left_during_count"
STARTS = np.arange(0, 40, 0.5)  # window centres 1.0 .. 40.5
THRESHOLD = 0.5
VOTE = Vote(enabled=True, n_sec=3.0, types=frozenset({OUTSIDER}))
SETTINGS = Settings(allowed=frozenset({OUTSIDER, OBJECT, LEFT}), vote=VOTE)


def scores_with(**peaks):
    """Window scores 0.1 everywhere; `peaks` maps a window centre (as w<centre*10>) to a score."""
    s = np.full(len(STARTS), 0.1)
    for key, value in peaks.items():
        s[int((int(key[1:]) / 10 - 1.0) / 0.5)] = value
    return s


def run(hits, scores, vote=VOTE, knn=None, peak_feat=None):
    return decide(STARTS, scores, THRESHOLD, hits, replace(SETTINGS, vote=vote, knn=knn), peak_feat)


def test_evidence_kind_is_state_for_scenarios_that_last_and_action_for_the_rest():
    assert evidence_kind(OUTSIDER) == "state"
    assert evidence_kind("keys_left_on_safe") == "state"
    assert evidence_kind(LEFT) == "action"
    assert evidence_kind("collector_id_not_shown") == "action"
    assert evidence_kind(FALLBACK_TYPE) == "action"


def test_nothing_flagged_and_no_vote_is_no_violation():
    assert run([], scores_with()) == []
    assert (
        run([Hit(OBJECT, 10.0, 20.0, "cup")], scores_with()) == []
    )  # a non-voting type never raises the verdict


def test_heuristic_type_beats_the_fallback_and_takes_the_start_of_the_state():
    found = run([Hit(OUTSIDER, 12.0, 20.0)], scores_with(w150=0.9))
    assert [(f.type_id, f.timecode_sec, f.kind) for f in found] == [(OUTSIDER, 12.0, "state")]


def test_action_type_takes_the_score_peak_inside_the_hit():
    found = run([Hit(LEFT, 10.0, 20.0)], scores_with(w150=0.9, w300=0.95))
    assert [(f.type_id, f.timecode_sec, f.kind) for f in found] == [(LEFT, 15.0, "action")]


def test_object_hit_carries_its_class_as_a_fact():
    found = run([Hit(OBJECT, 12.0, 20.0, "cup")], scores_with(w150=0.9))
    assert found[0].facts == {"object": "чашка"}


def test_type_outside_the_allowed_set_falls_through_to_the_fallback_at_the_peak():
    found = run([Hit("cash_on_desk_next_client", 5.0, 30.0)], scores_with(w250=0.9))
    assert [(f.type_id, f.timecode_sec, f.kind) for f in found] == [(FALLBACK_TYPE, 25.0, "action")]


def test_without_a_hit_the_fallback_is_one_violation_at_the_strongest_peak():
    found = run([], scores_with(w150=0.7, w300=0.9))
    assert [(f.type_id, f.timecode_sec) for f in found] == [(FALLBACK_TYPE, 30.0)]


def test_vote_raises_a_quiet_verdict_with_the_voting_hit_only():
    hits = [Hit(OUTSIDER, 12.0, 16.0), Hit(OBJECT, 20.0, 30.0, "cup")]
    found = run(hits, scores_with())
    assert [(f.type_id, f.timecode_sec) for f in found] == [(OUTSIDER, 12.0)]


def test_a_hit_shorter_than_n_does_not_vote():
    assert run([Hit(OUTSIDER, 12.0, 13.0)], scores_with()) == []


def test_disabled_vote_leaves_the_verdict_to_branch_a():
    off = Vote(enabled=False, n_sec=3.0, types=frozenset({OUTSIDER}))
    assert run([Hit(OUTSIDER, 12.0, 20.0)], scores_with(), vote=off) == []
    assert [f.type_id for f in run([Hit(OUTSIDER, 12.0, 20.0)], scores_with(w150=0.9), vote=off)] == [
        OUTSIDER
    ]


def test_two_types_are_two_violations_strongest_first_and_apart():
    hits = [Hit(OUTSIDER, 3.0, 9.0), Hit(OBJECT, 25.0, 33.0, "cup")]
    found = run(hits, scores_with(w60=0.7, w290=0.9))
    assert [f.type_id for f in found] == [OBJECT, OUTSIDER]


def test_two_types_closer_than_the_gap_keep_the_stronger():
    hits = [Hit(OUTSIDER, 4.0, 8.0), Hit(OBJECT, 7.0, 12.0, "cup")]  # starts 3 s apart, no window shared
    found = run(hits, scores_with(w50=0.7, w110=0.9))
    assert [f.type_id for f in found] == [OBJECT]


def test_a_type_with_several_hits_uses_the_one_with_the_stronger_window():
    hits = [Hit(OUTSIDER, 1.0, 6.0), Hit(OUTSIDER, 20.0, 26.0)]
    found = run(hits, scores_with(w40=0.6, w230=0.9))
    assert [f.timecode_sec for f in found] == [20.0]


def knn_bank():
    # two types on two axes
    x = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    return Knn(
        features=x, types=np.array(["keys_left_on_safe"] * 2 + ["laptop_left_open"] * 2), k=3, min_share=0.8
    )


def test_knn_names_the_type_of_the_nearest_windows_when_they_agree():
    assert knn_type(np.array([0.0, 1.0]), knn_bank()) == "laptop_left_open"


def test_knn_stays_silent_when_neighbours_disagree():
    bank = knn_bank()
    assert knn_type(np.array([1.0, 1.0]), bank) is None


def test_knn_never_names_a_type_that_needs_an_object_fact():
    bank = Knn(np.array([[1.0, 0.0]] * 3), np.array([OBJECT] * 3), k=3, min_share=0.5)
    assert knn_type(np.array([1.0, 0.0]), bank) is None


def test_knn_type_replaces_the_fallback_but_has_no_hit_so_it_keeps_the_peak_and_its_frames():
    found = run([], scores_with(w150=0.9), knn=knn_bank(), peak_feat=np.array([0.0, 1.0]))
    assert [(f.type_id, f.timecode_sec, f.kind) for f in found] == [("laptop_left_open", 15.0, "action")]


@pytest.mark.parametrize("peak_feat", [None, np.array([1.0, 1.0])])
def test_knn_without_a_confident_type_leaves_the_fallback(peak_feat):
    found = run([], scores_with(w150=0.9), knn=knn_bank(), peak_feat=peak_feat)
    assert [f.type_id for f in found] == [FALLBACK_TYPE]
