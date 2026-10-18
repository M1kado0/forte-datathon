"""The fold logic of the type and vote evaluation (#34): what is chosen, on which clips, and how it is counted."""

import numpy as np
import pytest

from src.heuristics import Hit
from src.models.evaluate import (
    Case,
    fit_allowed,
    fit_knn,
    fit_settings,
    fit_vote,
    lovo,
    summarize,
)
from src.models.violations import (
    FALLBACK_TYPE,
    HEURISTIC_CONFIGS,
    HEURISTIC_TYPES,
    STATE_TYPES,
    Found,
)
from src.rules.describe import rule_catalog

OUTSIDER, OBJECT = "outsider_in_cash_room", "foreign_object_at_desk"
STARTS = np.arange(0, 20, 0.5)
THRESHOLD = 0.5


def case(video_id, truth=(), *, hits=(), other_hits=None, flagged=True, feature=None):
    """`hits` are the default thresholds' hits, `other_hits` those of controller_exempt (default: the same)."""
    scores = np.full(len(STARTS), 0.1)
    if flagged:
        scores[20] = 0.9
    by = {"default": list(hits), "controller_exempt": list(hits if other_hits is None else other_hits)}
    return Case(video_id, STARTS, scores, feature, by["default"], list(truth), by)


def hit(type_id, start=4.0, end=9.0):
    return Hit(type_id, start, end)


def test_every_type_the_heuristics_emit_and_every_state_type_is_in_the_catalogue():
    ids = {t["id"] for t in rule_catalog()}
    assert set(HEURISTIC_TYPES) <= ids
    assert STATE_TYPES <= ids


def test_a_type_is_allowed_where_it_fires_mostly_on_its_own_videos_and_the_table_lists_silent_types():
    cases = [
        case("X-1", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-2", [(5, "other_type")], hits=[hit(OUTSIDER)]),
        case("X-3", [(5, OBJECT)], hits=[hit(OBJECT)]),
        case("X-8", hits=[hit(OBJECT)]),
        case("X-10", hits=[hit(OBJECT)]),
    ]
    allowed, table = fit_allowed(cases)
    assert allowed == {OUTSIDER}  # 1 of 2 videos; the object type is right on 1 of 3
    assert table["id_not_presented"] == {"right": 0, "fired": 0}
    assert table[OBJECT] == {"right": 1, "fired": 3}


def test_the_vote_takes_the_smallest_n_for_types_that_never_fire_alone_on_a_clean_video():
    cases = [
        case("X-1", [(5, OUTSIDER)], hits=[hit(OUTSIDER, 4, 9)]),
        case("X-2", [(5, OBJECT)], hits=[hit(OBJECT, 4, 9)]),
        case("X-8", hits=[hit(OBJECT, 4, 9)]),  # the object type fires on a clean video: it cannot vote
    ]
    chosen, notes = fit_vote(cases, THRESHOLD)
    assert chosen.enabled and chosen.n_sec == 3.0 and chosen.types == {OUTSIDER}
    assert notes["clean_videos"] == 1


def test_the_vote_is_off_when_a_voting_type_adds_a_false_positive_branch_a_did_not_make():
    cases = [
        case("X-1", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-8", hits=[hit(OUTSIDER)], flagged=False),  # only the vote would flag this clean video
    ]
    chosen, _ = fit_vote(cases, THRESHOLD)
    assert not chosen.enabled


def test_heuristic_thresholds_are_the_ones_naming_more_right_types_and_a_tie_keeps_the_default():
    default_ok = [case("X-1", [(5, OUTSIDER)], hits=[hit(OUTSIDER)])]
    assert fit_settings(default_ok, THRESHOLD)[0].heuristics == "default"

    exempt_better = [
        case("X-1", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-2", [(5, "other_type")], hits=[hit(OUTSIDER)], other_hits=[]),  # the default names it wrongly
    ]
    settings, notes = fit_settings(exempt_better, THRESHOLD)
    assert settings.heuristics == "controller_exempt"
    assert set(notes["heuristics"]["tried"]) == set(HEURISTIC_CONFIGS)


def test_a_clip_is_judged_without_its_own_group():
    """X-4 and X-47 are one clip: held out, neither may teach the settings that the outsider type is right."""
    cases = [
        case("X-4", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-47", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-8"),
    ]
    result = lovo(cases, THRESHOLD)
    assert result["types_by_video"]["X-4"]["found"] == FALLBACK_TYPE
    assert result["types_by_video"]["X-47"]["found"] == FALLBACK_TYPE
    assert result["n_typed_right"] == 0


def test_a_second_clip_with_the_type_teaches_it_to_the_held_out_one():
    cases = [
        case("X-4", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
        case("X-8", [(5, OUTSIDER)], hits=[hit(OUTSIDER)]),
    ]
    assert lovo(cases, THRESHOLD)["n_typed_right"] == 2


def test_summarize_counts_false_positives_and_type_accuracy_directly():
    cases = [
        case("X-1", [(5, OUTSIDER)]),
        case("X-2", [(5, OUTSIDER)]),
        case("X-8"),
        case("X-10"),
    ]
    first = [
        Found(OUTSIDER, 5.5, "state", onset_sec=5.0, peak_sec=6.0),
        None,
        Found(FALLBACK_TYPE, 3.0, "action"),
        None,
    ]
    out = summarize(cases, first)
    assert out["video_recall"] == 0.5 and out["type_accuracy"] == 0.5
    assert (out["n_clean_videos"], out["n_false_positives"], out["fp_rate"]) == (2, 1, 0.5)
    assert out["median_abs_dt"] == pytest.approx(0.5)
    assert out["timecode_choice"]["state"]["onset"] == 0.0 and out["timecode_choice"]["state"]["peak"] == 1.0


def test_knn_is_off_when_the_nearest_windows_name_the_wrong_type():
    """Windows of a scene are close to each other whatever the type: a bank of other clips says 'laptop'."""
    feature = np.array([1.0, 0.0])
    cases = [case(f"X-{i}", [(5, "keys_left_on_safe")], feature=feature) for i in range(1, 5)]
    banks = {"X-20": (np.array([[1.0, 0.0]] * 5), ["laptop_left_open"] * 5)}
    knn, notes = fit_knn(cases, banks)
    assert knn is None
    assert all(row["right"] == 0 for row in notes["grid"])


def test_knn_is_kept_when_it_names_the_right_type_and_carries_its_bank():
    feature = np.array([1.0, 0.0])
    cases = [case(f"X-{i}", [(5, "keys_left_on_safe")], feature=feature) for i in range(1, 5)]
    banks = {"X-20": (np.array([[1.0, 0.0]] * 5), ["keys_left_on_safe"] * 5)}
    knn, _ = fit_knn(cases, banks)
    assert knn is not None and len(knn.types) == 5
