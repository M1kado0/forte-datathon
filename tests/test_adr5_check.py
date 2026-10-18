"""Pure parts of the ADR 0005 check (#70): the shares and the pass rule, and which labels each item is judged on.
The detectors themselves are never asserted."""

import pytest

from src.features.adr5_check import ITEMS, matching_clean, summarize_item, sweep, union_score
from src.heuristics.thresholds import Thresholds

LAPTOP = next(i for i in ITEMS if i.name == "laptop")  # judged on X-28 only, threshold 0.3


def test_summarize_item_counts_frames_and_labels_over_the_threshold():
    violation = {
        "X-28": [0.9, 0.5, 0.1, 0.0],  # target label: 2 of 4 frames
        "X-15": [0.0, 0.0],  # in the COCO label set, but not this item's target
        "X-11": [0.9, 0.9],  # an OWLv2 label: outside the laptop's label set, never counted
    }
    clean = {"C-1": [0.1, 0.0, 0.0, 0.0], "C-2": [0.0, 0.0]}
    s = summarize_item(LAPTOP, violation, clean)
    assert s.violation_frames_all == pytest.approx(2 / 6)
    assert s.violation_frames_target == pytest.approx(0.5)
    assert s.target_labels_hit == pytest.approx(1.0)
    assert s.clean_frames == 0.0
    assert s.clean_videos_hit == 0.0
    assert s.passed


def test_threshold_follows_the_floor_of_the_heuristics():
    by_name = {i.name: i.threshold for i in ITEMS}
    th = Thresholds()
    assert (by_name["cell phone"], by_name["cup"], by_name["bottle"]) == (
        th.phone_conf,
        th.cup_conf,
        th.bottle_conf,
    )
    assert by_name["cell phone"] != by_name["cup"]  # the test would pass by accident on one shared floor
    phone = next(i for i in ITEMS if i.name == "cell phone")
    s = summarize_item(phone, {"X-15": [0.35, 0.45]}, {})
    assert s.violation_frames_target == pytest.approx(0.5)


def test_clean_frames_are_pooled_and_clean_videos_counted_once():
    violation = {"X-28": [0.9, 0.9]}
    clean = {"C-1": [0.4, 0.4, 0.0, 0.0], "C-2": [0.0, 0.0]}
    s = summarize_item(LAPTOP, violation, clean)
    assert s.clean_frames == pytest.approx(2 / 6)
    assert s.clean_videos_hit == pytest.approx(0.5)


def test_item_that_fires_as_often_on_clean_frames_does_not_pass():
    violation = {"X-28": [0.9, 0.9, 0.0, 0.0]}  # 50% of target frames
    clean = {"C-1": [0.9, 0.0, 0.0, 0.0]}  # 25% of clean frames: not under half of 50%
    assert not summarize_item(LAPTOP, violation, clean).passed
    clean = {"C-1": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}  # 11%: under half
    assert summarize_item(LAPTOP, violation, clean).passed


def test_item_missed_on_half_of_target_labels_does_not_pass():
    key = next(i for i in ITEMS if i.name == "key")  # targets X-11 and X-22
    violation = {"X-11": [0.5, 0.5], "X-22": [0.0, 0.0]}
    assert summarize_item(key, violation, {"C-1": [0.0]}).target_labels_hit == pytest.approx(0.5)
    assert not summarize_item(key, violation, {"C-1": [0.0]}).passed


def test_empty_inputs_find_nothing():
    s = summarize_item(LAPTOP, {}, {})
    assert (s.violation_frames_target, s.clean_frames, s.target_labels_hit) == (0.0, 0.0, 0.0)
    assert not s.passed


def test_every_item_is_judged_on_the_labels_of_adr_0005():
    targets = {i.name: set(i.targets) for i in ITEMS}
    assert targets == {
        "laptop": {"X-28"},
        "cup": {"X-15", "X-40", "X-46"},
        "cell phone": {"X-15", "X-40", "X-46"},
        "bottle": {"X-15", "X-40", "X-46"},
        "foreign object": {"X-15", "X-40", "X-46"},
        "book": {"X-24"},
        "key": {"X-11", "X-22"},
        "open safe": {"X-20"},
        "open door": {"X-32", "X-33"},
    }
    sources = {i.name: i.source for i in ITEMS}
    assert {n for n, s in sources.items() if s == "coco"} == {
        "laptop",
        "cup",
        "cell phone",
        "bottle",
        "foreign object",
        "book",
    }
    assert {n for n, s in sources.items() if s == "owl"} == {"key", "open safe", "open door"}


def test_foreign_object_is_found_when_any_class_clears_its_own_floor():
    foreign = next(i for i in ITEMS if i.name == "foreign object")
    th = Thresholds()
    best = {"cup": th.cup_conf - 0.01, "cell phone": th.phone_conf - 0.01, "bottle": th.bottle_conf - 0.01}
    assert union_score(foreign, best) < foreign.threshold  # every class one step under its floor
    assert union_score(foreign, best | {"bottle": th.bottle_conf}) >= foreign.threshold
    assert union_score(foreign, best | {"cell phone": 0.9}) >= foreign.threshold


def test_clean_videos_are_matched_to_the_camera_of_the_target_labels():
    arrangement = {"X-28": "facing", "C-1": "facing", "C-2": "back"}
    clean = {"C-1": [0.0], "C-2": [0.9]}
    assert list(matching_clean(LAPTOP, clean, arrangement)) == ["C-1"]


def test_sweep_reports_each_threshold_on_its_own():
    door = next(i for i in ITEMS if i.name == "open door")
    violation = {"X-32": [0.24, 0.24], "X-33": [0.24, 0.2]}
    clean = {"C-1": [0.12, 0.24, 0.0, 0.0]}
    rows = {t: s for t, s in sweep(door, violation, clean, (0.1, 0.2, 0.25))}
    assert rows[0.1].clean_frames == pytest.approx(0.5)
    assert rows[0.2].violation_frames_target == pytest.approx(1.0)
    assert rows[0.25].violation_frames_target == 0.0 and not rows[0.25].passed
