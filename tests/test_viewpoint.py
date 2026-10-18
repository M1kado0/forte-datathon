"""Viewpoint distortions (#43). The presets are stronger than the camera spread in training, and pooled sets
keep the threshold each of them was judged with."""

import numpy as np
import pytest

from scripts.viewpoint_eval import (
    MIN_ROTATION_DEG,
    MIN_SHIFT_PX,
    PRESETS,
    pooled,
    rotation_deg,
    shift_px,
    warp_frame,
)
from src.models.evaluate import Case

TRAIN_SIZE = (464, 848)  # width and height of the dataset videos


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_is_stronger_than_the_camera_spread_in_training(name):
    h = PRESETS[name].matrix(*TRAIN_SIZE)
    assert shift_px(h, *TRAIN_SIZE) > MIN_SHIFT_PX
    assert abs(rotation_deg(h, *TRAIN_SIZE)) > MIN_ROTATION_DEG


def test_known_homography_is_measured_in_pixels_and_degrees():
    theta = np.deg2rad(10)
    h = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=float
    )
    assert rotation_deg(h, 100, 100) == pytest.approx(10)
    # worked by hand, the centre 50, 50 of a 100 x 100 frame maps to 40.56, 57.92 and moves 12.32 px
    assert shift_px(h, 100, 100) == pytest.approx(12.32, abs=0.01)


def test_warp_keeps_the_frame_size_and_fills_new_area_with_black():
    frame = np.full((60, 40, 3), 200, dtype=np.uint8)
    out = warp_frame(frame, PRESETS["back"].matrix(40, 60))
    assert out.shape == frame.shape
    assert (out == 0).any() and (out == 200).any()


def violation_case(video_id: str, score: float) -> Case:
    return Case(video_id, np.array([0.0]), np.array([score]), None, [], [(5, "other")])


def test_pooled_sets_keep_the_threshold_each_was_judged_with():
    low = {"cases": [violation_case("a", 0.5)], "first": [None], "thresholds": [0.4]}
    high = {"cases": [violation_case("b", 0.5)], "first": [None], "thresholds": [0.6]}
    # 0.5 clears 0.4 and misses 0.6, so half of the violation videos are flagged
    assert pooled([low, high])["branch_a_recall"] == 0.5
