"""Decoder seam: iter_frames(path, fps) -> (second, frame) at native resolution."""

import cv2
import numpy as np
import pytest

from src.data.video import iter_frames

SRC_FPS = 30
SECONDS = 2
W, H = 96, 64


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "tiny.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    for i in range(SRC_FPS * SECONDS):
        writer.write(np.full((H, W, 3), i * 4, dtype=np.uint8))
    writer.release()
    return path


def test_samples_requested_rate_with_timestamps(clip):
    out = list(iter_frames(clip))  # default rate is 4/s
    assert [s for s, _ in out] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


def test_frames_keep_native_resolution(clip):
    frames = [f for _, f in iter_frames(clip, fps=4)]
    assert all(f.shape == (H, W, 3) for f in frames)


def test_frames_follow_the_video_in_time(clip):
    means = [f.mean() for _, f in iter_frames(clip, fps=4)]
    assert means == sorted(means) and means[0] < means[-1]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nope.mov"))
