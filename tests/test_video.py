"""Decoder seam: iter_frames(path, fps) -> (second, frame) at native resolution."""

import cv2
import numpy as np
import pytest

from src.data.video import duration, iter_frames

SRC_FPS = 30
SECONDS = 2
W, H = 96, 64


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / "tiny.mov", SRC_FPS)


def write_clip(path, src_fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
    for i in range(src_fps * SECONDS):
        writer.write(np.full((H, W, 3), i * 4, dtype=np.uint8))
    writer.release()
    return path


def test_samples_requested_rate_with_timestamps(clip):
    seconds = [s for s, _ in iter_frames(clip)]  # default rate is 4/s
    assert len(seconds) == 8
    # each sample is the first source frame at or after the grid time, so within one frame of it
    assert seconds == pytest.approx([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], abs=1 / SRC_FPS)


def test_duration_is_the_length_of_the_clip_not_the_last_sampled_frame(clip):
    assert duration(clip) == pytest.approx(SECONDS)  # the last frame sampled at 4/s is at 1.75 s


def test_duration_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        duration(tmp_path / "nope.mov")


def test_slow_source_keeps_true_frame_times(tmp_path):
    slow = write_clip(tmp_path / "slow.mov", 2)
    assert [s for s, _ in iter_frames(slow, fps=4)] == [0.0, 0.5, 1.0, 1.5]


def test_frames_keep_native_resolution(clip):
    frames = [f for _, f in iter_frames(clip, fps=4)]
    assert all(f.shape == (H, W, 3) for f in frames)


def test_frames_follow_the_video_in_time(clip):
    means = [f.mean() for _, f in iter_frames(clip, fps=4)]
    assert means == sorted(means) and means[0] < means[-1]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nope.mov"))


def test_non_positive_fps_raises(clip):
    with pytest.raises(ValueError):
        list(iter_frames(clip, fps=0))
