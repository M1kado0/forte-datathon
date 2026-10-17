"""Evidence seam: evidence_frames(path, timecode_sec, kind) -> 1-3 base64 JPEG frames around the timecode."""

import base64

import cv2
import numpy as np
import pytest

from src.data.evidence import evidence_frames

SRC_FPS = 10
SECONDS = 12
W, H = 96, 64
ONE_FRAME = 1 / SRC_FPS


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("evidence") / "tiny.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    for i in range(SRC_FPS * SECONDS):
        writer.write(np.full((H, W, 3), i * 2, dtype=np.uint8))
    writer.release()
    return path


def times(frames):
    return [f.timestamp_sec for f in frames]


def test_state_takes_start_then_1s_and_5s_later(clip):
    assert times(evidence_frames(clip, 3.0, "state")) == pytest.approx([3.0, 4.0, 8.0], abs=ONE_FRAME)


def test_action_takes_peak_then_1s_before_and_after(clip):
    assert times(evidence_frames(clip, 3.0, "action")) == pytest.approx([3.0, 2.0, 4.0], abs=ONE_FRAME)


def test_first_frame_is_at_the_timecode(clip):
    for kind in ("state", "action"):
        assert evidence_frames(clip, 6.5, kind)[0].timestamp_sec == pytest.approx(6.5, abs=ONE_FRAME)


def test_every_frame_decodes_as_native_resolution_jpeg(clip):
    for frame in evidence_frames(clip, 3.0, "action"):
        raw = base64.b64decode(frame.image_base64)
        assert raw[:2] == b"\xff\xd8"
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (H, W, 3)


def test_frames_come_from_their_own_moment(clip):
    first, later = evidence_frames(clip, 2.0, "state")[:2]

    def decode(f):
        return cv2.imdecode(np.frombuffer(base64.b64decode(f.image_base64), np.uint8), cv2.IMREAD_GRAYSCALE)

    assert decode(later).mean() > decode(first).mean()


def test_timecode_at_start_drops_frames_before_it(clip):
    assert times(evidence_frames(clip, 0.0, "action")) == pytest.approx([0.0, 1.0], abs=ONE_FRAME)


def test_timecode_near_end_drops_frames_after_it(clip):
    assert times(evidence_frames(clip, 11.0, "state")) == pytest.approx([11.0], abs=ONE_FRAME)
    assert len(evidence_frames(clip, 11.5, "action")) == 2


def test_timecode_past_the_end_snaps_to_the_last_frame(clip):
    frames = evidence_frames(clip, 99.0, "state")
    assert times(frames) == pytest.approx([SECONDS - ONE_FRAME])


def test_negative_timecode_raises(clip):
    with pytest.raises(ValueError):
        evidence_frames(clip, -1.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evidence_frames(tmp_path / "nope.mov", 1.0)
