"""Evidence frames: 1-3 JPEGs around a violation's timecode, so a human can check it without scrubbing."""

import base64
from pathlib import Path
from typing import Literal

import cv2

from contracts.schemas import EvidenceFrame

JPEG_QUALITY = 85
# Seconds from the timecode. The first frame is always the timecode itself.
OFFSETS_SEC = {
    "state": (0.0, 1.0, 5.0),  # start of the state, then how long it lasts
    "action": (0.0, -1.0, 1.0),  # peak of the action, then just before and just after
}
Kind = Literal["state", "action"]


def evidence_frames(path: str | Path, timecode_sec: float, kind: Kind = "state") -> list[EvidenceFrame]:
    """Frames at the timecode and its offsets for `kind`; offsets outside the video are dropped.

    Always returns at least the first frame: a timecode at or past the last frame snaps to the last one.
    Frames keep the native resolution and carry the time of the frame actually decoded.
    """
    if timecode_sec < 0:
        raise ValueError(f"timecode_sec must be >= 0, got {timecode_sec}")
    offsets = OFFSETS_SEC[kind]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot decode {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps > 0:  # 0 or NaN from a broken container header
            raise ValueError(f"{path} has no usable fps in its header")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[EvidenceFrame] = []
        taken: set[int] = set()
        for i, offset in enumerate(offsets):
            idx = round((timecode_sec + offset) * src_fps)
            if idx < 0 or idx in taken:
                continue
            if i == 0 and 0 < total <= idx:
                idx = total - 1
            frame = _read_at(cap, idx)
            if frame is None and i == 0:
                raise ValueError(f"no frame at {timecode_sec:g}s in {path}")
            if frame is None:
                continue
            taken.add(idx)
            frames.append(EvidenceFrame(timestamp_sec=idx / src_fps, image_base64=_to_base64_jpeg(frame)))
        return frames
    finally:
        cap.release()


def _read_at(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def _to_base64_jpeg(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("cannot encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
