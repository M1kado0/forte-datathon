"""Video decoding: frames at native resolution, sampled at a fixed rate, each with its second."""

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 4.0


def iter_frames(path: str | Path, fps: float = DEFAULT_FPS) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (second, BGR frame) every 1/fps seconds. Frames are never resized."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot decode {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps > 0:  # 0 or NaN from a broken container header
            src_fps = 30.0
        step = 1.0 / fps
        frame_idx = 0
        emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            # tolerance keeps float noise (e.g. 0.25 * 30 = 7.5 frames) from skipping a sample
            if frame_idx / src_fps + 1e-6 >= emitted * step:
                yield emitted * step, frame
                emitted += 1
            frame_idx += 1
    finally:
        cap.release()


def main(argv: list[str]) -> None:
    path = argv[0]
    fps = float(argv[1]) if len(argv) > 1 else DEFAULT_FPS
    start = time.perf_counter()
    n = 0
    shape = None
    for _, frame in iter_frames(path, fps):
        n += 1
        shape = frame.shape
    elapsed = time.perf_counter() - start
    print(f"{path}: {n} frames at {fps:g}/s, shape={shape}, {n / elapsed:.1f} frames/s decoded")


if __name__ == "__main__":
    main(sys.argv[1:])
