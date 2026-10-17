"""The heuristics' vote on the verdict (ADR 0005, ADR 0002).

A hit already passed its type's thresholds, so it is confident. The vote only raises the verdict to
"violation" and never lowers it: silence is not "no violation". N is passed in, #34 picks it on LOVO.
"""

from collections.abc import Iterable
from math import ceil

from src.data.video import DEFAULT_FPS
from src.heuristics.detect import Hit


def voice(hits: Iterable[Hit], n_sec: float, fps: float = DEFAULT_FPS) -> bool:
    """True when one hit lasts at least n_sec. A hit from first to last frame is (last - first + 1) / fps long."""
    need = ceil(n_sec * fps - 1e-9)
    return any(round((h.end_sec - h.start_sec) * fps) + 1 >= need for h in hits)
