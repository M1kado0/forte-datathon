"""The heuristics' vote on the verdict (ADR 0005, ADR 0002).

A hit already passed its type's thresholds, so it is confident. The vote only raises the verdict to
"violation" and never lowers it: silence is not "no violation". N is passed in, #34 picks it on LOVO.

Only a type that held at most 2 false positives in 19 clean videos on leave-one-video-out may vote
(docs/HEURISTICS.md). The others still report their hits as findings, but do not raise the verdict.
"""

from collections.abc import Collection, Iterable
from math import ceil

from src.data.video import DEFAULT_FPS
from src.heuristics.detect import CASH_ON_DESK_NEXT_CLIENT, Hit

VOTING_TYPES = frozenset({CASH_ON_DESK_NEXT_CLIENT})


def voice(
    hits: Iterable[Hit], n_sec: float, fps: float = DEFAULT_FPS, types: Collection[str] = VOTING_TYPES
) -> bool:
    """True when one hit of a voting type lasts at least n_sec. A hit from first to last frame is
    (last - first + 1) / fps long."""
    need = ceil(n_sec * fps - 1e-9)
    return any(h.type_id in types and round((h.end_sec - h.start_sec) * fps) + 1 >= need for h in hits)
