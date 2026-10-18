"""The heuristics' vote on the verdict (ADR 0005, ADR 0002).

A hit already passed its type's thresholds, so it is confident. The vote only raises the verdict to
"violation" and never lowers it: silence is not "no violation". N is passed in, #34 picks it on LOVO.

A type may vote only if it held at most 2 false positives in 19 clean videos (scripts/heuristics_lovo.py
report prints the count per type; on the stand-in tracks that is outsider_in_cash_room 0/19 and
foreign_object_at_desk 2/19, while cash_on_desk_next_client has 7/19 and client_left_during_count 4/19).
The hits of the other types are still reported as findings; they just do not raise the verdict. The set
rests on stand-in tracks, #34 rechecks it on the real ones.
"""

from collections.abc import Collection, Iterable
from math import ceil

from src.data.video import DEFAULT_FPS
from src.heuristics.detect import FOREIGN_OBJECT_AT_DESK, OUTSIDER_IN_CASH_ROOM, Hit

VOTING_TYPES: frozenset[str] = frozenset({OUTSIDER_IN_CASH_ROOM, FOREIGN_OBJECT_AT_DESK})


def vote(
    hits: Iterable[Hit], n_sec: float, fps: float = DEFAULT_FPS, types: Collection[str] = VOTING_TYPES
) -> bool:
    """True when one hit of a voting type lasts at least n_sec. A hit from first to last frame is
    (last - first + 1) / fps long."""
    need = ceil(n_sec * fps - 1e-9)
    return any(h.type_id in types and round((h.end_sec - h.start_sec) * fps) + 1 >= need for h in hits)
