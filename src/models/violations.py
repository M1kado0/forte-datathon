"""Which violations `analyze` reports, of which type and at which second (#34, ADR 0005, ADR 0002).

Type, in order: a heuristic hit of an allowed type, then kNN over the labelled windows, then the fallback type.
Timecode by scenario: the start of the state for state types, the score peak for actions and the fallback.
Verdict: a window at or above the threshold, OR a long enough hit of a voting type. The vote only raises the
verdict. Pure functions: `train` measures them on leave-one-video-out and saves the settings that won.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.data.evidence import Kind
from src.heuristics import Hit, vote
from src.heuristics.detect import FOREIGN_OBJECT_AT_DESK, OUTSIDER_IN_CASH_ROOM, VALUABLES_LEFT_ON_DESK
from src.models.windows import WINDOW_SEC, timecode
from src.rules.describe import OBJECT_NAMES

FALLBACK_TYPE = "other"  # rules.yaml `fallback: true`
MAX_VIOLATIONS = 2
MAX_FALLBACK_VIOLATIONS = 1  # ADR 0005: with no type, one neutral violation per video
MIN_GAP_SEC = 5.0

# Types that stand while they last: the timecode is the start of the state. Every other type, the fallback
# included, is an action and gets the score peak (docs/TYPES.md). `train` checks both choices by median |Δt|.
ONSET_TYPES = frozenset(
    {
        "keys_left_on_safe",
        "safe_left_open",
        "cash_room_door_open",
        "laptop_left_open",
        "valuables_left_on_desk",
        "foreign_object_at_desk",
        "cash_left_on_desk",
        "cash_on_desk_next_client",
        "outsider_in_cash_room",
        "workplace_left",
        "count_without_controller",
    }
)
# The description of these needs the object a heuristic saw, which a window embedding does not give.
NEEDS_OBJECT = frozenset({FOREIGN_OBJECT_AT_DESK, VALUABLES_LEFT_ON_DESK})


@dataclass(frozen=True)
class Vote:
    """Heuristics' vote on the verdict: a hit of one of `types` lasting at least `n_sec`."""

    enabled: bool = False
    n_sec: float = 3.0
    types: frozenset[str] = frozenset({OUTSIDER_IN_CASH_ROOM})


@dataclass(frozen=True)
class Knn:
    """Labelled positive windows (one row each) and the type of each; a type is named when `min_share` of the
    similarity-weighted votes of the `k` nearest windows agree."""

    features: np.ndarray
    types: np.ndarray
    k: int
    min_share: float


@dataclass(frozen=True)
class Settings:
    """What `train` saves next to the window classifier. The defaults are branch A alone: no heuristic names a
    type, the vote is off, no kNN, so every violation gets the fallback type."""

    allowed: frozenset[str] = frozenset()  # heuristic types that may name a violation
    vote: Vote = Vote()
    knn: Knn | None = None


@dataclass(frozen=True)
class Found:
    type_id: str
    timecode_sec: float
    kind: Kind  # which evidence frames to cut: "state" around the start, "action" around the peak
    facts: dict[str, str] = field(default_factory=dict)  # fills the description template
    score: float = 0.0  # branch A's strongest window over the violation, for ordering only
    onset_sec: float = (
        0.0  # both candidate timecodes, so `train` can compare them; timecode_sec is one of them
    )
    peak_sec: float = 0.0


def timecode_kind(type_id: str) -> Literal["onset", "peak"]:
    return "onset" if type_id in ONSET_TYPES else "peak"


def decide(
    starts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    hits: list[Hit],
    settings: Settings,
    knn: Knn | None = None,
    peak_feature: np.ndarray | None = None,
) -> list[Found]:
    """The violations of one video, strongest first; [] means no violation. `knn` is the bank for this clip;
    `settings.knn` only holds the chosen k and share."""
    starts = np.asarray(starts, dtype=float)
    scores = np.asarray(scores, dtype=float)
    flagged = bool(scores.size and scores.max() >= threshold)
    voting = _voting_hits(hits, settings.vote)
    if not flagged and not voting:
        return []
    typed = _from_hits(
        voting if not flagged else [h for h in hits if h.type_id in settings.allowed], starts, scores
    )
    if typed:
        return typed[:MAX_VIOLATIONS]
    peak = int(np.argmax(scores))
    type_id = knn_type(peak_feature, knn) if knn is not None and peak_feature is not None else None
    type_id = type_id or FALLBACK_TYPE
    centre = float(starts[peak] + WINDOW_SEC / 2)
    kind: Kind = "state" if timecode_kind(type_id) == "onset" else "action"
    onset = timecode(
        starts, scores, threshold, "onset"
    )  # not None: this path is only reached by a flagged video
    return [Found(type_id, centre, kind, {}, float(scores[peak]), onset, centre)]


def knn_type(feature: np.ndarray | None, bank: Knn | None) -> str | None:
    """The type the nearest labelled windows agree on, None when they do not or when it needs an object fact."""
    if feature is None or bank is None or not len(bank.types):
        return None
    sim = _unit(bank.features) @ _unit(np.asarray(feature, dtype=float))
    near = np.argsort(-sim)[: bank.k]
    weight: dict[str, float] = {}
    for i in near:
        weight[str(bank.types[i])] = weight.get(str(bank.types[i]), 0.0) + max(float(sim[i]), 0.0)
    total = sum(weight.values())
    best = max(weight, key=weight.__getitem__)
    if total <= 0 or weight[best] / total < bank.min_share or best in NEEDS_OBJECT:
        return None
    return best


def _unit(x: np.ndarray) -> np.ndarray:
    """Vector or rows scaled to length 1."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def _voting_hits(hits: list[Hit], settings: Vote) -> list[Hit]:
    if not settings.enabled:
        return []
    return [h for h in hits if vote([h], settings.n_sec, types=settings.types)]


def window_score(hit: Hit, starts: np.ndarray, scores: np.ndarray) -> tuple[float, float | None]:
    """(best window score over the hit, centre of that window); windows that overlap the hit count."""
    inside = (starts + WINDOW_SEC > hit.start_sec) & (starts < hit.end_sec)
    if not inside.any():
        return 0.0, None
    i = np.flatnonzero(inside)[int(np.argmax(scores[inside]))]
    return float(scores[i]), float(starts[i] + WINDOW_SEC / 2)


def _from_hits(hits: list[Hit], starts: np.ndarray, scores: np.ndarray) -> list[Found]:
    """One violation per type, from the hit of that type over the strongest window, MIN_GAP_SEC apart."""
    best: dict[str, tuple[float, Hit, float | None]] = {}
    for hit in hits:
        score, centre = window_score(hit, starts, scores)
        if hit.type_id not in best or score > best[hit.type_id][0]:  # a tie keeps the earlier hit
            best[hit.type_id] = (score, hit, centre)
    found: list[Found] = []
    for score, hit, centre in sorted(best.values(), key=lambda b: (-b[0], b[1].start_sec)):
        onset = timecode_kind(hit.type_id) == "onset"
        second = hit.start_sec if onset or centre is None else centre
        if any(abs(second - f.timecode_sec) < MIN_GAP_SEC for f in found):
            continue
        facts = {"object": OBJECT_NAMES[hit.label]} if hit.type_id in NEEDS_OBJECT and hit.label else {}
        found.append(
            Found(
                hit.type_id,
                second,
                "state" if onset else "action",
                facts,
                score,
                hit.start_sec,
                centre or hit.start_sec,
            )
        )
    return found
