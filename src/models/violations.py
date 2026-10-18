"""Which violations `analyze` reports, of which type and at which second (#34, ADR 0005, ADR 0002).

Type, in order: a heuristic hit of an allowed type, then kNN over the labelled windows, then the fallback type.
Timecode by scenario: the start of the state for state types, the score peak for actions and the fallback.
Verdict: a window at or above the threshold, OR a long enough hit of a voting type. The vote only raises the
verdict. Pure functions: `train` measures them on leave-one-video-out and saves the settings that won.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from src.data.evidence import Kind
from src.heuristics import Hit, vote
from src.heuristics.detect import (
    CASH_LEFT_ON_DESK,
    CASH_ON_DESK_NEXT_CLIENT,
    CLIENT_LEFT_DURING_COUNT,
    FOREIGN_OBJECT_AT_DESK,
    ID_NOT_PRESENTED,
    LAPTOP_LEFT_OPEN,
    OUTSIDER_IN_CASH_ROOM,
    VALUABLES_LEFT_ON_DESK,
)
from src.heuristics.thresholds import DEFAULT, Thresholds
from src.models.windows import WINDOW_SEC, timecode
from src.rules.describe import OBJECT_NAMES

FALLBACK_TYPE = "other"  # rules.yaml `fallback: true`
MAX_VIOLATIONS = 2
MIN_GAP_SEC = 5.0

# Every type `detect` can emit.
HEURISTIC_TYPES = (
    CASH_LEFT_ON_DESK,
    CASH_ON_DESK_NEXT_CLIENT,
    LAPTOP_LEFT_OPEN,
    ID_NOT_PRESENTED,
    OUTSIDER_IN_CASH_ROOM,
    FOREIGN_OBJECT_AT_DESK,
    VALUABLES_LEFT_ON_DESK,
    CLIENT_LEFT_DURING_COUNT,
)

# The heuristic thresholds `train` chooses between (inside the folds, see evaluate.fit_settings); the first is
# the default and wins ties. controller_exempt silences the outsider type while cash lies on the desk, because
# a second worker beside the cashier during a count is the controller, not an outsider.
HEURISTIC_CONFIGS: dict[str, Thresholds] = {
    "default": DEFAULT,
    "controller_exempt": replace(DEFAULT, controller_exempt=True),
}

# State types stand while they last: the timecode is the start of the state and the evidence frames follow it.
# These are the scenarios the ticket lists (keys, safe, door, laptop, object, cash on the desk, outsider). Every
# other type, the fallback included, is an action, a moment: the timecode is the score peak (client left, collector,
# cash put away, detector, identification, handing over). `train` checks both choices by median |Δt|
# (metrics.json, timecode_choice).
STATE_TYPES = frozenset(
    {
        "keys_left_on_safe",
        "safe_left_open",
        "cash_room_door_open",
        LAPTOP_LEFT_OPEN,
        CASH_LEFT_ON_DESK,
        VALUABLES_LEFT_ON_DESK,
        FOREIGN_OBJECT_AT_DESK,
        CASH_ON_DESK_NEXT_CLIENT,
        OUTSIDER_IN_CASH_ROOM,
    }
)
# The description of these needs the object a heuristic saw, which a window embedding does not give.
NEEDS_OBJECT = frozenset({FOREIGN_OBJECT_AT_DESK, VALUABLES_LEFT_ON_DESK})


@dataclass(frozen=True)
class Vote:
    """Heuristics' vote on the verdict: a hit of one of `types` lasting at least `n_sec`."""

    enabled: bool = False
    n_sec: float = 3.0
    types: frozenset[str] = frozenset()


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
    knn: Knn | None = None  # the labelled windows to look a type up in, with the k and share chosen for them
    heuristics: str = "default"  # key of HEURISTIC_CONFIGS


@dataclass(frozen=True)
class Found:
    type_id: str
    timecode_sec: float
    kind: (
        Kind  # "state": timecode is the start and the frames follow it; "action": the peak, frames around it
    )
    facts: dict[str, str] = field(default_factory=dict)  # fills the description template
    score: float = 0.0  # branch A's strongest window over the violation, for ordering only
    # Both candidate timecodes, so `train` can compare them; timecode_sec is one of them.
    onset_sec: float = 0.0
    peak_sec: float = 0.0
    from_hit: bool = False  # a heuristic saw it; False = the classifier's peak window (or no window at all)


def evidence_kind(type_id: str) -> Kind:
    return "state" if type_id in STATE_TYPES else "action"


def peak_feature(scores: np.ndarray, feats: np.ndarray) -> np.ndarray | None:
    """Features of the best-scoring window, what kNN looks the type up by; None for a clip with no windows."""
    return feats[int(np.argmax(scores))] if len(scores) else None


def decide(
    starts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    hits: list[Hit],
    settings: Settings,
    peak_feature: np.ndarray | None = None,
) -> list[Found]:
    """The violations of one video, strongest first; [] means no violation. `hits` come from the thresholds
    named by `settings.heuristics`."""
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
    type_id = knn_type(peak_feature, settings.knn)
    type_id = type_id or FALLBACK_TYPE
    centre = float(starts[peak] + WINDOW_SEC / 2)
    onset = timecode(starts, scores, threshold, "onset")  # not None: only a flagged video gets here
    # No hit, so no start of a state: the timecode is the peak for kNN's type too, and the frames follow it.
    return [Found(type_id, centre, "action", {}, float(scores[peak]), onset, centre)]


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
        kind = evidence_kind(hit.type_id)
        second = hit.start_sec if kind == "state" or centre is None else centre
        if any(abs(second - f.timecode_sec) < MIN_GAP_SEC for f in found):
            continue
        facts = {"object": OBJECT_NAMES[hit.label]} if hit.type_id in NEEDS_OBJECT and hit.label else {}
        found.append(
            Found(
                hit.type_id, second, kind, facts, score, hit.start_sec, centre or hit.start_sec, from_hit=True
            )
        )
    return found
