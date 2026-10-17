"""Type heuristics: tracks in, violation types out. Silent (empty list) when no condition holds.

`detect` only decides which type fits and when. The choice between heuristic, kNN and fallback is #34's.
Thresholds below are first guesses; #34 tunes them on leave-one-video-out.

Known limits, for #32 and #34 (no id merging is built before there is real tracker output):
- Anyone who is not the cashier is "someone else": clients and incassators are not told apart.
- If the tracker splits one person into two ids away from the seat, the second id reads as a new person.
- The cashier is the stillest of the longest-present people; a still client present for most of a clip in
  which the cashier is mostly away can be picked instead.
- A detector miss on the cashier of MIN_AWAY_SEC or more reads as "left".
"""

from dataclasses import dataclass
from itertools import pairwise
from math import ceil

import numpy as np
import pandas as pd

from src.heuristics.roles import Roles, find_roles
from src.heuristics.tracks import with_geometry

WORKPLACE_LEFT = "workplace_left"  # type 1, ids match src/rules/rules.yaml
CASH_ON_DESK_NEXT_CLIENT = "cash_on_desk_next_client"  # type 2
COUNT_WITHOUT_CONTROLLER = "count_without_controller"  # type 4

# Distances are in cashier box heights (Roles.distance).
SEAT_RADIUS = 0.75  # a person this close to the seat is the cashier
DESK_RADIUS = 2.0  # cash this close to the seat lies on the cashier's desk
COUNTER_RADIUS = 2.0  # a person this close to the seat is at the counter, not just somewhere in the frame
MIN_AWAY_SEC = 3.0  # type 1: shorter absences are a step to the shelf, not leaving the workplace
MIN_HOLD_SEC = 3.0  # type 2: cash must stay this long after the next person arrives
MIN_COUNT_SEC = 3.0  # type 4: shorter handling of cash is not counting


@dataclass(frozen=True)
class Hit:
    type_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class _Timeline:
    """One boolean per decoded frame."""

    fps: float
    at_seat: np.ndarray  # someone stands at the cashier's seat
    others: np.ndarray  # a person other than the cashier is in the frame, a client or an incassator
    at_counter: np.ndarray  # such a person is within COUNTER_RADIUS of the seat
    cash: np.ndarray  # cash lies on the cashier's desk
    visits: list[tuple[int, int]]  # (first, last) frame of each person at the counter, by arrival

    def runs(self, mask: np.ndarray, min_sec: float) -> list[tuple[int, int]]:
        """Inclusive (first, last) frame indices of each run of True lasting at least min_sec."""
        edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        found = zip(
            np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist(), strict=True
        )
        return [(a, b) for a, b in found if (b - a + 1) / self.fps >= min_sec]

    def hit(self, type_id: str, run: tuple[int, int]) -> Hit:
        return Hit(type_id, run[0] / self.fps, run[1] / self.fps)


def detect(tracks: pd.DataFrame, fps: float = 4.0) -> list[Hit]:
    roles = find_roles(tracks)
    if roles is None:
        return []
    tl = _timeline(tracks, roles, fps)
    arrived = np.maximum.accumulate(tl.at_seat)  # a cashier who was never at the seat has not left it
    left = arrived & ~tl.at_seat & tl.at_counter & tl.cash
    counting = tl.at_seat & tl.cash & ~tl.others
    return [
        *(tl.hit(WORKPLACE_LEFT, run) for run in tl.runs(left, MIN_AWAY_SEC)),
        *_cash_left_for_next_client(tl),
        *(tl.hit(COUNT_WITHOUT_CONTROLLER, run) for run in tl.runs(counting, MIN_COUNT_SEC)),
    ]


def _timeline(tracks: pd.DataFrame, roles: Roles, fps: float) -> _Timeline:
    d = with_geometry(tracks)
    d = d.assign(
        frame=(d["t"] * fps).round().astype(int), from_seat=lambda x: roles.distance(x["cx"], x["cy"])
    )
    n = int(d["frame"].max()) + 1
    people = d[d["kind"] == "person"]
    near = people["from_seat"] <= SEAT_RADIUS
    is_cashier = people["track_id"] == roles.cashier
    # Someone else at the seat counts only while the cashier's own track is there too; without it, a track
    # at the seat is the cashier coming back under a new id.
    cashier_here = _mask(n, people["frame"][near & is_cashier])[people["frame"].to_numpy()]
    others = people[~is_cashier & (~near | cashier_here)]
    at_counter = others[others["from_seat"] <= COUNTER_RADIUS]
    money = d[d["kind"] == "cash"]
    span = at_counter.groupby("track_id")["frame"].agg(["min", "max"]).sort_values("min")
    return _Timeline(
        fps=fps,
        at_seat=_mask(n, people["frame"][near]),
        others=_mask(n, others["frame"]),
        at_counter=_mask(n, at_counter["frame"]),
        cash=_mask(n, money["frame"][money["from_seat"] <= DESK_RADIUS]),
        visits=list(zip(span["min"].tolist(), span["max"].tolist(), strict=True)),
    )


def _cash_left_for_next_client(tl: _Timeline) -> list[Hit]:
    """The previous person left, the next one arrived, and the cash never left the desk (п. 20)."""
    hold = ceil(MIN_HOLD_SEC * tl.fps)
    cash_runs = tl.runs(tl.cash, 0)
    hits = []
    for (_, previous_last), (next_first, _) in pairwise(tl.visits):
        if next_first <= previous_last:
            continue
        for first, last in cash_runs:
            if first <= previous_last and last >= next_first + hold - 1:
                hits.append(tl.hit(CASH_ON_DESK_NEXT_CLIENT, (next_first, last)))
    return hits


def _mask(n: int, frames: pd.Series) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[frames.to_numpy()] = True
    return mask
