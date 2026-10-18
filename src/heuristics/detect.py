"""Type heuristics: tracks in, violation types out. Silent (empty list) when no condition holds.

`detect` only decides which type fits and when. The choice between heuristic, kNN and fallback is #34's.

Known limits, for #32 and #34 (no id merging is built before there is real tracker output):
- Anyone who is not the cashier is "someone else": clients and incassators are not told apart.
- Type 4 does not see counting. It fires on cashier at the seat + cash on the desk + nobody else in the
  frame, so a cashier sitting idle next to cash fires it too.
- Anyone at the seat counts as the cashier while the cashier's own track is absent. A client standing at
  the seat while the cashier is away hides type 1; a second worker who sits in the empty seat looks like
  the cashier, so type 4 can fire with "no second worker in the frame". This is the price of tolerating a
  cashier who returns under a new track id.
- If the tracker splits one person into two ids away from the seat, the second id reads as a new person.
- A visit is the first to last frame of one id at the counter. A client who leaves and comes back under the
  same id is one long visit, so the client served in between reads as a passer-by and type 2 is missed.
- The seat is where one track stays longest. A client who stays at the counter longer than the cashier's
  longest track at the seat takes the seat, and a real type 1 is reported as type 4. That happens when the
  cashier is away for much of the clip, or when the tracker splits the cashier into short ids (stable ids
  are #32's job).
- A detector miss on the cashier longer than Thresholds.gap_sec reads as "left".
- Type 3 is "a second person within Thresholds.room_radius of the seat while the cashier is at it, as tall as the cashier". An incassator or a client who
  leans over the desk reads as an outsider. The values rest on two positives (X-18, X-30) and are for #34 to recheck.
- Types 5 and 6 see an object within desk_radius of the seat, so a client's cup at the counter counts. Type 6
  is a book on the desk: the safe is not detected, so "outside the safe" is the same thing.
- Type 8 fires on cash left on the desk after the client went, with the cashier at the seat and nobody next
  at the counter. A cashier who forgets cash on the desk reads as that too (X-26, which has no heuristic).
- Type 7 does not see the identification itself, only whether a document shows up at the counter while
  cash is there. A document held below the counter, out of frame or shown as a digital document on a
  phone fires it too, and so does a document prompt missing from #32. It needs the closest look on LOVO.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from src.data.video import DEFAULT_FPS
from src.heuristics.roles import Roles, find_roles
from src.heuristics.thresholds import DEFAULT, Thresholds
from src.heuristics.tracks import with_geometry

WORKPLACE_LEFT = "workplace_left"  # type 1, ids match src/rules/rules.yaml
CASH_ON_DESK_NEXT_CLIENT = "cash_on_desk_next_client"  # type 2
COUNT_WITHOUT_CONTROLLER = "count_without_controller"  # type 4
ID_NOT_PRESENTED = "id_not_presented"  # type 7
OUTSIDER_IN_CASH_ROOM = "outsider_in_cash_room"  # type 3
FOREIGN_OBJECT_AT_DESK = "foreign_object_at_desk"  # type 5
VALUABLES_LEFT_ON_DESK = "valuables_left_on_desk"  # type 6
CLIENT_LEFT_DURING_COUNT = "client_left_during_count"  # type 8

# COCO classes looked for on the desk, and the Russian name a description gets for {object}.
# The name of the object only: whose it is and what it is worth is not visible (src/rules/rules.yaml).
OBJECT_NAMES = {"cup": "чашка", "cell phone": "телефон", "bottle": "бутылка", "book": "книга"}
FOREIGN_OBJECTS = ("cup", "cell phone", "bottle")
VALUABLES_OBJECTS = ("book",)


@dataclass(frozen=True)
class Hit:
    type_id: str
    start_sec: float
    end_sec: float
    label: str | None = None  # COCO class of the object, for the types about an object on the desk


@dataclass(frozen=True)
class _Timeline:
    """One boolean per decoded frame."""

    fps: float
    at_seat: np.ndarray  # someone stands at the cashier's seat
    others: np.ndarray  # a person other than the cashier is in the frame, a client or an incassator
    at_counter: np.ndarray  # such a person is within counter_radius of the seat
    cash: np.ndarray  # cash lies on the cashier's desk
    document: np.ndarray  # an identity document or paper is visible at the counter
    visits: list[tuple[int, int]]  # (first, last) frame of each person at the counter, by arrival
    outsider: np.ndarray  # a person other than the cashier stands beside the seat while the cashier is there
    objects: dict[str, np.ndarray]  # per COCO class: an object of it lies on the cashier's desk

    def runs(self, mask: np.ndarray, min_sec: float) -> list[tuple[int, int]]:
        """Inclusive (first, last) frame indices of each run of True lasting at least min_sec."""
        edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        found = zip(
            np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist(), strict=True
        )
        return [(a, b) for a, b in found if (b - a + 1) / self.fps >= min_sec]

    def hit(self, type_id: str, run: tuple[int, int], label: str | None = None) -> Hit:
        return Hit(type_id, run[0] / self.fps, run[1] / self.fps, label)

    def objects_on_desk(self, type_id: str, labels: tuple[str, ...], min_sec: float) -> list[Hit]:
        """Each run of at least min_sec with an object of the class on the desk, one hit per class."""
        return [
            self.hit(type_id, run, label)
            for label in labels
            if label in self.objects
            for run in self.runs(self.objects[label], min_sec)
        ]

    def client_left_during_count(self, min_sec: float) -> list[Hit]:
        """Cash lies on the desk with the cashier at the seat, the person at the counter leaves, and the cash
        stays at least min_sec with nobody at the counter (п. 20, часть первая). If the next person steps up
        before the cash is gone, it is a client change and type 2's business.
        """
        alone = self.cash & self.at_seat & ~self.at_counter
        served = self.cash & self.at_seat & self.at_counter
        hits = []
        for _, last in self.visits:
            if not served[last]:
                continue
            for first, end in self.runs(alone, min_sec):
                next_stepped_up = end + 1 < len(alone) and self.at_counter[end + 1]
                if first == last + 1 and not next_stepped_up:
                    hits.append(self.hit(CLIENT_LEFT_DURING_COUNT, (first, end)))
        return hits

    def cash_left_for_next_client(self, hold_sec: float) -> list[Hit]:
        """The previous person left, the next one arrived, and the cash never left the desk (п. 20).

        The previous person is the last to leave of those who came before the next one and left before them,
        so someone who comes and goes during a visit is not taken for it. The next person often steps up
        before the previous one leaves, so the client changes at the later of the two moments.
        """
        hold = ceil(hold_sec * self.fps)
        cash_runs = self.runs(self.cash, 0)
        hits = []
        for next_first, next_last in self.visits:
            earlier_exits = [last for first, last in self.visits if first < next_first and last < next_last]
            if not earlier_exits:
                continue
            previous_last = max(earlier_exits)
            changed_at = max(next_first, previous_last + 1)
            for first, last in cash_runs:
                if first <= previous_last and last >= changed_at + hold - 1:
                    hits.append(self.hit(CASH_ON_DESK_NEXT_CLIENT, (changed_at, last)))
        return hits

    def id_not_presented(self, min_sec: float) -> list[Hit]:
        """Cash lies in front of a client for a whole payout and no document is shown (п. 31 подп. 1)).

        One visit, one verdict: a document seen at any moment of the visit clears the whole visit, because
        the client shows it once and the пункт asks for one identification per operation.
        """
        hits = []
        for first, last in self.visits:
            if self.document[first : last + 1].any():
                continue
            visit = np.zeros(len(self.cash), dtype=bool)
            visit[first : last + 1] = True
            hits += [self.hit(ID_NOT_PRESENTED, run) for run in self.runs(visit & self.cash, min_sec)]
        return hits


def detect(tracks: pd.DataFrame, fps: float = DEFAULT_FPS, thresholds: Thresholds = DEFAULT) -> list[Hit]:
    tracks = _confident(tracks, thresholds)
    roles = find_roles(tracks, thresholds, fps)
    if roles is None:
        return []
    tl = _timeline(tracks, roles, fps, thresholds)
    arrived = np.maximum.accumulate(tl.at_seat)  # a cashier who was never at the seat has not left it
    left = arrived & ~tl.at_seat & tl.at_counter & tl.cash
    counting = tl.at_seat & tl.cash & ~tl.others
    return [
        *(tl.hit(WORKPLACE_LEFT, run) for run in tl.runs(left, thresholds.min_away_sec)),
        *tl.cash_left_for_next_client(thresholds.min_hold_sec),
        *(tl.hit(COUNT_WITHOUT_CONTROLLER, run) for run in tl.runs(counting, thresholds.min_count_sec)),
        *tl.id_not_presented(thresholds.min_payout_sec),
        *(tl.hit(OUTSIDER_IN_CASH_ROOM, run) for run in tl.runs(tl.outsider, thresholds.min_outsider_sec)),
        *tl.objects_on_desk(FOREIGN_OBJECT_AT_DESK, FOREIGN_OBJECTS, thresholds.min_object_sec),
        *tl.objects_on_desk(VALUABLES_LEFT_ON_DESK, VALUABLES_OBJECTS, thresholds.min_object_sec),
        *tl.client_left_during_count(thresholds.min_left_sec),
    ]


def _confident(tracks: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Drop the boxes below their kind's (or object class's) confidence floor."""
    floor = {
        "person": th.person_conf,
        "cash": th.cash_min,
        "document": th.doc_min,
    }
    off = 2.0  # above any confidence: the class is switched off
    by_class = {
        "cell phone": th.phone_conf,
        "cup": th.cup_conf,
        "bottle": th.bottle_conf,
        "book": off if th.book_conf is None else th.book_conf,
    }
    tracks = tracks.reset_index(drop=True)
    need = tracks["kind"].map(floor).fillna(0.0)
    is_object = tracks["kind"] == "object"
    need = need.where(~is_object, tracks["label"].map(by_class).fillna(0.0))
    return tracks[tracks["conf"] >= need].reset_index(drop=True)


def _timeline(tracks: pd.DataFrame, roles: Roles, fps: float, th: Thresholds) -> _Timeline:
    d = with_geometry(tracks)
    d = d.assign(
        frame=(d["t"] * fps).round().astype(int), from_seat=lambda x: roles.distance(x["cx"], x["cy"])
    )
    n = int(d["frame"].max()) + 1
    people = d[d["kind"] == "person"]
    near = people["from_seat"] <= th.seat_radius
    is_cashier = people["track_id"] == roles.cashier
    # Someone else at the seat counts only while the cashier's own track is there too; without it, a track
    # at the seat is the cashier coming back under a new id.
    cashier_here = _frames_to_mask(n, people["frame"][near & is_cashier])[people["frame"].to_numpy()]
    others = people[~is_cashier & (~near | cashier_here)]
    at_counter = others[others["from_seat"] <= th.counter_radius]
    money = d[d["kind"] == "cash"]
    papers = d[d["kind"] == "document"]
    span = at_counter.groupby("track_id")["frame"].agg(["min", "max"]).sort_values("min")
    gap = ceil(th.gap_sec * fps)
    cash = _frames_to_mask(n, money["frame"][money["from_seat"] <= th.desk_radius])
    return _Timeline(
        fps=fps,
        at_seat=_close_gaps(_frames_to_mask(n, people["frame"][near]), gap),
        others=_close_gaps(_frames_to_mask(n, others["frame"]), gap),
        at_counter=_close_gaps(_frames_to_mask(n, at_counter["frame"]), gap),
        cash=_dense(cash, round(th.cash_win_sec * fps), th.cash_frac)
        if th.cash_win_sec
        else _close_gaps(cash, gap),
        document=_close_gaps(
            _frames_to_mask(n, papers["frame"][papers["from_seat"] <= th.counter_radius]), gap
        ),
        visits=list(zip(span["min"].tolist(), span["max"].tolist(), strict=True)),
        outsider=_close_gaps(_frames_to_mask(n, _outsider_frames(people, is_cashier, near, roles, th)), gap),
        objects={label: _close_gaps(mask, gap) for label, mask in _objects_on_desk(d, n, fps, th).items()},
    )


def _outsider_frames(
    people: pd.DataFrame, is_cashier: pd.Series, near: pd.Series, roles: Roles, th: Thresholds
) -> pd.Series:
    """Frames of a second person beside the seat while the cashier is there (type 3). A box that overlaps the
    cashier's is the cashier seen twice (a tracker split or a duplicate box), and a box much smaller than the
    cashier's is a client across the counter, not someone on the cashier's side of it."""
    candidates = people[~is_cashier & (people["from_seat"] <= th.room_radius)]
    mine = people[is_cashier & near].drop_duplicates("frame")
    pair = candidates.merge(mine, on="frame", suffixes=("", "_c"))
    inter_w = (pair[["x2", "x2_c"]].min(axis=1) - pair[["x1", "x1_c"]].max(axis=1)).clip(lower=0)
    inter_h = (pair[["y2", "y2_c"]].min(axis=1) - pair[["y1", "y1_c"]].max(axis=1)).clip(lower=0)
    inter = inter_w * inter_h
    area = (pair["x2"] - pair["x1"]) * (pair["y2"] - pair["y1"])
    area_c = (pair["x2_c"] - pair["x1_c"]) * (pair["y2_c"] - pair["y1_c"])
    overlap = inter / (area + area_c - inter)
    h_rel = pair["h"] / roles.scale
    ok = (overlap < th.iou_dup) & h_rel.between(th.outsider_h_min, th.outsider_h_max)
    return pair["frame"][ok]


def _objects_on_desk(d: pd.DataFrame, n: int, fps: float, th: Thresholds) -> dict[str, np.ndarray]:
    """Per COCO class, the frames where an object of it was put on the cashier's desk (types 5 and 6)."""
    things = d[(d["kind"] == "object") & d["label"].isin(OBJECT_NAMES)]
    if th.static_frac is not None:
        things = things[~_is_static(things, n, th.static_dist, th.static_frac)]
    on_desk = things[things["from_seat"] <= th.object_radius]
    found = {}
    for label, rows in on_desk.groupby("label"):
        if th.arrive_after_sec and rows["frame"].min() < th.arrive_after_sec * fps:
            continue  # it was there before the shift started: not put on the desk
        found[label] = _frames_to_mask(n, rows["frame"])
    return found


def _is_static(things: pd.DataFrame, n: int, dist: float, frac: float) -> pd.Series:
    """True for a box whose centre has a box of the same class within `dist` of its own heights in at least
    `frac` of all n frames: something that stays where it is the whole clip."""
    static = pd.Series(False, index=things.index)
    for _, rows in things.groupby("label"):
        cx, cy, h, frame = (rows[c].to_numpy() for c in ("cx", "cy", "h", "frame"))
        close = np.hypot(cx[:, None] - cx, cy[:, None] - cy) <= dist * h[:, None]
        seen = np.array([np.unique(frame[row]).size for row in close])
        static[rows.index] = seen / n >= frac
    return static


def _dense(mask: np.ndarray, window: int, frac: float) -> np.ndarray:
    """True where at least `frac` of the `window` frames centred on a frame are True."""
    window = max(window, 1)
    share = np.convolve(mask.astype(float), np.ones(window) / window, mode="same")
    return share >= frac - 1e-9


def _frames_to_mask(n: int, frames: pd.Series) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[frames.to_numpy()] = True
    return mask


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False no longer than max_gap frames that sit between two Trues."""
    closed = mask.copy()
    on = np.flatnonzero(mask)
    for start, gap in zip(on[:-1], np.diff(on) - 1, strict=True):
        if 0 < gap <= max_gap:
            closed[start + 1 : start + 1 + gap] = True
    return closed
