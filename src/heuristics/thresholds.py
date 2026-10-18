"""Tunable numbers and switches of the type heuristics. Defaults are first guesses; #34 tunes them on
leave-one-video-out."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Thresholds:
    # Distances are in cashier box heights, so they hold for any camera angle and zoom.
    seat_radius: float = 0.75  # a person this close to the seat is the cashier
    desk_radius: float = 2.0  # cash this close to the seat lies on the cashier's desk
    counter_radius: float = 2.0  # a person this close to the seat is at the counter, not just in the frame
    min_away_sec: float = 3.0  # type 1: shorter absences are a step to the shelf, not leaving the workplace
    min_hold_sec: float = 3.0  # type 2: cash must stay this long after the next person arrives
    min_count_sec: float = 3.0  # type 4: shorter handling of cash is not counting
    min_payout_sec: float = 5.0  # type 7: cash and a client together for less than this is not a payout
    # type 3: two people at the seat at once. A person other than the cashier, no farther than room_radius from
    # the seat while the cashier's own track is within seat_radius of it, is in the cash room. Image distance
    # cannot tell an outsider beside the seat from a client at the counter, so the cashier has to be there too.
    room_radius: float = 0.75
    iou_dup: float = 0.5  # type 3: a box overlapping the cashier's by this much is the cashier seen twice
    outsider_h_min: float = (
        0.9  # type 3: the outsider's box height, in cashier box heights, stays in this band;
    )
    outsider_h_max: float = 1.5  # a much smaller or larger box is a detector artefact, not a second adult
    min_outsider_sec: float = 3.0  # type 3: shorter presence behind the desk is passing through, not a stay
    min_object_sec: float = 3.0  # types 5 and 6: an object on the desk for less than this is a passing hand
    object_radius: float = 0.5  # types 5 and 6: an object this close to the seat lies on the cashier's desk
    # types 5 and 6: a box below its class confidence is dropped, a phone is never named on a doubt
    phone_conf: float = 0.4
    cup_conf: float = 0.5
    bottle_conf: float = 0.3
    # type 6 is off (None): a book box is static clutter at conf 0.3-0.5 in the clips, 11 of 19 clean videos fired
    # for one hit, and no setting in the sweep hit X-24. A number switches it on.
    book_conf: float | None = None
    # types 5 and 6: an object whose centre stays within static_dist (in its own box heights) in static_frac of all
    # frames is furniture or clutter, not something put on the desk. None switches the mask off.
    static_dist: float = 0.15
    static_frac: float | None = None
    # types 5 and 6: a class seen on the desk during the first arrive_after_sec was there before the shift
    # started, so it is not "put on the desk". 0 switches the test off.
    arrive_after_sec: float = 2.0
    person_conf: float = 0.5  # a person box below this is dropped before anything else
    cash_min: float = 0.10  # an OWLv2 cash box below this is dropped
    doc_min: float = 0.10  # an OWLv2 document box below this is dropped
    # cash counts at a frame when at least cash_frac of the frames within cash_win_sec around it have cash;
    # 0 falls back to closing detector misses of up to gap_sec
    cash_win_sec: float = 0.0
    cash_frac: float = 0.5
    min_left_sec: float = 3.0  # type 8: cash that stays on the desk this long after the client left
    # a detector miss up to this long does not end an episode (a missed violation is the worst error)
    gap_sec: float = 1.0
    # how find_roles ranks candidate seats: "track" puts the spot where one track stays longest first, so
    # clients who come one after another do not add up; "boxes" puts the spot with the most person boxes
    # first, so a cashier split into short ids keeps the seat while a client waits longer than each id
    seat_by: Literal["track", "boxes"] = "track"


DEFAULT = Thresholds()
