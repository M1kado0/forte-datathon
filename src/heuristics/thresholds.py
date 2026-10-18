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
    # workplace_left: shorter absences are a step to the shelf, not leaving the workplace
    min_away_sec: float = 3.0
    # cash_on_desk_next_client: cash must stay this long after the next person arrives
    min_hold_sec: float = 3.0
    min_count_sec: float = 3.0  # count_without_controller: shorter handling of cash is not counting
    # id_not_presented: cash and a client together for less than this is not a payout
    min_payout_sec: float = 5.0
    # outsider_in_cash_room: two people at the seat at once. A person other than the cashier, no farther than room_radius from
    # the seat while the cashier's own track is within seat_radius of it, is in the cash room. Image distance
    # cannot tell an outsider beside the seat from a client at the counter, so the cashier has to be there too.
    room_radius: float = 0.75
    # outsider_in_cash_room: a box overlapping the cashier's by this much is the cashier seen twice
    iou_dup: float = 0.5
    # the outsider's box height, in cashier box heights, stays in this band; a much smaller or larger box is
    # a detector artefact, not a second adult
    outsider_h_min: float = 0.9
    outsider_h_max: float = 1.5
    # outsider_in_cash_room: shorter presence behind the desk is passing through
    min_outsider_sec: float = 3.0
    # outsider_in_cash_room: True takes a second person beside the cashier, while cash lies on the desk, for the
    # controller of a count (count_without_controller needs one). On since #34: on the real tracks the labelled
    # outsiders still fire and a hit on a video of another type goes (reports/metrics.json, types.per_type). On the
    # stand-in tracks it hid a labelled outsider, so with other tracks this needs the same check.
    controller_exempt: bool = True
    # foreign_object_at_desk and valuables_left_on_desk, up to book_conf:
    min_object_sec: float = 3.0  # an object on the desk for less than this is a passing hand
    object_radius: float = 0.5  # an object this close to the seat lies on the cashier's desk
    # a box below its class confidence is dropped, a phone is never named on a doubt
    phone_conf: float = 0.4
    cup_conf: float = 0.5
    bottle_conf: float = 0.3
    # valuables_left_on_desk is off (None): a book box is static clutter at conf 0.3-0.5 in the clips, 11 of 19 clean videos fired
    # for one hit, and no setting in the sweep hit X-24. A number switches it on.
    book_conf: float | None = None
    # an object whose centre stays within static_dist (in its own box heights) in static_frac of all
    # frames is furniture or clutter, not something put on the desk. None switches the mask off.
    static_dist: float = 0.15
    static_frac: float | None = None
    # an object track first seen during the first arrive_after_sec was there before the shift started, so it is
    # not "put on the desk". Off (0): an object that lies there from the first frame is still a violation, and a
    # recall-first default does not hide it. The test judges each track, not the whole class.
    arrive_after_sec: float = 0.0
    person_conf: float = 0.5  # a person box below this is dropped before anything else
    cash_min: float = 0.10  # an OWLv2 cash box below this is dropped
    doc_min: float = 0.10  # an OWLv2 document box below this is dropped
    # cash counts at a frame when at least cash_frac of the frames within cash_win_sec around it have cash;
    # 0 falls back to closing detector misses of up to gap_sec
    cash_win_sec: float = 0.0
    cash_frac: float = 0.5
    # client_left_during_count: cash that stays on the desk this long after the client left
    min_left_sec: float = 3.0
    # a detector miss up to this long does not end an episode (a missed violation is the worst error)
    gap_sec: float = 1.0
    # how find_roles ranks candidate seats: "track" puts the spot where one track stays longest first, so
    # clients who come one after another do not add up; "boxes" puts the spot with the most person boxes
    # first, so a cashier split into short ids keeps the seat while a client waits longer than each id
    seat_by: Literal["track", "boxes"] = "track"


DEFAULT = Thresholds()
