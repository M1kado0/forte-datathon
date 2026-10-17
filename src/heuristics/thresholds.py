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
    # type 3: a person other than the cashier, farther than seat_radius but no farther than this from the seat,
    # is behind the desk, in the cash room; clients stand beyond it. The seat itself is the cashier's.
    room_radius: float = 1.25
    min_outsider_sec: float = 3.0  # type 3: shorter presence behind the desk is passing through, not a stay
    min_object_sec: float = 3.0  # types 5 and 6: an object on the desk for less than this is a passing hand
    phone_conf: float = 0.5  # type 5: below this a "cell phone" box is dropped, never called a phone
    min_left_sec: float = 3.0  # type 8: cash that stays on the desk this long after the client left
    # a detector miss up to this long does not end an episode (a missed violation is the worst error)
    gap_sec: float = 1.0
    # how find_roles ranks candidate seats: "track" puts the spot where one track stays longest first, so
    # clients who come one after another do not add up; "boxes" puts the spot with the most person boxes
    # first, so a cashier split into short ids keeps the seat while a client waits longer than each id
    seat_by: Literal["track", "boxes"] = "track"


DEFAULT = Thresholds()
