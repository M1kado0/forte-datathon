"""Tunable numbers of the type heuristics. Defaults are first guesses; #34 tunes them on leave-one-video-out."""

from dataclasses import dataclass


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
    # a detector miss up to this long does not end an episode (a missed violation is the worst error)
    gap_sec: float = 1.0


DEFAULT = Thresholds()
