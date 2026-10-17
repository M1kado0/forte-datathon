"""Roles from tracks, with no hand-drawn zones (the jury's camera angle has none)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.heuristics.thresholds import DEFAULT, Thresholds
from src.heuristics.tracks import with_geometry

MAX_SEEDS = 1000  # candidate seats tried, spread evenly over all person boxes; keeps memory bounded


@dataclass(frozen=True)
class Roles:
    cashier: int  # track_id most present at the seat
    seat: tuple[float, float]  # median box centre of the people at the seat, in pixels
    scale: float  # median box height of the people at the seat, in pixels

    def distance(self, cx: pd.Series, cy: pd.Series) -> pd.Series:
        """Distance from the seat in cashier box heights, so thresholds hold for any camera angle and zoom."""
        return np.hypot(cx - self.seat[0], cy - self.seat[1]) / self.scale


def find_roles(tracks: pd.DataFrame, thresholds: Thresholds = DEFAULT) -> Roles | None:
    """The seat is the spot where one track stays longest, so clients who come one after another at the
    counter do not add up. A tie goes to the spot with more person boxes, then to the stillest one.

    The cashier is that longest track. A client who stays at the counter longer than the cashier's longest
    track at the seat takes the seat. That happens when the cashier is away for much of the clip, or when the
    tracker splits the cashier into short ids (stable ids are #32's job). None when no person is tracked.
    """
    people = with_geometry(tracks[tracks["kind"] == "person"])
    if people.empty:
        return None
    people = people.sort_values("track_id", kind="stable")  # one block of boxes per track, for reduceat
    cx, cy, h = (people[c].to_numpy() for c in ("cx", "cy", "h"))
    ids, track_starts = np.unique(people["track_id"].to_numpy(), return_index=True)
    seeds = np.unique(np.linspace(0, len(people) - 1, num=min(len(people), MAX_SEEDS)).astype(int))
    from_seed = np.hypot(cx[seeds, None] - cx, cy[seeds, None] - cy) / h[seeds, None]  # seeds x boxes
    near = from_seed <= thresholds.seat_radius
    per_track = np.add.reduceat(near, track_starts, axis=1)  # seeds x tracks: a track's boxes near a seed
    count = near.sum(axis=1)
    var_x = (near * cx**2).sum(axis=1) / count - ((near * cx).sum(axis=1) / count) ** 2
    var_y = (near * cy**2).sum(axis=1) / count - ((near * cy).sum(axis=1) / count) ** 2
    spread = np.sqrt(np.clip(var_x + var_y, 0, None)) / h[seeds]
    best = np.lexsort((spread, -count, -per_track.max(axis=1)))[0]
    members = near[best]
    return Roles(
        cashier=int(ids[per_track[best].argmax()]),
        seat=(float(np.median(cx[members])), float(np.median(cy[members]))),
        scale=float(np.median(h[members])),
    )
