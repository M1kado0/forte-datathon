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
    """The seat is where person boxes pile up most over the clip, whoever they are; the stillest pile wins a tie.

    Everyone at a spot counts, so clients who come one after another at the counter add up and can match or
    outweigh the cashier's time at the seat; then the counter becomes the seat. The cashier is the track most
    present there. A cashier who leaves and returns under a new id still gives one seat. None when no person
    is tracked.
    """
    people = with_geometry(tracks[tracks["kind"] == "person"])
    if people.empty:
        return None
    cx, cy, h = (people[c].to_numpy() for c in ("cx", "cy", "h"))
    seeds = np.unique(np.linspace(0, len(people) - 1, num=min(len(people), MAX_SEEDS)).astype(int))
    from_seed = np.hypot(cx[seeds, None] - cx, cy[seeds, None] - cy) / h[seeds, None]  # seeds x boxes
    near = from_seed <= thresholds.seat_radius
    count = near.sum(axis=1)
    var_x = (near * cx**2).sum(axis=1) / count - ((near * cx).sum(axis=1) / count) ** 2
    var_y = (near * cy**2).sum(axis=1) / count - ((near * cy).sum(axis=1) / count) ** 2
    spread = np.sqrt(np.clip(var_x + var_y, 0, None)) / h[seeds]
    members = near[np.lexsort((spread, -count))[0]]
    ids, per_id = np.unique(people["track_id"].to_numpy()[members], return_counts=True)
    return Roles(
        cashier=int(ids[per_id.argmax()]),
        seat=(float(np.median(cx[members])), float(np.median(cy[members]))),
        scale=float(np.median(h[members])),
    )
