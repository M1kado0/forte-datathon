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
    """The seat is the place where one track has the most frames near it, whoever it is; the stillest place wins a tie.

    Ranking by a single track, not by everyone at the spot, keeps clients who arrive one after another at
    the counter from adding up to a seat. The cost: a cashier who returns under a new id loses the seat to a
    client with more frames there than either fragment, so stable ids are #32's job. The cashier is the track most
    present at the seat. None when no person is tracked.
    """
    people = with_geometry(tracks[tracks["kind"] == "person"])
    if people.empty:
        return None
    cx, cy, h = (people[c].to_numpy() for c in ("cx", "cy", "h"))
    ids, codes = np.unique(people["track_id"].to_numpy(), return_inverse=True)
    seeds = np.unique(np.linspace(0, len(people) - 1, num=min(len(people), MAX_SEEDS)).astype(int))
    from_seed = np.hypot(cx[seeds, None] - cx, cy[seeds, None] - cy) / h[seeds, None]  # seeds x boxes
    near = from_seed <= thresholds.seat_radius
    # for each seed: boxes of the single track that is most present around it
    longest = np.array([np.bincount(codes[row]).max() for row in near])
    n_near = near.sum(axis=1)
    var_x = (near * cx**2).sum(axis=1) / n_near - ((near * cx).sum(axis=1) / n_near) ** 2
    var_y = (near * cy**2).sum(axis=1) / n_near - ((near * cy).sum(axis=1) / n_near) ** 2
    spread = np.sqrt(np.clip(var_x + var_y, 0, None)) / h[seeds]
    best = np.lexsort((spread, -longest))[0]
    members = near[best]
    return Roles(
        cashier=int(ids[np.bincount(codes[members]).argmax()]),
        seat=(float(np.median(cx[members])), float(np.median(cy[members]))),
        scale=float(np.median(h[members])),
    )
