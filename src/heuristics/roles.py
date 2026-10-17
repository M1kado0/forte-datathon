"""Roles from tracks, with no hand-drawn zones (the jury's camera angle has none)."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from src.data.video import DEFAULT_FPS
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


def find_roles(
    tracks: pd.DataFrame, thresholds: Thresholds = DEFAULT, fps: float = DEFAULT_FPS
) -> Roles | None:
    """The seat is the spot where one track stays longest, so clients who come one after another at the
    counter do not add up. A tie goes to the spot with more person boxes, then to the stillest one.

    A track's time at a spot counts detector misses up to Thresholds.gap_sec as present, as `detect` does,
    so a half-hidden cashier is not outweighed by a fully visible client.

    The cashier is that longest track. A client who stays at the counter longer than the cashier's longest
    track at the seat takes the seat. That happens when the cashier is away for much of the clip, or when the
    tracker splits the cashier into short ids (stable ids are #32's job). None when no person is tracked.
    """
    people = with_geometry(tracks[tracks["kind"] == "person"])
    if people.empty:
        return None
    people = people.sort_values(["track_id", "t"], kind="stable")  # one block per track, in time order
    cx, cy, h = (people[c].to_numpy() for c in ("cx", "cy", "h"))
    track = people["track_id"].to_numpy()
    frame = (people["t"].to_numpy() * fps).round().astype(np.int32)
    ids, track_starts = np.unique(track, return_index=True)
    seeds = np.unique(np.linspace(0, len(people) - 1, num=min(len(people), MAX_SEEDS)).astype(int))
    from_seed = np.hypot(cx[seeds, None] - cx, cy[seeds, None] - cy) / h[seeds, None]  # seeds x boxes
    near = from_seed <= thresholds.seat_radius
    held = near + _missed_frames(near, track, frame, ceil(thresholds.gap_sec * fps))  # seeds x boxes
    per_track = np.add.reduceat(held, track_starts, axis=1)  # seeds x tracks: frames a track is near a seed
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


def _missed_frames(near: np.ndarray, track: np.ndarray, frame: np.ndarray, max_gap: int) -> np.ndarray:
    """Per seed and box, the frames missed just before the box, when the same track was near the seed on both
    sides of a gap of at most max_gap frames. Boxes are sorted by track, then time."""
    index = np.arange(near.shape[1], dtype=np.int32)
    last_near = np.maximum.accumulate(np.where(near, index, -1), axis=1)
    # The last near box before each box, -1 when there is none.
    previous = np.pad(last_near[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    gap = frame - frame[previous] - 1
    closed = near & (previous >= 0) & (track[previous] == track) & (gap <= max_gap)
    return np.where(closed, gap, 0)
