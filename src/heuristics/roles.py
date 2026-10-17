"""Roles from tracks, with no hand-drawn zones (the jury's camera angle has none)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.heuristics.tracks import with_geometry

# A track competes for "cashier" if it is present at least this share as long as the longest person track.
# Clients stand still too, but they come and go; the cashier is there for the whole clip.
LONG_TRACK_SHARE = 0.8


@dataclass(frozen=True)
class Roles:
    cashier: int  # track_id of the cashier's track
    seat: tuple[float, float]  # median box centre of that track, in pixels
    scale: float  # median box height of that track, in pixels

    def distance(self, cx: pd.Series, cy: pd.Series) -> pd.Series:
        """Distance from the seat in cashier box heights, so thresholds hold for any camera angle and zoom."""
        return np.hypot(cx - self.seat[0], cy - self.seat[1]) / self.scale


def find_roles(tracks: pd.DataFrame) -> Roles | None:
    """The cashier is the stillest of the longest-present people. None when no person is tracked."""
    people = with_geometry(tracks[tracks["kind"] == "person"])
    if people.empty:
        return None
    by_track = people.groupby("track_id")
    frames = by_track["t"].size()
    long_tracks = frames[frames >= LONG_TRACK_SHARE * frames.max()].index
    spread = np.hypot(by_track["cx"].std(ddof=0), by_track["cy"].std(ddof=0)) / by_track["h"].median()
    cashier = int(spread[long_tracks].idxmin())
    own = people[people["track_id"] == cashier]
    return Roles(
        cashier=cashier,
        seat=(float(own["cx"].median()), float(own["cy"].median())),
        scale=float(own["h"].median()),
    )
