"""Box geometry for the heuristics. The track format itself is TRACKS_SCHEMA in contracts/data.py."""

import pandas as pd

from contracts.data import has_area


def with_geometry(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add box centre (cx, cy) and height (h) in pixels. Rows without area are dropped: TRACKS_SCHEMA rejects
    them, but a frame that skipped validation must not crash the heuristics."""
    tracks = tracks[has_area(tracks)]
    return tracks.assign(
        cx=(tracks["x1"] + tracks["x2"]) / 2,
        cy=(tracks["y1"] + tracks["y2"]) / 2,
        h=tracks["y2"] - tracks["y1"],
    )
