"""Track format: the contract between the detector/tracker (#32) and the heuristics (#24).

One row per track per decoded frame ("long" format). #32 caches it per video as parquet and must pass
`TRACKS_SCHEMA.validate(df)` before writing. Pixels are in the native 464x848 frame (no downscale, #20).

| column        | type  | meaning                                                                     |
|---------------|-------|-----------------------------------------------------------------------------|
| track_id      | int   | unique within a video across all kinds; may change if the tracker loses someone |
| kind          | str   | "person", "object" (COCO: phone, bag, cup), "cash" or "document"             |
| label         | str   | COCO class name, or "cash" / "document" for the open-vocabulary kinds        |
| t             | float | seconds from the start of the video, on the decoder grid (4 fps by default)  |
| x1,y1,x2,y2   | float | box corners in pixels, x2 > x1 and y2 > y1 (a box with no area is dropped by #32) |
| conf          | float | detector confidence in [0, 1]                                                |

People are boxes only: no faces, no identities (TZ-F.1). "cash" rows are a box around money found by OWLv2
or, if #19 falls back to desk-zone change, a box around the changed region.
"document" rows are a box around an identity document or paper held at the counter, also from OWLv2:
the kind is separate from "object" so that a document is not read as a personal item (type 5).
Type 7 reads silence as "no document shown", so #32 keeps the document prompt on even when it finds
nothing: a video without document rows is one where nobody showed one.
"""

import pandas as pd
import pandera.pandas as pa

KINDS = ("person", "object", "cash", "document")


def _has_area(boxes: pd.DataFrame) -> pd.Series:
    return (boxes["x2"] > boxes["x1"]) & (boxes["y2"] > boxes["y1"])


TRACKS_SCHEMA = pa.DataFrameSchema(
    {
        "track_id": pa.Column(int, pa.Check.ge(0)),
        "kind": pa.Column(str, pa.Check.isin(KINDS)),
        "label": pa.Column(str),
        "t": pa.Column(float, pa.Check.ge(0)),
        "x1": pa.Column(float),
        "y1": pa.Column(float),
        "x2": pa.Column(float),
        "y2": pa.Column(float),
        "conf": pa.Column(float, pa.Check.in_range(0, 1)),
    },
    checks=pa.Check(_has_area, error="box has no area"),
    unique=["track_id", "t"],
    strict=True,
    coerce=True,
)


def with_geometry(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add box centre (cx, cy) and height (h) in pixels. Rows without area are dropped: TRACKS_SCHEMA rejects
    them, but a frame that skipped validation must not crash the heuristics."""
    tracks = tracks[_has_area(tracks)]
    return tracks.assign(
        cx=(tracks["x1"] + tracks["x2"]) / 2,
        cy=(tracks["y1"] + tracks["y2"]) / 2,
        h=tracks["y2"] - tracks["y1"],
    )
