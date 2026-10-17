"""Data contracts (pandera). Owned by P1; P4 holds the labels contract (TZ-8.5, ticket allows it),
P3 the tracks contract (#24).

Labels contract (data/labels.csv), one row per label, 40 videos / 41 rows:
  video_id  normalized name, ASCII "X-<number>" (Cyrillic Х and Latin X unified, duplicates merged)
  file      name of the clip as distributed, in data/raw/videos (may start with Cyrillic Х)
  verdict   "violation" | "no_violation" (same words as the API)
  second    whole second of the violation; empty exactly when verdict is "no_violation"
  violation_type  type id from rules.yaml or "[<scenario>]", set by the team after review (ADR 0005);
            empty exactly when verdict is "no_violation". Optional: build_labels() output has no types
A video with two violations has two rows, with different seconds.
"""

import pandas as pd
import pandera.pandas as pa

LabelsSchema = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, pa.Check.str_matches(r"^X-\d+$")),
        "file": pa.Column(str, pa.Check.str_matches(f"^[X{chr(0x0425)}]-\\d+\\.mov$")),
        "verdict": pa.Column(str, pa.Check.isin(["violation", "no_violation"])),
        "second": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "violation_type": pa.Column(
            str, pa.Check.str_matches(r"^([a-z_]+|\[.+\])$"), nullable=True, required=False
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["second"].notna() == (df["verdict"] == "violation"),
            error="second must be set exactly when verdict is violation",
        ),
        pa.Check(
            lambda df: (
                "violation_type" not in df
                or (df["violation_type"].notna() == (df["verdict"] == "violation")).all()
            ),
            error="violation_type must be set exactly when verdict is violation",
        ),
    ],
    unique=["video_id", "second"],
    strict=True,
    coerce=True,
)


# Track format: the contract between the detector/tracker (#32) and the heuristics (#24).
#
# One row per track per decoded frame ("long" format). #32 caches it per video as parquet and must pass
# `TRACKS_SCHEMA.validate(df)` before writing. Pixels are in the native 464x848 frame (no downscale, #20).
#
# | column        | type  | meaning                                                                     |
# |---------------|-------|-----------------------------------------------------------------------------|
# | track_id      | int   | unique within a video across all kinds; may change if the tracker loses someone |
# | kind          | str   | "person", "object" (COCO: phone, bag, cup), "cash" or "document"             |
# | label         | str   | COCO class name, or "cash" / "document" for the open-vocabulary kinds        |
# | t             | float | seconds from the start of the video, on the decoder grid (4 fps by default)  |
# | x1,y1,x2,y2   | float | box corners in pixels, x2 > x1 and y2 > y1 (a box with no area is dropped by #32) |
# | conf          | float | detector confidence in [0, 1]                                                |
#
# People are boxes only: no faces, no identities (TZ-F.1). "cash" rows are a box around money found by OWLv2
# or, if #19 falls back to desk-zone change, a box around the changed region.
# "document" rows are a box around an identity document or paper held at the counter, also from OWLv2:
# the kind is separate from "object" so that a document is not read as a personal item (type 5).
# Type 7 reads silence as "no document shown", so #32 keeps the document prompt on even when it finds
# nothing: a video without document rows is one where nobody showed one.
KINDS = ("person", "object", "cash", "document")


def has_area(boxes: pd.DataFrame) -> pd.Series:
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
    checks=pa.Check(has_area, error="box has no area"),
    unique=["track_id", "t"],
    strict=True,
    coerce=True,
)


# Stub schema for the synthetic demo data. Replace columns once the case data is known.
StubSchema = pa.DataFrameSchema(
    {
        "client_id": pa.Column(str, unique=True),
        "age": pa.Column(int, pa.Check.in_range(18, 90)),
        "income_kzt": pa.Column(float, pa.Check.ge(0)),
        "debt_ratio": pa.Column(float, pa.Check.in_range(0, 5)),
        "days_overdue_max": pa.Column(int, pa.Check.ge(0)),
        "tenure_months": pa.Column(int, pa.Check.ge(0)),
        "region": pa.Column(str),
        "target": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)
