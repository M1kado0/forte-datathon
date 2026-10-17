"""Data contracts (pandera). Owned by P1; P4 holds the labels contract below (TZ-8.5, ticket allows it).

Labels contract (data/labels.csv), one row per label, 40 videos / 41 rows:
  video_id  normalized name, ASCII "X-<number>" (Cyrillic Х and Latin X unified, duplicates merged)
  file      name of the clip as distributed, in data/raw/videos (may start with Cyrillic Х)
  verdict   "violation" | "no_violation" (same words as the API)
  second    whole second of the violation; empty exactly when verdict is "no_violation"
A video with two violations has two rows, with different seconds.
"""

import pandera.pandas as pa

LabelsSchema = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, pa.Check.str_matches(r"^X-\d+$")),
        "file": pa.Column(str, pa.Check.str_matches(f"^[X{chr(0x0425)}]-\\d+\\.mov$")),
        "verdict": pa.Column(str, pa.Check.isin(["violation", "no_violation"])),
        "second": pa.Column("Int64", pa.Check.ge(0), nullable=True),
    },
    checks=pa.Check(
        lambda df: df["second"].notna() == (df["verdict"] == "violation"),
        error="second must be set exactly when verdict is violation",
    ),
    unique=["video_id", "second"],
    strict=True,
    coerce=True,
)
