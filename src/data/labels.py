"""Cashier-video labels: normalize the organizers' xlsx into data/labels.csv (TZ-8.5)."""

import re
from pathlib import Path

import pandas as pd

from contracts.data import ARRANGEMENTS, TYPE_IDS, LabelsSchema
from src.config import ARRANGEMENTS_CSV, LABELS_CSV, LABELS_XLSX, TYPES_CSV

# Cyrillic Х and х look identical to Latin X, so they are built from code points, not typed.
CYRILLIC_X = {chr(0x0425): "X", chr(0x0445): "X"}


def normalize_name(name: str) -> str:
    """'Х-47' (Cyrillic), 'X-47' (Latin) and ' х-47 ' all become 'X-47'."""
    return "".join(CYRILLIC_X.get(ch, ch) for ch in name.strip()).upper()


_TIMECODE = re.compile(r"^(?:(?P<min>\d+)\s*минут\w*)?\s*(?:(?P<sec>\d+)\s*секунд\w*)?$")


def parse_timecode(raw: str | None) -> int | None:
    """'1минута 8 секунда' -> 68, '22 секунда' -> 22, empty -> None; anything else raises."""
    if pd.isna(raw):  # None, or the NaN pandas reads from an empty xlsx cell
        return None
    text = raw.strip().lower()
    if not text:
        return None
    m = _TIMECODE.match(text)
    if m is None or (m["min"] is None and m["sec"] is None):
        raise ValueError(f"unrecognized timecode: {raw!r}")
    return int(m["min"] or 0) * 60 + int(m["sec"] or 0)


# The organizers sent the same clip twice under two names; the lower number is canonical (docs/DATA_CARD.md).
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}
VERDICTS = {"нарушение": "violation", "без нарушения": "no_violation"}


def build_labels(
    raw: pd.DataFrame, types: pd.DataFrame | None = None, arrangements: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Raw xlsx rows (file_name, timestamp, verdict_ru) -> one row per label.

    Columns: video_id (normalized, duplicates merged), file (name as distributed on disk),
    verdict (violation | no_violation), second (Int64, null for no_violation).
    With `types` (video_id, second, violation_type; see read_types) the violation_type column is added:
    every violation needs a type and every type row must match a violation, or this raises.
    With `arrangements` (video_id, arrangement; see read_arrangements) the arrangement column is added:
    every video needs one and every row must match a video, or this raises.
    """
    ids = raw["file_name"].map(normalize_name)
    file_of = {i: f"{name.strip()}.mov" for i, name in zip(ids, raw["file_name"], strict=True)}
    verdict_ru = raw["verdict_ru"].str.strip().str.lower()
    unknown = set(verdict_ru) - VERDICTS.keys()
    if unknown:
        raise ValueError(f"unknown verdict text: {sorted(unknown)}")
    canonical = ids.map(lambda i: DUPLICATES.get(i, i))
    out = pd.DataFrame(
        {
            "video_id": canonical,
            "file": canonical.map(file_of),
            "verdict": verdict_ru.map(VERDICTS),
            "second": pd.array([parse_timecode(t) for t in raw["timestamp"]], dtype="Int64"),
        }
    ).drop_duplicates()
    mixed = out.groupby("video_id")["verdict"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"merged videos disagree on verdict: {sorted(mixed[mixed > 1].index)}")
    number = out["video_id"].str.extract(r"(\d+)", expand=False).astype(int)
    out = out.assign(_n=number).sort_values(["_n", "second"]).drop(columns="_n").reset_index(drop=True)
    if types is not None:
        out = _add_types(out, types)
    if arrangements is not None:
        out = _add_arrangements(out, arrangements)
    return LabelsSchema.validate(out)


def _keys(rows: pd.DataFrame) -> list[tuple[str, int]]:
    return sorted(zip(rows["video_id"], rows["second"].astype(int), strict=True))


def _add_types(labels: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    types = types.astype({"second": "Int64"})
    dup = types[types.duplicated(["video_id", "second"], keep=False)]
    if not dup.empty:
        raise ValueError(f"two types for one label: {_keys(dup)}")
    merged = labels.merge(types, on=["video_id", "second"], how="left")
    known = set(zip(labels["video_id"], labels["second"], strict=True))
    orphans = types[[k not in known for k in zip(types["video_id"], types["second"], strict=True)]]
    if not orphans.empty:
        raise ValueError(f"type rows that match no violation label: {_keys(orphans)}")
    missing = merged[(merged["verdict"] == "violation") & merged["violation_type"].isna()]
    if not missing.empty:
        raise ValueError(f"violations without a type: {_keys(missing)}")
    return merged


def _add_arrangements(labels: pd.DataFrame, arrangements: pd.DataFrame) -> pd.DataFrame:
    dup = arrangements[arrangements.duplicated("video_id", keep=False)]
    if not dup.empty:
        raise ValueError(f"two arrangements for one video: {sorted(dup['video_id'].unique())}")
    orphans = sorted(set(arrangements["video_id"]) - set(labels["video_id"]))
    if orphans:
        raise ValueError(f"arrangement rows that match no video: {orphans}")
    missing = sorted(set(labels["video_id"]) - set(arrangements["video_id"]))
    if missing:
        raise ValueError(f"videos without an arrangement: {missing}")
    return labels.merge(arrangements, on="video_id", how="left", validate="many_to_one")


def load_labels(path: Path = LABELS_CSV) -> pd.DataFrame:
    """Read data/labels.csv and validate it against the contract."""
    return LabelsSchema.validate(pd.read_csv(path, dtype={"second": "Int64"}))


def count_types(labels: pd.DataFrame) -> dict[str, int]:
    """Violation labels per type id, in rules.yaml order; a type nobody is labelled with counts 0."""
    counted = labels["violation_type"].value_counts()
    return {type_id: int(counted.get(type_id, 0)) for type_id in TYPE_IDS}


def count_arrangements(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Videos (not label rows) per arrangement and verdict: a video with any violation label counts as violation."""
    per_video = labels.groupby("video_id").agg(
        arrangement=("arrangement", "first"), violation=("verdict", lambda v: (v == "violation").any())
    )
    return {
        name: {
            "violation": int(per_video.loc[per_video["arrangement"] == name, "violation"].sum()),
            "no_violation": int((~per_video.loc[per_video["arrangement"] == name, "violation"]).sum()),
        }
        for name in ARRANGEMENTS
    }


def read_types(path: Path = TYPES_CSV) -> pd.DataFrame:
    """data/violation_types.csv: the team's type per violation label, keyed by (video_id, second)."""
    return pd.read_csv(path, dtype={"second": "Int64"})


def read_arrangements(path: Path = ARRANGEMENTS_CSV) -> pd.DataFrame:
    """data/arrangements.csv: the team's arrangement per video, keyed by canonical video_id."""
    return pd.read_csv(path)


def read_xlsx() -> pd.DataFrame:
    """The organizers' sheet has three columns: file name, timestamp text, verdict text."""
    raw = pd.read_excel(LABELS_XLSX, dtype=str)
    raw.columns = ["file_name", "timestamp", "verdict_ru"]
    return raw


if __name__ == "__main__":
    labels = build_labels(read_xlsx(), read_types(), read_arrangements())
    labels.to_csv(LABELS_CSV, index=False)
    print(f"wrote {LABELS_CSV}: {len(labels)} rows, {labels['video_id'].nunique()} videos")
