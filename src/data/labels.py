"""Cashier-video labels: normalize the organizers' xlsx into data/labels.csv (TZ-8.5)."""

import re
from pathlib import Path

import pandas as pd

from contracts.data import LabelsSchema
from src.config import LABELS_CSV, LABELS_XLSX

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


def build_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw xlsx rows (file_name, timestamp, verdict_ru) -> one row per label.

    Columns: video_id (normalized, duplicates merged), file (name as distributed on disk),
    verdict (violation | no_violation), second (Int64, null for no_violation).
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
    return LabelsSchema.validate(out)


def load_labels(path: Path = LABELS_CSV) -> pd.DataFrame:
    """Read data/labels.csv and validate it against the contract."""
    return LabelsSchema.validate(pd.read_csv(path, dtype={"second": "Int64"}))


def read_xlsx() -> pd.DataFrame:
    """The organizers' sheet has three columns: file name, timestamp text, verdict text."""
    raw = pd.read_excel(LABELS_XLSX, dtype=str)
    raw.columns = ["file_name", "timestamp", "verdict_ru"]
    return raw


if __name__ == "__main__":
    labels = build_labels(read_xlsx())
    labels.to_csv(LABELS_CSV, index=False)
    print(f"wrote {LABELS_CSV}: {len(labels)} rows, {labels['video_id'].nunique()} videos")
