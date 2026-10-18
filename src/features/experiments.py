"""One row in the append-only experiments.csv for runs that have no model metrics (checks, tracks)."""

import csv
import datetime as dt
import json

from src.config import EXPERIMENTS


def append_row(note: str, params: dict) -> None:
    """Append one row keyed by the file's own header; the numbers go to params and the metric columns stay empty
    (the stub header has no slot for them)."""
    with EXPERIMENTS.open(newline="") as f:
        header = next(csv.reader(f))
    row = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "params": json.dumps(params, ensure_ascii=False),
    }
    with EXPERIMENTS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, restval="").writerow(row)
