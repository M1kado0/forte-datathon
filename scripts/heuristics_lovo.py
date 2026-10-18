"""Score the type heuristics (#33) on the labelled videos, per type.

Heuristic thresholds are not fitted, so a config is scored once on all videos. `nested` repeats the pick:
for each video, the config is chosen on the other videos (recall first, then fewest false positives) and
applied to the held-out one, which is the leave-one-video-out number that does not reuse the video it is
scored on.

A hit counts for a label when the hit's type is the labelled type and [start - WINDOW, end + WINDOW] holds the
label second. A false positive is a clean video with any hit of the type. Tracks come from
scripts/standin_tracks.py until #32 provides them.

Run: python -m scripts.heuristics_lovo report   (defaults, as in docs of the PR)
"""

import csv
import datetime as dt_
import json
import sys
from dataclasses import dataclass, field
from functools import cache, lru_cache

import numpy as np
import pandas as pd

from scripts.standin_tracks import build_tracks
from src.config import DATA_PROCESSED, EXPERIMENTS, LABELS_CSV
from src.heuristics import Hit, Thresholds, detect
from src.heuristics.thresholds import DEFAULT
from src.heuristics.vote import VOTING_TYPES, vote

WINDOW = 3.0
RAW_DIR = DATA_PROCESSED / "standin"
TYPES = {
    "cash_on_desk_next_client": 2,
    "outsider_in_cash_room": 3,
    "foreign_object_at_desk": 5,
    "valuables_left_on_desk": 6,
    "id_not_presented": 7,
    "client_left_during_count": 8,
}


@lru_cache(maxsize=1)
def labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_CSV, dtype={"second": "Int64"})


@cache
def raw(video_id: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{video_id}.parquet")


def video_ids() -> list[str]:
    return sorted(labels()["video_id"].unique(), key=lambda v: int(v.split("-")[1]))


@cache
def tracks(video_id: str, build: tuple) -> pd.DataFrame:
    kwargs = dict(build)
    if "object_min" in kwargs:
        kwargs["object_min"] = dict(kwargs["object_min"])
    return build_tracks(raw(video_id), **kwargs)


def hits_of(video_id: str, th: Thresholds, build: tuple = ()) -> list[Hit]:
    df = tracks(video_id, build)
    return detect(df, thresholds=th) if len(df) else []


@dataclass
class Outcome:
    """What one config did to one video, for one type."""

    video_id: str
    positives: list[int] = field(default_factory=list)  # labelled seconds of the type
    matched: list[tuple[int, float]] = field(default_factory=list)  # (second, |dt|)
    fired: bool = False  # any hit of the type
    clean: bool = False


def score_video(video_id: str, hits: list[Hit]) -> dict[str, Outcome]:
    rows = labels()[labels()["video_id"] == video_id]
    clean = bool((rows["verdict"] == "no_violation").all())
    out = {}
    for type_id in TYPES:
        o = Outcome(video_id, clean=clean)
        mine = [h for h in hits if h.type_id == type_id]
        o.fired = bool(mine)
        for sec in rows.loc[rows["violation_type"] == type_id, "second"].dropna().astype(int):
            o.positives.append(sec)
            near = [h for h in mine if h.start_sec - WINDOW <= sec <= h.end_sec + WINDOW]
            if near:
                o.matched.append((sec, min(abs(h.start_sec - sec) for h in near)))
        out[type_id] = o
    return out


def evaluate(
    th: Thresholds, build: tuple = (), videos: list[str] | None = None
) -> dict[str, dict[str, Outcome]]:
    return {v: score_video(v, hits_of(v, th, build)) for v in videos or video_ids()}


def tally(result: dict[str, dict[str, Outcome]], type_id: str) -> tuple[int, int, int, int]:
    """(true positives, positives, false positives, clean videos) of one type."""
    outs = [r[type_id] for r in result.values()]
    return (
        sum(len(o.matched) for o in outs),
        sum(len(o.positives) for o in outs),
        sum(o.fired for o in outs if o.clean),
        sum(o.clean for o in outs),
    )


def fmt(result: dict[str, dict[str, Outcome]], type_id: str) -> str:
    tp, pos, fp, clean = tally(result, type_id)
    return f"TP {tp}/{pos} FP {fp}/{clean}"


def rank(result: dict[str, dict[str, Outcome]], type_id: str, exclude: str | None = None) -> tuple:
    """Higher is better: recall first, then fewer false positives."""
    sub = {v: r for v, r in result.items() if v != exclude}
    tp, pos, fp, _ = tally(sub, type_id)
    return (tp / pos if pos else 0.0, -fp)


def nested(results: dict[str, dict[str, dict[str, Outcome]]], type_id: str) -> tuple[int, int, int, int]:
    """Leave-one-video-out pick over named configs: tally of the held-out outcomes of each video's pick."""
    held = {}
    for v in video_ids():
        best = max(results, key=lambda name: rank(results[name], type_id, exclude=v))
        held[v] = results[best][v]
    return tally(held, type_id)


def vote_scores(hits_by_video: dict[str, list[Hit]], n_sec: float, types=None) -> tuple[int, int, int, int]:
    """(violation videos called, violation videos, clean videos called, clean videos) of the vote."""
    lab = labels()
    clean = set(lab.loc[lab["verdict"] == "no_violation", "video_id"])
    kw = {} if types is None else {"types": types}
    called = {v: vote(h, n_sec, **kw) for v, h in hits_by_video.items()}
    return (
        sum(c for v, c in called.items() if v not in clean),
        sum(v not in clean for v in called),
        sum(c for v, c in called.items() if v in clean),
        sum(v in clean for v in called),
    )


def median_dt(result: dict[str, dict[str, Outcome]], type_id: str) -> float | None:
    dts = [dt for r in result.values() for _, dt in r[type_id].matched]
    return float(np.median(dts)) if dts else None


def log_rows(
    note: str, params: dict, recall: float, fp_rate: float, dt: float | None, acc: float | None = None
):
    """Append one row to experiments.csv, keyed by the file's own header."""
    with EXPERIMENTS.open(newline="") as f:
        header = next(csv.reader(f))
    row = {
        "ts": dt_.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "params": json.dumps(params, ensure_ascii=False),
        "video_recall": round(recall, 4),
        "fp_rate": round(fp_rate, 4),
        "median_abs_dt": "" if dt is None else round(dt, 2),
        "type_accuracy": "" if acc is None else round(acc, 4),
    }
    with EXPERIMENTS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, restval="").writerow(row)


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "report":
        raise SystemExit(__doc__)
    log = "--log" in argv
    result = evaluate(DEFAULT)
    print(f"stand-in tracks, {len(result)} videos, thresholds {DEFAULT}")
    for type_id, number in TYPES.items():
        tp, pos, fp, clean = tally(result, type_id)
        print(
            f"type {number} {type_id}: TP {tp}/{pos} FP {fp}/{clean} median|dt| {median_dt(result, type_id)}"
        )
        if log:
            log_rows(
                f"heuristic-type{number}-standin-tracks",
                {
                    "type": type_id,
                    "tp": tp,
                    "positives": pos,
                    "fp": fp,
                    "clean": clean,
                    "tracks": "greedy IoU 0.3, age 8",
                },
                tp / pos if pos else 0.0,
                fp / clean,
                median_dt(result, type_id),
            )
    hits = {v: hits_of(v, DEFAULT) for v in video_ids()}
    for n_sec in (3.0, 4.0, 5.0):
        for name, types in (("voting types", None), ("all types", set(TYPES))):
            tpv, pv, fpv, cv = vote_scores(hits, n_sec, types)
            print(f"vote n_sec={n_sec:g} {name}: violation videos {tpv}/{pv}, clean videos called {fpv}/{cv}")
            if log and types is None:
                log_rows(
                    "heuristic-vote-standin-tracks",
                    {"n_sec": n_sec, "voting": sorted(VOTING_TYPES)},
                    tpv / pv,
                    fpv / cv,
                    None,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
