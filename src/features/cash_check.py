"""One-hour check (TZ-4.4, #19): does D-FINE find the cashier and OWLv2 find the cash on label frames?

Frames are decoded and scored locally; nothing leaves the machine. Run:
    uv run python -m src.features.cash_check [--no-log]
"""

import csv
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS, VIDEOS_DIR, WEIGHTS
from src.data.labels import build_labels, load_labels, read_xlsx
from src.data.video import iter_frames

PERSON_THRESHOLD = 0.5
CASH_THRESHOLD = 0.1  # OWLv2 scores are low by design; 0.1 is the value its authors use for demos
CASH_QUERIES = ["a photo of banknotes", "a stack of cash"]
WINDOW_S = 3.0  # frames from [second - WINDOW_S, second + WINDOW_S] around a violation label
CLEAN_FRAMES = 12  # evenly spaced frames per video without violations
FPS = 2.0
CASHIER_GOAL, CASH_GOAL = 0.90, 0.70  # ticket pass criteria


class FrameScore(NamedTuple):
    person: float  # best D-FINE person score in the frame
    cash: float  # best OWLv2 cash-query score in the frame


@dataclass
class Row:
    video_id: str
    second: int | None
    n_frames: int
    cashier_frac: float  # share of frames with a person >= PERSON_THRESHOLD
    cashier_conf: float  # best person score in the window
    cash_frac: float
    cash_conf: float

    @property
    def cashier_found(self) -> bool:
        return self.cashier_frac > 0

    @property
    def cash_found(self) -> bool:
        return self.cash_frac > 0


def pick_frame_indices(n: int, k: int) -> list[int]:
    """k evenly spaced indices into n items (all of them when n <= k)."""
    if n <= k:
        return list(range(n))
    return sorted({int(i) for i in np.linspace(0, n - 1, k)})


def in_window(seconds: list[float], center: float, half: float = WINDOW_S) -> list[int]:
    return [i for i, s in enumerate(seconds) if abs(s - center) <= half]


def summarize(video_id: str, second: int | None, person: list[float], cash: list[float]) -> Row:
    """person/cash: best score per frame."""
    n = len(person)
    return Row(
        video_id,
        second,
        n,
        sum(s >= PERSON_THRESHOLD for s in person) / n if n else 0.0,
        max(person, default=0.0),
        sum(s >= CASH_THRESHOLD for s in cash) / n if n else 0.0,
        max(cash, default=0.0),
    )


def share(rows: list[Row], attr: str) -> float:
    return sum(getattr(r, attr) for r in rows) / len(rows) if rows else 0.0


class Detectors:
    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before transformers is imported
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            AutoProcessor,
            Owlv2ForObjectDetection,
        )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dfine_proc = AutoImageProcessor.from_pretrained(WEIGHTS / "dfine-small")
        self.dfine = (
            AutoModelForObjectDetection.from_pretrained(WEIGHTS / "dfine-small").to(self.device).eval()
        )
        self.owl_proc = AutoProcessor.from_pretrained(WEIGHTS / "owlv2-base")
        self.owl = Owlv2ForObjectDetection.from_pretrained(WEIGHTS / "owlv2-base").to(self.device).eval()
        self.person_id = next(i for i, n in self.dfine.config.id2label.items() if n == "person")

    def scores(self, frame_bgr: np.ndarray) -> FrameScore:
        """Best person and cash scores for one BGR frame."""
        from PIL import Image

        image = Image.fromarray(frame_bgr[:, :, ::-1])
        size = [(image.height, image.width)]
        with self.torch.no_grad():
            inputs = self.dfine_proc(images=image, return_tensors="pt").to(self.device)
            res = self.dfine_proc.post_process_object_detection(
                self.dfine(**inputs), target_sizes=size, threshold=0.0
            )[0]
            person = res["scores"][res["labels"] == self.person_id]
            person_best = float(person.max()) if len(person) else 0.0

            inputs = self.owl_proc(text=[CASH_QUERIES], images=image, return_tensors="pt").to(self.device)
            out = self.owl(**inputs)
            logits = out.logits.sigmoid()  # (1, boxes, queries)
            cash_best = float(logits.max())
        return FrameScore(person_best, cash_best)


def load_label_table() -> pd.DataFrame:
    try:
        return load_labels()
    except FileNotFoundError:
        return build_labels(read_xlsx())


def frame_seconds(path: Path) -> list[float]:
    """Times of the sampled frames; decodes once and drops each frame, so memory stays flat."""
    return [sec for sec, _ in iter_frames(path, FPS)]


def run() -> list[Row]:
    labels = load_label_table()
    det = Detectors()
    rows: list[Row] = []
    for (video_id, file), group in labels.groupby(["video_id", "file"], sort=False):
        path = VIDEOS_DIR / file
        seconds = frame_seconds(path)
        wanted: dict[tuple[str, int | None], list[int]] = {}
        for second in group["second"]:
            if pd.isna(second):
                wanted[video_id, None] = pick_frame_indices(len(seconds), CLEAN_FRAMES)
            else:
                wanted[video_id, int(second)] = in_window(seconds, int(second))
        needed = {i for idx in wanted.values() for i in idx}
        scores = {i: det.scores(f) for i, (_, f) in enumerate(iter_frames(path, FPS)) if i in needed}
        for (vid, second), idx in wanted.items():
            pairs = [scores[i] for i in idx]
            rows.append(summarize(vid, second, [p.person for p in pairs], [p.cash for p in pairs]))
        print(f"  {video_id}: {len(seconds)} frames sampled, {len(needed)} scored", flush=True)
    return rows


def summarize_rows(rows: list[Row]) -> dict:
    """Totals and the ticket's pass decision (cashier >= 90%, cash >= 70% on violation labels)."""
    viol = [r for r in rows if r.second is not None]
    clean = [r for r in rows if r.second is None]
    s = {
        "n_violation_labels": len(viol),
        "n_clean_videos": len(clean),
        "cashier_found_violation": share(viol, "cashier_found"),
        "cash_found_violation": share(viol, "cash_found"),
        "cashier_found_clean": share(clean, "cashier_found"),
        "cash_found_clean": share(clean, "cash_found"),
    }
    s["passed"] = bool(
        s["cashier_found_violation"] >= CASHIER_GOAL and s["cash_found_violation"] >= CASH_GOAL
    )
    return s


def report(rows: list[Row]) -> dict:
    print(f"\n{'метка':<8}{'сек':>5}{'кадров':>8}{'кассир':>8}{'conf':>7}{'деньги':>8}{'conf':>7}")
    for r in rows:
        sec = "-" if r.second is None else str(r.second)
        print(
            f"{r.video_id:<8}{sec:>5}{r.n_frames:>8}{'да' if r.cashier_found else 'нет':>8}"
            f"{r.cashier_conf:>7.2f}{'да' if r.cash_found else 'нет':>8}{r.cash_conf:>7.2f}"
        )
    s = summarize_rows(rows)
    print(
        f"\nМетки с нарушением ({s['n_violation_labels']}): кассир {s['cashier_found_violation']:.0%}, "
        f"деньги {s['cash_found_violation']:.0%}"
    )
    print(
        f"Видео без нарушений ({s['n_clean_videos']}): кассир {s['cashier_found_clean']:.0%}, "
        f"деньги в кадре {s['cash_found_clean']:.0%}"
    )
    verdict = (
        "проверка пройдена" if s["passed"] else "не пройдена, берём запасной вариант (изменения в зоне стола)"
    )
    print(f"Итог: {verdict}")
    print("Типы 1, 2, 4 в labels.csv не размечены (нет колонки type): цифры по всем меткам нарушений.")
    return s


def log_experiment(summary: dict) -> None:
    """Append one row keyed by the file's own header; the metrics go to params (stub header has no slot)."""
    with EXPERIMENTS.open(newline="") as f:
        header = next(csv.reader(f))
    row = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "cash-check-dfine-owlv2",
        "params": json.dumps(summary, ensure_ascii=False),
    }
    with EXPERIMENTS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, restval="").writerow(row)


if __name__ == "__main__":
    summary = report(run())
    if "--no-log" not in sys.argv:
        log_experiment(summary)
