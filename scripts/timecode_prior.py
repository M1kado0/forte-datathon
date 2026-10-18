"""Timecode prior from where the labels sit in their clips, scored leave-one-video-out (TZ-4.4, fix-cv-model).

    uv run python -m scripts.timecode_prior

The prior guesses `fraction x duration`, with `fraction` the median of label second / clip duration over the
training clips. It is the opt-in TIMECODE_MODE=prior of `analyze` (src/config.py): a prior from where the training
labels fall, not something seen in the video. Writes reports/timecode_prior.json: the fraction fit on all clips
(what serving uses), and the leave-one-out |dt| of the prior next to the clip-midpoint baseline. A video with two
labels counts its nearer one. Duplicate clips are already merged into one video id in data/labels.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import LABELS_CSV, REPORTS, VIDEOS_DIR
from src.data.video import duration

OUT = REPORTS / "timecode_prior.json"


def nearest_dt(guess: float, seconds: list[float]) -> float:
    return min(abs(guess - s) for s in seconds)


def share_within(dts: np.ndarray, sec: float) -> float:
    return float(np.mean(dts <= sec))


def main() -> dict:
    labels = pd.read_csv(LABELS_CSV)
    viol = labels[labels["verdict"] == "violation"]
    videos = {
        vid: (duration(VIDEOS_DIR / g["file"].iloc[0]), [float(s) for s in g["second"]])
        for vid, g in viol.groupby("video_id")
    }
    # one fraction per label: a video with two labels gives two, both left out together with their video
    fracs = {vid: [s / dur for s in secs] for vid, (dur, secs) in videos.items()}
    prior_dt, mid_dt = [], []
    for vid, (dur, secs) in videos.items():
        rest = [f for other, fs in fracs.items() if other != vid for f in fs]
        prior_dt.append(nearest_dt(float(np.median(rest)) * dur, secs))
        mid_dt.append(nearest_dt(dur / 2, secs))
    prior_dt, mid_dt = np.array(prior_dt), np.array(mid_dt)
    all_fracs = [f for fs in fracs.values() for f in fs]

    def summary(dts: np.ndarray) -> dict:
        return {"median_abs_dt": float(np.median(dts))} | {
            f"share_within_{s}s": share_within(dts, s) for s in (1, 2, 3, 5)
        }

    out = {
        "fraction": float(np.median(all_fracs)),
        "n_violation_videos": len(videos),
        "n_labels": len(all_fracs),
        "scheme": "leave-one-video-out, duplicates merged",
        "prior_loo": summary(prior_dt),
        "midpoint": summary(mid_dt),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
