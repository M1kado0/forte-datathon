"""Viewpoint robustness (#43, TZ-F.8): the same checks on warped training videos and on our own recordings.

    uv run python -m scripts.viewpoint_eval [--no-log]

All training videos share one camera (FAQ 3), so a new angle can only be imitated or recorded. Two checks:

1. Warped training videos. Every frame goes through one of the PRESETS: a rotation, a shift, a zoom and a
   perspective tilt, each stronger than the camera spread inside the training set (shift over MIN_SHIFT_PX,
   rotation over MIN_ROTATION_DEG, the numbers of the ticket). Every number is out of sample: the held-out
   clip's group is left out of the classifier, the heuristics' settings and the kNN bank. The classifier is fit
   on clean clips, and on clean clips plus the clips of the other presets (augmentation); the held-out preset is
   never in its own training set.
2. Own recordings (data/labels_self.csv): the model fit on all training videos, judged on clips shot from other
   places in the room, as recorded and scaled to the training size.

The threshold is always the one chosen on the clean leave-one-video-out run (ADR 0002), because on a new
camera that is the only threshold the service would have.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from contracts.data import TRACKS_SCHEMA
from src.config import DATA_PROCESSED, ROOT, VIDEOS_DIR
from src.data.labels import load_labels
from src.data.video import DEFAULT_FPS, iter_frames
from src.features import tracks as track_cache
from src.features.detectors import LocalDetector
from src.features.embeddings import EMBED_FPS, Embedder, load_or_embed, window_features
from src.heuristics import detect
from src.models.evaluate import Case, files_by_video, fit_settings, run_case, summarize
from src.models.train import Clip, build_cases, build_dataset, fit_on, log_experiment, nested_check
from src.models.violations import HEURISTIC_CONFIGS, Settings, peak_feature
from src.models.windows import (
    choose_threshold,
    group_of,
    labels_by_video,
    make_windows,
    shift_threshold,
    video_score,
)

MIN_SHIFT_PX = 40  # the ticket's bound for the camera spread inside the training videos
MIN_ROTATION_DEG = 7.0
SELF_LABELS = ROOT / "data" / "labels_self.csv"
SELF_DIR = ROOT / "data" / "raw" / "self_recorded"
WORK_DIR = DATA_PROCESSED / "viewpoint"  # warped copies, their embeddings and tracks; rebuilt on demand
TRAIN_LONG_SIDE = 848  # the dataset videos are 464 x 848

Transform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Preset:
    """A homography about the frame centre, in fractions of the frame so it fits any resolution.

    rotation_deg turns the frame, shift moves it (fractions of width and height), zoom scales it, and tilt
    is the perspective term (kx, ky): a corner is scaled by about 1 +- tilt / 2.
    """

    rotation_deg: float
    shift: tuple[float, float]
    zoom: float
    tilt: tuple[float, float]

    def matrix(self, width: int, height: int) -> np.ndarray:
        centre = np.array([width / 2, height / 2])
        theta = np.deg2rad(self.rotation_deg)
        turn = np.array(
            [
                [self.zoom * np.cos(theta), -self.zoom * np.sin(theta), 0],
                [self.zoom * np.sin(theta), self.zoom * np.cos(theta), 0],
                [self.tilt[0] / width, self.tilt[1] / height, 1],
            ]
        )
        to_centre = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]])
        back = np.array(
            [
                [1, 0, centre[0] + self.shift[0] * width],
                [0, 1, centre[1] + self.shift[1] * height],
                [0, 0, 1],
            ]
        )
        return back @ turn @ to_centre


# Three camera positions: left of the dataset's, right of it, and farther back. Each is a different mix, so a
# classifier trained on two of them is judged on the third.
PRESETS: dict[str, Preset] = {
    "left": Preset(9.0, (0.12, 0.04), 1.0, (0.25, 0.10)),
    "right": Preset(-9.0, (-0.12, -0.04), 1.0, (-0.25, 0.10)),
    "back": Preset(14.0, (0.10, -0.06), 0.9, (0.0, -0.20)),
}


# Identity through the same pipeline (4 fps MJPG copy): what the re-encoding alone does to the metrics.
CONTROL = Preset(0.0, (0.0, 0.0), 1.0, (0.0, 0.0))


def _project(h: np.ndarray, x: float, y: float) -> np.ndarray:
    p = h @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def shift_px(h: np.ndarray, width: int, height: int) -> float:
    """How far the frame centre moves, in pixels."""
    centre = np.array([width / 2, height / 2])
    return float(np.linalg.norm(_project(h, *centre) - centre))


def rotation_deg(h: np.ndarray, width: int = 100, height: int = 100) -> float:
    """Rotation of the map at the frame centre, from its numeric Jacobian (the perspective term is ignored)."""
    cx, cy, eps = width / 2, height / 2, 1e-3
    dx = (_project(h, cx + eps, cy) - _project(h, cx - eps, cy)) / (2 * eps)
    dy = (_project(h, cx, cy + eps) - _project(h, cx, cy - eps)) / (2 * eps)
    return float(np.degrees(np.arctan2(dx[1] - dy[0], dx[0] + dy[1])))


def warp_frame(frame: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The frame seen through the homography, same size; the area the camera would not see is black."""
    height, width = frame.shape[:2]
    return cv2.warpPerspective(frame, h, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)


def scale_to_long_side(frame: np.ndarray, long_side: int = TRAIN_LONG_SIDE) -> np.ndarray:
    """The frame scaled so its longer side is `long_side`; the aspect ratio is kept."""
    h, w = frame.shape[:2]
    k = long_side / max(h, w)
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


# --- variants of a video on disk ---------------------------------------------------------------------------


def variant_key(name: str, params: object) -> str:
    """Folder name of one variant: a changed preset gets a new folder, so no cache is read stale."""
    return f"{name}-{hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"


def frame_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def write_variants(source: Path, targets: dict[Path, Transform]) -> None:
    """Decode `source` once at DEFAULT_FPS and write one MJPG clip per target; existing clips are kept.

    A clip holds the sampled frames at DEFAULT_FPS, so reading it back yields the same seconds on a regular grid.
    """
    todo = {t: f for t, f in targets.items() if not t.exists()}
    if not todo:
        return
    writers: dict[Path, cv2.VideoWriter] = {}
    try:
        for _, frame in iter_frames(source, DEFAULT_FPS):
            for target, transform in todo.items():
                out = transform(frame)
                if target not in writers:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    h, w = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writers[target] = cv2.VideoWriter(str(_part(target)), fourcc, DEFAULT_FPS, (w, h))
                writers[target].write(out)
    finally:
        for writer in writers.values():
            writer.release()
    for target in writers:
        _part(target).rename(target)  # a clip is complete once it has its final name


def _part(target: Path) -> Path:
    return target.with_name(target.stem + ".part.avi")


# --- clips, tracks and cases of a variant ------------------------------------------------------------------


def variant_clips(
    key: str, files: dict[str, Path], by_video: dict[str, list[int]], embedder: Callable[[], Embedder]
) -> dict[str, Clip]:
    """Windows with labels and pooled embeddings of every video of one variant, as `build_dataset` makes them."""
    clips = {}
    for vid, path in files.items():
        seconds, emb, elapsed = load_or_embed(vid, path, embedder, WORK_DIR / key / "embeddings")
        duration = float(seconds[-1]) + 1.0 / EMBED_FPS if len(seconds) else 0.0
        windows = make_windows(duration, by_video[vid])
        feats, kept = window_features(seconds, emb, windows)
        clips[vid] = Clip(windows[kept].reset_index(drop=True), feats[kept], elapsed, duration, by_video[vid])
    return clips


def variant_tracks(key: str, vid: str, path: Path, detector: Callable[[], LocalDetector]) -> pd.DataFrame:
    cached = WORK_DIR / key / "tracks" / track_cache.setting_id() / f"{vid}.parquet"
    if cached.is_file():
        return TRACKS_SCHEMA.validate(pd.read_parquet(cached))
    table = track_cache.build_tracks(path, detector())
    cached.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cached, index=False)
    return table


def make_case(
    vid: str, clip: Clip, scores: np.ndarray, tracks: pd.DataFrame, truth: list[tuple[int, str]]
) -> Case:
    by_config = {name: detect(tracks, thresholds=th) for name, th in HEURISTIC_CONFIGS.items()}
    return Case(
        vid,
        clip.windows["start"].to_numpy(),
        scores,
        peak_feature(scores, clip.feats),
        by_config["default"],
        truth,
        by_config,
    )


def rescore(case: Case, clip: Clip, scores: np.ndarray) -> Case:
    """The same video and hits under another classifier."""
    return replace(case, scores=scores, peak_feature=peak_feature(scores, clip.feats))


# --- scoring and judging -----------------------------------------------------------------------------------


def lovo_scores(
    ids: list[str],
    clips: Sequence[Clip],
    extra: Sequence[tuple[str, Clip]],
    evals: dict[str, Sequence[Clip]],
) -> dict[str, list[np.ndarray]]:
    """Window scores of every clip in `evals` (each aligned with `ids`), from a model fit without its group.

    The model sees `clips` and `extra` (video id, clip) of the other groups only.
    """
    out = {key: [np.zeros(len(c.windows)) for c in group] for key, group in evals.items()}
    groups = np.array([group_of(v) for v in ids])
    for g in sorted(set(groups)):
        others = np.flatnonzero(groups != g)
        clf = fit_on([clips[i] for i in others] + [c for v, c in extra if group_of(v) != g])
        for key, group in evals.items():
            for i in np.flatnonzero(groups == g):
                if len(group[i].feats):
                    out[key][i] = clf.predict_proba(group[i].feats)[:, 1]
    return out


def decide_all(
    clean_cases: Sequence[Case], test_cases: Sequence[Case], threshold: float, banks: dict
) -> dict:
    """The decision on every test case, each judged with settings fitted on the clean cases of the other groups:
    the camera is the only thing that differs between the fit and the test. Keeps the cases and the first
    violation of each so several sets can be pooled."""
    first, seen = [], []
    for c in test_cases:
        rest = [o for o in clean_cases if o.group != c.group]
        rest_banks = {v: b for v, b in banks.items() if group_of(v) != c.group}
        settings, _ = fit_settings(rest, threshold, rest_banks)
        found = run_case(c, threshold, settings)
        first.append(found[0] if found else None)
        seen.append(c.view(settings.heuristics))
    return {"cases": seen, "first": first, "threshold": threshold}


def judge(clean_cases: Sequence[Case], test_cases: Sequence[Case], threshold: float, banks: dict) -> dict:
    """Metrics of `decide_all`."""
    return metrics_of(decide_all(clean_cases, test_cases, threshold, banks))


def metrics_of(part: dict) -> dict:
    return report(part["cases"], part["first"], part["threshold"])


def report(cases: Sequence[Case], first: list, threshold: float) -> dict:
    """The decision's metrics next to the window classifier's alone."""
    out = summarize(cases, first)
    flagged = [bool(c.scores.size and c.scores.max() >= threshold) for c in cases]
    bad = [c.is_violation for c in cases]
    n_bad = sum(bad)
    out["branch_a_recall"] = (
        sum(f and b for f, b in zip(flagged, bad, strict=True)) / n_bad if n_bad else None
    )
    out["branch_a_fp_rate"] = (
        sum(f and not b for f, b in zip(flagged, bad, strict=True)) / (len(bad) - n_bad)
        if len(bad) > n_bad
        else None
    )
    return out


def pooled(parts: list[dict]) -> dict:
    """Metrics over several sets at once; `parts` are `decide_all` results."""
    cases = [c for p in parts for c in p["cases"]]
    first = [f for p in parts for f in p["first"]]
    return report(cases, first, parts[0]["threshold"])


NUMBERS = ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy", "branch_a_recall", "branch_a_fp_rate")


def numbers(metrics: dict) -> dict:
    """The figures a table or an experiments row keeps."""
    return {k: metrics[k] for k in NUMBERS} | {"n_false_positives": metrics["n_false_positives"]}


# --- the run -----------------------------------------------------------------------------------------------


def truth_of(labels: pd.DataFrame, vid: str) -> list[tuple[int, str]]:
    rows = labels[(labels["video_id"] == vid) & (labels["verdict"] == "violation")]
    return [(int(s), t) for s, t in zip(rows["second"], rows["violation_type"], strict=True)]


def run(log: bool = True) -> dict:
    labels = load_labels()
    ids, clips = build_dataset(labels)
    files = files_by_video(labels)
    by_video = labels_by_video(labels)
    embedder, detector = functools.cache(Embedder), functools.cache(LocalDetector)
    sources = {v: VIDEOS_DIR / files[v] for v in ids}

    # warped copies of every training video, one decode pass per video
    variants = {**PRESETS, "control": CONTROL}
    keys = {p: variant_key(p, {"preset": asdict(variants[p]), "fps": DEFAULT_FPS}) for p in variants}
    paths = {p: {v: WORK_DIR / keys[p] / "videos" / f"{v}.avi" for v in ids} for p in variants}
    for v in ids:
        size = frame_size(sources[v])
        write_variants(sources[v], {paths[p][v]: _transform(variants[p], size) for p in variants})
    warped = {p: variant_clips(keys[p], paths[p], by_video, embedder) for p in variants}
    warped_clips = {p: [warped[p][v] for v in ids] for p in variants}

    # the clean run: out-of-fold scores, the ADR 0002 threshold, cases and kNN banks
    plain = lovo_scores(ids, clips, [], {"clean": clips, **warped_clips})
    threshold = choose_threshold([video_score(s) for s in plain["clean"]], [c.is_violation for c in clips])
    clean_cases, banks = build_cases(labels, ids, clips, plain["clean"])
    clean = judge(clean_cases, clean_cases, threshold, banks)

    # each warped video with its tracks; the hits do not depend on the classifier
    base = {
        p: [
            make_case(
                v,
                warped[p][v],
                plain[p][i],
                variant_tracks(keys[p], v, paths[p][v], detector),
                clean_cases[i].truth,
            )
            for i, v in enumerate(ids)
        ]
        for p in variants
    }
    control = judge(clean_cases, base["control"], threshold, banks)

    # augmentation: a model per held-out preset, fit on clean clips and the clips of the other presets
    plain_parts, aug_parts, aug_threshold = [], [], {}
    for p in PRESETS:
        extra = [(v, warped[q][v]) for q in PRESETS if q != p for v in ids]
        aug = lovo_scores(ids, clips, extra, {"clean": clips, p: warped_clips[p]})
        aug_threshold[p] = choose_threshold(
            [video_score(s) for s in aug["clean"]], [c.is_violation for c in clips]
        )
        aug_clean = [rescore(c, clips[i], aug["clean"][i]) for i, c in enumerate(clean_cases)]
        aug_cases = [rescore(c, warped[p][c.video_id], aug[p][i]) for i, c in enumerate(base[p])]
        plain_parts.append(decide_all(clean_cases, base[p], threshold, banks))
        aug_parts.append(decide_all(aug_clean, aug_cases, aug_threshold[p], banks))
    warped_plain = {p: numbers(metrics_of(part)) for p, part in zip(PRESETS, plain_parts, strict=True)}
    warped_aug = {p: numbers(metrics_of(part)) for p, part in zip(PRESETS, aug_parts, strict=True)}

    # the model that would ship with augmentation: fit on everything, what it costs on clean clips
    extra_all = [(v, warped[q][v]) for q in PRESETS for v in ids]
    aug_all = lovo_scores(ids, clips, extra_all, {"clean": clips})
    aug_all_threshold = choose_threshold(
        [video_score(s) for s in aug_all["clean"]], [c.is_violation for c in clips]
    )
    aug_all_cases = [rescore(c, clips[i], aug_all["clean"][i]) for i, c in enumerate(clean_cases)]
    clean_with_aug = judge(aug_all_cases, aug_all_cases, aug_all_threshold, banks)

    own = judge_own(
        labels, ids, clips, warped_clips, threshold, aug_all_threshold, clean_cases, aug_all_cases, banks,
        embedder, detector,
    )  # fmt: skip

    result = {
        "presets": {p: _preset_facts(PRESETS[p], frame_size(sources[ids[0]])) for p in PRESETS},
        "threshold": threshold,
        "clean": numbers(clean),
        "reencoded_control": numbers(control),
        "warped_clean_trained": {"pooled": numbers(pooled(plain_parts)), "by_preset": warped_plain},
        "warped_augmented": {
            "pooled": numbers(pooled(aug_parts)),
            "by_preset": warped_aug,
            "thresholds": aug_threshold,
        },
        "clean_with_augmentation": numbers(clean_with_aug) | {"threshold": aug_all_threshold},
        "own_recordings": own,
    }
    if log:
        log_rows(result)
    return result


def _transform(preset: Preset, size: tuple[int, int]) -> Transform:
    h = preset.matrix(*size)
    return lambda frame: warp_frame(frame, h)


def _preset_facts(preset: Preset, size: tuple[int, int]) -> dict:
    h = preset.matrix(*size)
    return {"shift_px": round(shift_px(h, *size), 1), "rotation_deg": round(rotation_deg(h, *size), 1)}


def judge_own(
    labels,
    ids,
    clips,
    warped_clips,
    threshold,
    aug_threshold,
    clean_cases,
    aug_cases,
    banks,
    embedder,
    detector,
) -> dict:
    """Own recordings under the model fit on all training videos, clean and augmented, as recorded and scaled."""
    self_labels = pd.read_csv(SELF_LABELS)
    by_video = labels_by_video(self_labels)
    nested = nested_check(ids, clips)
    delta = nested["threshold_shift_logit"]  # the shift `train` saves into window_clf.pkl (#63)
    models = {
        "clean": (fit_on(clips), shift_threshold(threshold, delta), clean_cases),
        "augmented": (
            fit_on(clips + [c for p in PRESETS for c in warped_clips[p]]),
            shift_threshold(aug_threshold, delta),
            aug_cases,
        ),
    }
    settings: dict[str, Settings] = {
        m: fit_settings(cases, thr, banks)[0] for m, (_, thr, cases) in models.items()
    }
    files = {v: SELF_DIR / f for v, f in zip(self_labels["video_id"], self_labels["file"], strict=True)}
    scaled = variant_key("own-scaled", {"long_side": TRAIN_LONG_SIDE, "fps": DEFAULT_FPS})
    scaled_files = {v: WORK_DIR / scaled / "videos" / f"{v}.avi" for v in files}
    for v, path in files.items():
        write_variants(path, {scaled_files[v]: scale_to_long_side})
    out: dict = {"threshold_shift_logit": delta, "n_videos": len(files)}
    for name, key, paths in (("as_recorded", "own-native", files), ("scaled", scaled, scaled_files)):
        own_clips = variant_clips(key, paths, by_video, embedder)
        for model, (clf, thr, _) in models.items():
            cases, first, rows = [], [], []
            for v, clip in own_clips.items():
                tracks = variant_tracks(key, v, paths[v], detector)
                scores = clf.predict_proba(clip.feats)[:, 1] if len(clip.feats) else np.zeros(0)
                case = make_case(v, clip, scores, tracks, truth_of(self_labels, v))
                found = run_case(case, thr, settings[model])
                cases.append(case.view(settings[model].heuristics))
                first.append(found[0] if found else None)
                rows.append(_own_row(case, found[0] if found else None, thr))
            out[f"{name}_{model}"] = {**numbers(report(cases, first, thr)), "threshold": thr, "videos": rows}
    return out


def _own_row(case: Case, found, threshold: float) -> dict:
    return {
        "video": case.video_id,
        "truth": case.truth,
        "found": None if found is None else {"type": found.type_id, "second": round(found.timecode_sec, 1)},
        "max_window_score": round(float(case.scores.max()), 4) if case.scores.size else None,
        "branch_a_flags": bool(case.scores.size and case.scores.max() >= threshold),
        "hits": sorted({h.type_id for h in case.hits}),
    }


def log_rows(result: dict) -> None:
    """One experiments.csv row per question; the figures of the metric columns are pooled over the presets."""
    plain, aug = result["warped_clean_trained"], result["warped_augmented"]
    for note, part in (("viewpoint-warped-lovo", plain), ("viewpoint-warped-lovo-augmented", aug)):
        log_experiment(
            note,
            {
                **part["pooled"],
                "by_preset": part["by_preset"],
                "presets": result["presets"],
                "clean": result["clean"],
                "reencoded_control": result["reencoded_control"],
                "threshold": result["threshold"],
            },
        )
    own = result["own_recordings"]
    log_experiment(
        "viewpoint-own-recordings",
        {
            **own["as_recorded_clean"],
            "scaled_to_training_size": {k: v for k, v in own["scaled_clean"].items() if k != "videos"},
            "augmented_as_recorded": {k: v for k, v in own["as_recorded_augmented"].items() if k != "videos"},
            "augmented_scaled": {k: v for k, v in own["scaled_augmented"].items() if k != "videos"},
            "n_videos": own["n_videos"],
            "videos": [
                {k: r[k] for k in ("video", "branch_a_flags", "found")}
                for r in own["as_recorded_clean"]["videos"]
            ],
        },
    )


def main(argv: list[str]) -> None:
    print(json.dumps(run(log="--no-log" not in argv), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
