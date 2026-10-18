"""Branch A under CV grouped by scene (ADR 0006): the numbers leave-one-video-out cannot give.

    uv run python -m src.models.scene_cv                     # production cache, data/processed/embeddings
    uv run python -m src.models.scene_cv --cache DIR --warp DIR --note proxy-448px-2fps --log

The organizers filmed each violation together with a clean take of the same scene (same people, clothes, objects),
so leave-one-video-out keeps the held-out clip's twin in training and the classifier answers with the twin's
label. Here a fold holds out a whole scene (data/scenes.csv). Per fold scheme: video AUC of the out-of-fold scores
and clean videos flagged at recall 1.0 (ADR 0002). For scenes, also the nested check: threshold from the other
scenes' out-of-fold scores, a model fit on the other scenes, the verdict on the held-out scene. With --warp, the
held-out videos are scored a second time from a cache built under a virtual camera, against the same threshold.
Embeddings only, no tracks, so it runs wherever the cache is. Writes reports/scene_cv.json; --log appends rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from src.config import REPORTS, SCENES_CSV
from src.data.labels import load_labels
from src.features.embeddings import CACHE_DIR, FINGERPRINT
from src.models.train import fit_on, log_experiment
from src.models.windows import choose_threshold, labels_by_video, make_windows, video_score
from src.models.windows import group_of as video_group

OUT = REPORTS / "scene_cv.json"
N_JOBS = -1  # folds run in parallel; tests set 1 to watch them in-process


class Clip:
    """Windows with labels and pooled features, as `train.fit_on` takes them."""

    def __init__(self, windows: pd.DataFrame, feats: np.ndarray, label_seconds: list[int]):
        self.windows, self.feats, self.label_seconds = windows, feats, label_seconds

    @property
    def is_violation(self) -> bool:
        return bool(self.label_seconds)


def load_scenes(path: Path = SCENES_CSV) -> dict[str, str]:
    """video_id -> scene: the takes that share people, clothes and objects (data/scenes.csv)."""
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["video_id"], df["scene"], strict=True))


def read_cache(cache: Path, video_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(seconds, frame embeddings) from a production cache file `X-n.npz` or a proxy file `n.npz` (cls, meanp)."""
    file = cache / f"{video_id}.npz"
    if not file.exists():
        file = cache / f"{video_id.split('-')[1]}.npz"
    z = np.load(file)
    emb = z["embeddings"] if "embeddings" in z.files else np.concatenate([z["cls"], z["meanp"]], axis=1)
    return z["seconds"], emb.astype(np.float32)


def window_feats(
    seconds: np.ndarray, emb: np.ndarray, label_seconds: list[int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Windows as in training, mean embedding of the frames in each; windows without a frame are dropped."""
    step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 0.25
    windows = make_windows(float(seconds[-1]) + step, label_seconds)
    feats = np.zeros((len(windows), emb.shape[1]), dtype=np.float32)
    kept = np.zeros(len(windows), dtype=bool)
    for i, (a, b) in enumerate(zip(windows["start"], windows["end"], strict=True)):
        m = (seconds >= a) & (seconds < b)
        if m.any():
            feats[i], kept[i] = emb[m].mean(0), True
    return windows[kept].reset_index(drop=True), feats[kept]


def load_clips(cache: Path, ids: list[str], by_video: dict[str, list[int]]) -> dict[str, Clip]:
    out = {}
    for vid in ids:
        seconds, emb = read_cache(cache, vid)
        windows, feats = window_feats(seconds, emb, by_video[vid])
        out[vid] = Clip(windows, feats, by_video[vid])
    return out


def _score(model, clip: Clip) -> float:
    return video_score(model.predict_proba(clip.feats)[:, 1]) if len(clip.feats) else 0.0


def out_of_fold(ids: list[str], clips: dict[str, Clip], groups: dict[str, str]) -> dict[str, float]:
    """Video score of each clip from a model fit on the clips of every other group."""

    def fold(g: str) -> dict[str, float]:
        with threadpool_limits(1):
            model = fit_on([clips[v] for v in ids if groups[v] != g])
            return {v: _score(model, clips[v]) for v in ids if groups[v] == g}

    parts = Parallel(n_jobs=N_JOBS)(delayed(fold)(g) for g in sorted({groups[v] for v in ids}))
    return {v: s for part in parts for v, s in part.items()}


def nested(
    ids: list[str], clips: dict[str, Clip], groups: dict[str, str], warped: dict[str, Clip]
) -> list[dict]:
    """Per held-out group: ADR 0002 threshold on the other groups' out-of-fold scores, model fit on them."""

    def outer(g: str) -> list[dict]:
        inner = [v for v in ids if groups[v] != g]
        with threadpool_limits(1):
            inner_scores = {}
            for h in sorted({groups[v] for v in inner}):
                model = fit_on([clips[v] for v in inner if groups[v] != h])
                inner_scores |= {v: _score(model, clips[v]) for v in inner if groups[v] == h}
            threshold = choose_threshold(
                [inner_scores[v] for v in inner], [clips[v].is_violation for v in inner]
            )
            full = fit_on([clips[v] for v in inner])
            return [
                {
                    "video_id": v,
                    "is_violation": clips[v].is_violation,
                    "threshold": threshold,
                    "score": _score(full, clips[v]),
                    "warped_score": _score(full, warped[v]) if v in warped else None,
                }
                for v in ids
                if groups[v] == g
            ]

    parts = Parallel(n_jobs=N_JOBS)(delayed(outer)(g) for g in sorted({groups[v] for v in ids}))
    return [r for part in parts for r in part]


def summarize(scores: dict[str, float], clips: dict[str, Clip]) -> dict:
    ids = list(scores)
    y = np.array([clips[v].is_violation for v in ids])
    s = np.array([scores[v] for v in ids])
    threshold = choose_threshold(s, y)
    return {
        "video_auc": float(roc_auc_score(y, s)),
        "clean_flagged_at_recall_1": int((s[~y] >= threshold).sum()),
        "n_clean": int((~y).sum()),
    }


def summarize_nested(rows: list[dict]) -> dict:
    y = np.array([r["is_violation"] for r in rows])
    hit = np.array([r["score"] >= r["threshold"] for r in rows])
    out = {
        "nested_video_recall": float(hit[y].mean()),
        "nested_fp_rate": float(hit[~y].mean()),
        "nested_missed": [
            r["video_id"] for r, h in zip(rows, hit, strict=True) if r["is_violation"] and not h
        ],
    }
    warped = [r for r in rows if r["warped_score"] is not None]
    if warped:
        out["virtual_camera"] = {
            "n_videos": len(warped),
            "n_violation": sum(r["is_violation"] for r in warped),
            "violation_flagged_native": sum(
                r["is_violation"] and r["score"] >= r["threshold"] for r in warped
            ),
            "violation_flagged_warped": sum(
                r["is_violation"] and r["warped_score"] >= r["threshold"] for r in warped
            ),
            "median_score_native": float(np.median([r["score"] for r in warped])),
            "median_score_warped": float(np.median([r["warped_score"] for r in warped])),
        }
    return out


def run(cache: Path, warp: Path | None, note: str) -> dict:
    by_video = labels_by_video(load_labels())
    ids = sorted(by_video, key=lambda v: int(v.split("-")[1]))
    clips = load_clips(cache, ids, by_video)
    warped = {}
    if warp is not None:
        have = [v for v in ids if (warp / f"{v}.npz").exists() or (warp / f"{v.split('-')[1]}.npz").exists()]
        warped = load_clips(warp, have, by_video)
    scenes = load_scenes()
    by_scene = {v: scenes[v] for v in ids}
    by_video_group = {v: video_group(v) for v in ids}
    result = {
        "features": note,
        "n_scenes": len(set(by_scene.values())),
        "video": summarize(out_of_fold(ids, clips, by_video_group), clips),  # what train.py reports
        "scene": summarize(out_of_fold(ids, clips, by_scene), clips)
        | summarize_nested(nested(ids, clips, by_scene, warped)),
    }
    return result


def log_rows(result: dict) -> None:
    """Two rows: the out-of-fold numbers per fold scheme. Verdict metrics are the nested ones for scenes."""
    for scheme in ("video", "scene"):
        r = result[scheme]
        metrics = {
            "video_recall": r.get("nested_video_recall"),
            "fp_rate": r.get("nested_fp_rate"),
            "median_abs_dt": None,
            "type_accuracy": None,
            "cv": scheme,
            "features": result["features"],
            **{
                k: v
                for k, v in r.items()
                if not k.startswith("nested_video_recall") and k != "nested_fp_rate"
            },
        }
        log_experiment(f"dinov2-small-logreg-{scheme}-cv|{result['features']}", metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DIR, help="frame embeddings, one npz per video")
    ap.add_argument("--warp", type=Path, default=None, help="the same videos embedded under a virtual camera")
    ap.add_argument("--note", default=FINGERPRINT, help="what the cache holds, for the report and the log")
    ap.add_argument("--log", action="store_true", help="append the rows to experiments.csv")
    args = ap.parse_args()
    result = run(args.cache, args.warp, args.note)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.log:
        log_rows(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
