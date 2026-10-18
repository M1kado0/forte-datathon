"""Run SigLIP and compare against archived DINOv2 metrics without rerunning DINOv2.

    uv run python -m src.models.compare

The fitted SigLIP artifact and full metrics are archived. The self recordings
are evaluated once, outside all fitting and threshold selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from src.config import LABELS_CSV, MODELS, REPORTS, SELF_LABELS_CSV, SELF_VIDEOS_DIR

OUTPUT = REPORTS / "embedding_comparison"


def read_holdout():
    """Validate external labels without imposing organizer-only file names/arrangements."""
    import pandas as pd

    from contracts.data import TYPE_IDS
    from src.data.video import duration

    labels = pd.read_csv(SELF_LABELS_CSV, dtype={"second": "Int64"})
    if labels.empty or not {"video_id", "file", "verdict", "second", "violation_type"} <= set(labels):
        raise ValueError("Self labels need video_id, file, verdict, second, violation_type")
    if labels[["video_id", "file", "verdict"]].isna().any().any():
        raise ValueError("Self labels contain missing identifiers or verdicts")
    if not labels.verdict.isin(["violation", "no_violation"]).all():
        raise ValueError("Unknown self-label verdict")
    positive = labels.verdict.eq("violation")
    if not (labels.second.notna() == positive).all() or not (labels.violation_type.notna() == positive).all():
        raise ValueError("Positive labels require timecode and type; negative labels require neither")
    if not labels.loc[positive, "violation_type"].isin(TYPE_IDS).all():
        raise ValueError("Unknown self-label type")
    if labels.duplicated(["video_id", "second"]).any():
        raise ValueError("Duplicate self labels")
    organizer = pd.read_csv(LABELS_CSV)
    if set(labels.video_id) & set(organizer.video_id) or set(labels.file) & set(organizer.file):
        raise ValueError("Self holdout overlaps organizer identifiers/files")
    for _, rows in labels.groupby("video_id", sort=False):
        if rows.file.nunique() != 1 or rows.verdict.nunique() != 1:
            raise ValueError("Conflicting file/verdict for one self video")
        path = (SELF_VIDEOS_DIR / rows.file.iloc[0]).resolve()
        if path.parent != SELF_VIDEOS_DIR.resolve():
            raise ValueError("Self video must be directly inside SELF_VIDEOS_DIR")
        seconds = rows.second.dropna()
        if ((seconds < 0) | (seconds >= duration(path))).any():
            raise ValueError(f"Label outside video: {path.name}")
    return labels


def evaluate_holdout(artifact: dict) -> dict:
    """No fit or threshold selection here: use the model trained on organizer videos."""
    from src.features.detectors import LocalDetector
    from src.features.embeddings import Embedder, embed_video
    from src.features.tracks import get_tracks
    from src.heuristics import detect
    from src.models.analyze import score_windows
    from src.models.evaluate import Case, run_case, summarize
    from src.models.violations import HEURISTIC_CONFIGS, peak_feature

    labels = read_holdout()
    settings = artifact["settings"]
    embedder = Embedder()
    detector = None
    if settings.allowed or settings.vote.enabled:
        detector = LocalDetector()
    cases, found, timings = [], [], {}
    for vid, rows in labels.groupby("video_id", sort=False):
        print(f"holdout {artifact['embedding']}: {vid}", flush=True)
        path = SELF_VIDEOS_DIR / rows.file.iloc[0]
        start = time.perf_counter()
        seconds, emb, elapsed = embed_video(path, embedder)
        starts, scores, feats = score_windows(seconds, emb, artifact["model"])
        hits = (
            detect(get_tracks(path, detector), thresholds=HEURISTIC_CONFIGS[settings.heuristics])
            if detector
            else []
        )
        truth = [
            (int(row.second), row.violation_type) for row in rows.itertuples() if row.verdict == "violation"
        ]
        case = Case(vid, starts, scores, peak_feature(scores, feats), hits, truth)
        predictions = run_case(case, artifact["threshold"], settings)
        cases.append(case)
        found.append(predictions[0] if predictions else None)
        timings[vid] = {"embedding_seconds": elapsed, "seconds_with_track_cache": time.perf_counter() - start}
    return summarize(cases, found) | {
        "n_videos": len(cases),
        "threshold": artifact["threshold"],
        "timings": timings,
        "protocol": "Organizer-only fitting and threshold selection; self videos never used for tuning. Track cache may be warm; timing excludes model load and JPEG evidence.",
    }


def worker() -> None:
    from src.features.embeddings import MODEL_NAME
    from src.models import train

    read_holdout()  # fail before spending GPU time if input labels are invalid
    target = OUTPUT / MODEL_NAME
    target.mkdir(parents=True, exist_ok=True)
    # This is also the acceptance command for #27.
    subprocess.run([sys.executable, "-u", "-m", "src.models.train"], check=True)
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    shutil.copy2(train.MODEL_PATH, MODELS / f"window_clf_{MODEL_NAME}.pkl")
    (target / "organizer.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    holdout = evaluate_holdout(train.load_model())
    (target / "self.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")
    train.log_experiment(f"{MODEL_NAME}-self-held-out", holdout)
    print(json.dumps({k: holdout[k] for k in train.HEADLINE_KEYS}, indent=2), flush=True)


def write_comparison() -> dict:
    import torch
    import transformers

    keys = ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy")
    baseline = json.loads((OUTPUT / "dinov2-small" / "organizer.json").read_text(encoding="utf-8"))
    siglip = {
        split: json.loads((OUTPUT / "siglip-base" / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("organizer", "self")
    }
    runs = {"dinov2-small": {"organizer": baseline}, "siglip-base": siglip}
    result = {
        "protocol": "SigLIP run only; DINOv2 organizer metrics are the existing repository baseline, not a new run. Same 4 fps, windows, grouped LOVO, train-only scalers, logistic C, type/heuristic selection and nested threshold procedure. No DINOv2 self baseline. Self videos held out from all fitting. LOVO recall is optimistic because its threshold is selected on those OOF scores. Historical timings were measured on a different machine and are not a speed comparison.",
        "labels_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (LABELS_CSV, SELF_LABELS_CSV)
        },
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "runs": runs,
        "siglip_minus_dinov2": {
            "organizer": {
                k: siglip["organizer"][k] - baseline[k]
                if siglip["organizer"][k] is not None and baseline[k] is not None
                else None
                for k in keys
            },
        },
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# SigLIP versus DINOv2",
        "",
        result["protocol"],
        "",
        "| Split | Encoder | Recall | False-positive rate | Median time error (s) | Type accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, splits in runs.items():
        for split in splits:
            values = [
                f"{runs[model][split][k]:.4f}" if runs[model][split][k] is not None else "n/a" for k in keys
            ]
            lines.append(f"| {split} | {model} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Full per-video results, nested checks, layout checks and timings: `comparison.json`. Runtime excludes model loading. Self-video timings may reuse detector tracks and are not end-to-end API latency.",
        "",
        f"Hardware: {result['hardware']}; torch {result['torch']}; transformers {result['transformers']}.",
    ]
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    if "--worker" in sys.argv:
        worker()
        return
    baseline = OUTPUT / "dinov2-small" / "organizer.json"
    if not baseline.exists():
        metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("embedding", "dinov2-small").startswith("siglip"):
            raise ValueError("Archive the original DINOv2 metrics before running SigLIP")
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPORTS / "metrics.json", baseline)
    subprocess.run(
        [sys.executable, "-u", "-m", "src.models.compare", "--worker"],
        check=True,
        env={**os.environ, "FORTE_EMBEDDER": "siglip-base", "HF_HUB_OFFLINE": "1"},
    )
    print(json.dumps(write_comparison()["siglip_minus_dinov2"], indent=2))


if __name__ == "__main__":
    main()
