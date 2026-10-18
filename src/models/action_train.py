"""Evaluate two action-first pipelines without training on the self-recorded set.

    uv run python -m src.models.action_train

Model selection uses nested organizer CV, before any self-video evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import MODELS, REPORTS, SELF_VIDEOS_DIR
from src.data.labels import load_labels
from src.features.actions import BACKBONES, ActionEncoder, cached, fingerprint
from src.features.embeddings import video_path
from src.models.action_pipeline import Clip, binary_scores, classify_events, fit_heads
from src.models.compare import read_holdout
from src.models.train import log_experiment
from src.models.windows import choose_threshold, lovo_folds

OUTPUT = REPORTS / "action_comparison"
KEYS = ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy")


def build_dataset(labels, encoder, dataset: str) -> list[Clip]:
    clips = []
    for vid, rows in labels.groupby("video_id", sort=False):
        path = (
            video_path(rows.file.iloc[0]) if dataset == "organizer" else SELF_VIDEOS_DIR / rows.file.iloc[0]
        )
        print(f"{encoder.name} {dataset}: {vid}", flush=True)
        proposals = cached(path, dataset, encoder)
        truth = [
            (int(row.second), row.violation_type) for row in rows.itertuples() if row.verdict == "violation"
        ]
        clips.append(Clip(vid, proposals, truth))
    return clips


def calibrate(clips: list[Clip]) -> float:
    scores = np.zeros(len(clips))
    for train, valid in lovo_folds([c.video_id for c in clips]):
        heads = fit_heads([clips[i] for i in train], binary_only=True)
        for i in valid:
            scores[i] = binary_scores(clips[i].proposals, heads).max()
    return choose_threshold(scores, [c.positive for c in clips])


def summarize(clips: list[Clip], events: list[list[dict]], maxima: list[float]) -> dict:
    positives = [(c, e) for c, e in zip(clips, events, strict=True) if c.positive]
    negatives = [(c, e) for c, e in zip(clips, events, strict=True) if not c.positive]
    detected = [(c, e[0]) for c, e in positives if e]
    typed = sum(e["type"] in {t for _, t in c.truth} for c, e in detected)
    errors = [min(abs(e["timecode_sec"] - sec) for sec, _ in c.truth) for c, e in detected]
    # Match every labelled event independently to any emitted event within 5 seconds.
    # Report separately from strongest-event metrics used by the previous baseline.
    localized = sum(
        any(abs(e["timecode_sec"] - sec) <= 5 for e in found) for c, found in positives for sec, _ in c.truth
    )
    return {
        "video_recall": len(detected) / len(positives) if positives else None,
        "fp_rate": sum(bool(e) for _, e in negatives) / len(negatives) if negatives else None,
        "median_abs_dt": float(np.median(errors)) if errors else None,
        "type_accuracy": typed / len(positives) if positives else None,
        "n_videos": len(clips),
        "n_violation_videos": len(positives),
        "n_clean_videos": len(negatives),
        "n_false_positives": sum(bool(e) for _, e in negatives),
        "n_typed_right": typed,
        "localized_labels_within_5s": localized,
        "n_violation_labels": sum(len(c.truth) for c in clips),
        "by_video": {
            c.video_id: {"truth": c.truth, "score": float(s), "events": e}
            for c, e, s in zip(clips, events, maxima, strict=True)
        },
    }


def evaluate_cv(clips: list[Clip]) -> tuple[dict, dict, float]:
    held_scores, held_heads, thresholds = {}, {}, {}
    for number, (train, valid) in enumerate(lovo_folds([c.video_id for c in clips]), 1):
        print(f"nested fold {number}/{len(clips)}", flush=True)
        training = [clips[i] for i in train]
        heads = fit_heads(training)
        threshold = calibrate(training)
        for i in valid:
            held_scores[i] = binary_scores(clips[i].proposals, heads)
            held_heads[i] = heads
            thresholds[i] = threshold
    maxima = [float(held_scores[i].max()) for i in range(len(clips))]
    threshold = choose_threshold(maxima, [c.positive for c in clips])
    lovo_events = [
        classify_events(c.proposals, held_scores[i], held_heads[i], threshold) for i, c in enumerate(clips)
    ]
    nested_events = [
        classify_events(c.proposals, held_scores[i], held_heads[i], thresholds[i])
        for i, c in enumerate(clips)
    ]
    lovo = summarize(clips, lovo_events, maxima) | {
        "threshold": threshold,
        "protocol": "OOF heads; threshold selected on these same scores (optimistic recall).",
    }
    nested = summarize(clips, nested_events, maxima) | {
        "thresholds": {c.video_id: thresholds[i] for i, c in enumerate(clips)},
        "protocol": "Outer video excluded from both heads, scaler and inner-LOVO threshold selection.",
    }
    return lovo, nested, threshold


def dump(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_self(name: str) -> dict:
    encoder = ActionEncoder(name)
    clips = build_dataset(read_holdout(), encoder, "self")
    with (MODELS / f"action_{name}.pkl").open("rb") as file:
        artifact = pickle.load(file)
    scores = [binary_scores(c.proposals, artifact) for c in clips]
    events = [
        classify_events(c.proposals, s, artifact, artifact["threshold"])
        for c, s in zip(clips, scores, strict=True)
    ]
    result = summarize(clips, events, [float(s.max()) for s in scores])
    result["embedding_seconds"] = {c.video_id: c.proposals.elapsed for c in clips}
    result["protocol"] = (
        "Previously used self benchmark; no fitting, selection or threshold adjustment on it in this experiment."
    )
    dump(OUTPUT / name / "self.json", result)
    log_experiment(f"action-{name}-self-held-out", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    # Limit BLAS threads: each nested classifier is small; oversubscription is costly.
    with threadpool_limits(1):
        for name in args.models:
            encoder = ActionEncoder(name)
            clips = build_dataset(load_labels(), encoder, "organizer")
            lovo, nested, threshold = evaluate_cv(clips)
            artifact = fit_heads(clips) | {
                "threshold": threshold,
                "fingerprint": fingerprint(name),
                "name": name,
            }
            with (MODELS / f"action_{name}.pkl").open("wb") as file:
                pickle.dump(artifact, file)
            results[name] = {
                "lovo": lovo,
                "nested": nested,
                "embedding_seconds": {c.video_id: c.proposals.elapsed for c in clips},
            }
            dump(OUTPUT / name / "organizer.json", results[name])
            log_experiment(f"action-{name}-lovo", lovo)
            log_experiment(f"action-{name}-nested", nested)
            print(
                json.dumps(
                    {
                        "model": name,
                        "lovo": {k: lovo[k] for k in KEYS},
                        "nested": {k: nested[k] for k in KEYS},
                    }
                ),
                flush=True,
            )
            del encoder, clips, artifact
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        winner = min(
            results,
            key=lambda name: (
                -results[name]["nested"]["video_recall"],
                results[name]["nested"]["fp_rate"],
                results[name]["nested"]["median_abs_dt"]
                if results[name]["nested"]["median_abs_dt"] is not None
                else float("inf"),
            ),
        )
        dump(
            OUTPUT / "selection.json",
            {
                "selected": winner,
                "basis": "Organizer nested recall, then FP, then time error; selected before self evaluation.",
            },
        )
        for name in args.models:
            results[name]["self"] = evaluate_self(name)
    report = {
        "selected": winner,
        "results": results,
        "limits": "Sparse violation point labels do not measure generic action-recognition accuracy. Dense proposals preserve static scenes. Type classes absent from a training fold cannot be learned. Historical baselines differ in both features and decision pipeline.",
    }
    dump(OUTPUT / "comparison.json", report)
    lines = [
        "# Action-first comparison",
        "",
        report["limits"],
        "",
        "| Model | Evaluation | Recall | FP rate | Time error (s) | Type accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for split in ("lovo", "nested", "self"):
            lines.append(
                f"| {name} | {split} | "
                + " | ".join(
                    f"{result[split][k]:.4f}" if result[split][k] is not None else "n/a" for k in KEYS
                )
                + " |"
            )
    lines += [
        "",
        f"Selected on organizer nested CV before self evaluation: `{winner}`.",
        "",
        "Only supervised violation/type heads are fitted; frozen action models receive no violation prompts. Times are score-peak centres, not claimed action onset annotations.",
    ]
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
