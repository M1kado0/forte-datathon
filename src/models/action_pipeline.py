"""Action proposals → binary verdict → type and time. No legal text generation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS, SEED
from src.features.actions import ActionEncoder, Proposals, extract, fingerprint
from src.models.violations import MAX_VIOLATIONS, MIN_GAP_SEC, NEEDS_OBJECT
from src.models.windows import NEG_BEFORE, POS_AFTER, POS_BEFORE


@dataclass
class Clip:
    video_id: str
    proposals: Proposals
    truth: list[tuple[int, str]]

    @property
    def positive(self):
        return bool(self.truth)


def targets(clip: Clip) -> tuple[np.ndarray, np.ndarray]:
    p = clip.proposals
    positive = np.zeros(len(p.starts), dtype=bool)
    negative = np.ones(len(p.starts), dtype=bool)
    for second, _ in clip.truth:
        positive |= (p.starts < second + POS_AFTER) & (p.ends > second - POS_BEFORE)
        negative &= p.ends < second - NEG_BEFORE
    types = np.array(
        [
            min(clip.truth, key=lambda item: abs(item[0] - (a + b) / 2))[1] if clip.truth else "other"
            for a, b in zip(p.starts, p.ends, strict=True)
        ]
    )
    return np.where(positive, 1, np.where(negative, 0, -1)), types


def classifier():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED)
    )


def fit_heads(clips: list[Clip], *, binary_only=False) -> dict:
    """All fitted state (including scalers) comes exclusively from these clips."""
    x = np.concatenate([c.proposals.features for c in clips])
    labels = [targets(c) for c in clips]
    y = np.concatenate([t[0] for t in labels])
    types = np.concatenate([t[1] for t in labels])
    labelled = y >= 0
    if len(np.unique(y[labelled])) != 2:
        raise ValueError("Training fold needs positive and negative windows")
    binary = classifier().fit(x[labelled], y[labelled])
    if binary_only:
        return {"binary": binary}
    positive = y == 1
    type_model = (
        classifier() if len(np.unique(types[positive])) > 1 else DummyClassifier(strategy="most_frequent")
    )
    type_model.fit(x[positive], types[positive])
    return {"binary": binary, "type": type_model}


def binary_scores(proposals: Proposals, heads: dict) -> np.ndarray:
    return heads["binary"].predict_proba(proposals.features)[:, 1]


def classify_events(proposals: Proposals, scores: np.ndarray, heads: dict, threshold: float) -> list[dict]:
    """Negative actions stop at stage two; only accepted events invoke the type head.

    Adjacent accepted windows of the same action form an event. Time is the centre
    of its highest-scoring observed window; point labels do not supervise onsets.
    """
    accepted = np.flatnonzero(scores >= threshold)
    if not len(accepted):
        return []
    groups = [[int(accepted[0])]]
    for value in accepted[1:]:
        i, previous = int(value), groups[-1][-1]
        if i == previous + 1 and proposals.action_ids[i] == proposals.action_ids[previous]:
            groups[-1].append(i)
        else:
            groups.append([i])
    peaks = [max(g, key=lambda i: scores[i]) for g in groups]
    type_ids = heads["type"].predict(proposals.features[peaks])
    events = []
    for group, peak, type_id in zip(groups, peaks, type_ids, strict=True):
        # These templates require a specific observed object. Neither action model
        # provides object boxes, so do not fabricate that fact.
        reported_type = "other" if type_id in NEEDS_OBJECT else str(type_id)
        events.append(
            {
                "type": reported_type,
                "raw_type": str(type_id),
                "timecode_sec": float((proposals.starts[peak] + proposals.ends[peak]) / 2),
                "start_sec": float(proposals.starts[group[0]]),
                "end_sec": float(proposals.ends[group[-1]]),
                "action": proposals.action_names[int(proposals.action_ids[peak])],
                "action_score": float(proposals.action_confidence[peak]),
                "violation_score": float(scores[peak]),
            }
        )
    kept = []
    for event in sorted(events, key=lambda e: (-e["violation_score"], e["timecode_sec"])):
        if all(abs(event["timecode_sec"] - other["timecode_sec"]) >= MIN_GAP_SEC for other in kept):
            kept.append(event)
        if len(kept) == MAX_VIOLATIONS:
            break
    return kept


class ActionAnalyzer:
    """Warm, local serving adapter. Uploaded videos never enter a feature cache."""

    def __init__(self, name: str):
        expected = fingerprint(name)
        with (MODELS / f"action_{name}.pkl").open("rb") as file:
            self.artifact = pickle.load(file)
        if self.artifact["fingerprint"] != expected:
            raise ValueError("Action artifact is stale: rerun src.models.action_train")
        self.encoder = ActionEncoder(name)

    def __call__(self, path: Path):
        from contracts.schemas import Violation
        from src.data.evidence import evidence_frames
        from src.rules.describe import describe_violation

        proposals = extract(path, self.encoder)
        scores = binary_scores(proposals, self.artifact)
        events = classify_events(proposals, scores, self.artifact, self.artifact["threshold"])
        result = []
        for event in events:
            frames = evidence_frames(path, event["timecode_sec"], "action")
            name, description = describe_violation(event["type"], {})
            result.append(
                Violation(
                    type=name,
                    timecode_sec=frames[0].timestamp_sec,
                    description=description,
                    evidence_frames=frames,
                )
            )
        return result
