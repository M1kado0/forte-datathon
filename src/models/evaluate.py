"""Leave-one-video-out evaluation of the type choice and the heuristics' vote (#34), and the choice of their settings.

Everything that is chosen is chosen on the videos outside the held-out clip's group and applied to it, so every
number is out of sample: the heuristic thresholds, the types the heuristics may name, the vote's N and types, and
kNN's k and share. `fit_settings` on all videos gives the settings `train` saves. Branch A's window scores are the
out-of-fold ones from `train.cross_validate`.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from statistics import median

import numpy as np
import pandas as pd

from src.features import tracks as track_cache
from src.features.detectors import LocalDetector
from src.features.embeddings import video_path
from src.heuristics import Hit, Thresholds, detect, vote
from src.heuristics.detect import ID_NOT_PRESENTED
from src.heuristics.thresholds import DEFAULT
from src.models.violations import (
    FALLBACK_TYPE,
    HEURISTIC_CONFIGS,
    HEURISTIC_TYPES,
    Found,
    Knn,
    Settings,
    Vote,
    decide,
    evidence_kind,
    knn_type,
    window_score,
)
from src.models.windows import POSITIVE, WINDOW_SEC, group_of

VOTE_GRID_SEC = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)  # a hit is at least 3 s long already (Thresholds.min_*_sec)
KNN_GRID = [(k, share) for k in (5, 15) for share in (0.6, 0.8, 1.0)]
# A heuristic names a type only if, where it fires, the type was the labelled one half the time and on at least
# one video. One video is thin evidence; with two, a fold that holds one of two positives out keeps nothing.
MIN_TYPE_PRECISION = 0.5
MIN_TYPE_VIDEOS = 1
MIN_KNN_ANSWERS = 3  # kNN is judged on at least this many answers
# min_payout_sec values tried for id_not_presented (#54). Nothing is selected from them, so there is no fold.
PAYOUT_GRID_SEC = (3.0, 5.0, 8.0, 15.0)


@dataclass
class Case:
    """One video: branch A's out-of-fold window scores, the heuristics' hits and the labelled violations."""

    video_id: str
    starts: np.ndarray
    scores: np.ndarray
    peak_feature: np.ndarray | None  # features of the best-scoring window
    hits: list[Hit]  # of the default heuristic thresholds
    truth: list[tuple[int, str]]  # (second, type) of each violation label; [] for a clean video
    hits_by: dict[str, list[Hit]] = field(default_factory=dict)  # per key of HEURISTIC_CONFIGS

    @property
    def types(self) -> set[str]:
        return {t for _, t in self.truth}

    @property
    def is_violation(self) -> bool:
        return bool(self.truth)

    @property
    def group(self) -> str:
        return group_of(self.video_id)

    def view(self, config: str) -> Case:
        """This video as the heuristics of `config` see it."""
        return replace(self, hits=self.hits_by[config]) if self.hits_by else self


# --- inputs -----------------------------------------------------------------------------------------------


def files_by_video(labels: pd.DataFrame) -> dict[str, str]:
    return labels.drop_duplicates("video_id").set_index("video_id")["file"].to_dict()


def heuristic_hits(files: dict[str, str], thresholds: Thresholds = DEFAULT) -> dict[str, list[Hit]]:
    """Heuristic hits of each video, from its cached tracks (built once on a miss)."""
    detector = functools.cache(LocalDetector)
    hits = {}
    for vid, file in files.items():
        tracks = track_cache.load_tracks(video_path(file))
        hits[vid] = detect(
            tracks if tracks is not None else track_cache.get_tracks(video_path(file), detector()),
            thresholds=thresholds,
        )
    return hits


def payout_sweep(labels: pd.DataFrame, ids: list[str]) -> list[dict]:
    """Videos where id_not_presented fires at each min_payout_sec, split by what the video is labelled with."""
    files = files_by_video(labels)
    clean = set(labels.loc[labels["verdict"] == "no_violation", "video_id"])
    right = set(labels.loc[labels["violation_type"] == ID_NOT_PRESENTED, "video_id"])
    sweep = []
    for min_payout in PAYOUT_GRID_SEC:
        hits = heuristic_hits({v: files[v] for v in ids}, replace(DEFAULT, min_payout_sec=min_payout))
        fired = {v for v in ids if any(h.type_id == ID_NOT_PRESENTED for h in hits[v])}
        sweep.append(
            {
                "min_payout_sec": min_payout,
                "fired_right": len(fired & right),
                "fired_clean": len(fired & clean),
                "fired_other_type": len(fired - right - clean),
            }
        )
    return sweep


def bank_windows(
    windows, feats: np.ndarray, truth: Sequence[tuple[int, str]]
) -> tuple[np.ndarray, list[str]]:
    """Positive windows of one video and the type of the label nearest to each."""
    rows = np.flatnonzero(windows["label"].to_numpy() == POSITIVE)
    centre = windows["start"].to_numpy()[rows] + WINDOW_SEC / 2
    types = [min(truth, key=lambda lab: abs(lab[0] - c))[1] for c in centre] if truth else []
    return feats[rows], types


def run_case(case: Case, threshold: float, settings: Settings) -> list[Found]:
    view = case.view(settings.heuristics)
    return decide(view.starts, view.scores, threshold, view.hits, settings, view.peak_feature)


# --- choosing the settings --------------------------------------------------------------------------------


def fit_allowed(cases: Sequence[Case]) -> tuple[frozenset[str], dict]:
    """Heuristic types that may name a violation, and per type (right videos, videos where it fires)."""
    fired: Counter[str] = Counter()
    right: Counter[str] = Counter()
    for c in cases:
        for t in {h.type_id for h in c.hits}:
            fired[t] += 1
            right[t] += t in c.types
    allowed = frozenset(
        t for t in fired if right[t] >= MIN_TYPE_VIDEOS and right[t] / fired[t] >= MIN_TYPE_PRECISION
    )
    return allowed, {t: {"right": right[t], "fired": fired[t]} for t in sorted({*HEURISTIC_TYPES, *fired})}


def fit_vote(cases: Sequence[Case], threshold: float) -> tuple[Vote, dict]:
    """Smallest N at which the vote, limited to types with no false positive of their own on clean videos, adds
    no false positive to branch A's verdict. Off when there is no such N.

    A type that fires alone on a clean video is left out even where branch A flags that video anyway: on
    these clips branch A flags nearly every clean video, so "adds no false positive" alone tests almost nothing.
    """
    clean = [c for c in cases if not c.is_violation]
    unflagged = [c for c in clean if not (c.scores.size and c.scores.max() >= threshold)]
    seen = {h.type_id for c in cases for h in c.hits}
    table = []
    for n_sec in VOTE_GRID_SEC:
        alone = {t: [c for c in cases if vote(c.hits, n_sec, types=frozenset({t}))] for t in seen}
        # a type votes if it fires on some violation video and on no clean one
        types = frozenset(t for t, fired in alone.items() if fired and all(c.is_violation for c in fired))
        added = sum(vote(c.hits, n_sec, types=types) for c in unflagged) if types else 0
        table.append({"n_sec": n_sec, "types": sorted(types), "added_false_positives": added})
    ok = [row for row in table if row["types"] and row["added_false_positives"] == 0]
    notes = {"grid": table, "clean_videos": len(clean), "clean_videos_branch_a_misses": len(unflagged)}
    if not ok:
        return Vote(enabled=False), notes
    return Vote(enabled=True, n_sec=ok[0]["n_sec"], types=frozenset(ok[0]["types"])), notes


def fit_knn(cases: Sequence[Case], banks: dict[str, tuple[np.ndarray, list[str]]]) -> tuple[Knn | None, dict]:
    """kNN with the bank of all of `banks`, or None when no k and share name the right type for at least half of
    their answers.

    Each answer is out of sample: the bank for a clip leaves out its group. `banks` maps video id -> (features,
    types) of its positive windows.
    """
    grid = []
    for k, share in KNN_GRID:
        right = answered = 0
        for c in cases:
            if not c.is_violation or c.peak_feature is None:
                continue
            bank = _bank(banks, c.group)
            if bank is None:
                continue
            answer = knn_type(c.peak_feature, Knn(bank[0], bank[1], k, share))
            answered += answer is not None
            right += answer in c.types
        grid.append({"k": k, "min_share": share, "answered": answered, "right": right})
    good = [
        g
        for g in grid
        if g["answered"] >= MIN_KNN_ANSWERS and g["right"] / g["answered"] >= MIN_TYPE_PRECISION
    ]
    notes = {"grid": grid}
    if not good:
        return None, notes
    best = max(good, key=lambda g: (g["right"], -g["answered"]))
    features, types = _bank(banks)  # not None: a setting that answered had a bank to answer from
    return Knn(features, types, best["k"], best["min_share"]), notes


def _bank(
    banks: dict[str, tuple[np.ndarray, list[str]]], without_group: str | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Positive windows of every video outside `without_group` (all of them by default) and their types."""
    parts = [(x, t) for v, (x, t) in banks.items() if len(t) and group_of(v) != without_group]
    if not parts:
        return None
    return np.concatenate([x for x, _ in parts]), np.array([s for _, t in parts for s in t])


def _typed(cases: Sequence[Case], threshold: float, settings: Settings) -> tuple[int, int]:
    """(violation videos whose first violation has the labelled type, videos named with a wrong type)."""
    right = wrong = 0
    for c in cases:
        found = run_case(c, threshold, settings)
        if found and found[0].type_id in c.types:
            right += 1
        elif found and found[0].type_id != FALLBACK_TYPE:
            wrong += 1
    return right, wrong


def fit_settings(cases: Sequence[Case], threshold: float, banks: dict | None = None) -> tuple[Settings, dict]:
    """Settings fitted on `cases`, and the numbers behind each choice for metrics.json.

    The heuristic thresholds are the HEURISTIC_CONFIGS key with the most right types on `cases`, then the fewest
    wrong ones; the first key (the default) wins a tie.
    """
    tried = {}
    for name in HEURISTIC_CONFIGS:
        view = [c.view(name) for c in cases]
        allowed, per_type = fit_allowed(view)
        chosen_vote, vote_notes = fit_vote(view, threshold)
        settings = Settings(allowed, chosen_vote, None, name)
        tried[name] = (settings, per_type, vote_notes, *_typed(view, threshold, settings))
    best = max(tried, key=lambda name: (tried[name][3], -tried[name][4]))
    settings, per_type, vote_notes, _, _ = tried[best]
    knn, knn_notes = fit_knn(cases, banks) if banks else (None, {})
    notes = {
        "heuristics": {
            "chosen": best,
            "tried": {name: {"typed_right": t[3], "named_wrong": t[4]} for name, t in tried.items()},
        },
        "per_type": per_type,
        "vote": vote_notes,
        "knn": knn_notes,
    }
    return replace(settings, knn=knn), notes


# --- leave-one-video-out ----------------------------------------------------------------------------------


def lovo(cases: Sequence[Case], threshold: float, banks: dict | None = None) -> dict:
    """Metrics of the whole decision when every clip is judged with settings fitted on the other clips: the
    held-out clip's group is out of the cases and out of kNN's bank."""
    first: list[Found | None] = []
    seen: list[Case] = []  # each clip as the heuristics chosen for its fold saw it
    for c in cases:
        rest = [o for o in cases if o.group != c.group]
        rest_banks = {v: b for v, b in banks.items() if group_of(v) != c.group} if banks else None
        settings, _ = fit_settings(rest, threshold, rest_banks)
        found = run_case(c, threshold, settings)
        first.append(found[0] if found else None)
        seen.append(c.view(settings.heuristics))
    return summarize(seen, first)


def _delta(truth: Sequence[tuple[int, str]], second: float) -> float:
    return min(abs(second - sec) for sec, _ in truth)


def summarize(cases: Sequence[Case], first: Sequence[Found | None]) -> dict:
    """Recall, FP rate, median |Δt| and type accuracy of the strongest violation of each video."""
    violation = [(c, f) for c, f in zip(cases, first, strict=True) if c.is_violation]
    clean = [(c, f) for c, f in zip(cases, first, strict=True) if not c.is_violation]
    flagged = [(c, f) for c, f in violation if f is not None]
    typed = [(c, f) for c, f in flagged if f.type_id in c.types]
    named = [(c, f) for c, f in flagged if f.type_id != FALLBACK_TYPE]
    false_positives = sum(f is not None for _, f in clean)
    return {
        "video_recall": len(flagged) / len(violation) if violation else None,
        "fp_rate": false_positives / len(clean) if clean else None,
        "n_clean_videos": len(clean),
        "n_false_positives": false_positives,
        "median_abs_dt": _median([_delta(c.truth, f.timecode_sec) for c, f in flagged]),
        "type_accuracy": len(typed) / len(violation) if violation else None,
        "n_typed_right": len(typed),
        "n_named": len(named),  # videos whose first violation has a type other than the fallback
        "n_violation_videos": len(violation),
        "types_by_video": {
            c.video_id: {"truth": sorted(c.types), "found": f.type_id if f else None} for c, f in violation
        },
        "timecode_choice": _timecode_choice(cases, flagged),
    }


def _timecode_choice(cases: Sequence[Case], flagged: Sequence[tuple[Case, Found]]) -> dict:
    """Median |Δt| of the start and of the peak, by kind of type, so the choice by scenario rests on numbers.

    `state` is judged on the episodes `decide` reports: violations of a state type whose type is the labelled one
    (the hit's start against the centre of its strongest window). `nearest_episode` is the same on the hit nearest
    each label, which assumes the right episode was picked and so flatters both; the two differ when a clip has
    several episodes of one type. `action`: every flagged video whose first violation is an action type, the
    fallback included; the start is the end of the first window over the threshold, the peak the best window's centre.
    """
    onset, peak = [], []
    for c in cases:
        for sec, type_id in c.truth:
            hits = [h for h in c.hits if h.type_id == type_id]
            if evidence_kind(type_id) != "state" or not hits:
                continue
            onset.append(min(abs(h.start_sec - sec) for h in hits))
            peak.append(min(abs((window_score(h, c.starts, c.scores)[1] or h.start_sec) - sec) for h in hits))
    picked = [(c, f) for c, f in flagged if f.kind == "state" and f.type_id in c.types]
    action = [(c, f) for c, f in flagged if f.kind == "action"]
    return {
        "state": {
            "n": len(picked),
            "onset": _median([_delta(c.truth, f.onset_sec) for c, f in picked]),
            "peak": _median([_delta(c.truth, f.peak_sec) for c, f in picked]),
            "nearest_episode": {"n": len(onset), "onset": _median(onset), "peak": _median(peak)},
        },
        "action": {
            "n": len(action),
            "onset": _median([_delta(c.truth, f.onset_sec) for c, f in action]),
            "peak": _median([_delta(c.truth, f.peak_sec) for c, f in action]),
        },
    }


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None
