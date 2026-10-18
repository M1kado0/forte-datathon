"""Is the video AUC 0.33 of the window classifier real, or made by leave-one-video-out?

Diagnostics on the cached embeddings, no model and no GPU:

    uv run python -m scripts.lovo_bias d2    balanced LOVO: every fold trains on the same number of clips per class
    uv run python -m scripts.lovo_bias d3    leave-pair-out: one violation and one clean clip held out, 21 x 19 folds
    uv run python -m scripts.lovo_bias d5    do the nearest training clips predict the held-out score?
    uv run python -m scripts.lovo_bias dn2   kNN distance to a bank of normal windows of the same layout
    uv run python -m scripts.lovo_bias all   the four above
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import SEED
from src.data.labels import load_labels
from src.models.train import Clip, build_dataset, fit_on
from src.models.windows import NEGATIVE, group_of, lovo_folds, video_score

SEEDS = range(SEED, SEED + 10)
NEIGHBOURS = 5  # D5: nearest training clips
KNN_K = (1, 5)  # DN2
TOP_N = 5  # DN2: top-n mean video score


def _groups(video_ids: Sequence[str]) -> np.ndarray:
    return np.array([group_of(v) for v in video_ids])


def balanced_folds(
    video_ids: Sequence[str],
    is_violation: Sequence[bool],
    seed: int,
    layouts: Sequence[str] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """LOVO where the training side always holds the same number of clips per class.

    Plain LOVO takes the held-out clip out of its own class, so the training class ratio moves against the
    held-out label, which pulls pooled AUC below 0.5 (Parker et al. 2007, Austin et al. 2024). Here one random
    clip of the other class is dropped as well, so every fold trains on n_violation - 1 and n_clean - 1 clips.
    With `layouts` (one furniture arrangement per clip) the dropped clip comes from the held-out clip's own layout,
    so the clips of that layout, the only ones DN2 draws its bank from, are balanced the same way.
    """
    rng = np.random.default_rng(seed)
    groups = _groups(video_ids)
    bad = np.asarray(is_violation, dtype=bool)
    same_layout = np.ones(len(groups), dtype=bool)
    for g in sorted(set(groups)):
        val = groups == g
        held_bad = bool(bad[val].any())
        if layouts is not None:
            same_layout = np.asarray(layouts) == np.asarray(layouts)[val][0]
        others = sorted(set(groups[(bad != held_bad) & same_layout]) - {g})
        dropped = rng.choice(others)
        train = ~val & (groups != dropped)
        yield np.flatnonzero(train), np.flatnonzero(val)


def pair_folds(
    video_ids: Sequence[str], is_violation: Sequence[bool]
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Leave-pair-out: hold out one violation clip and one clean clip, train on the rest.

    Yields (train_idx, violation_idx, clean_idx) for every pair. AUC is the share of pairs where the violation
    clip scores higher, and it does not move with the class ratio of a fold (Airola et al. 2011).
    """
    groups = _groups(video_ids)
    bad = np.asarray(is_violation, dtype=bool)
    for i in np.flatnonzero(bad):
        for j in np.flatnonzero(~bad):
            train = ~np.isin(groups, [groups[i], groups[j]])
            yield np.flatnonzero(train), int(i), int(j)


def pair_auc(violation_scores: Sequence[float], clean_scores: Sequence[float]) -> float:
    """Share of (violation, clean) score pairs ordered the right way; a tie counts half."""
    pos = np.asarray(violation_scores, dtype=float)[:, None]
    neg = np.asarray(clean_scores, dtype=float)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def normal_bank(
    feats: Sequence[np.ndarray], labels: Sequence[np.ndarray], train_idx: np.ndarray
) -> np.ndarray:
    """Windows that count as normal: the NEGATIVE-labelled windows of the training clips, held-out clips never.

    A violation clip contributes the windows that end well before its label; positive and unlabeled windows stay out.
    """
    return np.concatenate([feats[i][labels[i] == NEGATIVE] for i in train_idx])


def knn_scores(bank: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Per query window, the mean Euclidean distance to its k nearest bank windows (DN2, Bergman et al. 2020)."""
    dist = cdist(queries, bank)
    k = min(k, dist.shape[1])
    return np.partition(dist, k - 1, axis=1)[:, :k].mean(axis=1)


def top_mean(scores: Sequence[float], n: int) -> float:
    """Mean of the n highest window scores: a video score less brittle than the single maximum; 0.0 without windows."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1][:n]
    return float(s.mean()) if len(s) else 0.0


# --- runners: real data from the embedding cache, printed as tables ---------------------------------------


@dataclass
class Cohort:
    """The labelled clips with what every diagnostic needs about them, in `ids` order."""

    ids: list[str]
    clips: list[Clip]
    violation: np.ndarray
    layout: np.ndarray  # furniture arrangement of each clip


def load_cohort() -> Cohort:
    labels = load_labels()
    ids, clips = build_dataset(labels)
    arrangement = labels.drop_duplicates("video_id").set_index("video_id")["arrangement"]
    return Cohort(
        ids, clips, np.array([c.is_violation for c in clips]), np.array([arrangement[v] for v in ids])
    )


def _clip_score(clf, clip: Clip) -> float:
    return video_score(clf.predict_proba(clip.feats)[:, 1]) if len(clip.feats) else 0.0


def _fit_fold(clips: list[Clip], train: np.ndarray, held_out: Sequence[int]) -> list[float]:
    with threadpool_limits(1):
        clf = fit_on([clips[i] for i in train])
        return [_clip_score(clf, clips[i]) for i in held_out]


def _oof_scores(clips: list[Clip], folds: Iterator[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Out-of-fold video score of every clip under the given folds."""
    folds = list(folds)
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(clips, train, val) for train, val in folds)
    scores = np.full(len(clips), np.nan)
    for (_, val), got in zip(folds, out, strict=True):
        scores[val] = got
    return scores


def _auc(scores: np.ndarray, violation: np.ndarray) -> float:
    return float(roc_auc_score(violation, scores))


def run_baseline(c: Cohort) -> np.ndarray:
    scores = _oof_scores(c.clips, lovo_folds(c.ids))
    print(f"plain LOVO         video AUC {_auc(scores, c.violation):.3f}   (the 0.33 under test)")
    return scores


def run_d2(c: Cohort) -> None:
    """Balanced LOVO: the same model, every fold trained on 20 violation + 18 clean clips."""
    aucs = [_auc(_oof_scores(c.clips, balanced_folds(c.ids, c.violation, s)), c.violation) for s in SEEDS]
    print(f"D2 balanced LOVO   video AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}   over {len(aucs)} seeds")
    print(f"                   min {min(aucs):.3f}  max {max(aucs):.3f}")


def run_d3(c: Cohort) -> None:
    """Leave-pair-out: 21 x 19 folds, AUC = share of pairs ordered the right way."""
    pairs = list(pair_folds(c.ids, c.violation))
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(c.clips, train, (i, j)) for train, i, j in pairs)
    auc = float(np.mean([pair_auc([pos], [neg]) for pos, neg in out]))
    print(f"D3 leave-pair-out  video AUC {auc:.3f}   over {len(pairs)} pairs")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def run_d5(c: Cohort, oof: np.ndarray) -> None:
    """Is the held-out score explained by who its neighbours are, rather than by the clip?"""
    dim = c.clips[0].feats.shape[1]
    mean = np.stack([x.feats.mean(axis=0) if len(x.feats) else np.zeros(dim) for x in c.clips])
    unit = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :NEIGHBOURS]
    neighbour_share = c.violation[nearest].mean(axis=1)
    fold_share = (c.violation.sum() - c.violation) / (
        len(c.violation) - 1
    )  # violation share of the training side
    print(
        f"D5 neighbours      share of violations among the {NEIGHBOURS} nearest clips (cosine, mean embedding)"
    )
    print(f"                   Spearman with the label            {_rho(neighbour_share, c.violation):+.3f}")
    print(f"                   Spearman with the LOVO score       {_rho(neighbour_share, oof):+.3f}")
    print(f"                   training share vs LOVO score       {_rho(fold_share, oof):+.3f}")


def _dn2_score(c: Cohort, train: np.ndarray, held_out: Sequence[int], k: int) -> list[tuple[float, float]]:
    """(max, top-n mean) of the window distances of each held-out clip to the bank of its own layout."""
    train = train[c.layout[train] == c.layout[held_out[0]]]
    bank = normal_bank([x.feats for x in c.clips], [x.windows["label"].to_numpy() for x in c.clips], train)
    scaler = StandardScaler().fit(bank)
    bank = scaler.transform(bank)
    out = []
    for i in held_out:
        if not len(c.clips[i].feats):
            out.append((0.0, 0.0))
            continue
        d = knn_scores(bank, scaler.transform(c.clips[i].feats), k)
        out.append((video_score(d), top_mean(d, TOP_N)))
    return out


def run_dn2(c: Cohort) -> None:
    """kNN distance to a bank of normal windows of the same layout, balanced inside that layout."""
    print("DN2 kNN bank       video AUC, bank = normal windows of training clips with the held-out layout")
    for k in KNN_K:
        results = []
        for seed in SEEDS:
            folds = list(balanced_folds(c.ids, c.violation, seed, layouts=c.layout))
            got = Parallel(n_jobs=-1)(delayed(_dn2_score)(c, train, val, k) for train, val in folds)
            scores = np.zeros((len(c.clips), 2))
            for (_, val), g in zip(folds, got, strict=True):
                scores[val] = g
            results.append([_auc(scores[:, 0], c.violation), _auc(scores[:, 1], c.violation)])
        mean, sd = np.mean(results, axis=0), np.std(results, axis=0)
        print(
            f"                   k={k}  max {mean[0]:.3f} +- {sd[0]:.3f}   top-{TOP_N} mean {mean[1]:.3f} +- {sd[1]:.3f}"
        )


COMMANDS = ("d2", "d3", "d5", "dn2", "all")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m scripts.lovo_bias {{{'|'.join(COMMANDS)}}}")
    cmd = argv[1]
    c = load_cohort()
    windows = sum(len(x.feats) for x in c.clips)
    print(f"{len(c.ids)} clips, {int(c.violation.sum())} with a violation, {windows} windows")
    if cmd in ("d5", "all"):  # only D5 needs the plain LOVO scores
        oof = run_baseline(c)
        run_d5(c, oof)
    if cmd in ("d2", "all"):
        run_d2(c)
    if cmd in ("d3", "all"):
        run_d3(c)
    if cmd in ("dn2", "all"):
        run_dn2(c)


if __name__ == "__main__":
    main(sys.argv)
