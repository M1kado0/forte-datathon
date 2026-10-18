"""Is the video AUC 0.33 of the window classifier real, or made by leave-one-video-out? (docs/research/fp-generalisation.md)

Diagnostics on the cached embeddings, no model and no GPU:

    uv run python -m scripts.lovo_bias d2    balanced LOVO: every fold trains on the same number of clips per class
    uv run python -m scripts.lovo_bias d3    leave-pair-out: one violation and one clean clip held out, 21 x 19 folds
    uv run python -m scripts.lovo_bias d5    do the nearest training clips predict the held-out score?
    uv run python -m scripts.lovo_bias dn2   kNN distance to a bank of normal windows instead of a classifier
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.labels import load_labels
from src.models.train import Clip, build_dataset, fit_on
from src.models.windows import NEGATIVE, group_of, lovo_folds, video_score

SEEDS = range(10)
NEIGHBOURS = 5  # D5: nearest training clips
KNN_K = (1, 5)  # DN2
TOP_N = 5  # DN2: top-n mean video score


def _groups(video_ids: Sequence[str]) -> np.ndarray:
    return np.array([group_of(v) for v in video_ids])


def balanced_folds(
    video_ids: Sequence[str], is_violation: Sequence[bool], seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """LOVO where the training side always holds the same number of clips per class.

    Plain LOVO takes the held-out clip out of its own class, so the training class ratio moves against the
    held-out label, which pulls pooled AUC below 0.5 (Parker et al. 2007, Austin et al. 2024). Here one random
    clip of the other class is dropped as well, so every fold trains on n_violation - 1 and n_clean - 1 clips.
    """
    rng = np.random.default_rng(seed)
    groups = _groups(video_ids)
    bad = np.asarray(is_violation, dtype=bool)
    for g in sorted(set(groups)):
        val = groups == g
        held_bad = bool(bad[val].any())
        others = sorted(set(groups[bad != held_bad]) - {g})
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


def _load() -> tuple[list[str], list[Clip], np.ndarray]:
    ids, clips = build_dataset(load_labels())
    return ids, clips, np.array([c.is_violation for c in clips])


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


def run_baseline(ids, clips, violation) -> np.ndarray:
    scores = _oof_scores(clips, lovo_folds(ids))
    print(f"plain LOVO         video AUC {_auc(scores, violation):.3f}   (the 0.33 under test)")
    return scores


def run_d2(ids, clips, violation) -> None:
    """Balanced LOVO: the same model, every fold trained on 20 violation + 18 clean clips."""
    aucs = [_auc(_oof_scores(clips, balanced_folds(ids, violation, seed)), violation) for seed in SEEDS]
    print(f"D2 balanced LOVO   video AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}   over {len(aucs)} seeds")
    print(f"                   min {min(aucs):.3f}  max {max(aucs):.3f}")


def run_d3(ids, clips, violation) -> None:
    """Leave-pair-out: 21 x 19 folds, AUC = share of pairs ordered the right way."""
    pairs = list(pair_folds(ids, violation))
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(clips, train, (i, j)) for train, i, j in pairs)
    auc = float(np.mean([pair_auc([pos], [neg]) for pos, neg in out]))
    print(f"D3 leave-pair-out  video AUC {auc:.3f}   over {len(pairs)} pairs")


def run_d5(ids, clips, violation, oof: np.ndarray) -> None:
    """Is the held-out score explained by who its neighbours are, rather than by the clip?"""
    mean = np.stack(
        [c.feats.mean(axis=0) if len(c.feats) else np.zeros(clips[0].feats.shape[1]) for c in clips]
    )
    unit = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :NEIGHBOURS]
    neighbour_share = violation[nearest].mean(axis=1)
    train_share = (violation.sum() - violation) / (
        len(violation) - 1
    )  # class ratio of the fold, global channel
    rho = lambda a, b: spearmanr(a, b).statistic  # noqa: E731
    print(f"D5 neighbours      share of violations among the {NEIGHBOURS} nearest clips")
    print(f"                   Spearman with the label            {rho(neighbour_share, violation):+.3f}")
    print(f"                   Spearman with the LOVO score       {rho(neighbour_share, oof):+.3f}")
    print(f"                   fold class ratio vs LOVO score     {rho(train_share, oof):+.3f}")
    print(f"                   fold class ratio vs the label      {rho(train_share, violation):+.3f}")


def _dn2_score(
    clips: list[Clip], train: np.ndarray, held_out: Sequence[int], k: int
) -> list[tuple[float, float]]:
    bank = normal_bank([c.feats for c in clips], [c.windows["label"].to_numpy() for c in clips], train)
    scaler = StandardScaler().fit(bank)
    bank = scaler.transform(bank)
    out = []
    for i in held_out:
        if not len(clips[i].feats):
            out.append((0.0, 0.0))
            continue
        d = knn_scores(bank, scaler.transform(clips[i].feats), k)
        out.append((video_score(d), top_mean(d, TOP_N)))
    return out


def run_dn2(ids, clips, violation) -> None:
    """kNN distance to a bank of normal windows. Banks hold the same number of clips in every fold."""
    print("DN2 kNN bank       video AUC, balanced folds (bank built from the fold's training clips only)")
    for k in KNN_K:
        results = []
        for seed in SEEDS:
            folds = list(balanced_folds(ids, violation, seed))
            got = Parallel(n_jobs=-1)(delayed(_dn2_score)(clips, train, val, k) for train, val in folds)
            scores = np.zeros((len(clips), 2))
            for (_, val), g in zip(folds, got, strict=True):
                scores[val] = g
            results.append([_auc(scores[:, 0], violation), _auc(scores[:, 1], violation)])
        mx, tm = np.mean(results, axis=0), np.std(results, axis=0)
        print(
            f"                   k={k}  max {mx[0]:.3f} +- {tm[0]:.3f}   top-{TOP_N} mean {np.mean(results, axis=0)[1]:.3f} +- {tm[1]:.3f}"
        )


COMMANDS = ("d2", "d3", "d5", "dn2", "all")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m scripts.lovo_bias {{{'|'.join(COMMANDS)}}}")
    cmd = argv[1]
    ids, clips, violation = _load()
    print(
        f"{len(ids)} clips, {int(violation.sum())} with a violation, {sum(len(c.feats) for c in clips)} windows"
    )
    oof = run_baseline(ids, clips, violation)
    if cmd in ("d2", "all"):
        run_d2(ids, clips, violation)
    if cmd in ("d3", "all"):
        run_d3(ids, clips, violation)
    if cmd in ("d5", "all"):
        run_d5(ids, clips, violation, oof)
    if cmd in ("dn2", "all"):
        run_dn2(ids, clips, violation)


if __name__ == "__main__":
    main(sys.argv)
