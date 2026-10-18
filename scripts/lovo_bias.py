"""Is the video AUC 0.33 of the window classifier real, or made by leave-one-video-out?

Diagnostics on the cached embeddings, no model and no GPU:

    uv run python -m scripts.lovo_bias d2    balanced LOVO: every fold trains on the same number of clips per class
    uv run python -m scripts.lovo_bias d3    leave-pair-out: one violation and one clean clip held out, 21 x 19 folds
    uv run python -m scripts.lovo_bias d5    do the nearest training clips predict the held-out score?
    uv run python -m scripts.lovo_bias dn2   kNN distance to a bank of normal windows of the same layout
    uv run python -m scripts.lovo_bias lto   leave-twin-out: a clip is scored without its nearest lookalike in training
    uv run python -m scripts.lovo_bias ref   score each clip against its own start: centred features, max minus median
    uv run python -m scripts.lovo_bias all   everything above
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import SEED
from src.data.labels import load_labels
from src.models.train import Clip, build_dataset, fit_on
from src.models.windows import NEGATIVE, choose_threshold, group_of, lovo_folds, video_score

SEEDS = range(SEED, SEED + 10)
NEIGHBOURS = 5  # D5: nearest training clips
KNN_K = (1, 5)  # DN2
TOP_N = 5  # DN2: top-n mean video score
CONTROL_SEEDS = SEEDS[:5]  # LTO control draws
REF_WINDOWS = (2, 4, 8)  # REF: windows at the start of a clip that serve as its scene reference (0.5 s apart)
RECALL_AT_FP = 9  # REF: false positives allowed when reporting recall (half of the clean clips)


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


def nearest_twin(mean_embeddings: np.ndarray) -> np.ndarray:
    """Per clip, the index of the most similar other clip (cosine of the clip's mean embedding)."""
    unit = mean_embeddings / np.maximum(np.linalg.norm(mean_embeddings, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    return np.argmax(sim, axis=1)


def random_stand_in(twin: np.ndarray, is_violation: Sequence[bool], seed: int) -> np.ndarray:
    """Per clip, a random clip of its twin's class that is neither the clip nor the twin.

    The control for a twin drop: the training side loses a clip of the same class, but not the lookalike.
    """
    rng = np.random.default_rng(seed)
    bad = np.asarray(is_violation, dtype=bool)
    out = np.empty(len(twin), dtype=int)
    for i, t in enumerate(twin):
        pool = [j for j in np.flatnonzero(bad == bad[t]) if j not in (i, t)]
        out[i] = rng.choice(pool)
    return out


def drop_folds(video_ids: Sequence[str], dropped: Sequence[int]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-video-out where clip `dropped[i]` also leaves the training side whenever clip i is held out.

    With `dropped` = each clip's nearest twin this is leave-twin-out: a clip is never scored by a model that
    saw its lookalike.
    """
    groups = _groups(video_ids)
    for g in sorted(set(groups)):
        val = groups == g
        out = {groups[dropped[i]] for i in np.flatnonzero(val)} | {g}
        yield np.flatnonzero(~np.isin(groups, list(out))), np.flatnonzero(val)


def pair_folds(
    video_ids: Sequence[str], is_violation: Sequence[bool], dropped: Sequence[int] | None = None
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Leave-pair-out: hold out one violation clip and one clean clip, train on the rest.

    Yields (train_idx, violation_idx, clean_idx) for every pair. AUC is the share of pairs where the violation
    clip scores higher, and it does not move with the class ratio of a fold (Airola et al. 2011).
    With `dropped`, the clip `dropped[i]` leaves the training side whenever clip i is held out.
    """
    groups = _groups(video_ids)
    bad = np.asarray(is_violation, dtype=bool)
    for i in np.flatnonzero(bad):
        for j in np.flatnonzero(~bad):
            out = [groups[i], groups[j]]
            if dropped is not None:
                out += [groups[dropped[i]], groups[dropped[j]]]
            yield np.flatnonzero(~np.isin(groups, out)), int(i), int(j)


def pair_auc(violation_scores: Sequence[float], clean_scores: Sequence[float]) -> float:
    """Share of (violation, clean) score pairs ordered the right way; a tie counts half."""
    pos = np.asarray(violation_scores, dtype=float)[:, None]
    neg = np.asarray(clean_scores, dtype=float)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def fp_at_full_recall(scores: Sequence[float], is_violation: Sequence[bool]) -> int:
    """Clean videos flagged at the ADR 0002 threshold: the lowest violation score, so every violation is caught."""
    s = np.asarray(scores, dtype=float)
    bad = np.asarray(is_violation, dtype=bool)
    return int((s[~bad] >= choose_threshold(s, bad)).sum())


def recall_at_fp(scores: Sequence[float], is_violation: Sequence[bool], max_fp: int) -> float:
    """Share of violation videos scoring strictly above the (max_fp + 1)-th highest clean video: recall with max_fp FP."""
    s = np.asarray(scores, dtype=float)
    bad = np.asarray(is_violation, dtype=bool)
    clean = np.sort(s[~bad])[::-1]
    floor = clean[max_fp] if max_fp < len(clean) else -np.inf
    return float((s[bad] > floor).mean())


def centre_on_start(feats: np.ndarray, n_windows: int) -> np.ndarray:
    """Window features minus the mean of the clip's first `n_windows` windows: change from the scene at the start.

    Uses the clip alone and no labels, so a service can do the same at inference. It assumes nothing violating
    happens inside the reference (the earliest label in data/labels.csv is at 5 s).
    """
    return feats - feats[:n_windows].mean(axis=0) if len(feats) else feats


def max_minus_median(window_scores: Sequence[float]) -> float:
    """Peak window logit above the clip's own median window logit: a video score that ignores the clip's level."""
    if not len(window_scores):
        return 0.0
    z = logit(np.clip(np.asarray(window_scores, dtype=float), 1e-6, 1 - 1e-6))
    return float(z.max() - np.median(z))


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


def _fp_text(fps: Sequence[int], n_clean: int) -> str:
    """FP count at recall 1.0 as 'x/19'; over seeds the mean with the range."""
    if len(set(fps)) == 1:
        return f"{fps[0]}/{n_clean}"
    return f"{np.mean(fps):.1f}/{n_clean} (min {min(fps)}, max {max(fps)})"


def run_baseline(c: Cohort) -> np.ndarray:
    scores = _oof_scores(c.clips, lovo_folds(c.ids))
    fp = _fp_text([fp_at_full_recall(scores, c.violation)], int((~c.violation).sum()))
    print(f"plain LOVO         video AUC {_auc(scores, c.violation):.3f}   FP at recall 1.0 {fp}")
    return scores


def run_d2(c: Cohort) -> None:
    """Balanced LOVO: the same model, every fold trained on 20 violation + 18 clean clips."""
    runs = [_oof_scores(c.clips, balanced_folds(c.ids, c.violation, s)) for s in SEEDS]
    aucs = [_auc(r, c.violation) for r in runs]
    fps = [fp_at_full_recall(r, c.violation) for r in runs]
    print(f"D2 balanced LOVO   video AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}   over {len(aucs)} seeds")
    print(
        f"                   min {min(aucs):.3f}  max {max(aucs):.3f}   FP at recall 1.0 {_fp_text(fps, int((~c.violation).sum()))}"
    )


def run_d3(c: Cohort) -> None:
    """Leave-pair-out: 21 x 19 folds, AUC = share of pairs ordered the right way.

    FP at recall 1.0 needs one score per video: the mean over the pairs the video was held out in.
    """
    pairs = list(pair_folds(c.ids, c.violation))
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(c.clips, train, (i, j)) for train, i, j in pairs)
    auc = float(np.mean([pair_auc([pos], [neg]) for pos, neg in out]))
    total, count = np.zeros(len(c.clips)), np.zeros(len(c.clips))
    for (_, i, j), (pos, neg) in zip(pairs, out, strict=True):
        total[i] += pos
        total[j] += neg
        count[i] += 1
        count[j] += 1
    fp = _fp_text([fp_at_full_recall(total / count, c.violation)], int((~c.violation).sum()))
    print(f"D3 leave-pair-out  video AUC {auc:.3f}   over {len(pairs)} pairs")
    print(f"                   FP at recall 1.0 {fp}   (score = mean over the video's pairs)")


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


def _dn2_oof(c: Cohort, folds: list[tuple[np.ndarray, np.ndarray]], k: int) -> np.ndarray:
    """Out-of-fold DN2 scores, one row per clip: (max, top-n mean) of its window distances."""
    got = Parallel(n_jobs=-1)(delayed(_dn2_score)(c, train, val, k) for train, val in folds)
    scores = np.zeros((len(c.clips), 2))
    for (_, val), g in zip(folds, got, strict=True):
        scores[val] = g
    return scores


def run_dn2(c: Cohort) -> None:
    """kNN distance to a bank of normal windows of the same layout, balanced inside that layout."""
    n_clean = int((~c.violation).sum())
    print("DN2 kNN bank       video AUC and FP at recall 1.0; bank = normal windows of the held-out layout")
    for k in KNN_K:
        auc, fp = [], []
        for seed in SEEDS:
            folds = list(balanced_folds(c.ids, c.violation, seed, layouts=c.layout))
            scores = _dn2_oof(c, folds, k)
            auc.append([_auc(scores[:, m], c.violation) for m in (0, 1)])
            fp.append([fp_at_full_recall(scores[:, m], c.violation) for m in (0, 1)])
        mean, sd, fps = np.mean(auc, axis=0), np.std(auc, axis=0), np.array(fp)
        for m, name in enumerate(("max", f"top-{TOP_N} mean")):
            print(
                f"                   k={k}  {name:<11} AUC {mean[m]:.3f} +- {sd[m]:.3f}   "
                f"FP at recall 1.0 {_fp_text(fps[:, m].tolist(), n_clean)}"
            )


def _clip_twins(c: Cohort) -> np.ndarray:
    dim = c.clips[0].feats.shape[1]
    mean = np.stack([x.feats.mean(axis=0) if len(x.feats) else np.zeros(dim) for x in c.clips])
    return nearest_twin(mean)


def _d3_scores(c: Cohort, dropped: np.ndarray | None) -> tuple[float, int]:
    """Leave-pair-out pair AUC and FP at recall 1.0 (score = mean over the video's pairs)."""
    pairs = list(pair_folds(c.ids, c.violation, dropped))
    out = Parallel(n_jobs=-1)(delayed(_fit_fold)(c.clips, train, (i, j)) for train, i, j in pairs)
    total, count = np.zeros(len(c.clips)), np.zeros(len(c.clips))
    for (_, i, j), (pos, neg) in zip(pairs, out, strict=True):
        total[[i, j]] += (pos, neg)
        count[[i, j]] += 1
    auc = float(np.mean([pair_auc([pos], [neg]) for pos, neg in out]))
    return auc, fp_at_full_recall(total / count, c.violation)


def run_lto(c: Cohort) -> None:
    """Leave-twin-out: the classifier, DN2 and D3 when the held-out clip's nearest lookalike is not in training.

    Three arms per method: plain, twin dropped, and a control that drops a random clip of the twin's class
    instead, so a change in class counts is not mistaken for a twin effect.
    """
    n_clean = int((~c.violation).sum())
    twin = _clip_twins(c)
    stand_ins = [random_stand_in(twin, c.violation, s) for s in CONTROL_SEEDS]
    print(
        f"LTO leave-twin-out nearest twin is of the other class for {np.mean(c.violation[twin] != c.violation):.0%} of clips"
    )
    print("                   method           arm        AUC             FP at recall 1.0")

    def line(method: str, arm: str, aucs: list[float], fps: list[int]) -> None:
        spread = f" +- {np.std(aucs):.3f}" if len(aucs) > 1 else "      "
        print(
            f"                   {method:<16} {arm:<10} {np.mean(aucs):.3f}{spread}   {_fp_text(fps, n_clean)}"
        )

    def folds_of(dropped: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(lovo_folds(c.ids) if dropped is None else drop_folds(c.ids, dropped))

    arms = [("plain", [None]), ("twin out", [twin]), ("control", stand_ins)]
    for arm, droppeds in arms:
        runs = [_oof_scores(c.clips, folds_of(d)) for d in droppeds]
        line(
            "classifier",
            arm,
            [_auc(r, c.violation) for r in runs],
            [fp_at_full_recall(r, c.violation) for r in runs],
        )
    for k in KNN_K:
        for arm, droppeds in arms:
            runs = [_dn2_oof(c, folds_of(d), k) for d in droppeds]
            for m, name in enumerate(("max", f"top-{TOP_N}")):
                line(
                    f"DN2 k={k} {name}",
                    arm,
                    [_auc(r[:, m], c.violation) for r in runs],
                    [fp_at_full_recall(r[:, m], c.violation) for r in runs],
                )
    for arm, droppeds in arms:
        runs = [_d3_scores(c, d) for d in droppeds]
        line("D3 leave-pair", arm, [r[0] for r in runs], [r[1] for r in runs])


def _fit_windows(clips: list[Clip], train: np.ndarray, held_out: Sequence[int]) -> dict[int, np.ndarray]:
    with threadpool_limits(1):
        clf = fit_on([clips[i] for i in train])
        return {
            i: clf.predict_proba(clips[i].feats)[:, 1] if len(clips[i].feats) else np.zeros(0)
            for i in held_out
        }


def _oof_windows(clips: list[Clip], folds: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Out-of-fold probability of every window of every clip."""
    got = Parallel(n_jobs=-1)(delayed(_fit_windows)(clips, train, val) for train, val in folds)
    out: list[np.ndarray] = [np.zeros(0)] * len(clips)
    for fold in got:
        for i, w in fold.items():
            out[i] = w
    return out


def run_ref(c: Cohort) -> None:
    """Does scoring a clip against its own start help? Features centred on the first windows, max minus median.

    Label-free and clip-local. Around 25 variants were tried on these 40 clips before this table, so read the
    plateau over reference lengths and the twin-out column, not the best single number.
    """
    n_clean = int((~c.violation).sum())
    twin = _clip_twins(c)
    arms = (("plain", list(lovo_folds(c.ids))), ("twin out", list(drop_folds(c.ids, twin))))
    variants = [("raw features", c.clips)] + [
        (f"first {n} centred", [replace(x, feats=centre_on_start(x.feats, n)) for x in c.clips])
        for n in REF_WINDOWS
    ]
    print(
        f"REF reference-centred scoring; recall is measured with {RECALL_AT_FP} of {n_clean} clean clips flagged"
    )
    print("                   features          score         arm        AUC     FP at recall 1.0   recall")
    for name, clips in variants:
        for arm, folds in arms:
            windows = _oof_windows(clips, folds)
            for score, fn in (("max", video_score), ("max-median", max_minus_median)):
                s = np.array([fn(w) for w in windows])
                print(
                    f"                   {name:<17} {score:<13} {arm:<10} {_auc(s, c.violation):.3f}   "
                    f"{fp_at_full_recall(s, c.violation):>3}/{n_clean}              "
                    f"{recall_at_fp(s, c.violation, RECALL_AT_FP):.2f}"
                )


COMMANDS = ("d2", "d3", "d5", "dn2", "lto", "ref", "all")


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
    if cmd in ("lto", "all"):
        run_lto(c)
    if cmd in ("ref", "all"):
        run_ref(c)


if __name__ == "__main__":
    main(sys.argv)
