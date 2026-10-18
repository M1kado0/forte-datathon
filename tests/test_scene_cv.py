"""Scene folds (ADR 0006): every video has one scene, a fold never trains on the scene it scores, nested counts."""

import numpy as np

from src.data.labels import DUPLICATES, load_labels
from src.models import scene_cv
from src.models.scene_cv import Clip, load_scenes, summarize_nested
from src.models.windows import make_windows


def test_every_labelled_video_has_one_scene():
    scenes = load_scenes()
    assert set(scenes) == set(load_labels()["video_id"])
    assert not set(DUPLICATES) & set(scenes)  # merged duplicates are not takes of their own


def test_violation_takes_share_a_scene_with_their_clean_twins():
    scenes = load_scenes()
    # nearest take by median frame, the next clip number, the other verdict (docs/adr/0006)
    for violation, clean in [
        ("X-11", "X-12"),
        ("X-20", "X-21"),
        ("X-22", "X-23"),
        ("X-26", "X-27"),
        ("X-38", "X-39"),
    ]:
        assert scenes[violation] == scenes[clean]


class _Model:
    def __init__(self, trained):
        self.trained = trained

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def test_a_fold_trains_on_every_scene_but_the_one_it_scores(monkeypatch):
    trained = []
    monkeypatch.setattr(scene_cv, "N_JOBS", 1)
    monkeypatch.setattr(
        scene_cv, "fit_on", lambda clips: trained.append({c.vid for c in clips}) or _Model(clips)
    )
    groups = {"a1": "A", "a2": "A", "b1": "B", "c1": "C", "c2": "C"}
    clips = {}
    for vid in groups:
        clip = Clip(make_windows(6.0), np.zeros((9, 2), dtype=np.float32), [])
        clip.vid = vid
        clips[vid] = clip
    scores = scene_cv.out_of_fold(list(groups), clips, groups)
    assert set(scores) == set(groups)
    scenes = sorted(set(groups.values()))
    assert sorted(trained, key=sorted) == sorted(
        ({v for v, g in groups.items() if g != held_out} for held_out in scenes), key=sorted
    )


def test_nested_summary_matches_a_hand_count():
    rows = [
        {"video_id": "A", "is_violation": True, "threshold": 0.5, "score": 0.9, "warped_score": 0.1},
        {"video_id": "B", "is_violation": True, "threshold": 0.5, "score": 0.2, "warped_score": None},
        {"video_id": "C", "is_violation": False, "threshold": 0.5, "score": 0.7, "warped_score": 0.6},
        {"video_id": "D", "is_violation": False, "threshold": 0.5, "score": 0.1, "warped_score": None},
    ]
    out = summarize_nested(rows)
    assert out["nested_video_recall"] == 0.5
    assert out["nested_fp_rate"] == 0.5
    assert out["nested_missed"] == ["B"]
    camera = out["virtual_camera"]
    assert (camera["n_videos"], camera["n_violation"]) == (2, 1)
    assert (camera["violation_flagged_native"], camera["violation_flagged_warped"]) == (1, 0)
