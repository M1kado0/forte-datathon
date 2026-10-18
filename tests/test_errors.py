"""The error table and the limitations text (#38): rows from per-video verdicts, numbers filled from metrics."""

import numpy as np
import pandas as pd

from src.models import train
from src.models.errors import error_rows, limitations_text, render_table
from src.models.evaluate import Case
from src.models.train import Clip

ARR = {"X-1": "back", "X-2": "back", "X-3": "facing", "X-8": "facing"}
TRUTH = {"X-1": "id_not_presented", "X-2": "cash_partially_given"}
THRESHOLDS = {"LOVO": 0.3, "back": 0.4, "facing": 0.5}


def verdict(flagged, type_=None, score=0.5, hits=()):
    return {
        "flagged": flagged,
        "type": type_,
        "timecode_sec": 4.0 if flagged else None,
        "video_score": score,
        "hit_types": list(hits),
    }


def test_rows_list_false_negatives_before_false_positives_with_the_checks_that_failed():
    lovo = {
        "X-1": verdict(True, "other"),
        "X-2": verdict(False, score=0.01),
        "X-3": verdict(True, "other", 0.9),
        "X-8": verdict(False),
    }
    lolo = {
        "X-1": verdict(False, score=0.02),
        "X-2": verdict(False, score=0.03),
        "X-3": verdict(True, "other", 0.7),
        "X-8": verdict(False),
    }
    rows = error_rows(lovo, lolo, {}, TRUTH, ARR, THRESHOLDS)
    assert [(r["video_id"], r["kind"]) for r in rows] == [("X-1", "FN"), ("X-2", "FN"), ("X-3", "FP")]
    by_id = {r["video_id"]: r for r in rows}
    assert by_id["X-1"]["checks"] == ["leave-one-layout-out"]
    assert by_id["X-2"]["checks"] == ["LOVO", "leave-one-layout-out"]
    assert by_id["X-3"]["checks"] == ["LOVO", "leave-one-layout-out"]
    assert by_id["X-2"]["arrangement"] == "back" and by_id["X-2"]["label_type"] == "cash_partially_given"
    assert by_id["X-3"]["label_type"] is None and by_id["X-3"]["found_type"] == "other"


def test_a_nested_miss_is_a_row_even_when_the_other_checks_pass():
    ok = {"X-1": verdict(True, "other")}
    nested = {"X-1": {"score": 0.1, "threshold": 0.2}}
    rows = error_rows(ok, ok, nested, TRUTH, ARR, THRESHOLDS, nested_missed=["X-1"])
    assert [(r["video_id"], r["kind"], r["checks"]) for r in rows] == [("X-1", "FN", ["nested"])]
    assert rows[0]["found_type"] is None  # LOVO flagged it, but the nested check is what missed it
    assert "0.1000" in rows[0]["explanation"] and "0.2000" in rows[0]["explanation"]


def test_explanation_says_what_the_system_saw_not_why_it_is_wrong():
    lovo = {
        "X-3": verdict(True, "other", 0.9),
        "X-8": verdict(True, "outsider_in_cash_room", 0.1, ["outsider_in_cash_room"]),
    }
    rows = error_rows(lovo, {}, {}, {}, ARR, THRESHOLDS)
    by_id = {r["video_id"]: r["explanation"] for r in rows}
    assert "0.9000" in by_id["X-3"] and "0.3000" in by_id["X-3"]  # window score over the threshold
    assert "эвристик" in by_id["X-8"]  # score under the threshold: only the heuristics' vote flagged it


def test_table_renders_one_markdown_row_per_error():
    rows = error_rows({"X-3": verdict(True, "other", 0.9)}, {}, {}, {}, ARR, THRESHOLDS)
    text = render_table(rows)
    assert text.count("\n") == 2 + len(rows) - 1 and "X-3" in text and "facing" in text


def test_limitations_text_takes_every_number_from_the_metrics():
    layouts = {
        "back": {"n_videos": 14, "n_violation_videos": 7, "n_clean_videos": 7, "video_recall": 0.857, "fp_rate": 1.0},
        "facing": {"n_videos": 26, "n_violation_videos": 14, "n_clean_videos": 12, "video_recall": 1.0, "fp_rate": 0.75},
    }  # fmt: skip
    metrics = {
        "n_videos": 40,
        "n_violation_videos": 21,
        "n_clean_videos": 19,
        "fp_rate": 0.947,
        "video_recall": 1.0,
        "type_accuracy": 0.095,
        "leave_one_layout_out": {"layouts": layouts},
    }
    text = limitations_text(metrics)
    assert "один ракурс" in text
    for number in ("40", "21", "95%", "86%", "75%", "10%"):
        assert number in text


def _clip(rng, positive):
    n = 12
    windows = pd.DataFrame(
        {"start": np.arange(n) * 0.5, "end": np.arange(n) * 0.5 + 2.0, "label": [int(positive)] * n}
    )
    return Clip(windows, rng.normal(size=(n, 3)) + positive * 2, 0.0, 8.0, [3] if positive else [])


def test_layout_check_trains_each_fold_on_the_other_layout_only(monkeypatch):
    rng = np.random.default_rng(0)
    ids = ["X-1", "X-2", "X-3", "X-4", "X-5", "X-8", "X-10", "X-11"]
    layout = dict(zip(ids, ["back"] * 4 + ["facing"] * 4, strict=True))
    clips = [_clip(rng, i % 2 == 0) for i in range(len(ids))]
    cases = [
        Case(
            v, c.windows["start"].to_numpy(), np.zeros(12), None, [], [(3, "other")] if c.is_violation else []
        )
        for v, c in zip(ids, clips, strict=True)
    ]
    fitted = []
    real_fit = train.fit_on
    monkeypatch.setattr(train, "fit_on", lambda cs: fitted.append({id(c) for c in cs}) or real_fit(cs))
    out = train.layout_check(ids, clips, cases, {}, layout)
    by_layout = {
        n: {id(c) for v, c in zip(ids, clips, strict=True) if layout[v] == n} for n in ("back", "facing")
    }
    assert all(
        f <= by_layout["back"] or f <= by_layout["facing"] for f in fitted
    )  # a fit never mixes layouts
    assert set(out["layouts"]) == {"back", "facing"}
    assert out["layouts"]["back"]["n_videos"] == 4 and out["layouts"]["back"]["trained_on"] == "facing"
    assert set(out["layouts"]["back"]["verdicts_by_video"]) == {"X-1", "X-2", "X-3", "X-4"}
    assert out["pooled"]["n_violation_videos"] == 4
