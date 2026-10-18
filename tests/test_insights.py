"""docs/INSIGHTS.md: one verdict per hypothesis, computed from experiments.csv rows (TZ-8.3, #36).

Tests feed a tiny synthetic experiments table in both schemas the real file has had; they never read the
real experiments.csv (P2 appends to it) and never look at the PNGs.
"""

import json

import pandas as pd

from src.data.insights import BASELINE_NOTE, build_hypotheses, load_runs, render_md

COLUMNS = ["ts", "note", "params", "video_recall", "fp_rate", "median_abs_dt"]


def table(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def baseline(ts="2026-10-18T03:00:00", recall=1.0, fp=0.9, dt=5.0, **params) -> dict:
    params = {"median_abs_dt_by_mode": {"onset": 9.5, "peak": dt}, "n_videos": 40} | params
    return {
        "ts": ts, "note": BASELINE_NOTE, "params": json.dumps(params),
        "video_recall": recall, "fp_rate": fp, "median_abs_dt": dt,
    }  # fmt: skip


def verdicts(runs: pd.DataFrame) -> dict[str, str]:
    return {h.id: h.verdict for h in build_hypotheses(load_runs(runs))}


def test_peak_beats_onset_is_confirmed_from_the_latest_baseline_row():
    h = {h.id: h for h in build_hypotheses(load_runs(table(baseline(dt=5.0))))}["peak_vs_onset"]
    assert h.verdict == "подтверждена"
    assert "5,0" in h.numbers and "9,5" in h.numbers


def test_onset_not_worse_than_peak_is_rejected():
    runs = table(baseline(dt=12.0))  # onset 9.5 s is closer to the label than peak 12 s
    assert verdicts(runs)["peak_vs_onset"] == "отвергнута"


def test_stub_rows_are_dropped_and_the_latest_row_per_note_wins():
    stub = {
        "ts": "2026-10-19",
        "note": "stub-baseline",
        "params": "{}",
        "video_recall": 1.0,
        "fp_rate": 0.0,
        "median_abs_dt": 0.0,
    }
    old = baseline(ts="2026-10-17T20:00:00", dt=20.0)  # would reject the claim if it were used
    runs = load_runs(table(old, baseline(dt=5.0), stub))
    assert [r["note"] for r in runs] == [BASELINE_NOTE]
    assert runs[0]["median_abs_dt"] == 5.0


def test_metrics_missing_from_columns_come_from_params_json():
    early = baseline() | {"video_recall": None, "fp_rate": None, "median_abs_dt": None}
    early["params"] = json.dumps({"video_recall": 1.0, "fp_rate": 0.5, "median_abs_dt": 7.5})
    (run,) = load_runs(table(early))
    assert (run["video_recall"], run["fp_rate"], run["median_abs_dt"]) == (1.0, 0.5, 7.5)


def variant(note: str, ts="2026-10-18T05:00:00", recall=1.0, fp=0.9, dt=5.0, **params) -> dict:
    return {
        "ts": ts, "note": note, "params": json.dumps(params),
        "video_recall": recall, "fp_rate": fp, "median_abs_dt": dt,
    }  # fmt: skip


def test_without_a_matching_row_every_variant_hypothesis_is_untested():
    v = verdicts(table(baseline()))
    for hid in (
        "dinov2_vs_siglip",
        "whole_frame_vs_crops",
        "window_length",
        "median_subtraction",
        "heuristic_vote",
    ):
        assert v[hid] == "не проверена", hid


def test_siglip_row_decides_dinov2_vs_siglip_and_recall_comes_before_fp():
    # SigLIP has far fewer false positives but misses a violation: a missed violation is unacceptable (ADR 0002)
    siglip = variant("siglip-base-logreg-lovo", recall=0.95, fp=0.1)
    assert verdicts(table(baseline(recall=1.0, fp=0.9), siglip))["dinov2_vs_siglip"] == "подтверждена"
    better = variant("siglip-base-logreg-lovo", recall=1.0, fp=0.5)
    assert verdicts(table(baseline(recall=1.0, fp=0.9), better))["dinov2_vs_siglip"] == "отвергнута"


def test_equal_recall_and_fp_fall_back_to_timecode_error():
    siglip = variant("siglip-base-logreg-lovo", fp=0.9, dt=7.0)
    assert verdicts(table(baseline(fp=0.9, dt=5.0), siglip))["dinov2_vs_siglip"] == "подтверждена"


def test_siglip_comparison_does_not_mix_holdout_and_layout_with_lovo():
    runs = table(
        baseline(),
        variant("siglip-base-self-held-out", fp=0.0),
        variant("siglip-base-logreg-leave-one-layout-out", fp=0.0),
    )
    assert verdicts(runs)["dinov2_vs_siglip"] == "не проверена"


def test_a_tie_confirms_no_claim_that_one_side_is_better():
    tie = variant("siglip-base-logreg-lovo", fp=0.9, dt=5.0)  # same as the baseline on every metric
    assert verdicts(table(baseline(fp=0.9, dt=5.0), tie))["dinov2_vs_siglip"] == "отвергнута"


def test_crop_matcher_ignores_a_center_crop_switch_that_is_off():
    runs = table(baseline(), variant("dinov2-x", fp=0.1, center_crop=False))
    assert verdicts(runs)["whole_frame_vs_crops"] == "не проверена"
    runs = table(baseline(), variant("dinov2-x", fp=0.1, crops=True))
    assert verdicts(runs)["whole_frame_vs_crops"] == "отвергнута"


def test_window_matcher_ignores_the_default_window_length():
    runs = table(baseline(), variant("dinov2-w2", fp=0.1, window_sec=2.0))
    assert verdicts(runs)["window_length"] == "не проверена"


def test_variants_are_found_by_note_or_params_key():
    runs = table(
        baseline(),
        variant("dinov2-crops", fp=0.5),
        variant("dinov2-w4", fp=0.95, window_sec=4.0),
        variant("dinov2-median", fp=0.4, subtract_median=True),
        variant("dinov2-vote", fp=0.9, heuristic_vote=True),
    )
    v = verdicts(runs)
    assert v["whole_frame_vs_crops"] == "отвергнута"  # crops: 0.5 vs 0.9 is better, so whole frame loses
    assert v["window_length"] == "подтверждена"  # 2 s stays: 4 s window is worse (0.95 vs 0.9)
    assert v["median_subtraction"] == "подтверждена"  # claim is that subtracting helps; 0.4 vs 0.9 does
    assert v["heuristic_vote"] == "отвергнута"  # nothing changed, so the vote did not help


def cash_check(passed: bool, found=0.55) -> dict:
    params = {"cash_found_violation": found, "cash_found_clean": 0.74, "passed": passed}
    return {
        "ts": "2026-10-17T18:19:06", "note": "cash-check-dfine-owlv2", "params": json.dumps(params),
        "video_recall": None, "fp_rate": None, "median_abs_dt": None,
    }  # fmt: skip


def test_cash_check_verdict_follows_its_own_pass_flag():
    assert verdicts(table(cash_check(False)))["cash_check"] == "отвергнута"
    assert verdicts(table(cash_check(True, found=0.8)))["cash_check"] == "подтверждена"
    assert verdicts(table(baseline()))["cash_check"] == "не проверена"


def test_there_is_no_hypothesis_with_a_threshold_nobody_agreed_on():
    ids = [h.id for h in build_hypotheses(load_runs(table(baseline(fp=0.95))))]
    assert ids == [
        "dinov2_vs_siglip", "whole_frame_vs_crops", "window_length", "median_subtraction",
        "peak_vs_onset", "heuristic_vote", "cash_check",
    ]  # fmt: skip


def test_numbers_use_the_runs_values():
    h = {h.id: h for h in build_hypotheses(load_runs(table(baseline(dt=5.0), cash_check(False))))}
    assert "5,0" in h["peak_vs_onset"].numbers
    assert "55%" in h["cash_check"].numbers


def test_md_has_one_row_per_hypothesis_and_counts_the_tested_ones():
    hyps = build_hypotheses(load_runs(table(baseline(), cash_check(False))))
    md = render_md(hyps)
    rows = [line for line in md.splitlines() if line.startswith("| H")]
    assert len(rows) == len(hyps)
    tested = sum(h.verdict != "не проверена" for h in hyps)
    assert f"Проверено гипотез: **{tested}** из {len(hyps)}" in md
    assert all(any(v in row for v in ("подтверждена", "отвергнута", "не проверена")) for row in rows)


def test_md_links_a_figure_only_for_hypotheses_that_have_a_chart():
    hyps = build_hypotheses(load_runs(table(baseline(), cash_check(False))))
    md = render_md(hyps)
    row = {h.id: next(line for line in md.splitlines() if h.claim in line) for h in hyps}
    assert "peak_vs_onset.png" in row["peak_vs_onset"] and "cash_check.png" in row["cash_check"]
    assert ".png" not in row["dinov2_vs_siglip"]


def test_chart_titles_state_the_conclusion_with_the_runs_numbers():
    hyps = {h.id: h for h in build_hypotheses(load_runs(table(baseline(dt=5.0), cash_check(False))))}
    assert "5,0" in hyps["peak_vs_onset"].chart.title and "9,5" in hyps["peak_vs_onset"].chart.title
    assert "55%" in hyps["cash_check"].chart.title


def test_a_hypothesis_without_data_gets_no_chart():
    assert all(h.chart is None for h in build_hypotheses(load_runs(table(variant("stub-x")))))
