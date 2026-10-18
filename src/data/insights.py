"""`python -m src.data.insights`: docs/INSIGHTS.md and reports/figures/ from experiments.csv (TZ-8.3)."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import EXPERIMENTS, FIGURES, INSIGHTS_MD
from src.features.cash_check import CASH_GOAL

BASELINE_NOTE = "dinov2-small-logreg-lovo"  # the run every variant is compared with
CONFIRMED, REJECTED, UNTESTED = "подтверждена", "отвергнута", "не проверена"
METRICS = ("video_recall", "fp_rate", "median_abs_dt")
FP_LIMIT = 0.5  # team bar for "the verdict says something": at most half of the clean videos flagged


@dataclass(frozen=True)
class Hypothesis:
    id: str
    claim: str
    verdict: str
    number: str
    conclusion: str


def _num(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def load_runs(df: pd.DataFrame) -> list[dict]:
    """Latest row per note, stubs dropped; a metric empty in its column is read from the params JSON."""
    runs: dict[str, dict] = {}
    for row in df.sort_values("ts").to_dict("records"):
        if str(row["note"]).startswith("stub"):
            continue
        params = json.loads(row["params"]) if isinstance(row["params"], str) and row["params"] else {}
        run = {"ts": row["ts"], "note": row["note"], "params": params}
        for m in METRICS:
            value = row.get(m)
            run[m] = params.get(m) if pd.isna(value) else float(value)
        runs[row["note"]] = run
    return list(runs.values())


def _has_metrics(run: dict) -> bool:
    return all(run[m] is not None for m in METRICS)


def _base(runs: list[dict]) -> dict | None:
    return next((r for r in runs if r["note"] == BASELINE_NOTE and _has_metrics(r)), None)


def _peak_vs_onset(runs: list[dict]) -> Hypothesis:
    claim = "Таймкод по пику окон ближе к метке, чем по началу серии"
    by_mode = ((_base(runs) or {}).get("params") or {}).get("median_abs_dt_by_mode")
    if not by_mode:
        return Hypothesis("peak_vs_onset", claim, UNTESTED, "—", "")
    peak, onset = by_mode["peak"], by_mode["onset"]
    number = f"медиана ошибки таймкода: пик {_num(peak)} с, начало {_num(onset)} с"
    verdict = CONFIRMED if peak < onset else REJECTED
    return Hypothesis(
        "peak_vs_onset", claim, verdict, number, "Лучше пик" if peak < onset else "Лучше начало"
    )


@dataclass(frozen=True)
class Comparison:
    """A claim settled by one variant run against the baseline run (ADR 0002 order of metrics)."""

    id: str
    claim: str
    variant_wins: bool  # True: the claim says the variant is better; False: it says the baseline is
    matches: Callable[[dict], bool]
    needs: str  # what a run must look like to settle the claim, shown while it is untested


COMPARISONS = [
    Comparison(
        "dinov2_vs_siglip", "DINOv2-small лучше SigLIP-base", False,
        lambda r: r["note"].startswith("siglip"), "note начинается с `siglip`",
    ),
    Comparison(
        "whole_frame_vs_crops", "Эмбеддинг всего кадра лучше эмбеддингов кропов", False,
        lambda r: "crop" in r["note"] or any("crop" in k for k in r["params"]),
        "note или ключ params со словом `crop`",
    ),
    Comparison(
        "window_length", "Окно 2 с лучше других длин окна", False,
        lambda r: "window_sec" in r["params"], "в params ключ `window_sec` с другой длиной окна",
    ),
    Comparison(
        "median_subtraction", "Вычитание медианного эмбеддинга видео улучшает результат", True,
        lambda r: bool(r["params"].get("subtract_median")), "в params `subtract_median: true`",
    ),
    Comparison(
        "heuristic_vote", "Голос эвристик за вердикт улучшает результат (ADR 0003)", True,
        lambda r: bool(r["params"].get("heuristic_vote")) or "vote" in r["note"],
        "в params `heuristic_vote: true` или `vote` в note",
    ),
]  # fmt: skip


def _rank(run: dict) -> tuple[float, float, float]:
    """Smaller is better: recall first (a missed violation is unacceptable), then FP, then timecode error."""
    return (-run["video_recall"], run["fp_rate"], run["median_abs_dt"])


def _compare(c: Comparison, runs: list[dict]) -> Hypothesis:
    base = _base(runs)
    candidates = [r for r in runs if r["note"] != BASELINE_NOTE and _has_metrics(r) and c.matches(r)]
    if base is None or not candidates:
        return Hypothesis(c.id, c.claim, UNTESTED, "—", f"Нужна строка в experiments.csv: {c.needs}")
    other = max(candidates, key=lambda r: r["ts"])
    variant_better = _rank(other) < _rank(base)
    verdict = CONFIRMED if variant_better == c.variant_wins else REJECTED
    number = f"база: {_summary(base)}; {other['note']}: {_summary(other)}"
    return Hypothesis(c.id, c.claim, verdict, number, f"Лучше {other['note'] if variant_better else 'база'}")


def _summary(run: dict) -> str:
    return f"recall {run['video_recall']:.0%}, FP {run['fp_rate']:.0%}, ошибка таймкода {_num(run['median_abs_dt'])} с"


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _threshold_fp(runs: list[dict]) -> Hypothesis:
    claim = f"Порог с recall 100% оставляет ложными не больше {_pct(FP_LIMIT)} чистых видео (ADR 0002)"
    base = _base(runs)
    if base is None:
        return Hypothesis("threshold_fp", claim, UNTESTED, "—", "")
    ok = base["video_recall"] >= 1.0 and base["fp_rate"] <= FP_LIMIT
    number = f"recall {_pct(base['video_recall'])}, ложных {_pct(base['fp_rate'])} чистых видео"
    conclusion = (
        "Вердикт различает видео"
        if ok
        else "Цена recall 100% выше ориентира; ложные срабатывания показываем честно (ADR 0002)"
    )
    return Hypothesis("threshold_fp", claim, CONFIRMED if ok else REJECTED, number, conclusion)


def _cash_check(runs: list[dict]) -> Hypothesis:
    claim = f"OWLv2 находит деньги не менее чем на {_pct(CASH_GOAL)} кадров с нарушениями (проверка #19)"
    run = next((r for r in runs if r["note"].startswith("cash-check") and "passed" in r["params"]), None)
    if run is None:
        return Hypothesis("cash_check", claim, UNTESTED, "—", "")
    p = run["params"]
    number = f"деньги на {_pct(p['cash_found_violation'])} кадров нарушений, на {_pct(p['cash_found_clean'])} чистых"
    conclusion = (
        "Деньги в кадре годятся как признак"
        if p["passed"]
        else "Деньги в кадре не опора; запасной вариант — изменения в зоне стола"
    )
    return Hypothesis("cash_check", claim, CONFIRMED if p["passed"] else REJECTED, number, conclusion)


def build_hypotheses(runs: list[dict]) -> list[Hypothesis]:
    compared = {c.id: _compare(c, runs) for c in COMPARISONS}
    vote = compared.pop("heuristic_vote")
    return [
        *compared.values(),
        _peak_vs_onset(runs),
        vote,
        _threshold_fp(runs),
        _cash_check(runs),
    ]


@dataclass(frozen=True)
class Chart:
    """What one figure shows; `file` is the hypothesis id plus .png. Bars are grey, `accent` is magenta."""

    file: str
    title: str  # states the conclusion, built from the run's numbers
    labels: list[str]
    values: list[float]
    accent: int
    unit: str  # "%" (values are shares) or "с"
    reference: tuple[str, float] | None = None  # dashed line: label, value


def build_charts(runs: list[dict]) -> list[Chart]:
    """One chart for each hypothesis that has numbers; the others get none."""
    charts = []
    base = _base(runs)
    by_mode = ((base or {}).get("params") or {}).get("median_abs_dt_by_mode")
    if by_mode:
        peak, onset = by_mode["peak"], by_mode["onset"]
        better = "пику" if peak < onset else "началу"
        title = (
            f"Таймкод по {better} ближе к метке: {_num(min(peak, onset))} с против {_num(max(peak, onset))} с"
        )
        charts.append(
            Chart(
                "peak_vs_onset.png",
                title,
                ["по началу серии", "по пику"],
                [onset, peak],
                int(peak < onset),
                "с",
            )
        )
    if base:
        recall, fp = base["video_recall"], base["fp_rate"]
        title = (
            f"Recall {_pct(recall)} куплен ценой: ложными стали {_pct(fp)} чистых видео"
            if fp > FP_LIMIT
            else f"Recall {_pct(recall)} при ложных {_pct(fp)} чистых видео"
        )
        charts.append(
            Chart(
                "threshold_fp.png",
                title,
                ["Recall", "Ложные\nсрабатывания"],
                [recall, fp],
                1,
                "%",
                ("ориентир", FP_LIMIT),
            )
        )
    cash = next(
        (r["params"] for r in runs if r["note"].startswith("cash-check") and "passed" in r["params"]), None
    )
    if cash:
        found, clean = cash["cash_found_violation"], cash["cash_found_clean"]
        title = (
            f"Деньги находятся на {_pct(found)} кадров нарушений — цель {_pct(CASH_GOAL)} достигнута"
            if cash["passed"]
            else f"Деньги находятся лишь на {_pct(found)} кадров нарушений при цели {_pct(CASH_GOAL)}"
        )
        charts.append(
            Chart(
                "cash_check.png",
                title,
                ["Кадры нарушений", "Чистые кадры"],
                [found, clean],
                0,
                "%",
                ("цель", CASH_GOAL),
            )
        )
    return charts


HEADER = """# Insights

Собрано командой `uv run python -m src.data.insights` (входит в `make all`) из `experiments.csv`. Руками не правим.
Одна строка на гипотезу. Вердикт считается по строкам `experiments.csv`: «подтверждена» или «отвергнута», если строка с
числами есть, «не проверена», если её нет. Появится строка с нужным `note` или ключом в `params`, и вердикт сменится
при следующей сборке. Варианты сравниваются с базой `{base}` в порядке ADR 0002: recall, потом доля ложных, потом
медиана ошибки таймкода. Графики: серый и один акцент, заголовок называет вывод.

| # | Гипотеза | Числа | График | Вердикт | Вывод |
|---|---|---|---|---|---|
"""


def render_md(hypotheses: list[Hypothesis], charts: list[Chart]) -> str:
    figures = {c.file for c in charts}
    rows = []
    for i, h in enumerate(hypotheses, 1):
        fig = f"[{h.id}.png](../reports/figures/{h.id}.png)" if f"{h.id}.png" in figures else "—"
        rows.append(f"| H{i} | {h.claim} | {h.number} | {fig} | {h.verdict} | {h.conclusion} |")
    tested = sum(h.verdict != UNTESTED for h in hypotheses)
    footer = f"\nПроверено гипотез: **{tested}** из {len(hypotheses)} (столько называем, если спросят про множественные проверки).\n"
    return HEADER.format(base=BASELINE_NOTE) + "\n".join(rows) + "\n" + footer


PALETTE = {
    "grey": "#9A9A9A",
    "accent": "#A31551",
    "ink": "#2B2B2B",
    "grid": "#E4E4E4",
}  # accent = Forte magenta


def draw(chart: Chart, out: Path) -> None:
    """Static PNG: grey bars, one magenta accent, value labels, title = the conclusion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    has_open_sans = any(f.name == "Open Sans" for f in font_manager.fontManager.ttflist)
    plt.rcParams["font.family"] = "Open Sans" if has_open_sans else "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [PALETTE["accent"] if i == chart.accent else PALETTE["grey"] for i in range(len(chart.values))]
    bars = ax.bar(chart.labels, chart.values, color=colors, width=0.5)
    share = chart.unit == "%"
    top = max([*chart.values, chart.reference[1] if chart.reference else 0]) * 1.25
    ax.set_ylim(0, 1.2 if share else top)
    if chart.reference:
        name, value = chart.reference
        ax.axhline(value, color=PALETTE["ink"], linestyle="--", linewidth=1)
        ax.text(
            (len(chart.values) - 1) / 2,
            value,
            f"{name} {_pct(value)}",
            ha="center",
            va="bottom",
            color=PALETTE["ink"],
            fontsize=10,
        )
    for bar, value in zip(bars, chart.values, strict=True):
        text = _pct(value) if share else f"{_num(value)} с"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            text,
            ha="center",
            va="bottom",
            color=PALETTE["ink"],
            fontsize=12,
            fontweight="bold",
        )
    ax.set_title(
        "\n".join(textwrap.wrap(chart.title, 58)),
        loc="left",
        color=PALETTE["ink"],
        fontsize=13,
        fontweight="bold",
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=PALETTE["ink"], length=0, labelsize=11)
    fig.tight_layout()
    fig.savefig(
        out, dpi=150, metadata={"Software": None}
    )  # no version stamp, so rebuilds do not churn the file
    plt.close(fig)


def main() -> None:
    runs = load_runs(pd.read_csv(EXPERIMENTS))
    hypotheses, charts = build_hypotheses(runs), build_charts(runs)
    for h in hypotheses:  # a figure that lost its data must not linger
        (FIGURES / f"{h.id}.png").unlink(missing_ok=True)
    for chart in charts:
        draw(chart, FIGURES / chart.file)
    INSIGHTS_MD.write_text(render_md(hypotheses, charts), encoding="utf-8")
    print(f"wrote {INSIGHTS_MD} and {len(charts)} figures in {FIGURES}")
    for h in hypotheses:
        print(f"  {h.id}: {h.verdict}")


if __name__ == "__main__":
    main()
