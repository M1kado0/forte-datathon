"""`python -m src.data.insights`: docs/INSIGHTS.md and reports/figures/ from experiments.csv (TZ-8.3)."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import EXPERIMENTS, FIGURES, FONTS, INSIGHTS_MD
from src.features.cash_check import CASH_GOAL
from src.models.windows import WINDOW_SEC

BASELINE_NOTE = "dinov2-small-logreg-lovo"  # the run every variant is compared with
CONFIRMED, REJECTED, UNTESTED = "подтверждена", "отвергнута", "не проверена"
METRICS = ("video_recall", "fp_rate", "median_abs_dt")


@dataclass(frozen=True)
class Chart:
    """What one figure shows. Bars are grey, the bar at `accent` is Sky Blue."""

    title: str  # states the conclusion, built from the run's numbers
    labels: list[str]
    values: list[float]
    accent: int
    unit: str  # "%" (values are shares) or "с"
    reference: tuple[str, float] | None = None  # dashed line: label, value


@dataclass(frozen=True)
class Hypothesis:
    id: str
    claim: str
    verdict: str
    numbers: str  # the observed values, as text for the table
    conclusion: str
    chart: Chart | None = None

    @property
    def figure(self) -> str | None:
        """File name of the chart in reports/figures/; derived from the id here and nowhere else."""
        return f"{self.id}.png" if self.chart else None


def _num(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{x:.0%}"


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
    numbers = f"медиана ошибки таймкода: пик {_num(peak)} с, начало {_num(onset)} с"
    if peak == onset:
        return Hypothesis("peak_vs_onset", claim, REJECTED, numbers, "Одинаково")
    better = "пику" if peak < onset else "началу"
    title = f"Таймкод по {better} ближе к метке: {_num(min(peak, onset))} с против {_num(max(peak, onset))} с"
    chart = Chart(title, ["по началу серии", "по пику"], [onset, peak], int(peak < onset), "с")
    return Hypothesis(
        "peak_vs_onset",
        claim,
        CONFIRMED if peak < onset else REJECTED,
        numbers,
        "Лучше пик" if peak < onset else "Лучше начало",
        chart,
    )


@dataclass(frozen=True)
class Comparison:
    """A claim settled by one variant run against the baseline run (ADR 0002 order of metrics)."""

    id: str
    claim: str
    claim_favors_variant: bool  # True: the claim says the variant is better; False: it says the baseline is
    matches: Callable[[dict], bool]
    needs: str  # what a run must look like to settle the claim, shown while it is untested


def _switched_on(params: dict, word: str) -> bool:
    return any(word in key and value is True for key, value in params.items())


COMPARISONS = [
    Comparison(
        "dinov2_vs_siglip", "DINOv2-small лучше SigLIP-base", False,
        lambda r: r["note"] == "siglip-base-logreg-lovo", "note = `siglip-base-logreg-lovo` (тот же LOVO протокол)",
    ),
    Comparison(
        "whole_frame_vs_crops", "Эмбеддинг всего кадра лучше эмбеддингов кропов", False,
        lambda r: "crop" in r["note"] or _switched_on(r["params"], "crop"),
        "`crop` в note или ключ params со словом `crop` и значением `true`",
    ),
    Comparison(
        "window_length", f"Окно {WINDOW_SEC:g} с лучше других длин окна", False,
        lambda r: r["params"].get("window_sec", WINDOW_SEC) != WINDOW_SEC,
        f"в params `window_sec` не равен {WINDOW_SEC:g}",
    ),
    Comparison(
        "median_subtraction", "Вычитание медианного эмбеддинга видео улучшает результат", True,
        lambda r: r["params"].get("subtract_median") is True, "в params `subtract_median: true`",
    ),
    Comparison(
        "heuristic_vote", "Голос эвристик за вердикт улучшает результат", True,
        lambda r: r["params"].get("heuristic_vote") is True or "vote" in r["note"],
        "в params `heuristic_vote: true` или `vote` в note",
    ),
]  # fmt: skip


def _rank(run: dict) -> tuple[float, float, float]:
    """Smaller is better: recall first (a missed violation is unacceptable), then FP, then timecode error."""
    return (-run["video_recall"], run["fp_rate"], run["median_abs_dt"])


def _summary(run: dict) -> str:
    return f"recall {_pct(run['video_recall'])}, FP {_pct(run['fp_rate'])}, ошибка таймкода {_num(run['median_abs_dt'])} с"


def _compare(c: Comparison, runs: list[dict]) -> Hypothesis:
    base = _base(runs)
    candidates = [r for r in runs if r["note"] != BASELINE_NOTE and _has_metrics(r) and c.matches(r)]
    if base is None or not candidates:
        return Hypothesis(c.id, c.claim, UNTESTED, "—", f"Нужна строка в experiments.csv: {c.needs}")
    other = max(candidates, key=lambda r: r["ts"])
    numbers = f"база: {_summary(base)}; {other['note']}: {_summary(other)}"
    if _rank(other) == _rank(base):  # a tie confirms no claim that one side is better
        return Hypothesis(c.id, c.claim, REJECTED, numbers, "Одинаково по всем метрикам")
    variant_better = _rank(other) < _rank(base)
    verdict = CONFIRMED if variant_better == c.claim_favors_variant else REJECTED
    return Hypothesis(c.id, c.claim, verdict, numbers, f"Лучше {other['note'] if variant_better else 'база'}")


def _cash_check(runs: list[dict]) -> Hypothesis:
    claim = f"OWLv2 находит деньги не менее чем на {_pct(CASH_GOAL)} кадров с нарушениями (проверка #19)"
    run = next((r for r in runs if r["note"].startswith("cash-check") and "passed" in r["params"]), None)
    if run is None:
        return Hypothesis("cash_check", claim, UNTESTED, "—", "")
    p = run["params"]
    found = p["cash_found_violation"]
    numbers = f"деньги на {_pct(found)} кадров нарушений, на {_pct(p['cash_found_clean'])} чистых"
    if p["passed"]:
        title = f"Деньги находятся на {_pct(found)} кадров нарушений: цель {_pct(CASH_GOAL)} достигнута"
        conclusion = "Деньги в кадре годятся как признак"
    else:
        title = f"Деньги находятся лишь на {_pct(found)} кадров нарушений при цели {_pct(CASH_GOAL)}"
        conclusion = "Деньги в кадре не опора; запасной вариант — изменения в зоне стола"
    chart = Chart(
        title,
        ["Кадры нарушений", "Чистые кадры"],
        [found, p["cash_found_clean"]],
        0,
        "%",
        ("цель", CASH_GOAL),
    )
    return Hypothesis("cash_check", claim, CONFIRMED if p["passed"] else REJECTED, numbers, conclusion, chart)


def build_hypotheses(runs: list[dict]) -> list[Hypothesis]:
    """In the order of the ticket: backbone, frame vs crops, window, median, onset vs peak, vote; then cash."""
    compared = {c.id: _compare(c, runs) for c in COMPARISONS}
    return [
        compared["dinov2_vs_siglip"],
        compared["whole_frame_vs_crops"],
        compared["window_length"],
        compared["median_subtraction"],
        _peak_vs_onset(runs),
        compared["heuristic_vote"],
        _cash_check(runs),
    ]


HEADER = """# Insights

Собрано командой `uv run python -m src.data.insights` (входит в `make all`) из `experiments.csv`. Руками не правим.
Одна строка на гипотезу. Вердикт считается по строкам `experiments.csv`: «подтверждена» или «отвергнута», если строка с
числами есть, «не проверена», если её нет. Появится строка с нужным `note` или ключом в `params`, и вердикт сменится
при следующей сборке. Варианты сравниваются с базой `{base}` в порядке ADR 0002: recall, потом доля ложных, потом
медиана ошибки таймкода. Графики: серый и один акцент, заголовок называет вывод.

| # | Гипотеза | Числа | График | Вердикт | Вывод |
|---|---|---|---|---|---|
"""


def render_md(hypotheses: list[Hypothesis]) -> str:
    rows = []
    for i, h in enumerate(hypotheses, 1):
        fig = f"[{h.figure}](../reports/figures/{h.figure})" if h.figure else "—"
        rows.append(f"| H{i} | {h.claim} | {h.numbers} | {fig} | {h.verdict} | {h.conclusion} |")
    tested = sum(h.verdict != UNTESTED for h in hypotheses)
    footer = f"\nПроверено гипотез: **{tested}** из {len(hypotheses)} (столько называем, если спросят про множественные проверки).\n"
    return HEADER.format(base=BASELINE_NOTE) + "\n".join(rows) + "\n" + footer


# Brand tokens (docs/IMPECCABLE_BRAND.md): Sky Blue is the chart accent, neutrals are ours.
PALETTE = {"grey": "#8E8E93", "accent": "#61CFFF", "ink": "#1A1A1A", "grid": "#E7E5EA"}


def draw(chart: Chart, out: Path) -> None:
    """Static PNG: grey bars, one Sky Blue accent, value labels, title = the conclusion. Open Sans or an error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for ttf in ("OpenSans-Regular.ttf", "OpenSans-Bold.ttf"):
        font_manager.fontManager.addfont(str(FONTS / ttf))  # raises if the file is missing
    plt.rcParams["font.family"] = "Open Sans"
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
    hypotheses = build_hypotheses(load_runs(pd.read_csv(EXPERIMENTS)))
    for png in FIGURES.glob("*.png"):  # a figure whose hypothesis lost its data or its id must not linger
        png.unlink()
    for h in hypotheses:
        if h.chart:
            draw(h.chart, FIGURES / h.figure)
    INSIGHTS_MD.write_text(render_md(hypotheses), encoding="utf-8")
    print(f"wrote {INSIGHTS_MD} and {sum(h.chart is not None for h in hypotheses)} figures in {FIGURES}")
    for h in hypotheses:
        print(f"  {h.id}: {h.verdict}")


if __name__ == "__main__":
    main()
