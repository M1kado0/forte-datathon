"""Where the service is wrong (#38): the FP/FN table by video name and the limitations text for the README.

Both are built from `reports/metrics.json` data (per-video verdicts of LOVO, leave-one-layout-out and the nested
threshold check), so every number in them comes from `train`. An explanation says what the system saw (window score
against threshold, heuristics that fired), not why the video deserves a different verdict: nobody labelled that.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Mapping

from src.config import REPORTS

LOVO, LOLO, NESTED = "LOVO", "leave-one-layout-out", "nested"
LAYOUT_NAMES = {"back": "кассир спиной к камере", "facing": "кассир лицом к камере"}
COLUMNS = ["video_id", "arrangement", "kind", "label_type", "found_type", "checks", "explanation"]
HEADERS = [
    "Видео",
    "Расстановка",
    "Ошибка",
    "Тип по разметке",
    "Тип в ответе",
    "Где ошибка",
    "Что видела система",
]
KINDS = {"FN": "FN, пропуск", "FP": "FP, ложное срабатывание"}


def _number(video_id: str) -> int:
    return int(video_id.split("-")[1])


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{int(x * 100 + 0.5 + 1e-9)}%"


def _auc(layout: Mapping) -> str:
    auc = layout.get("branch_a", {}).get("roc_auc")
    return "" if auc is None else f", ROC AUC лучшей оценки окна видео {auc:.2f}"


def _explain(
    kind: str, check: str, v: Mapping | None, threshold: float | None, nested: Mapping | None
) -> str:
    """One short sentence about the first check that got the video wrong."""
    if check == NESTED and nested:
        return f"Модель, обученная на остальных видео, дала {nested['score']:.4f} при пороге {nested['threshold']:.4f}"
    if v is None or threshold is None:
        return ""
    score = f"{v['video_score']:.4f}"
    if kind == "FN":
        return f"Все окна ниже порога: лучшее {score} при пороге {threshold:.4f}"
    when = f" на {v['timecode_sec']:.0f} с" if v["timecode_sec"] is not None else ""
    if v["video_score"] >= threshold:
        return f"Окно с оценкой {score} выше порога {threshold:.4f}{when}"
    hits = ", ".join(v["hit_types"]) or "тип не указан"
    return f"Окна ниже порога ({score} при {threshold:.4f}), нарушение назвал голос эвристик: {hits}{when}"


def error_rows(
    lovo: Mapping[str, Mapping],
    lolo: Mapping[str, Mapping],
    nested: Mapping[str, Mapping],
    truth: Mapping[str, str],
    arrangement: Mapping[str, str],
    thresholds: Mapping[str, float],
    nested_missed: Collection[str] = (),
) -> list[dict]:
    """One row per video the system got wrong in at least one check, misses first.

    `lovo` and `lolo` map video id -> verdict record (`evaluate.summarize`, "verdicts_by_video"); `truth` maps a
    violation video to its labelled type(s); `thresholds` has the LOVO threshold under "LOVO" and, under each layout
    name, the one chosen on the other layout. `nested` has the nested check's score and threshold per video.
    """
    checks: dict[tuple[str, str], list[str]] = {}
    for check, verdicts in ((LOVO, lovo), (LOLO, lolo)):
        for vid, v in verdicts.items():
            if vid in truth and not v["flagged"]:
                checks.setdefault((vid, "FN"), []).append(check)
            elif vid not in truth and v["flagged"]:
                checks.setdefault((vid, "FP"), []).append(check)
    for vid in nested_missed:
        checks.setdefault((vid, "FN"), []).append(NESTED)
    rows = []
    for (vid, kind), failed in checks.items():
        first = failed[0]
        v = (lolo if first == LOLO else lovo).get(vid)
        threshold = thresholds.get(arrangement[vid] if first == LOLO else first)
        found = v["type"] if v and first != NESTED else None  # what the first failing check answered
        if kind == "FP" and found == "other":
            found = None  # a clean video has no type to match: "other" says nothing
        rows.append(
            {
                "video_id": vid,
                "arrangement": arrangement[vid],
                "kind": kind,
                "label_type": truth.get(vid),
                "found_type": found,
                "checks": failed,
                "explanation": _explain(kind, first, v, threshold, nested.get(vid)),
            }
        )
    return sorted(rows, key=lambda r: (r["kind"] != "FN", _number(r["video_id"])))


def render_table(rows: list[dict]) -> str:
    """The rows as a markdown table, ready for the README."""
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for r in rows:
        cells = [
            r["video_id"], r["arrangement"], KINDS[r["kind"]], r["label_type"] or "—", r["found_type"] or "—",
            ", ".join(r["checks"]), r["explanation"],
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def limitations_text(metrics: Mapping) -> str:
    """The limitations paragraph for the README (P4, #42). Every number is read from `metrics`."""
    layouts = metrics["leave_one_layout_out"]["layouts"]
    names = ", ".join(f"«{k}» ({LAYOUT_NAMES.get(k, k)}, {v['n_videos']} видео)" for k, v in layouts.items())
    held_out = "\n".join(
        f"  - расстановка «{k}»: recall {_pct(v['video_recall'])} на {v['n_violation_videos']} видео с нарушением, "
        f"FP rate {_pct(v['fp_rate'])} на {v['n_clean_videos']} чистых{_auc(v)}"
        for k, v in layouts.items()
    )
    nested = metrics.get("nested", {})
    baseline = (
        f"Для сравнения, та же модель на невиданных видео с порогом, выбранным без них (nested-проверка): "
        f"recall {_pct(nested['nested_video_recall'])}, FP rate {_pct(nested['nested_fp_rate'])}; "
        "цифры leave-one-video-out выше, потому что порог подобран на тех же видео."
        if "nested_video_recall" in nested and "nested_fp_rate" in nested
        else ""
    )
    return f"""## Ограничения

- **В обучении один ракурс камеры.** Все {metrics["n_videos"]} обучающих видео сняты одной камерой сверху (FAQ 3), \
меняется только расстановка мебели: {names}. Работу на другом ракурсе камеры эти данные не проверяют.
- **Мало данных.** {metrics["n_videos"]} видео: {metrics["n_violation_videos"]} с нарушением и \
{metrics["n_clean_videos"]} без. Все метрики посчитаны на них же (leave-one-video-out), на отложенной выборке их не проверяли.
- **Много ложных срабатываний.** Порог выбран так, чтобы не пропустить ни одного нарушения (ADR 0002). Цена: \
FP rate {_pct(metrics["fp_rate"])} на чистых видео при recall {_pct(metrics["video_recall"])}.
- **Невиданная расстановка.** Модель обучена на одной расстановке и проверена на другой \
(leave-one-layout-out, `reports/metrics.json`). {baseline}
{held_out}
- **Тип нарушения.** Верный тип назван у {_pct(metrics["type_accuracy"])} видео с нарушением; \
у остальных тип неверный или не назван (список: `reports/metrics.json`, `types.by_video`).
- Список ошибок по именам видео: `reports/errors.md`.
"""


def write_report(metrics: Mapping, labels) -> list[dict]:
    """Write errors.csv, errors.md and limitations.md next to metrics.json; returns the error rows."""
    nested = metrics.get("nested", {})
    layouts = metrics["leave_one_layout_out"]["layouts"]
    arrangement = labels.drop_duplicates("video_id").set_index("video_id")["arrangement"].to_dict()
    truth = {
        vid: "; ".join(dict.fromkeys(g["violation_type"]))
        for vid, g in labels[labels["verdict"] == "violation"].groupby("video_id")
    }
    rows = error_rows(
        metrics["verdicts_by_video"],
        {vid: v for layout in layouts.values() for vid, v in layout["verdicts_by_video"].items()},
        nested.get("by_video", {}),
        truth,
        arrangement,
        {LOVO: metrics["threshold"]} | {name: layout["threshold"] for name, layout in layouts.items()},
        nested.get("nested_missed", []),
    )
    with open(REPORTS / "errors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows({**r, "checks": "; ".join(r["checks"])} for r in rows)
    (REPORTS / "errors.md").write_text(render_table(rows) + "\n")
    (REPORTS / "limitations.md").write_text(limitations_text(metrics))
    return rows
