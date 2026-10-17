"""Turn a score + contributions into Russian UI text. P4 reviews the wording."""

from __future__ import annotations

from src.config import FEATURE_LABELS


def fmt_kzt(x: float) -> str:
    return f"{round(x):,}".replace(",", " ") + " ₸"


def fmt_value(feature: str, value) -> str:
    if feature == "income_kzt":
        return fmt_kzt(float(value))
    if feature == "debt_ratio":
        return f"{float(value) * 100:.0f}% дохода"
    if feature in ("days_overdue_max", "tenure_months", "age"):
        return str(int(value))
    return str(value)


def reason_text(feature: str, value, effect: float) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    direction = "повышает риск" if effect > 0 else "снижает риск"
    return f"{label}: {fmt_value(feature, value)} — {direction}"


def verdict(p: float) -> tuple[str, str]:
    """(label, action). Thresholds are placeholders: set them on cost tomorrow."""
    if p >= 0.35:
        return "Высокий риск", "Отказать или запросить залог"
    if p >= 0.15:
        return "Средний риск", "Одобрить с уменьшенным лимитом"
    return "Низкий риск", "Одобрить заявку"


def impact_kzt(p: float, income: float) -> float:
    """STUB: expected loss avoided ≈ p × exposure (6 × income) × LGD 45%. Replace with the case economics."""
    return p * income * 6 * 0.45
