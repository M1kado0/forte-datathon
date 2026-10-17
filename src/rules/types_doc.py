"""`python -m src.rules.types_doc`: write docs/TYPES.md, one row per type of rules.yaml with its label count (TZ-8.5)."""

from collections import defaultdict

import pandas as pd
import yaml

from src.config import RULES_YAML, TYPES_MD
from src.data.labels import count_types, load_labels

HEADER = """# Типы нарушений

Собрано командой `uv run python -m src.rules.types_doc` (входит в `make all`) из `src/rules/rules.yaml` и `data/violation_types.csv`. Руками не правим.
Тип называем словами сценария из видео, норма идёт в пункт (ADR 0005). В описании стоит только то, что видно в кадре; `…` вместо предмета, который эвристика подставляет сама.
Число меток считается по {total} меткам нарушений в `data/labels.csv`, у типа без примеров стоит 0.

| id | Название | Описание | Пункт | Меток |
|---|---|---|---|---|
"""


def _punkts(rules: list[dict], docs: dict) -> str:
    cited = (
        f"{docs[r['doc']]['short']}, {r['punkt']}" if "punkt" in r else f"{docs[r['doc']]['short']} целиком"
        for r in rules
    )
    return "; ".join(dict.fromkeys(cited))


def build_types_md(labels: pd.DataFrame) -> str:
    rules = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
    counts = count_types(labels)
    rows = []
    for t in rules["types"]:
        description = " ".join(t["description"].split()).format_map(defaultdict(lambda: "…"))
        cells = [
            f"`{t['id']}`",
            t["name"],
            description,
            _punkts(t["rules"], rules["documents"]),
            str(counts[t["id"]]),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return HEADER.format(total=sum(counts.values())) + "\n".join(rows) + "\n"


def main() -> None:
    TYPES_MD.write_text(build_types_md(load_labels()), encoding="utf-8")
    print(f"wrote {TYPES_MD}")


if __name__ == "__main__":
    main()
