"""docs/TYPES.md: every type of rules.yaml with its description and label count (TZ-8.5, #76)."""

import pandas as pd
import yaml

from src.config import RULES_YAML, TYPES_MD
from src.data.labels import load_labels
from src.rules.types_doc import build_types_md

TYPE_IDS = [t["id"] for t in yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))["types"]]


def table_rows(md: str) -> dict[str, list[str]]:
    """Table rows by the type id in the first cell, header and separator skipped."""
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in md.splitlines()
        if line.startswith("|")
    ]
    return {row[0].strip("`"): row for row in rows if row[0].strip("`") in TYPE_IDS}


def tiny_labels():
    return pd.DataFrame(
        {
            "verdict": ["violation", "violation", "violation", "no_violation"],
            "violation_type": ["cash_partially_given", "laptop_left_open", "laptop_left_open", None],
        }
    )


def test_every_type_gets_a_row_and_a_type_nobody_has_counts_zero():
    rows = table_rows(build_types_md(tiny_labels()))
    assert set(rows) == set(TYPE_IDS)
    assert rows["laptop_left_open"][-1] == "2"
    assert rows["cash_partially_given"][-1] == "1"
    assert rows["other"][-1] == "0"


def test_row_has_name_description_and_punkt():
    row = table_rows(build_types_md(tiny_labels()))["laptop_left_open"]
    assert "Ноутбук оставлен открытым" in row[1]
    assert row[2] and row[3]  # description and norm are never empty


def test_committed_types_md_counts_the_22_labels():
    """22 comes from issue #21, not from the code."""
    rows = table_rows(TYPES_MD.read_text(encoding="utf-8"))
    assert sum(int(row[-1]) for row in rows.values()) == 22


def test_committed_types_md_is_what_the_build_command_writes():
    assert TYPES_MD.read_text(encoding="utf-8") == build_types_md(load_labels())
