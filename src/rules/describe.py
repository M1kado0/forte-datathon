"""One lookup for the API's `type` and `description`: name and cited description of a violation type (TZ-4.6)."""

import logging
from typing import NamedTuple

import yaml

from src.config import RULES_YAML

log = logging.getLogger(__name__)

_RULES = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
_DOCS = _RULES["documents"]
_TYPES = {t["id"]: t for t in _RULES["types"]}
_FALLBACK = next(t for t in _RULES["types"] if t.get("fallback"))


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _format_citation(rule: dict) -> str:
    doc = _DOCS[rule["doc"]]
    parts = [_one_line(doc["title"]), "утв. " + _one_line(doc["approved_by"])]
    if rule.get("punkt"):
        parts.append(rule["punkt"])
    return ", ".join(parts)


class Violation(NamedTuple):
    name: str
    description: str


def describe_violation(type_id: str, facts: dict[str, str]) -> Violation:
    if type_id not in _TYPES:
        log.warning("unknown violation type %r, using fallback", type_id)
    t = _TYPES.get(type_id, _FALLBACK)
    citations = list(dict.fromkeys(_format_citation(r) for r in t["rules"]))
    try:
        text = _one_line(t["description"]).format(**facts)
    except (KeyError, IndexError) as missing:
        raise ValueError(f"description of {t['id']} needs fact {missing}") from None
    return Violation(t["name"], f"{text} {'; '.join(citations)}")
