"""One lookup for the API's `type` and `description`: name and cited description of a violation type (TZ-4.6)."""

import logging
from functools import cache
from typing import NamedTuple

import yaml

from src.config import RULES_YAML

log = logging.getLogger(__name__)

_RULES = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
_DOCS = _RULES["documents"]
_TYPES = {t["id"]: t for t in _RULES["types"]}
_FALLBACK = next((t for t in _RULES["types"] if t.get("fallback")), None)
if _FALLBACK is None:
    raise RuntimeError(f"{RULES_YAML.name} has no type with fallback: true")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _format_citation(rule: dict) -> str:
    doc = _DOCS[rule["doc"]]
    parts = [_one_line(doc["title"]), "утв. " + _one_line(doc["approved_by"])]
    if rule.get("punkt"):
        parts.append(rule["punkt"])
    return ", ".join(parts)


@cache
def rule_catalog() -> list[dict]:
    """UI metadata from the same cached rules and citation formatter as descriptions."""
    catalog = []
    for item in _TYPES.values():
        rules = []
        for rule in item["rules"]:
            citation = _format_citation(rule)
            if any(existing["citation"] == citation for existing in rules):
                continue
            doc = _DOCS[rule["doc"]]
            rules.append(
                {
                    "title": _one_line(doc["title"]),
                    "approved_by": _one_line(doc["approved_by"]),
                    "punkt": rule.get("punkt"),
                    "url": doc["url"],
                    "citation": citation,
                }
            )
        catalog.append(
            {
                "id": item["id"],
                "name": item["name"],
                "rules": rules,
                "citation_suffix": "; ".join(dict.fromkeys(rule["citation"] for rule in rules)),
            }
        )
    return catalog


class TypeDescription(NamedTuple):
    name: str
    description: str


def describe_violation(type_id: str, facts: dict[str, str]) -> TypeDescription:
    if type_id not in _TYPES:
        log.warning("unknown violation type %r, using fallback", type_id)
    t = _TYPES.get(type_id, _FALLBACK)
    citations = list(dict.fromkeys(_format_citation(r) for r in t["rules"]))
    try:
        text = _one_line(t["description"]).format(**facts)
    except (KeyError, IndexError, AttributeError) as missing:
        raise ValueError(f"description of {t['id']} needs fact {missing}") from None
    return TypeDescription(t["name"], f"{text} {'; '.join(citations)}")
