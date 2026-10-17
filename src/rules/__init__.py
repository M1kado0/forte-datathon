"""Violation type reference: one function gives `analyze(path)` the type name and the cited description."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from string import Formatter

import yaml

from src.config import RULES_YAML


@dataclass(frozen=True)
class TypeDescription:
    name: str
    description: str


@cache
def _reference() -> dict:
    return yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))


def _citations(type_: dict, documents: dict) -> str:
    """One "<title>, утв. <approved_by>, <пункт>, ..." per document, in the order the type lists them, joined by "; "."""
    puncts: dict[str, list[str]] = {}
    for rule in type_["rules"]:
        puncts.setdefault(rule["doc"], [])
        if rule.get("punkt") and rule["punkt"] not in puncts[rule["doc"]]:
            puncts[rule["doc"]].append(rule["punkt"])
    refs = []
    for doc_id, doc_puncts in puncts.items():
        doc = documents[doc_id]
        refs.append(", ".join([f"{doc['title']}, утв. {doc['approved_by']}", *doc_puncts]))
    return "; ".join(refs)


def describe(type_id: str, facts: Mapping[str, str] | None = None) -> TypeDescription:
    """Russian type name and description of `type_id` filled with observed `facts`, ending with the norm citation.

    An unknown id gives the fallback type. A placeholder without a fact raises ValueError.
    """
    reference = _reference()
    types = reference["types"]
    type_ = next((t for t in types if t["id"] == type_id), None) or next(
        t for t in types if t.get("fallback")
    )
    template = " ".join(type_["description"].split())
    facts = {k: v.strip() for k, v in (facts or {}).items() if isinstance(v, str)}
    missing = [f for _, f, _, _ in Formatter().parse(template) if f and not facts.get(f)]
    if missing:
        raise ValueError(f"type {type_['id']!r} needs facts: {', '.join(missing)}")
    body = template.format_map(facts)
    return TypeDescription(type_["name"], f"{body} {_citations(type_, reference['documents'])}")
