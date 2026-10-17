"""Rules contract: every violation type cites a real document, пункт, URL and quote (T05, T09)."""

import re
import string

import pytest
import yaml

from src.config import RULES_YAML

RULES = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
DOCS = RULES["documents"]
TYPES = RULES["types"]
FALLBACK_NAME = "Иное подозрительное действие"
ALLOWED_FIELDS = {"object"}
# Words that claim data a camera cannot see (TZ-7.1) or a device feature we cannot verify.
UNSUPPORTED = re.compile(r"проводк|сверк|журнал|учет|учёт|недостач|камер|фото", re.IGNORECASE)


@pytest.mark.parametrize("doc", DOCS.values(), ids=list(DOCS))
def test_document_has_title_act_and_adilet_url(doc):
    assert doc["title"].strip() and doc["approved_by"].strip() and doc["registration"].strip()
    assert re.fullmatch(r"https://adilet\.zan\.kz/rus/docs/V\d{10}", doc["url"])


def test_type_ids_and_names_unique():
    assert len({t["id"] for t in TYPES}) == len(TYPES)
    assert len({t["name"] for t in TYPES}) == len(TYPES)


def test_exactly_one_fallback_type():
    fallback = [t for t in TYPES if t.get("fallback")]
    assert [t["name"] for t in fallback] == [FALLBACK_NAME]


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_type_cites_documents(t):
    assert t["rules"], "type has no rule"
    for rule in t["rules"]:
        assert rule["doc"] in DOCS
        if not t.get("fallback"):
            assert rule["punkt"].startswith("п. ") and rule["quote"].strip()
    if not t.get("fallback"):
        assert any(not DOCS[r["doc"]]["secondary"] for r in t["rules"]), "only secondary sources"


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_description_states_only_visible_facts(t):
    text = t["description"]
    fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
    assert fields <= ALLOWED_FIELDS
    assert not UNSUPPORTED.search(text)
