"""Rules contract: every violation type cites a real document, пункт, URL and quote (T05, T09)."""

import re
import string

import pytest
import yaml

from src.config import RULES_YAML
from src.rules import describe

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


FACTS = {"object": "сумка"}


def citation(doc_id: str, puncts: list[str]) -> str:
    doc = DOCS[doc_id]
    return ", ".join([f"{doc['title']}, утв. {doc['approved_by']}", *puncts])


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_describe_gives_name_and_cites_every_document(t):
    result = describe(t["id"], FACTS)
    assert result.name == t["name"]
    for doc_id in dict.fromkeys(r["doc"] for r in t["rules"]):
        puncts = list(dict.fromkeys(r["punkt"] for r in t["rules"] if r["doc"] == doc_id and r.get("punkt")))
        assert citation(doc_id, puncts) in result.description


def test_describe_ends_with_citation_to_the_punkt():
    text = describe("count_without_controller").description
    assert text.endswith(
        "утв. постановлением Правления Национального Банка Республики Казахстан "
        "от 29 ноября 2019 года № 231, п. 42"
    )


def test_object_placeholder_is_filled():
    text = describe("foreign_object_at_desk", FACTS).description
    assert "посторонний предмет: сумка." in text
    assert "{" not in text


@pytest.mark.parametrize("facts", [None, {}, {"object": ""}, {"object": "  "}, {"thing": "сумка"}])
def test_unfilled_placeholder_is_an_error(facts):
    with pytest.raises(ValueError, match="object"):
        describe("foreign_object_at_desk", facts)


@pytest.mark.parametrize("type_id", ["no_such_type", "", "OTHER"])
def test_unknown_id_gives_fallback_type_citing_the_whole_act(type_id):
    result = describe(type_id, FACTS)
    assert result.name == FALLBACK_NAME
    assert result.description.endswith(
        f"{DOCS['pravila_231']['title']}, утв. {DOCS['pravila_231']['approved_by']}"
    )
    assert "п. " not in result.description


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_finished_description_states_only_visible_facts(t):
    text = describe(t["id"], FACTS).description
    assert not UNSUPPORTED.search(text), UNSUPPORTED.search(text)


def test_same_punkt_is_cited_once():
    assert "п. 12, п. 12" not in describe("foreign_object_at_desk", FACTS).description
