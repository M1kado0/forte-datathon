"""Rules contract: every violation type cites a real document, пункт, URL and quote (T05, T09)."""

import re
import string

import pytest
import yaml

from src.config import RULES_YAML
from src.rules.describe import describe_violation

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


# --- describe_violation: the one place analyze(path) takes `type` and `description` from (TZ-4.6) ---
ACT_231_CITATION = (
    "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей "
    "в банках второго уровня, филиалах банков-нерезидентов Республики Казахстан, Национальном операторе почты "
    "и юридических лицах, исключительной деятельностью которых является инкассация банкнот, монет и ценностей, "
    "утв. постановлением Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231"
)


def test_known_type_returns_name_and_description_ending_with_citation():
    name, description = describe_violation("foreign_object_at_desk", {"object": "телефон"})
    assert name == "Посторонние предметы на рабочем месте кассира"
    assert description.startswith(
        "На рабочем месте кассира в кассовом узле виден посторонний предмет: телефон."
    )
    assert description.endswith(f"{ACT_231_CITATION}, п. 12")


def test_type_with_two_punkts_cites_each():
    _, description = describe_violation("workplace_left", {})
    assert description.endswith(f"{ACT_231_CITATION}, п. 22; {ACT_231_CITATION}, п. 20")


def test_unfilled_placeholder_is_an_error():
    with pytest.raises(ValueError, match="object"):
        describe_violation("foreign_object_at_desk", {})


def test_unknown_id_gives_fallback_citing_whole_act():
    name, description = describe_violation("no_such_type", {"object": "x"})
    assert name == FALLBACK_NAME
    assert description.endswith(f"Нужна проверка человеком. {ACT_231_CITATION}")


def test_type_citing_two_acts_cites_each_with_its_own_title():
    _, description = describe_violation("outsider_in_cash_room", {})
    assert description.count("утв. постановлением Правления Национального Банка") == 1
    assert description.count("утв. постановлением Правления Агентства") == 2
    assert description.endswith("п. 15 подп. 3), 5)")
    assert "п. 8; " in description and "п. 14 подп. 1); " in description


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_every_type_renders_name_and_cites_each_document_and_punkt(t):
    name, description = describe_violation(t["id"], {"object": "предмет"})
    assert name == t["name"]
    assert not UNSUPPORTED.search(description)
    for rule in t["rules"]:
        title = " ".join(DOCS[rule["doc"]]["title"].split())
        assert title in description
        assert rule.get("punkt", "") in description


def test_unknown_id_is_logged(caplog):
    describe_violation("no_such_type", {})
    assert "no_such_type" in caplog.text
