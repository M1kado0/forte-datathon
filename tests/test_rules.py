"""Rules contract: every violation type cites a real document, пункт, URL and quote (T05, T09, ADR 0005)."""

import re
import string

import pytest
import yaml

from src.config import RULES_MD, RULES_YAML
from src.rules.describe import describe_violation
from src.rules.verify_quotes import quote_matches, verify

RULES = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
DOCS = RULES["documents"]
TYPES = RULES["types"]
FALLBACK_NAME = "Иное подозрительное действие"
ALLOWED_FIELDS = {"object"}
# Words that claim data a camera cannot see (TZ-7.1) or a device feature we cannot verify.
UNSUPPORTED = re.compile(r"проводк|сверк|журнал|учет|учёт|недостач|камер|фото", re.IGNORECASE)
# A description names what is seen; the norm is in the citation. No legal conclusion (TZ-5.6, ADR 0005).
LEGAL_CLAIM = re.compile(
    r"нарушен|должен|обязан|требуется|не провел|не провёл|не предъявил|не проверил", re.IGNORECASE
)
# Table of ADR 0005: scenario name -> videos.
ADR_0005_TYPES = {
    "Ключи оставлены на сейфе": ["X-11", "X-22"],
    "Сейф оставлен открытым": ["X-20"],
    "Наличные оставлены на столе вместо сейфа": ["X-26"],
    "Важные предметы оставлены на столе вместо сейфа": ["X-24"],
    "Наличные на столе при обслуживании следующего клиента": ["X-38", "X-4"],
    "Посторонний в кассовом помещении": ["X-18", "X-30", "X-33"],
    "Посторонний предмет на столе кассира": ["X-15", "X-40", "X-46"],
    "Дверь кассового помещения оставлена открытой": ["X-32"],
    "Ноутбук оставлен открытым": ["X-28"],
    "Наличные убраны без пересчёта": ["X-13"],
    "Банкноты не проверены на детекторе подлинности": ["X-44"],
    "Клиент ушёл во время пересчёта": ["X-34"],
    "Инкассатор не показал удостоверение": ["X-17"],
    "Выдача наличных без документа, удостоверяющего личность": ["X-2"],
    "Наличные выданы клиенту не полностью": ["X-1"],
    "Иное подозрительное действие": [],
}
NORMS = {"точная", "ближайшая", "акт целиком"}


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


def test_every_type_of_adr_0005_is_in_the_catalog_with_its_videos():
    catalog = {t["name"]: t.get("videos", []) for t in TYPES}
    assert {name: catalog.get(name) for name in ADR_0005_TYPES} == ADR_0005_TYPES


def test_collector_alone_is_removed():
    assert "Доставка ценностей одним работником" not in {t["name"] for t in TYPES}


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_type_cites_documents(t):
    assert t["rules"], "type has no rule"
    for rule in t["rules"]:
        assert rule["doc"] in DOCS
        if "punkt" in rule:
            assert rule["punkt"].startswith("п. ") and rule["quote"].strip()
        else:
            assert "quote" not in rule and rule["whole_act"].strip(), "a whole-act citation says why"
    assert any(not DOCS[r["doc"]]["secondary"] for r in t["rules"]), "only secondary sources"


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_type_states_how_close_its_norm_is(t):
    if t.get("fallback"):
        assert "norm" not in t
    else:
        assert t["norm"] in NORMS
        assert (t["norm"] == "акт целиком") == all("punkt" not in r for r in t["rules"])


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_description_states_only_visible_facts(t):
    text = t["description"]
    fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
    assert fields <= ALLOWED_FIELDS
    assert not UNSUPPORTED.search(text)
    assert not LEGAL_CLAIM.search(text)


def _rules_md_row(t):
    citations = "; ".join(
        f"{DOCS[r['doc']]['short']}, {r['punkt']}" if "punkt" in r else f"{DOCS[r['doc']]['short']} целиком"
        for r in t["rules"]
    )
    return (
        f"| {t['name']} | {', '.join(t.get('videos', []))} | {citations} | {t.get('norm', 'запасной тип')} |"
    )


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_rules_md_lists_type_with_videos_punkts_and_norm(t):
    assert _rules_md_row(t) in RULES_MD.read_text(encoding="utf-8").splitlines()


def test_rules_md_type_tables_have_no_row_without_a_type():
    text = RULES_MD.read_text(encoding="utf-8")
    section = text[text.index("## Типы нарушений и пункты") : text.index("## Только контекст")]
    rows = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith("| Тип |")]
    assert sorted(rows) == sorted(_rules_md_row(t) for t in TYPES)


# --- quotes vs the adilet texts in data/external/adilet (acceptance of #69) ---
ACT = ["Постановление.", "1. Утвердить Правила.", "Глава 1. Общие положения", "5. Кассир делает:"]
ACT += ["1) первое действие;", "", "2) второе действие;", "6. Другое.", "Глава 2. Конец"]


def test_every_quote_matches_its_punkt_in_the_adilet_text():
    assert verify() == 0


def test_quote_inside_its_subpunkt_matches():
    assert quote_matches("Кассир делает: 1) первое действие;", "п. 5 подп. 1)", ACT)
    assert quote_matches("2) второе действие;", "п. 5 подп. 2)", ACT)


def test_quote_from_another_subpunkt_does_not_match():
    assert not quote_matches("1) первое ... второе действие;", "п. 5 подп. 1)", ACT)


@pytest.mark.parametrize("punkt", ["п. 9", "п. 5 подп. 3)", "подп. 1)"])
def test_missing_punkt_is_a_mismatch_not_a_crash(punkt):
    assert not quote_matches("1) первое действие;", punkt, ACT)


def test_act_without_chapter_heading_is_a_mismatch_not_a_crash():
    assert not quote_matches("Другое.", "п. 6", ["6. Другое."])


# --- describe_violation: the one place analyze(path) takes `type` and `description` from (TZ-4.6) ---
ACT_231_CITATION = (
    "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей "
    "в банках второго уровня, филиалах банков-нерезидентов Республики Казахстан, Национальном операторе почты "
    "и юридических лицах, исключительной деятельностью которых является инкассация банкнот, монет и ценностей, "
    "утв. постановлением Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231"
)


def test_known_type_returns_name_and_description_ending_with_citation():
    name, description = describe_violation("foreign_object_at_desk", {"object": "кружка"})
    assert name == "Посторонний предмет на столе кассира"
    assert description.startswith("На столе кассира виден посторонний предмет: кружка.")
    assert description.endswith(f"{ACT_231_CITATION}, п. 12")


def test_type_with_two_punkts_cites_each():
    _, description = describe_violation("cash_put_away_uncounted", {})
    assert description.endswith(
        f"{ACT_231_CITATION}, п. 20, часть первая; {ACT_231_CITATION}, п. 19 подп. 2)"
    )


def test_whole_act_type_cites_the_act_without_a_punkt():
    _, description = describe_violation("laptop_left_open", {})
    assert description.endswith(
        "Требования к обеспечению информационной безопасности банков, филиалов банков-нерезидентов "
        "Республики Казахстан и организаций, осуществляющих отдельные виды банковских операций, "
        "утв. постановлением Правления Агентства Республики Казахстан по регулированию и развитию "
        "финансового рынка от 3 апреля 2026 года № 53"
    )
    assert "п. 71" not in description


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
    assert description.count("утв. постановлением Правления Агентства") == 1
    assert "№ 14, п. 8; " in description
    assert description.endswith("№ 21, п. 14 подп. 1)")


@pytest.mark.parametrize("t", TYPES, ids=[t["id"] for t in TYPES])
def test_every_type_renders_name_and_cites_each_document_and_punkt(t):
    name, description = describe_violation(t["id"], {"object": "предмет"})
    assert name == t["name"]
    assert not UNSUPPORTED.search(description)
    for rule in t["rules"]:
        doc = DOCS[rule["doc"]]
        citation = ", ".join(
            [" ".join(doc["title"].split()), "утв. " + " ".join(doc["approved_by"].split())]
            + ([rule["punkt"]] if rule.get("punkt") else [])
        )
        assert citation in description


def test_unknown_id_is_logged(caplog):
    describe_violation("no_such_type", {})
    assert "no_such_type" in caplog.text


def test_only_the_types_of_adr_0005_are_in_the_catalog():
    types = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))["types"]

    assert {"workplace_left", "count_without_controller", "delivery_by_one_worker"}.isdisjoint(
        t["id"] for t in types
    )
    assert [t["id"] for t in types if not t.get("videos") and not t.get("fallback")] == []
