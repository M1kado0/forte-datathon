"""Check every quote in rules.yaml against the adilet text: the пункт exists and holds each fragment in order.

Texts in data/external/adilet/<registration>.txt are the page text of each act, saved from adilet.zan.kz
on 18.10.2026 with playwright-cli. Run: uv run python -m src.rules.verify_quotes (exit code = mismatches).
"""

import re
import sys
from pathlib import Path

import yaml

from src.config import ADILET_TEXTS, RULES_YAML


def _norm(text: str) -> str:
    return " ".join(text.split())


def _text_path(doc: dict, texts: Path) -> Path:
    return texts / (doc["url"].rsplit("/", 1)[1] + ".txt")


def _punkt_block(lines: list[str], punkt: str) -> str:
    """Text of one пункт; 'часть первая/вторая' keeps the first paragraph or the rest."""
    n = int(re.match(r"п\. (\d+)", punkt).group(1))
    # The act's own пункты come after the resolution, which ends before "Глава 1."
    start = next(i for i, line in enumerate(lines) if re.match(r"\s*Глава 1\.", line))
    i = next(k for k in range(start, len(lines)) if lines[k].startswith(f"{n}. "))
    j = next(k for k in range(i + 1, len(lines)) if re.match(rf"({n + 1}\. |Глава )", lines[k]))
    block = [line for line in lines[i:j] if line.strip()]
    if "часть первая" in punkt:
        block = block[:1]
    elif "часть вторая" in punkt:
        block = block[1:]
    return _norm(" ".join(block))


def quote_matches(quote: str, punkt: str, lines: list[str]) -> bool:
    """Fragments split by ' ... ' appear in order inside the пункт, and each подпункт number is quoted."""
    body, quote = _punkt_block(lines, punkt), _norm(quote)
    pos = 0
    for fragment in quote.split(" ... "):
        k = body.find(fragment, pos)
        if k < 0:
            return False
        pos = k + len(fragment)
    return all(f"{sub}) " in quote for sub in re.findall(r"(\d+)\)", punkt))


def verify(rules_path: Path = RULES_YAML, texts: Path = ADILET_TEXTS) -> int:
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    bad = 0
    for doc_id, doc in rules["documents"].items():
        page = _norm(_text_path(doc, texts).read_text(encoding="utf-8"))
        number = doc["registration"].rsplit("№ ", 1)[1]
        ok = _norm(doc["title"]) in page and number in page
        print(f"{'OK ' if ok else 'BAD'} {doc_id}: title and registration № {number}")
        bad += not ok
    for t in rules["types"]:
        for rule in t["rules"]:
            if "punkt" not in rule:
                print(f"--  {t['id']}: {rule['doc']} whole act")
                continue
            path = _text_path(rules["documents"][rule["doc"]], texts)
            ok = quote_matches(rule["quote"], rule["punkt"], path.read_text(encoding="utf-8").split("\n"))
            print(f"{'OK ' if ok else 'BAD'} {t['id']}: {rule['doc']} {rule['punkt']}")
            bad += not ok
    return bad


if __name__ == "__main__":
    sys.exit(verify())
