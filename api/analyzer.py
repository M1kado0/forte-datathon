"""Local placeholder analyzer; never decodes uploaded videos."""

import base64
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from contracts.schemas import EvidenceFrame, Violation
from src.config import RULES_YAML


def analyze_stub(path: Path) -> list[Violation]:
    rules = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
    fallback = next(item for item in rules["types"] if item.get("fallback"))
    image = BytesIO()
    Image.new("RGB", (64, 64), color="white").save(image, format="JPEG")
    return [
        Violation(
            type=fallback["name"],
            timecode_sec=0,
            description=fallback["description"],
            evidence_frames=[
                EvidenceFrame(
                    timestamp_sec=0, image_base64=base64.b64encode(image.getvalue()).decode("ascii")
                )
            ],
        )
    ]
