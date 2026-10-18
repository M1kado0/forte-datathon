"""Load the local model for serving; keep an explicit placeholder for contract tests."""

import base64
import os
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from api.jobs import Analyzer
from contracts.schemas import EvidenceFrame, Violation
from src.config import RULES_YAML


def load_analyzer() -> Analyzer:
    """Warm the model's process-wide caches before accepting uploads; fail if weights are missing."""
    if name := os.environ.get("FORTE_ACTION_MODEL"):
        from src.models.action_pipeline import ActionAnalyzer

        return ActionAnalyzer(name)

    from src.models.analyze import analyze, default_detector, default_embedder, default_model

    default_model()
    default_embedder()
    default_detector()
    return analyze


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
