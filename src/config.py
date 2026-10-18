"""Paths, seeds and product copy. Everything else imports from here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
MODELS = ROOT / "models"
WEIGHTS = MODELS / "weights"  # filled once by `make weights`; loaded offline afterwards
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FONTS = REPORTS / "fonts"  # Open Sans TTF for the figures, so PNGs rebuild the same on every machine
WEB_DIST = ROOT / "web" / "dist"
INSIGHTS_MD = ROOT / "docs" / "INSIGHTS.md"  # hypothesis table, built by src/data/insights.py (make all)
EXPERIMENTS = ROOT / "experiments.csv"  # append-only run log; tests redirect it to a tmp file
RULES_YAML = ROOT / "src" / "rules" / "rules.yaml"  # violation type -> norm (T05)
TYPES_MD = (
    ROOT / "docs" / "TYPES.md"
)  # type list with label counts, built by src/rules/types_doc.py (make all)
RULES_MD = ROOT / "docs" / "RULES.md"  # human copy of the type table, synced by tests/test_rules.py
ADILET_TEXTS = DATA_EXTERNAL / "adilet"  # page text of each act cited in rules.yaml, saved 18.10.2026
LABELS_XLSX = DATA_RAW / "open_labels.xlsx"  # organizers' sheet (T04)
VIDEOS_DIR = DATA_RAW / "videos"
TRACKS_DIR = DATA_PROCESSED / "tracks"  # per-video track caches (#32), one folder per detector setting
LABELS_CSV = ROOT / "data" / "labels.csv"  # cleaned labels, contract in contracts/data.py
TYPES_CSV = (
    ROOT / "data" / "violation_types.csv"
)  # team-set type per violation label, kept apart so a rebuild keeps it (#26)
ARRANGEMENTS_CSV = (
    ROOT / "data" / "arrangements.csv"
)  # team-set furniture arrangement per video, kept apart so a rebuild keeps it (#26)

TIMECODE_PRIOR_JSON = REPORTS / "timecode_prior.json"  # label position prior, scripts/timecode_prior.py

# Serving switches (fix-cv-model, docs/research/cv-audit.md). The defaults are main's behaviour; flipping one is
# P1's decision. VERDICT_POLICY: "classifier" = saved threshold OR heuristic vote; "always_violation" = every
# video is flagged and the pipeline only picks type and timecode. TIMECODE_MODE: "peak" = classifier peak;
# "prior" = fraction-of-duration prior from the training labels, kept for heuristic hits (their onset).
VERDICT_POLICY = os.environ.get("VERDICT_POLICY", "classifier")
TIMECODE_MODE = os.environ.get("TIMECODE_MODE", "peak")

SEED = 42
VERSION = "0.1.0-stub"

# Product copy (Russian, verb-led). P4 owns the wording.
PRODUCT_NAME = "Forte Риск-скоринг"

# Stub feature set. Replace with the real case features tomorrow.
TARGET = "target"
ID_COL = "client_id"
CATEGORICAL = ["region"]
REGIONS = ["Алматы", "Астана", "Шымкент", "Караганда", "Актобе", "Другие"]
FEATURES = ["age", "income_kzt", "debt_ratio", "days_overdue_max", "tenure_months", "region"]

FEATURE_LABELS = {
    "age": "Возраст",
    "income_kzt": "Доход в месяц",
    "debt_ratio": "Долговая нагрузка",
    "days_overdue_max": "Макс. просрочка, дн.",
    "tenure_months": "Клиент банка, мес.",
    "region": "Регион",
}

for p in (DATA_PROCESSED, MODELS, WEIGHTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
