"""Paths, seeds and product copy. Everything else imports from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
MODELS = ROOT / "models"
WEIGHTS = MODELS / "weights"  # filled once by `make weights`; loaded offline afterwards
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
WEB_DIST = ROOT / "web" / "dist"
EXPERIMENTS = ROOT / "experiments.csv"  # append-only run log; tests redirect it to a tmp file
RULES_YAML = ROOT / "src" / "rules" / "rules.yaml"  # violation type -> norm (T05)
RULES_MD = ROOT / "docs" / "RULES.md"  # human copy of the type table, synced by tests/test_rules.py
ADILET_TEXTS = DATA_EXTERNAL / "adilet"  # page text of each act cited in rules.yaml, saved 18.10.2026
LABELS_XLSX = DATA_RAW / "open_labels.xlsx"  # organizers' sheet (T04)
VIDEOS_DIR = DATA_RAW / "videos"
LABELS_CSV = ROOT / "data" / "labels.csv"  # cleaned labels, contract in contracts/data.py

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
