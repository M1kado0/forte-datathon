# {{PRODUCT_NAME}} — DATATHON '26 × ForteBank

> {{who}} gets {{what}} in {{how fast}}.

**Live demo:** {{tunnel URL}} · **Video:** {{backup video link}} · **Slides:** `deck/`

## What it does
{{2–3 sentences: input → result + why → action. Screenshot below.}}

![screen](reports/screens/desktop-3-result.png)

## Brief coverage
Every requirement of the brief and where to see it: [`docs/acceptance.md`](docs/acceptance.md).

## Key finding
{{finding}} — **{{BIG NUMBER}}**. Holds on {{validity check}}. Details: [`docs/INSIGHTS.md`](docs/INSIGHTS.md).

## Run it
```bash
make setup     # uv sync + npm ci
make all       # rebuild data → model → numbers (source of truth)
make app       # http://localhost:8000
make test
```
Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+.

## How it works
| Step | File |
|---|---|
| Load + validate data | `src/data/`, `contracts/data.py` |
| Features | `src/features/` |
| Model (LightGBM → calibrated → SHAP reasons) | `src/models/` |
| API | `api/main.py`, contract in `contracts/schemas.py` |
| UI | `web/` (React + Tailwind, Forte brand tokens) |

Validation: {{time-based / grouped split}}. Metrics: `reports/metrics.json`. Runs log: `experiments.csv`.

## Limitations
{{honest, short}}

## Team
{{names · roles}}
