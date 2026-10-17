# AGENTS.md — Forte Datathon '26 (team repo)

Source of truth for every coding agent (Claude Code reads it via `CLAUDE.md` → `@AGENTS.md`; Codex reads it directly).
Keep this file ≤ 150 lines. The brief checklist `docs/acceptance.md` overrides everything else.

## Mission
24-hour datathon, one common ForteBank case: a **local** AI service that finds cash-discipline violations
in cashier videos (.mov → verdict, type, timecode, evidence frame, description, НБРК rule). Plan: `docs/PLAN.md`.
The jury scores our **API automatically** on hidden videos (85/100 pts); only the shortlist pitches (5 min).
**Never build what the brief didn't ask for.** A ticket without a `TZ-x.y` ID is out of scope.
**Cloud AI is forbidden (disqualification):** no OpenAI/Gemini/Claude/any remote model call anywhere in `src/` or `api/`.

## Layout and ownership (touch only your folders unless the ticket says otherwise)
| Path | Owner | What |
|---|---|---|
| `contracts/` | P1 | Pydantic API schemas + pandera data schemas. Change = tell everyone |
| `api/` | P1 | FastAPI app; serves `web/dist` on the same port |
| `web/` | P1 (+P4 design QA) | Vite + React + TS + Tailwind v4, Forte tokens in `web/src/index.css` |
| `src/data/`, `data/labels.csv` | P3 | video decoding, labels, checks |
| `src/features/` | P3 | embeddings, person/pose/object features, evidence frames |
| `src/models/` | P2 | train / threshold / `analyze(path)`; writes `models/` and `reports/metrics.json` |
| `src/rules/rules.yaml`, `docs/RULES.md`, `docs/TYPES.md` | P4 | type → НБРК document, пункт, quote, URL, description template |
| `reports/figures/` | P3 | every figure has a takeaway title |
| `docs/INSIGHTS.md` | P3 | one row per hypothesis with verdict |
| `docs/acceptance.md`, `docs/PITCH.md`, `deck/` | P4 | brief checklist, script, slides |
| `experiments.csv` | P2 | one row per run, append-only |

## Commands (all must work from a clean clone)
```bash
make setup      # uv sync + npm ci in web/
make test       # pytest -q (deterministic tests only)
make app        # build web, run API on http://localhost:8000
make dev        # API with reload on :8000 + Vite dev server on :5173 (proxy /api)
make demo       # stub pipeline -> artifacts -> app
make all        # rebuild every number from raw data (source of truth for slides)
make tunnel     # public URL via cloudflared quick tunnel
```

## Workflow per ticket
1. Read the ticket: TZ IDs, owned folders, acceptance command, timebox (≤ 2 h).
2. Explore → plan (plan mode only if the change touches 3+ files) → implement → run the check.
3. **Finish by running the ticket's acceptance command and pasting its output.**
4. Commit: `feat(#12): short imperative summary` (conventional commits, issue number in scope).
5. Push the branch, open a draft PR right after the first push, squash-merge within 15 min of green.

## Git rules
- `main` is protected and always demoable. Never push to `main` directly.
- Branch: `<p1|p2|p3|p4>/<issue#>-<slug>`, lives ≤ 2 h.
- Worktree: `git worktree add ../wt-<issue#> -b p3/12-slug` then run the agent inside it. Copy `.env` yourself.
- Tags (`mvp0`, `mvp1`, `demo-v1`, `final`) are created by P1 only.
- No Git Flow, no CHANGELOG, no force-push to shared branches.

## Testing rules
Test only deterministic code:
- data contracts (pandera schemas in `contracts/data.py`)
- feature functions (pure, input → output)
- leakage guards (CV grouped by video; windows of one clip never in train and val at once)
- the metric vs a hand-computed value
- submission file format
- API contract (`tests/test_api.py`)
- 30-second smoke test of the whole pipeline on a tiny sample

**Never assert a model score.** Never add slow tests (> 5 s each).

## Code rules
- Python 3.11+, `uv` only (`uv add <pkg>`, never bare `pip install`). Ruff-format before commit.
- Scripts over notebooks: `src/**/*.py`; exploration in jupytext `# %%` files under `notebooks/`.
  Notebook outputs are stripped (`nbstripout`).
- Every number shown on a slide or in the UI must be produced by `make all`. No hand-typed numbers.
- Paths come from `src/config.py`. No absolute paths, no hard-coded user dirs.
- Random seeds fixed (`SEED = 42` in `src/config.py`).
- Validation: leave-one-video-out (grouped by clip). Fit any scaler on train folds only.
- Models run locally from `models/weights` (`make weights` downloads once; then `HF_HUB_OFFLINE=1` must work).
- Licenses: open source that allows commercial use only (Apache-2.0, MIT, BSD). **No Ultralytics YOLO (AGPL).**
  Add each model to `docs/MODELS.md` with its license before `uv add`.
- No face recognition or identification. No audio. Uploaded videos are deleted after the job, never used for training.
- Descriptions come from templates in `src/rules/rules.yaml`, filled with observed facts. No generated legal claims.

## API contract (from brief section 10; field names are fixed, the jury parses them automatically)
- All `/api/v1/*` need header `X-API-Key` (value from `.env` `API_KEY`); wrong or missing key → 401.
- `POST /api/v1/analyze` multipart field `file` (.mov, ~29 MB) → `{job_id}`
- `GET  /api/v1/jobs/{job_id}` → `{job_id, status: processing|done|failed, verdict: violation|no_violation,
  violations[{type, timecode_sec, description, evidence_frames[{timestamp_sec, image_base64}] (1–3 JPEG)}]}`
- Result within 5 min per video, assume 5 parallel jobs → budget ≤ 60 s per clip.
- `GET /api/health` → `{status, version, model_loaded}`. UI uses `/api/ui/*` (same handlers, no key).

## Front-end rules (Forte brand — non-negotiable)
- Tokens live only in `web/src/index.css`. Use them via Tailwind classes (`bg-primary`, `text-forte-dark`…).
- Font: Open Sans (800/700 headings, 600 labels, 400 body). Palette: Magenta `#A31551` primary,
  Dark Magenta `#4A0221`, Sky Blue `#61CFFF` (charts, sparingly), Coral `#FC9385`.
  Hero gradient Dark Magenta → Coral. Never recolor the logo.
- One primary (magenta) action per screen. Nothing above the fold except the logo, title, one input, one CTA.
- UI text in Russian, short verb-led labels ("Проверить клиента"). Money: `1 250 000 ₸` (thin-space thousands; use `formatKzt`).
- Banned: purple-blue gradients, glassmorphism, cards inside cards, grey text on colour,
  emoji/icon tile above every heading, lorem ipsum, rainbow charts (charts = grey + one accent).
- Every screen has loading, empty and error states.
- Design work goes through Impeccable (`/impeccable craft|critique|audit|polish`); brand text in `docs/IMPECCABLE_BRAND.md`.

## Analysis rules (P2/P3)
- Every model change logs a row in `experiments.csv`: video recall, FP rate, median |Δt| s, type accuracy (LOVO CV).
- Threshold: recall 1.0 on CV first (a missed violation is unacceptable), then lowest FP (FAQ 16).
- Chart titles state the takeaway, not the variable name.

## Limits and sessions
- Default model Sonnet or `opusplan`; Haiku for subagents. `/clear` between tickets.
- After 2 failed corrections: `/clear` and re-prompt with what you learned.
- Check `/usage` (Claude) or `/status` (Codex) at each checkpoint (17:00, 00:00, 06:00).
- No agent teams, no `claude -p` fan-out. Don't update skills/tools during the event.
- Before stopping mid-ticket: run `/handoff` and post it on the issue.

## Freeze
After **06:00**: no new features, no number changes. Fixes only, reviewed by P1.
