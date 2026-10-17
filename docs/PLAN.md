# PLAN. Forte Datathon '26 (Planner seat, 17 Oct 12:45)

**Decision:** локальный видео-сервис «подсказка проверяющему»: .mov → вердикт + нарушения (тип, таймкод, кадр, описание, пункт НБРК), API `/api/v1` с ключом.
**Next action:** все четверо смотрят видео датасета (T04, 45 мин), P1 параллельно начинает T01.
**Owner:** P1 ведёт интеграцию; критический путь T04 → T03 → T01 → T10.

## What changed vs master prompt v4
| Master prompt said | Brief says | So we |
|---|---|---|
| Tabular case, LightGBM + SHAP | Video of a cashier, overhead camera, ~40 clips 10–145 s | Replace the tabular pipeline with a video pipeline |
| "Use Gemini/Claude API for text" | Cloud AI is forbidden, disqualification (FAQ 21, 35) | Every model runs inside our service. No API calls in `src/` |
| 90-s pitch, judged on stage | Automatic scoring through our API; 5-min pitch only for the shortlist (FAQ 18, 19) | 85 of 100 points come from the API JSON. Engineering and ML first, pitch after 06:00 |
| `/predict`, `/insights`, what-if slider | `POST /api/v1/analyze`, `GET /api/v1/jobs/{id}`, `X-API-Key` | Drop insights and what-if screens |
| Deadline ~10:30 | Deadline 12:00 (FAQ 33) | Submit at 11:00 |

## The decision the bank makes
The reviewer (внутренний контроль) decides which video fragment to watch and whether to open a case.
Our output tells them where to look and which rule applies. It is a hint, never a verdict (TZ-7.2).

## Approach (locked by the grill 17 Oct; decisions in `GLOSSARY.md`, `docs/adr/0001`–`0003`)
1. **Decode.** OpenCV, 4 fps, native 464×848, no downscale. A 145-s clip gives ~580 frames.
2. **Branch A: "something is wrong here, at second X".** Frame embeddings (DINOv2-small, center crop off, padded to square; SigLIP-base as backup) → 2-s windows, 0.5-s step → logistic regression. In a violation video a window is positive if it overlaps [N−1, N+4] s around label N, negative if it ends before N−3 s, and unlabeled after N+4 s (lasting states). Videos without a violation give negatives over their full length. Timecode = onset of the above-threshold run for lasting states, peak for short actions (T06 measures both). At most 2 violations per video, ≥ 5 s apart.
3. **Branch B: "what exactly happened".** D-FINE-S people and COCO objects (phone, bag, cup) + ByteTrack tracks + cash detection (OWLv2 zero-shot → desk-zone change → D-FINE fine-tune) → zone features. No pose: RTMPose and default rtmlib weights are trained on non-commercial data (`docs/research/models.md` §8). Heuristics give the type; kNN on labeled windows is the fallback; "Иное подозрительное действие" when both are unsure (TZ-F.9). A heuristic firing ≥ N s in a row can also raise the verdict (ADR 0003).
4. **Explain.** Template description per type from `src/rules/rules.yaml`: what the features saw + document + пункт. No free text generation (TZ-5.6, criterion 4).
5. **Threshold.** Pick it on leave-one-video-out CV so recall = 1.0 at the lowest FP rate (FAQ 16).

- Validation: grouped by video (leave-one-video-out). Metrics: video recall, FP rate, median |Δt| s, type accuracy vs our labels.
- ASSUMPTION: the labels give one timecode per violation; we check in T04 whether some videos have several.
- RISK: Ultralytics YOLO is AGPL-3.0. Do not use it. Check every LICENSE before `uv add` (TZ-F.3).
- RISK: a local VLM (Qwen2.5-VL-7B, Apache-2.0) needs ~6.9 GB in 4-bit (AWQ); it likely fits on the 16 GB RTX 5080 next to the detector, to be measured. Qwen2.5-VL-3B has a non-commercial license, so it is out. Stretch only, and only on the RTX machine.
- RISK: the test angle differs (FAQ 4). The training set already has 7 camera angles. T03 reports leave-one-angle-out next to LOVO; T13 (warps, own recordings) stays stretch.

## Grill agenda (path lock 13:30, 25 min hard stop)
1. Serving machine. **Decided in ADR 0001: Ubuntu + RTX 5080 16 GB, torch on `cu132`.** Original agenda: P1 has a Mac (M1, 8 GB); one teammate has an RTX 5070 on Windows. Default: the RTX machine serves via `cloudflared` and runs embedding extraction; the Mac is the CPU fallback. It is a laptop; we plan for 8 GB VRAM (worst case), so no VLM in the base plan. The RTX owner is P2 (Алдияр С.); P3 (Ескендир) sets up serving and the tunnel on that machine.
   - RISK: RTX 50xx (Blackwell) needs PyTorch ≥ 2.7. The `cu128` index stops at torch 2.11; torch 2.14.x ships on `cu132`, which the serving driver 595.99.02 supports. Plain `uv add torch` may not pick the CUDA build. T02 pins torch to the `cu132` index for Linux (`docs/research/models.md`, section 3).
   - RISK: the serving box must not sleep or update from 11:00 until results: disable suspend and unattended-upgrades; API and tunnel run under systemd.
2. Laptop + tunnel or our own VM? Brief: models must run inside our service, public URL required, teams use own resources. ASSUMPTION: a self-rented VM running our own models is allowed. Default: laptop + tunnel; ask organizers before paying for a VM (QUESTIONS #8).
3. Windows dev setup: `make` is not native. Default: P2–P4 use Git Bash or WSL; every Makefile target also has a plain `uv run ...` line in README.
4. Git repo: this folder is not a git repo yet. Who creates the remote, and when?
5. Type taxonomy from T04: which 4–7 types, which are rule-based, which need the window classifier.

## Timeline (deadline 12:00, 18 Oct)
| Time | Phase | Exit criterion |
|---|---|---|
| 12:45–13:30 | T04 everyone watches clips · P1 starts T01 · P2 runs T02 throughput test | `data/labels.csv` + draft types |
| 13:30–14:00 | Path lock (all, 30 min hard stop) | 4–7 types, approach confirmed, tickets assigned |
| 14:00–17:00 | T01, T02, T03, T05, T08 | **17:00 #1:** public URL returns valid JSON from the baseline. Tag `mvp0` |
| 17:00–21:00 | T06, T07, T08, T09, T13 | **21:00:** first good version on the public URL; jury simulation run #1 |
| 21:00–00:00 | Close every 🟥 row, README | **00:00 #2:** tag `mvp1`; jury simulation run #2 |
| 00:00–03:00 | Sleep A: P1 + P3. P2 error analysis; P4 verifies RULES.md | `/handoff` on open tickets |
| 03:00–06:00 | Sleep B: P2 + P4. P1 hardens the service; P3 runs robustness eval | all numbers from `make all` |
| **06:00** | **FREEZE** features and numbers. Tag `demo-v1` | acceptance all 🟩 or ⬜ |
| 06:00–09:00 | README final, limitations, backup video, 5-min deck | deck v1 |
| 09:00–11:00 | Rehearse 5-min pitch ×3 · **submit at 11:00** | tag `final`; nobody touches the serving laptop |

## Lanes
| Window | P1 Тамерлан, Fullstack | P2 Алдияр С., ML + CV | P3 Ескендир, Infra + heuristics | P4 Алдияр М., Data/Rules/Pitch |
|---|---|---|---|---|
| 12:45–14:00 | T01 | 10 clips of T04 | T02 | T04 lead (labels), QUESTIONS.md |
| 14:00–17:00 | T01 | T03 | T02 → T10 (quick tunnel) | T05 |
| 17:00–21:00 | T07 | cash check → T08 detections → T06 | T15, Tailscale Funnel + systemd by 21:00 | T04 angle column + duplicates, T09 |
| 21:00–00:00 | T07 polish, fixes | T06 tuning | T15 vote, T10 rerun | T11, INSIGHTS |
| 00:00–06:00 | sleep, then hardening | T12, then sleep | sleep, then T13 / VLM stretch | RULES check, then sleep |
| 06:00–11:00 | submit, UI fixes | T12 numbers into README, machine stays on | keep the public URL up | T14 |

## Tickets (≤ 2 h each; every ticket ends by running its check)
| # | Title | TZ | Owner | Folders | Acceptance command | Label |
|---|---|---|---|---|---|---|
| T01 | Walking skeleton: API v1 + key + job queue + stub analyzer | 4.8, 10.1–10.3, 5.1, F.5 | P1 | `contracts/`, `api/`, `tests/` | `uv run pytest tests/test_api.py` + curl below | mvp0 |
| T02 | Video deps (torch cu132 on Linux), `make weights`, offline run, license table | 7.3, 7.4, F.3 | P3 | `pyproject.toml`, `scripts/`, `docs/MODELS.md` | `HF_HUB_OFFLINE=1 uv run python -m src.data.video data/raw/<clip>.mov` prints fps processed | mvp0 |
| T03 | Baseline: embeddings → window classifier → LOVO CV → `analyze(path)` | 4.4, 5.2, 5.4 | P2 | `src/features/`, `src/models/`, `experiments.csv` | `uv run python -m src.models.train` writes `reports/metrics.json` | mvp0 |
| T04 | Data card + labels + type taxonomy | 8.5 | P4 (+all watch) | `src/data/`, `data/labels.csv`, `docs/DATA_CARD.md`, `docs/TYPES.md` | `uv run python -m src.data.check_labels` prints clips per type, fails on missing files | mvp0 |
| T05 | Normative research: type → document, пункт, quote, URL | 4.7, 6.1, 6.2, 7.1, 8.4 | P4 (P3 verifies links) | `src/rules/rules.yaml`, `docs/RULES.md` | `uv run pytest tests/test_rules.py` (every type has doc, пункт, URL, quote) | mvp0 |
| T06 | Types, multi-event split, threshold policy, fallback type | 4.4, F.6, F.7, F.9 | P2 | `src/models/` | `uv run python -m src.models.train` shows recall, FP, type acc | mvp1 |
| T07 | UI: upload → status → result list with frames, rule, disclaimer | 4.1–4.7, 7.2 | P1 (+P4 QA) | `web/` | `make app`, upload a clip, Playwright screenshots 1440/390 | mvp1 |
| T08 | Person/object/cash features + 1–3 evidence frames | 4.5, 10.4, F.1 | P2 features, P3 evidence frames | `src/features/`, `src/data/` | `uv run pytest tests/test_features.py` (frame count 1–3, valid JPEG) | mvp1 |
| T09 | Template descriptions linked to rules | 4.6, 5.6, 6.2 | P4 | `src/rules/` | `uv run pytest tests/test_rules.py` (no template claims invisible data) | mvp1 |
| T10 | Public URL + jury simulation (5 parallel jobs) | 8.1, 10.5, F.4 | P3 | `scripts/` | `make jury-sim PUBLIC_URL=$PUBLIC_URL` → p95 < 240 s for 5 parallel jobs, ≤ 45 s per 145-s clip, 0 schema errors | mvp0 |
| T11 | README: run, use, architecture, models | 8.2, 8.3 | P4 | `README.md` | a teammate follows it on a clean clone | mvp1 |
| T12 | Error analysis + limitations | 8.6 | P2 | `README.md`, `reports/` | FP/FN table with clip names | mvp1 |
| T13 | Robustness: angle augmentation + own recordings | F.8 | P3 | `src/features/`, `data/external/` | eval script prints metrics on warped clips | stretch |
| T15 | Type heuristics + heuristic verdict vote | 4.4, F.9 | P3 | `src/heuristics/` | `uv run pytest tests/test_heuristics.py` | mvp1 |
| T14 | 5-min pitch + Q&A (shortlist only) | P.1 | P4 | `deck/`, `docs/PITCH.md` | 3 timed runs ≤ 4:45 | after 06:00 |

### T01 details (P1 starts now, needs no data)
- `contracts/schemas.py`: `EvidenceFrame{timestamp_sec, image_base64}`, `Violation{type, timecode_sec, description, evidence_frames[1..3]}`, `JobResult{job_id, status: processing|done|failed, verdict: violation|no_violation|None, violations[]}`. Field names exactly as section 10.
- `X-API-Key` from `.env` (`API_KEY=`), 401 without it. UI calls `/api/ui/*` handlers that share the code and need no key.
- One background worker, in-memory job dict, upload to a temp file, delete it after the job (TZ-F.5). `failed` on any exception.
- Stub analyzer: returns `violation` with one frame grabbed by OpenCV at mid-video, so the full JSON path works before the model exists.
- Remove `/predict`, `/insights`, `/meta` and their tests.
- Check: `curl -X POST localhost:8000/api/v1/analyze -H "X-API-Key: $API_KEY" -F "file=@data/raw/<clip>.mov"` then `curl localhost:8000/api/v1/jobs/<id> -H "X-API-Key: $API_KEY"`.

### T05 search starting points (titles must come from adilet, never from memory)
- adilet.zan.kz: «кассовые операции банков второго уровня», «инкассация банкнот и монет», «кассовые помещения требования», «техническая укреплённость банков», «охрана объектов банков».
- For each type: exact title, number and date of the act, пункт, a 1–2 sentence quote, URL. A second person opens every URL before it ships (criterion 4 punishes invented links).
