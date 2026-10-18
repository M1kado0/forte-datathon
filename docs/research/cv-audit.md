# CV pipeline audit (branch `fix-cv-model`, against main `5b0a6e2`, 18 Oct 2026)

Scope: what the served model does from upload to JSON, what it gets wrong, and what each defect is expected to cost
on the jury's hidden videos. Every claim cites `path:line` or a command whose output is in the PR.
"UNVERIFIED" means nobody here measured it. Three evaluation settings are kept apart:
- **in-sample**: fit or chosen on the same clips it is scored on;
- **OOF**: out-of-fold, leave-one-video-out (LOVO) unless stated;
- **held-out**: never touched while tuning. **Nothing in this repo is held-out yet.**

Machine caveat. This audit ran on a laptop: RTX 3050 Ti, 4 GB, WSL, with torch on CPU because CUDA init crashes
here. That laptop does **not** hold the served `models/window_clf.pkl` or the D-FINE and OWLv2 weights (only
`dinov2-small`). So:
- the real analyzer was not run;
- no timing below is serving-machine timing;
- the served settings are read from `reports/metrics.json` on main, and whether the serving pickle matches them
  is UNVERIFIED.

## 1. Served path (main)

```
POST /api/v1/analyze            api/main.py:90  (X-API-Key check :73-79)
  save upload                   api/main.py:100-102
  probe is_video (15 s, FFmpeg) api/main.py:104-113 -> api/uploads.py:56-58
  JobQueue.submit               api/jobs.py:137-139   one queue task (main.py:60), ONE worker process (jobs.py:131)
  worker, 55 s per job          api/jobs.py:127; timeout -> kill, status failed, model reload (jobs.py:118,159-161,175-180)
  analyze(path)                 api/analyzer.py:15-22 -> src/models/analyze.py:77
    branch A  embed_video       src/features/embeddings.py:85  (4 fps, DINOv2-small, native size padded square :34-57)
              score_windows     src/models/analyze.py:113 -> windows.py:24 (2 s windows, 0.5 s step) -> logreg
    branch B  build_tracks      src/features/tracks.py:71 (2nd full decode; D-FINE-S 640x640 stretch detectors.py:110,
                                OWLv2 960 padded :137-143; in-house ByteTrack) -> src/heuristics detect
    decide                      src/models/violations.py:131-158
                                verdict = max window p >= threshold (:143) OR voting heuristic hit (:144-145)
                                type    = allowed heuristic hit -> kNN -> "other" (:147-154)
                                time    = state onset for hits (:207), classifier peak otherwise (:155)
    evidence_frames             src/data/evidence.py:21 (1-3 JPEG, offsets by kind :14-17)
    describe_violation          src/rules/describe.py:73-82 (template + citation from rules.yaml)
  Violation(type = Russian name, timecode = frame actually decoded)  analyze.py:103-108 -> contracts/schemas.py:38-49
GET /api/v1/jobs/{id}           api/main.py:122-127 -> in-memory dict api/jobs.py:134
```

Served settings, as `reports/metrics.json` on main records them (the pickle itself is UNVERIFIED):
- **thresholds:** `threshold` 0.00792 (OOF) and `saved_threshold` 0.00451 (#92, shifted by −0.568 logit);
- **vote:** on, `outsider_in_cash_room`, ≥ 3 s;
- **types:** `allowed` = {outsider_in_cash_room}, `heuristics` = `controller_exempt`, kNN **disabled**;
- **timecode:** `timecode_mode` = peak.

**Exists but not on the served path:**
- **scripts:** `scripts/jury_sim.py`, `scripts/heuristics_lovo.py`, `scripts/standin_tracks.py`;
- **training and evaluation only:** `src/models/evaluate.py`, `src/models/errors.py`, `src/models/explain.py`;
- **feature checks:** `src/features/cash_check.py`, `src/features/adr5_check.py`;
- **unused functions in served modules:** `load_or_embed` (`embeddings.py:105`) and the fold and threshold
  helpers in `windows.py:58-111`.

Scene-grouped CV (`src/models/scene_cv.py`, `data/scenes.csv`) is not on main. It is in draft PR #108
(`fix-the-cv-model`), together with a default flip (`FAIL_SAFE_VERDICT = True`, citing an ADR 0006 that is not in
`docs/adr/` on main). #108 and this branch both change `analyze.py`, so only one of them should merge.

## 2. Requirements vs implementation

| Requirement (BRIEF) | Implementation | Status | Risk on hidden videos |
|---|---|---|---|
| Verdict per video (crit. 1, 25 pts) | `violations.py:143-146`, `jobs.py:156` | met, no signal | **High**: OOF AUC 0.33, flags 18/19 clean; on a new view the scores collapse (§4 D1) |
| Several violations | `MAX_VIOLATIONS=2` `violations.py:32` | partly | the classifier path returns exactly 1 (`:158`), and only outsider may be typed, so 1 in practice |
| Type (crit. 3, 15) | hit → kNN (off) → `other` `violations.py:147-154` | partly | OOF type accuracy 2/21 (`metrics.json` `type_accuracy` 0.095) |
| Timecode (crit. 2, 15) | `analyze.py:105`, peak `violations.py:155` | met, weak | peak median \|Δt\| 5.0 s, chosen in-sample between 2 modes (`train.py:376`), vs midpoint 3.60 s |
| Evidence 1-3 JPEG | `schemas.py:42`, `evidence.py:21-63` | met | a raise in `evidence_frames` fails the whole job (`analyze.py:99`) |
| Description of visible facts | templates `describe.py:73-82`, facts = `{object}` only | met | low |
| НБРК document + пункт (crit. 4) | `rules.yaml` | met | `other`, `laptop_left_open`, `cash_partially_given` cite a whole act (`rules.yaml:205-211,304-310,318-323`) |
| No invented facts | no generation (`describe.py:79`) | met | low |
| Local only | no remote client in `src/`, `api/`; `HF_HUB_OFFLINE` `embeddings.py:21`, `detectors.py:63` | met | low |
| Result ≤ 5 min (FAQ 14) | 1 worker, 55 s timeout (`jobs.py:127,131`) | partly | 5 × 55 s = 275 s worst case; one timeout plus reload (≤ 180 s, `main.py:33`) pushes later jobs past 300 s |
| New camera angle (crit. 5, 10) | none for branch A; heuristics in cashier-box-height units | **not met** | leave-one-layout-out pooled recall 16/21 (`metrics.json` `leave_one_layout_out`) |
| Unknown types | fallback `other` | met | low |
| API contract | `contracts/schemas.py:33-66` (extra=forbid) | met | `type` is the Russian name (`describe.py:82`); whether the jury wants the id is UNVERIFIED |
| Licenses | D-FINE, OWLv2, DINOv2 Apache-2.0 (`docs/MODELS.md:11-14`); in-house ByteTrack | met | doc gaps only |

## 3. Status of the earlier "known findings" (re-checked on main)

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1 | Single classifier chain, heuristics not served | **STALE** | #94: branch B runs, and the outsider vote can raise the verdict (`analyze.py:91-95`, `violations.py:144`) |
| 2 | LOVO AUC 0.33, flags 18/19 clean | CONFIRMED | recomputed from `metrics.json` `video_scores`: AUC 0.3308; at 0.00792 and at 0.00451, 21/21 violations and 18/19 clean flagged; only X-8 is unflagged |
| 2b | Scene-grouped nested recall 0.90, D-rel AUC ~0.5 | UNVERIFIED by me | only in #108's `reports/scene_cv.json` (proxy features `proxy-448px-2fps-cpu`, not the served embeddings) |
| 3 | Scores follow the scene | PARTLY | X-3 (clean) 0.9997 and X-12 (clean) 1.0 in OOF scores; the scene-level claim is UNVERIFIED |
| 4 | Saturation, collapse on a warped view | CONFIRMED / UNVERIFIED | 28/40 OOF scores ≥ 0.5 with the threshold near 1e-2; warp collapse only in #108's proxy run (0/3 warped violations flagged) |
| 5 | Paired takes, duplicates | twins CONFIRMED, open; duplicates **handled** | X-47→X-4, X-48→X-5 merged in `labels.py:59-67`; no scene grouping on main |
| 6 | Labels: one second, 1 camera | CONFIRMED | `labels.py:20-33`; X-4 has 57 and 68 s; states after N+4 are UNLABELED, never positive (`windows.py:24-40`) |
| 7 | Timecode baselines | MEASURED here | `scripts/timecode_prior.py`: prior 2.83 s, midpoint 3.60 s, peak 5.0 s (`metrics.json`). Within 1 s: **19 %** (4/21) for both, not "about 5 %" |
| 8 | Every type "other", ≤ 1 violation | PARTLY | outsider can be named (3/40 OOF verdicts); 2 allowed but 1 in practice |
| 9 | Native size, 1080p blows the timeout, rotation implicit | CONFIRMED (mechanism), timing UNVERIFIED | 1920×1080 pads to 1922² = 18769 tokens vs 3721 for 464×848; fixed on this branch (§5) |
| 10a | OOF threshold on refit model | PARTLY STALE | #92 shifts it (`train.py:389-406`), but the shift is estimated on all folds (in-sample); nested still misses X-11 |
| 10b | Max over windows → length bias | mechanism CONFIRMED, effect not measurable | `violations.py:143`; clean clips: Spearman(duration, OOF score) −0.23, p 0.34 |
| 10c | X-11 / X-12 darkness shortcut | UNVERIFIED (no frames viewed) | X-11 is the lowest violation score and **is** the threshold; X-12 (clean) scores 1.0 |
| 10d | Constant-fps timestamps | CONFIRMED, harmless on this data | `video.py:40`; measured max \|Δt − 1/fps\| over the first 300 frames of each of 42 files: 0.27 ms |
| 10e | Cyrillic Х / Latin X mix | **STALE** (fixed) | `labels.py:11-17` |
| 10f | experiments.csv 2-6, insights.json are stubs | CONFIRMED | `experiments.csv:2-6`; `reports/insights.json` is the credit-risk template, not rebuilt by `make all` (`pipeline.py:45-52`) |
| 10g | README stale | CONFIRMED | `README.md:109` promises no missed violation (nested recall 20/21); `:113` names kNN typing, which is disabled |

## 4. Defects, ranked by expected points lost on the hidden test

The hidden set is "5+ videos" (FAQ 12) from at least one new camera angle (FAQ 4). The point estimates below are
reasoning, not measurements.

| # | Defect | Evidence | Criterion at risk | Fix on this branch |
|---|---|---|---|---|
| **D1** | **The verdict has no signal and misses on a new view.** In-sample LOVO threshold equal to X-11's score; OOF AUC 0.33; layout shift misses 5/21; proxy warp misses 3/3 | `metrics.json` `threshold`, `video_scores`, `leave_one_layout_out.pooled`; #108 `scene_cv.json` | 1 (25), 5 (10). A missed violation is "недопустим" (FAQ 16) | `VERDICT_POLICY=always_violation` (flag, P1 decides) |
| **D2** | **Large uploads time out → `failed`.** No resize before DINOv2; ~5× tokens at 1080p, more at 4K | `embeddings.py:34-57` (main) | every criterion for that video | **fixed**: long side ≤ 848 in `embed_video`, no-op on training clips |
| **D3** | **Queue is sequential with a 55 s timeout and a reload after each timeout** | `jobs.py:127,131,175-180`; `main.py:33` | 6 (10), and 1-4 for late jobs | none (P1 owns `api/`). D2 removes the most likely timeout |
| **D4** | **Timecode is the classifier peak, as good as a constant** | median \|Δt\| 5.0 s vs midpoint 3.60 s, prior 2.83 s (LOVO) | 2 (15) | `TIMECODE_MODE=prior` (flag) |
| **D5** | **Type is `other` for everything except outsider** | type accuracy 2/21 OOF; kNN 0 right in every grid cell (`metrics.json` `knn.grid`) | 3 (15) | none: no validated cue. Do not enable kNN |
| **D6** | **Served settings may predate #102.** `metrics.json` still lists `workplace_left` and `count_without_controller`, which #102 removed; vote and allowed types were fit at the unshifted threshold, while serving uses the shifted one | `train.py:340`, `analyze.py:96`; audit of `metrics.json` | 1, 3 | none: P2 must confirm which pickle is served (UNVERIFIED) |
| **D7** | **An evidence-decode error loses the whole verdict** | `analyze.py:99` (no try) | 1-4 for that video | not changed (behaviour change, needs P1) |
| **D8** | **Heuristic geometry in cashier-box heights survives zoom, not perspective; D-FINE stretches frames to 640×640** | `thresholds.py:11-54`, `roles.py:22-24,67-68`, `detectors.py:110-112` | 5 (10) | none (Phase 4) |
| **D9** | **Rotation tag implicit** | `video.py`, `evidence.py` (main) | 5 | **fixed**: `CAP_PROP_ORIENTATION_AUTO=1` set explicitly. All 42 files have tag 0, so training frames are unchanged |
| **D10** | **Rule grounding gaps.** The fallback and 2 types cite a whole act without a пункт | `rules.yaml:205-211,304-310,318-323` | 4 (15) | none (P4 content) |
| D11 | Determinism not enforced: fp16 detectors, no torch seed | `detectors.py:69` | 7 | UNVERIFIED; check same clip twice on the serving box |
| D12 | `jury_sim` checks 45 s, the server allows 55 s; `VERSION = "0.1.0-stub"` in `/api/health` | `scripts/jury_sim.py:30`, `config.py:35` | 6 | none |
| D13 | Stale docs: `vote.py:6-12` docstring, `thresholds.py:32-36` comment, README `:109,:113`, `make demo` overwrites `reports/metrics.json` | audit §C | 7 | listed only |
| D14 | `/api/ui/*` runs the same analysis without a key over the public tunnel | `main.py:130` | 6 (abuse could fill the single queue) | none (P1) |

## 5. What this branch changes (all tests pass; full output in the PR)

1. **Input cap** (`src/features/embeddings.py`). `cap_long_side` shrinks a frame to long side 848 inside
   `embed_video`, before batching, so the patch mask follows the resized frame. `FINGERPRINT` is unchanged, so the
   saved classifier still loads. On the 464×848 training clips it returns the same array object.
   Tests: `test_1080p_upload_is_capped_to_the_training_long_side`, `test_training_size_frames_are_not_resized`,
   `test_fingerprint_is_unchanged_by_the_cap`.
2. **Rotation** set explicitly in `iter_frames` and `evidence_frames`. Measured `CAP_PROP_ORIENTATION_META` = 0
   on all 42 files, and OpenCV 5.0 already defaults `ORIENTATION_AUTO` = 1, so no training frame changes.
3. **`VERDICT_POLICY`** (env var, read on every call by `src/config.py:verdict_policy`): `classifier` (default,
   main) | `always_violation`. To flip it on the serving box, add `VERDICT_POLICY=always_violation` to `.env` and
   run `sudo systemctl restart forte-api`. `api/main.py:35` loads `.env` before the worker is spawned.
4. **`TIMECODE_MODE`** (env var): `peak` (default, main) | `prior`. `prior` puts a violation no heuristic saw at
   `0.669 × duration`: the median label fraction, from `scripts/timecode_prior.py` →
   `reports/timecode_prior.json`. Its evidence frames sit at −3, 0 and +3 s. A heuristic hit keeps its onset.

### P1 decision memo: `VERDICT_POLICY=always_violation`

- **What it does.** Every video gets `violation`. The pipeline then only chooses the type, the timecode and the
  frames.
- **Why.**
  - **OOF on this camera:** the classifier flags 18/19 clean clips (LOVO, threshold chosen in-sample) and has
    AUC 0.33, below chance.
  - **Recall drops whenever the view changes.** Nested LOVO 20/21 (X-11 missed). Leave-one-layout-out pooled
    16/21. Scene-grouped nested 19/21 and 0/3 warped violations flagged (#108's proxy run, not the served
    embeddings).
  - **always_violation is 21/21 under every scheme, by construction.** Its cost in false positives:
    - LOVO: one more, X-8 (19/19 against 18/19);
    - leave-one-layout-out: six more (19/19 against 13/19).
- **Trade-off.**
  - It forfeits criterion 1 on every clean hidden video.
  - The classifier already forfeits it on about 95 % of clean videos on the training camera, and on a new
    camera it starts missing violations, which FAQ 16 calls unacceptable.
  - "Злоупотребление без обоснования штрафуется" (FAQ 15) concerns "Недостаточно данных", not this verdict. Still,
    the README must state the policy openly.
- **Recommendation:** flip to `always_violation` and state it in the README. **P1 decides.**

### P1 decision: `TIMECODE_MODE=prior`

- **Numbers (LOVO, 21 violation videos):** prior 2.83 s median |Δt|, midpoint 3.60 s, peak 5.0 s.
- **Shares within 1, 2, 3 and 5 s:**
  - prior: 4, 9, 12 and 18 of 21;
  - midpoint: 4, 8, 8 and 14 of 21.
- **Caveat.** It is a prior from where the training labels sit, not something seen in the video. If the hidden
  clips are cut differently, it is no better than the midpoint. The README must say so.
- **Recommendation:** flip, with that disclosure. **P1 decides.**

## 6. NOT RUN in this pass (time-boxed to the 11:00 stop)

- `scripts/eval_cv.py`:
  - the fold schemes (scene, session blocks, arrangement, camera pose, leave-one-type-out);
  - the untouched final-test hold-out;
  - stress probes;
  - one `experiments.csv` row per design and probe.
- The jury simulation with the real analyzer (#67): needs the serving box.
- An offline end-to-end start with `HF_HUB_OFFLINE=1` and the same clip analyzed twice: needs the served pickle and
  the D-FINE and OWLv2 weights.
- Serving-machine timing of a 1080p clip before and after the cap.
- Phase 4 (object- and event-centric model): not started. It must not touch main before P1 approves.
