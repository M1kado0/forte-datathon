# Data card (P4)

Labels for the ForteBank cashier videos. Source: `data/raw/open_labels.xlsx` (organizers) and `data/raw/videos/*.mov`
(42 files, not in git, see `docs/SETUP.md`). Cleaned labels: `data/labels.csv`, contract: `contracts/data.py`.

| Item | Value |
|---|---|
| Files | `open_labels.xlsx` (42 rows × 3 columns: file name, timestamp text, verdict text), 42 `.mov` clips |
| Cleaned output | `data/labels.csv`: 41 rows, 40 unique videos. `violation_type` comes from `data/violation_types.csv`, `arrangement` from `data/arrangements.csv` |
| Grain (one row = ?) | One label. A video with two violations has two rows (different `second`) |
| Target | `verdict` = `violation` / `no_violation`; for violations also `second` (whole second of the event) |
| Counts | 40 videos: 21 with violations (22 labels), 19 without. Printed by `uv run python -m src.data.check_labels` |
| ID columns / repeat videos | `video_id` (normalized, ASCII `X-<n>`). Two clips were delivered twice, see below |
| Date columns → split scheme | None. Leave-one-video-out grouped by `video_id`; merged duplicates keep a clip from landing on both sides |
| Missingness | `second` is empty exactly when `verdict` is `no_violation` (enforced by the contract) |
| Suspicious / leaky | Duplicate clips (below): unmerged, the same footage would sit in train and validation |

## Cleaning steps (`src/data/labels.py`)

- **Names.** 40 of 42 file names start with Cyrillic `Х` (U+0425); `X-1` and `X-2` use Latin `X`. `video_id` unifies both to Latin
  `X-<n>`. Column `file` keeps the name exactly as distributed, so the existence check is literal.
- **Timecodes.** Text like `22 секунда` becomes `22`; `1минута 8 секунда` (no space in the sheet) becomes `68`. Unknown text raises.
- **Verdicts.** `нарушение` → `violation`, `без нарушения` → `no_violation` (the API's words).

## Merged duplicates

The organizers' sheet lists the same footage under two names. Evidence: the files are byte-identical
(`sha256sum` on `data/raw/videos`).

| Kept as | Dropped alias | File size, bytes | sha256 (first 12) | Labels after merge |
|---|---|---|---|---|
| `X-4` (`Х-4.mov`) | `X-47` (`Х-47.mov`) | 14 707 558 | `5385be9420e6` | two violations: **57 s** (sheet row Х-4) and **68 s** (sheet row Х-47, "1минута 8 секунда") |
| `X-5` (`Х-5.mov`) | `X-48` (`Х-48.mov`) | 14 448 055 | `1f5840bff58f` | `no_violation` (both sheet rows agree, collapsed into one) |

The lower number is canonical. 42 sheet rows − 2 merged duplicates = 40 videos. The alias files stay on disk but
never appear in `labels.csv`. If a merged pair ever disagreed on the verdict, `build_labels` raises instead of guessing.

## Regenerate and check

```bash
uv run python -m src.data.labels        # open_labels.xlsx + violation_types.csv + arrangements.csv -> data/labels.csv
uv run python -m src.data.check_labels  # counts, videos per arrangement and verdict, violations per type;
                                        # fails on a contract breach or a missing clip
```

## Violation types (`data/violation_types.csv`)

The organizers' sheet has no types, so the team sets one per violation label after watching the video (ADR 0005).
One row per label, keyed by `(video_id, second)`, value is a type `id` from `src/rules/rules.yaml`. The file is
separate from `labels.csv` so a rebuild from the xlsx keeps it. The rebuild raises if a violation has no type, if a
type row matches no violation label, or if one label has two types. The labels contract rejects any value that is
not an `id` in `rules.yaml`.

## Arrangements (`data/arrangements.csv`)

All training videos come from one camera (FAQ 3); what changes between them is the arrangement (GLOSSARY): how the desk and
safe stand and which way the cashier sits. The organizers' sheet has no arrangement, so the team sets one per video after
watching it. One row per video, keyed by the canonical `video_id` (no `X-47`, `X-48`), 40 rows. The file is separate from
`labels.csv` so a rebuild from the xlsx keeps it. The rebuild raises if a video has no arrangement, if a row matches no
video or if a video has two. The labels contract accepts only the two values below and requires one value per video.

| Value | Meaning | Videos |
|---|---|---|
| `back` | the cashier sits with the back to the camera | 14 |
| `facing` | the cashier sits facing the camera | 26 |

Both arrangements have videos with both verdicts, so recall and FP rate can be computed on each (#38):

| Arrangement | Videos | With violation | Clean |
|---|---|---|---|
| `back` | 14 | 7 | 7 |
| `facing` | 26 | 14 | 12 |

The counts are printed by `uv run python -m src.data.check_labels`. TODO (after the freeze, separate ticket): where the desk
and the safe stand in each arrangement, so that any video can be assigned from the description alone.
