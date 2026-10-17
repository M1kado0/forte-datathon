# Data card (P4)

Labels for the ForteBank cashier videos. Source: `data/raw/open_labels.xlsx` (organizers) and `data/raw/videos/*.mov`
(42 files, not in git, see `docs/SETUP.md`). Cleaned labels: `data/labels.csv`, contract: `contracts/data.py`.

| Item | Value |
|---|---|
| Files | `open_labels.xlsx` (42 rows × 3 columns: file name, timestamp text, verdict text), 42 `.mov` clips |
| Cleaned output | `data/labels.csv`: 41 rows, 40 unique videos |
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
uv run python -m src.data.labels        # open_labels.xlsx -> data/labels.csv
uv run python -m src.data.check_labels  # counts; fails on a contract breach or a missing clip
```
