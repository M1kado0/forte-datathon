"""`python -m src.data.check_labels`: validate data/labels.csv and that every named clip exists (TZ-8.5)."""

import argparse
from pathlib import Path

import pandera.errors

from src.config import LABELS_CSV, VIDEOS_DIR
from src.data.labels import load_labels


def check_labels(labels_csv: Path = LABELS_CSV, videos_dir: Path = VIDEOS_DIR) -> dict[str, int]:
    """Raises SchemaError on a contract breach and FileNotFoundError on a missing clip."""
    labels = load_labels(labels_csv)
    missing = sorted(name for name in labels["file"].unique() if not (videos_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clip(s) from labels.csv not in {videos_dir}: {', '.join(missing)}"
        )
    violations = labels[labels["verdict"] == "violation"]
    videos = labels["video_id"].nunique()
    return {
        "videos": videos,
        "violation_videos": violations["video_id"].nunique(),
        "violation_labels": len(violations),
        "clean_videos": videos - violations["video_id"].nunique(),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_csv", nargs="?", type=Path, default=LABELS_CSV)
    ap.add_argument("videos_dir", nargs="?", type=Path, default=VIDEOS_DIR)
    args = ap.parse_args(argv)
    try:
        c = check_labels(args.labels_csv, args.videos_dir)
    except (FileNotFoundError, pandera.errors.SchemaError) as e:
        raise SystemExit(f"FAIL: {e}") from e
    print(
        f"{c['videos']} видео: {c['violation_videos']} с нарушениями ({c['violation_labels']} меток), "
        f"{c['clean_videos']} без нарушений"
    )


if __name__ == "__main__":
    main()
