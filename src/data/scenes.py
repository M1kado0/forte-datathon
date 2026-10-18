"""Scene groups: the takes that share people, clothes and objects (ADR 0006). Writes data/scenes.csv.

    uv run python -m src.data.scenes

The organizers filmed each violation with a clean take of the same scene, so CV must hold the whole scene out,
not one video. Numbers only, from tiny downsampled frames (no frame leaves the machine): each video's median frame
is compared with the others after the shared background of its pixel cluster is removed; takes whose residuals
correlate at TAU or more, in the same cluster, are joined (connected components). TAU = 0.5 merges more than the
0.7 at which the groups first appeared (12 groups): a twin pair split across folds would leak the scene again.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist

from src.config import SCENES_CSV, VIDEOS_DIR
from src.data.labels import load_labels

TAU = 0.5
CLUSTER_SIZE, CLUSTER_STEP = (29, 53), 0.5  # (w, h) = native 464x848 / 16, a frame every 0.5 s
SCENE_SIZE, SCENE_STEP = (58, 106), 1.0  # native / 8, a frame every 1 s


def median_frames(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of the small frames for the background cluster and for the scene residual."""
    cap = cv2.VideoCapture(str(path))
    small, mid, next_small, next_mid = [], [], 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t + 1e-6 >= next_small:
            small.append(
                cv2.resize(frame, CLUSTER_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            )
            next_small += CLUSTER_STEP
        if t >= next_mid:
            mid.append(cv2.resize(frame, SCENE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32))
            next_mid += SCENE_STEP
    cap.release()
    return np.median(np.stack(small), axis=0), np.median(np.stack(mid), axis=0)


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def scene_groups(small: np.ndarray, mid: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Rows = videos (flattened median frames). Group index per video."""
    cluster = fcluster(linkage(pdist(_standardize(small), "correlation"), "average"), 2, "maxclust")
    residual = _standardize(mid)
    for c in np.unique(cluster):  # remove the background every take of the cluster shares
        residual[cluster == c] -= residual[cluster == c].mean(0)
    residual /= np.linalg.norm(residual, axis=1, keepdims=True)
    joined = (residual @ residual.T >= tau) & (cluster[:, None] == cluster[None, :])
    np.fill_diagonal(joined, False)
    return connected_components(joined, directed=False)[1]


def build() -> pd.DataFrame:
    files = load_labels().drop_duplicates("video_id").set_index("video_id")["file"]
    ids = sorted(files.index, key=lambda v: int(v.split("-")[1]))
    medians = [median_frames(VIDEOS_DIR / files[v]) for v in ids]
    small = np.stack([m[0].ravel() for m in medians])
    mid = np.stack([m[1].ravel() for m in medians])
    groups = scene_groups(small, mid)
    # name each scene after its lowest-numbered take, so the names do not depend on the component order
    first = {g: ids[int(np.flatnonzero(groups == g)[0])] for g in np.unique(groups)}
    return pd.DataFrame({"video_id": ids, "scene": [f"scene-{first[g]}" for g in groups]})


if __name__ == "__main__":
    df = build()
    df.to_csv(SCENES_CSV, index=False)
    sizes = df["scene"].value_counts()
    print(f"{len(sizes)} scenes, sizes {sorted(sizes.tolist(), reverse=True)} -> {SCENES_CSV.name}")
