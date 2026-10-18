"""Offline SigLIP frame embeddings; FORTE_EMBEDDER=dinov2-small reproduces the baseline.

SigLIP uses its pretrained 224-square image transform (whole frame, no crop). DINOv2
keeps native pixels with square padding. Caches are isolated by model and fingerprint.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATA_PROCESSED, VIDEOS_DIR, WEIGHTS
from src.data.video import iter_frames

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # transformers is imported lazily, after this line

EMBED_FPS = 4.0
BATCH = 8
PATCH = 14  # DINOv2 baseline patch size
MODEL_NAME = os.environ.get("FORTE_EMBEDDER", "siglip-base")
if MODEL_NAME not in {"dinov2-small", "siglip-base"}:
    raise ValueError(f"Unknown FORTE_EMBEDDER: {MODEL_NAME!r}")
EMBED_DIM = 768  # SigLIP pooler; DINOv2 baseline is CLS (384) ++ mean patch (384)
FINGERPRINT = (
    f"dinov2-small|cls+meanpatch-on-frame|fps={EMBED_FPS}|pad-to-multiple-of-{PATCH}"
    if MODEL_NAME == "dinov2-small"
    else f"siglip-base|pooler|fps={EMBED_FPS}|rgb-bicubic-224|mean=.5|std=.5|fp32"
)
CACHE_DIR = DATA_PROCESSED / "embeddings" / MODEL_NAME
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_to_square(frame: np.ndarray, multiple: int = 1) -> np.ndarray:
    """Pad with black to a square whose side is a multiple of `multiple`, content centred; no crop, no resize."""
    h, w = frame.shape[:2]
    side = -(-max(h, w) // multiple) * multiple
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(frame, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)


def frame_patch_mask(h: int, w: int, multiple: int = PATCH) -> np.ndarray:
    """Flat bool mask over the padded square's patch grid: True where a patch overlaps the real frame."""
    side = -(-max(h, w) // multiple) * multiple
    top, left = (side - h) // 2, (side - w) // 2
    grid = side // multiple
    rows = np.arange(grid)
    in_rows = (rows * multiple < top + h) & ((rows + 1) * multiple > top)
    in_cols = (rows * multiple < left + w) & ((rows + 1) * multiple > left)
    return (in_rows[:, None] & in_cols[None, :]).ravel()


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR native frame -> normalized CHW float32, padded to a patch-multiple square."""
    square = pad_to_square(frame_bgr, PATCH)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def preprocess_siglip(frame_bgr: np.ndarray) -> np.ndarray:
    """Official SigLIP RGB/bicubic 224 transform; all source pixels participate, no crop."""
    rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    pixels = np.asarray(rgb.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32)
    return ((pixels / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def cache_path(video_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{video_id}.npz"


class Embedder:
    """Selected vision encoder from local weights, returning 768 features per frame."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, SiglipVisionModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cls = SiglipVisionModel if MODEL_NAME == "siglip-base" else AutoModel
        self.model = (
            model_cls.from_pretrained(WEIGHTS / MODEL_NAME, local_files_only=True).to(self.device).eval()
        )

    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        if not frames_bgr:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        transform = preprocess_siglip if MODEL_NAME == "siglip-base" else preprocess
        x = torch.from_numpy(np.stack([transform(f) for f in frames_bgr])).to(self.device)
        with torch.inference_mode():
            output = self.model(pixel_values=x)
        if MODEL_NAME == "siglip-base":
            return output.pooler_output.float().cpu().numpy()
        h = output.last_hidden_state
        # frames of one video share a size, so one mask serves the batch; black padding is left out of the mean
        mask = torch.from_numpy(frame_patch_mask(*frames_bgr[0].shape[:2])).to(self.device)
        return torch.cat([h[:, 0], h[:, 1:][:, mask].mean(1)], dim=1).float().cpu().numpy()


def embed_video(
    path: Path, embedder: Embedder, fps: float = EMBED_FPS
) -> tuple[np.ndarray, np.ndarray, float]:
    """(seconds, embeddings, wall seconds spent) for one video, decoding included."""
    start = time.perf_counter()
    seconds: list[float] = []
    chunks: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for second, frame in iter_frames(path, fps):
        seconds.append(second)
        batch.append(frame)
        if len(batch) == BATCH:
            chunks.append(embedder(batch))
            batch = []
    if batch:
        chunks.append(embedder(batch))
    emb = np.concatenate(chunks) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.array(seconds), emb, time.perf_counter() - start


def load_or_embed(
    video_id: str, path: Path, embedder_factory=Embedder, cache_dir: Path = CACHE_DIR
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cached (seconds, embeddings, embed time). The model is only built on a miss or when the cache is stale."""
    file = cache_path(video_id, cache_dir)
    if file.exists():
        z = np.load(file)
        if "fingerprint" in z.files and str(z["fingerprint"]) == FINGERPRINT:
            return z["seconds"], z["embeddings"], float(z["elapsed"])
    seconds, emb, elapsed = embed_video(path, embedder_factory())
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(file, seconds=seconds, embeddings=emb, elapsed=elapsed, fingerprint=FINGERPRINT)
    return seconds, emb, elapsed


def window_features(
    seconds: np.ndarray, emb: np.ndarray, windows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding of frames in [start, end) per window. Returns (features, kept) where kept marks windows with frames."""
    feats = np.zeros((len(windows), emb.shape[1]), dtype=np.float32)
    kept = np.zeros(len(windows), dtype=bool)
    for i, (a, b) in enumerate(zip(windows["start"], windows["end"], strict=True)):
        m = (seconds >= a) & (seconds < b)
        if m.any():
            feats[i] = emb[m].mean(0)
            kept[i] = True
    return feats, kept


def video_path(file: str) -> Path:
    return VIDEOS_DIR / file
