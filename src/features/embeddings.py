"""Branch A: DINOv2-small frame embeddings at native resolution, cached per video.

Frames are padded to a square (no center crop, so nothing at the edge of the counter is cut) and then
resized to the model input. The cache holds the frame seconds and embeddings, so a rerun costs nothing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, VIDEOS_DIR, WEIGHTS
from src.data.video import iter_frames

EMBED_FPS = 4.0
BATCH = 32
INPUT_SIZE = 224
CACHE_DIR = DATA_PROCESSED / "embeddings"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_to_square(frame: np.ndarray) -> np.ndarray:
    """Pad the short side with black so the frame becomes square, content centred; no crop, no stretch."""
    h, w = frame.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(frame, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR native frame -> normalized CHW float32 at INPUT_SIZE."""
    square = cv2.resize(pad_to_square(frame_bgr), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def cache_path(video_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{video_id}.npz"


class Embedder:
    """DINOv2-small from models/weights, offline. Embedding = CLS token ++ mean of patch tokens (768-d)."""

    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before transformers is imported
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(WEIGHTS / "dinov2-small").to(self.device).eval()

    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.stack([preprocess(f) for f in frames_bgr])).to(self.device)
        with torch.inference_mode():
            h = self.model(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float().cpu().numpy()


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
    emb = np.concatenate(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)
    return np.array(seconds), emb, time.perf_counter() - start


def load_or_embed(
    video_id: str, path: Path, embedder_factory=Embedder, cache_dir: Path = CACHE_DIR
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cached (seconds, embeddings, embed time). The model is only built on a cache miss."""
    file = cache_path(video_id, cache_dir)
    if file.exists():
        z = np.load(file)
        return z["seconds"], z["embeddings"], float(z["elapsed"])
    seconds, emb, elapsed = embed_video(path, embedder_factory())
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(file, seconds=seconds, embeddings=emb, elapsed=elapsed)
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
