"""Frozen temporal action recognizers: dense proposals before any violation decision.

The labels here are activities, never legal verdicts. All windows are retained so
stationary violations are not silently removed by a motion/confidence gate.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import DATA_PROCESSED, WEIGHTS
from src.data.video import duration, iter_frames

WINDOW_SEC = 2.0
STEP_SEC = 1.0
FPS = 8.0
BATCH = 4
BACKBONES = ("r3d-18", "xclip-base-patch32")
ACTIVITIES = (
    "a person sitting and working at a desk",
    "a person counting banknotes with their hands",
    "a person handing banknotes to another person",
    "a person receiving banknotes from another person",
    "a person placing banknotes on a desk",
    "a person putting banknotes into a safe",
    "a person inspecting an identity document",
    "a person handing over a bag",
    "a person opening a door",
    "a person closing a door",
    "a person opening a safe",
    "a person closing a safe",
    "a person leaving a desk and walking away",
    "a person entering a room",
    "two people standing beside a desk",
    "a person using a laptop computer",
    "a person using a mobile phone",
    "a person placing an object on a desk",
    "a person handling keys",
    "an empty room with a desk and a safe",
)


def fingerprint(name: str) -> str:
    if name not in BACKBONES:
        raise ValueError(f"Unknown action backbone: {name}")
    return hashlib.sha256(
        json.dumps(
            {
                "model": name,
                "window": WINDOW_SEC,
                "step": STEP_SEC,
                "fps": FPS,
                "transform": "rgb-letterbox-pil-bicubic-v1-fp32",
                "activities": ACTIVITIES,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


@dataclass
class Proposals:
    starts: np.ndarray
    ends: np.ndarray
    features: np.ndarray
    action_ids: np.ndarray
    action_confidence: np.ndarray
    action_names: tuple[str, ...]
    elapsed: float
    duration: float


def window_starts(length: float) -> np.ndarray:
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Video duration must be finite and positive")
    last = max(0.0, length - WINDOW_SEC)
    starts = np.arange(0.0, last + 1e-6, STEP_SEC)
    if last - starts[-1] > 1e-6:
        starts = np.append(starts, last)
    return starts


def sample_indices(seconds: np.ndarray, start: float, end: float, count: int) -> np.ndarray:
    """Nearest available frame at/before each target; repeated padding only at clip ends."""
    targets = np.linspace(start, end, count, endpoint=False)
    return np.clip(np.searchsorted(seconds, targets, side="right") - 1, 0, len(seconds) - 1)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Whole native RGB frame letterboxed to square; no field-of-view crop."""
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scale = size / max(rgb.size)
    resized = rgb.resize(tuple(max(1, round(v * scale)) for v in rgb.size), Image.Resampling.BICUBIC)
    square = Image.new("RGB", (size, size))
    square.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(square)


class ActionEncoder:
    def __init__(self, name: str):
        import torch

        self.name = name
        self.fingerprint = fingerprint(name)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "r3d-18":
            from torchvision.models.video import r3d_18

            self.model = r3d_18(weights=None)
            self.model.load_state_dict(
                torch.load(WEIGHTS / name / "r3d_18-b3b3357e.pth", map_location="cpu", weights_only=True)
            )
            self.classifier = self.model.fc.to(self.device).eval()
            self.model.fc = torch.nn.Identity()
            self.names = tuple(json.loads((WEIGHTS / name / "categories.json").read_text(encoding="utf-8")))
            self.size, self.frames = 112, 16
            mean, std = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
        else:
            from transformers import AutoTokenizer, XCLIPModel

            self.model = XCLIPModel.from_pretrained(WEIGHTS / name, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(WEIGHTS / name, local_files_only=True)
            self.tokens = tokenizer(list(ACTIVITIES), padding=True, return_tensors="pt").to(self.device)
            self.names = ACTIVITIES
            self.size, self.frames = 224, 8
            mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor(mean, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 1, 3, 1, 1)

    def __call__(self, clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        x = torch.from_numpy(clips).to(self.device).permute(0, 1, 4, 2, 3).float() / 255
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            if self.name == "r3d-18":
                features = self.model(x.permute(0, 2, 1, 3, 4))
                logits = self.classifier(features)
            else:
                out = self.model(pixel_values=x, **self.tokens)
                features, logits = out.video_embeds, out.logits_per_video
            probabilities = logits.softmax(-1)
        return features.float().cpu().numpy(), probabilities.float().cpu().numpy()


def extract(path: Path, encoder: ActionEncoder) -> Proposals:
    start_time = time.perf_counter()
    length = duration(path)
    seconds, frames = [], []
    for second, frame in iter_frames(path, FPS):
        seconds.append(second)
        frames.append(resize_frame(frame, encoder.size))
    if not frames:
        raise ValueError(f"No decoded frames: {path.name}")
    seconds, frames = np.asarray(seconds), np.stack(frames)
    starts = window_starts(length)
    ends = np.minimum(starts + WINDOW_SEC, length)
    feats, probs = [], []
    for first in range(0, len(starts), BATCH):
        clips = np.stack(
            [
                frames[sample_indices(seconds, a, b, encoder.frames)]
                for a, b in zip(starts[first : first + BATCH], ends[first : first + BATCH], strict=True)
            ]
        )
        f, p = encoder(clips)
        feats.append(f)
        probs.append(p)
    features, probabilities = np.concatenate(feats), np.concatenate(probs)
    return Proposals(
        starts,
        ends,
        np.concatenate([features, probabilities], axis=1),
        probabilities.argmax(1),
        probabilities.max(1),
        encoder.names,
        time.perf_counter() - start_time,
        length,
    )


def cached(path: Path, dataset: str, encoder: ActionEncoder) -> Proposals:
    """Dataset-only cache; source content identity and preprocessing invalidate it."""
    source = hashlib.sha256(path.read_bytes()).hexdigest()
    file = DATA_PROCESSED / "actions" / encoder.fingerprint / dataset / f"{path.stem}.npz"
    if file.exists():
        with np.load(file, allow_pickle=False) as z:
            if str(z["source"]) == source:
                return Proposals(
                    z["starts"],
                    z["ends"],
                    z["features"],
                    z["action_ids"],
                    z["action_confidence"],
                    tuple(z["action_names"].tolist()),
                    float(z["elapsed"]),
                    float(z["duration"]),
                )
    result = extract(path, encoder)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, source=source, **vars(result))
    return result
