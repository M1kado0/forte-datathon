"""D-FINE-S (people and COCO objects) and OWLv2 (cash, documents, optional queries) on batches of BGR frames.

Everything runs locally from models/weights. Frames are resized on the GPU by hand: the Hugging Face image
processors do it on the CPU, one frame at a time, and that alone would eat the time budget (#32).
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.config import WEIGHTS

# COCO classes kept besides person (ADR 0005): laptop X-28, cup/cell phone/bottle X-15, X-40, X-46, book X-24.
OBJECT_CLASSES = ("laptop", "cup", "cell phone", "bottle", "book")
COCO_MIN_CONF = 0.2  # D-FINE boxes below this are never emitted; the heuristics set their own floors above it

# OWLv2 text queries by track kind. The prompt for a document is always on: type 7 reads silence as "nobody
# showed one" (#54), so a video without document rows must mean the prompt ran and found nothing.
CASH_QUERIES = ("a photo of banknotes", "a stack of cash")  # as in the #19 check
DOCUMENT_QUERIES = ("an identity document", "a passport")
# Queries added only when they passed #70 (ADR 0005: keys on the safe, open safe, open door). Each becomes
# kind "object" with the query text as its label. #70 ran (src.features.adr5_check): no query passed, so none is on.
EXTRA_QUERIES: tuple[str, ...] = ()
OWL_MIN_CONF = 0.1  # OWLv2 scores are low by design; the heuristics' cash_min and doc_min are the same
OWL_NMS_IOU = 0.5
OWL_TOP = 10  # boxes kept per group and frame


@dataclass(frozen=True)
class Box:
    kind: str  # "person", "object", "cash" or "document", as in TRACKS_SCHEMA
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


class Detector(Protocol):
    def coco(self, frames: Sequence[np.ndarray]) -> list[list[Box]]: ...

    def open_vocab(self, frames: Sequence[np.ndarray]) -> list[list[Box]]: ...


def owl_groups(extra: Sequence[str] | None = None) -> dict[tuple[str, str], tuple[str, ...]]:
    """(kind, label) -> the queries whose best score decides the box. `extra` replaces EXTRA_QUERIES (checks)."""
    groups: dict[tuple[str, str], tuple[str, ...]] = {
        ("cash", "cash"): CASH_QUERIES,
        ("document", "document"): DOCUMENT_QUERIES,
    }
    groups.update({("object", q): (q,) for q in (EXTRA_QUERIES if extra is None else extra)})
    return groups


class LocalDetector:
    """Loads both models once. `coco` and `open_vocab` take a batch of BGR frames of one size."""

    def __init__(self, extra_queries: Sequence[str] | None = None) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before transformers is imported
        import torch
        from transformers import AutoModelForObjectDetection, AutoTokenizer, Owlv2ForObjectDetection

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dfine = (
            AutoModelForObjectDetection.from_pretrained(WEIGHTS / "dfine-small")
            .to(self.device, self.dtype)
            .eval()
        )
        id2label = self.dfine.config.id2label
        wanted = ("person", *OBJECT_CLASSES)
        self.coco_ids = {i: n for i, n in id2label.items() if n in wanted}
        self.coco_id_list = torch.tensor(sorted(self.coco_ids), device=self.device)
        self.owl = (
            Owlv2ForObjectDetection.from_pretrained(WEIGHTS / "owlv2-base").to(self.device, self.dtype).eval()
        )
        self.groups = owl_groups(extra_queries)
        queries = [q for qs in self.groups.values() for q in qs]
        tokens = AutoTokenizer.from_pretrained(WEIGHTS / "owlv2-base")(
            queries, padding="max_length", max_length=16, truncation=True, return_tensors="pt"
        )
        self.queries = queries
        self.query_tokens = tokens["input_ids"].to(self.device)
        self.query_mask = tokens["attention_mask"].to(self.device)
        self.owl_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.owl_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def _rgb(self, frames: Sequence[np.ndarray]):
        """Batch of BGR uint8 frames -> float RGB (n, 3, h, w) in [0, 1] on the device."""
        batch = self.torch.from_numpy(np.stack(frames)).to(self.device)
        return batch.flip(-1).permute(0, 3, 1, 2).float().div_(255)

    def _pixel_boxes(self, centre_wh, scale, w: int, h: int):
        """Normalised (cx, cy, w, h) -> pixel (x1, y1, x2, y2) inside the w x h frame."""
        centre_wh = centre_wh.float()
        xy, size = centre_wh[..., :2], centre_wh[..., 2:]
        xyxy = (self.torch.cat([xy - size / 2, xy + size / 2], -1) * scale).clamp_(min=0)
        xyxy[..., 0::2].clamp_(max=w)
        xyxy[..., 1::2].clamp_(max=h)
        return xyxy

    def coco(self, frames: Sequence[np.ndarray]) -> list[list[Box]]:
        torch = self.torch
        h, w = frames[0].shape[:2]
        x = torch.nn.functional.interpolate(
            self._rgb(frames), size=(640, 640), mode="bilinear", antialias=True, align_corners=False
        )
        with torch.no_grad():
            out = self.dfine(pixel_values=x.to(self.dtype))
        scores = out.logits.float().sigmoid()[:, :, self.coco_id_list]  # (n, queries, classes kept)
        xyxy = self._pixel_boxes(out.pred_boxes, torch.tensor([w, h, w, h], device=self.device), w, h)
        result: list[list[Box]] = []
        for b in range(len(frames)):
            q, c = (scores[b] >= COCO_MIN_CONF).nonzero(as_tuple=True)
            boxes, conf = xyxy[b, q].tolist(), scores[b, q, c].tolist()
            names = [self.coco_ids[int(self.coco_id_list[i])] for i in c.tolist()]
            result.append(
                [
                    Box("person" if n == "person" else "object", n, *box, s)
                    for n, box, s in zip(names, boxes, conf, strict=True)
                ]
            )
        return result

    def open_vocab(self, frames: Sequence[np.ndarray]) -> list[list[Box]]:
        from torchvision.ops import nms

        torch = self.torch
        n, (h, w) = len(frames), frames[0].shape[:2]
        side = max(h, w)
        x = self._rgb(frames)
        padded = torch.full(
            (n, 3, side, side), 0.5, device=self.device
        )  # OWLv2 pads bottom and right with grey
        padded[:, :, :h, :w] = x
        x = torch.nn.functional.interpolate(
            padded, size=(960, 960), mode="bilinear", antialias=True, align_corners=False
        )
        x = ((x - self.owl_mean) / self.owl_std).to(self.dtype)
        q = len(self.queries)
        with torch.no_grad():
            out = self.owl(
                input_ids=self.query_tokens.repeat(n, 1),
                attention_mask=self.query_mask.repeat(n, 1),
                pixel_values=x,
            )
        logits = out.logits.float().sigmoid().view(n, -1, q)  # (n, patches, queries)
        xyxy = self._pixel_boxes(out.pred_boxes, side, w, h)  # OWLv2 boxes are relative to the padded square
        result: list[list[Box]] = [[] for _ in range(n)]
        start = 0
        for (kind, label), queries in self.groups.items():
            best = logits[:, :, start : start + len(queries)].max(-1).values  # (n, patches)
            start += len(queries)
            for b in range(n):
                keep = (best[b] >= OWL_MIN_CONF).nonzero(as_tuple=True)[0]
                if not len(keep):
                    continue
                kept = keep[nms(xyxy[b, keep], best[b, keep], OWL_NMS_IOU)][:OWL_TOP]
                result[b] += [
                    Box(kind, label, *box, s)
                    for box, s in zip(xyxy[b, kept].tolist(), best[b, kept].tolist(), strict=True)
                ]
        return result
