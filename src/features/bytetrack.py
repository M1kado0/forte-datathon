"""ByteTrack (Zhang et al., MIT): boxes in, stable track ids out.

Written here because the usual packages are AGPL (ultralytics, boxmot) or pull in a stack of dependencies for
80 lines of logic. Two passes per frame: confident boxes are matched to tracks first, then boxes of low
confidence pick up the tracks that are left, so a half-hidden person is not lost. A box of low confidence
never starts a track.
"""

from collections.abc import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (n, 4) with every box in b (m, 4); boxes are x1, y1, x2, y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(tracks: np.ndarray, dets: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """Best one-to-one pairing of track boxes and detection boxes, keeping pairs with IoU >= min_iou."""
    iou = iou_matrix(tracks, dets)
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]


class _Track:
    """Box with a constant-velocity guess of where it moves next (centre, size and their speeds)."""

    def __init__(self, track_id: int, box: np.ndarray) -> None:
        self.id = track_id
        self.state = self._to_state(box)
        self.seen = self.state  # where the last match put it
        self.velocity = np.zeros(4)
        self.lost = 0  # frames missed since the last match

    @staticmethod
    def _to_state(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.state
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> None:
        self.state = self.state + self.velocity
        self.state[2:] = np.maximum(self.state[2:], 1.0)

    def update(self, box: np.ndarray) -> None:
        new = self._to_state(box)
        per_frame = (new - self.seen) / (self.lost + 1)  # a match after a gap spans the missed frames too
        self.velocity = 0.5 * self.velocity + 0.5 * per_frame
        self.state = self.seen = new
        self.lost = 0


class ByteTracker:
    """One tracker per kind of object. `ids` hands out track ids, so several trackers can share one counter and
    keep ids unique across kinds."""

    def __init__(
        self,
        ids: Iterator[int],
        high: float = 0.5,
        low: float = 0.1,
        min_iou: float = 0.2,
        min_iou_low: float = 0.4,
        max_lost: int = 8,
    ) -> None:
        self.ids = ids
        self.high, self.low = high, low
        self.min_iou, self.min_iou_low = min_iou, min_iou_low
        self.max_lost = max_lost
        self.tracks: list[_Track] = []

    def update(self, boxes: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Track id of every input box (n, 4); -1 for a box of low confidence that matched no track."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        conf = np.asarray(conf, dtype=float)
        ids = np.full(len(boxes), -1, dtype=int)
        for track in self.tracks:
            track.predict()
        strong = np.flatnonzero(conf >= self.high)
        weak = np.flatnonzero((conf >= self.low) & (conf < self.high))
        free = list(range(len(self.tracks)))
        unmatched_strong = list(strong)
        for pool, min_iou, is_strong in ((strong, self.min_iou, True), (weak, self.min_iou_low, False)):
            # a lost track needs a confident box to come back; a weak one may only continue an active track
            candidates = [i for i in free if is_strong or self.tracks[i].lost == 0]
            if not len(pool) or not candidates:
                continue
            pairs = match(np.array([self.tracks[i].box for i in candidates]), boxes[pool], min_iou)
            for r, c in pairs:
                track = self.tracks[candidates[r]]
                track.update(boxes[pool[c]])
                ids[pool[c]] = track.id
            taken_tracks = {candidates[r] for r, _ in pairs}
            free = [i for i in free if i not in taken_tracks]
            if is_strong:
                taken = {int(pool[c]) for _, c in pairs}
                unmatched_strong = [i for i in strong if int(i) not in taken]
        for i in unmatched_strong:
            track = _Track(next(self.ids), boxes[i])
            self.tracks.append(track)
            ids[i] = track.id
        for i in free:
            self.tracks[i].lost += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return ids
