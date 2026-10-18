"""Track building (#32): the tracker keeps ids, build_tracks writes a schema-valid cache. No model is run."""

import itertools

import cv2
import numpy as np
import pytest

from contracts.data import TRACKS_SCHEMA
from src.features import detectors, tracks
from src.features.bytetrack import ByteTracker
from src.features.detectors import Box


def boxes(*centres, size=20.0):
    return np.array([[x - size / 2, y - size / 2, x + size / 2, y + size / 2] for x, y in centres])


def tracker():
    return ByteTracker(itertools.count())


def test_two_people_keep_their_ids_while_they_move():
    t = tracker()
    first = t.update(boxes((50, 50), (300, 50)), np.array([0.9, 0.9]))
    for step in range(1, 6):
        ids = t.update(boxes((50 + 5 * step, 50), (300 - 5 * step, 50)), np.array([0.9, 0.9]))
        assert list(ids) == list(first)
    assert first[0] != first[1]


def test_a_person_missed_for_two_frames_keeps_the_id():
    t = tracker()
    (first,) = t.update(boxes((100, 100)), np.array([0.9]))
    t.update(boxes((105, 100)), np.array([0.9]))
    t.update(np.zeros((0, 4)), np.zeros(0))
    t.update(np.zeros((0, 4)), np.zeros(0))
    (back,) = t.update(boxes((120, 100)), np.array([0.9]))
    assert back == first


def test_a_track_is_dropped_after_too_long_a_miss():
    t = ByteTracker(itertools.count(), max_lost=2)
    (first,) = t.update(boxes((100, 100)), np.array([0.9]))
    for _ in range(4):
        t.update(np.zeros((0, 4)), np.zeros(0))
    (again,) = t.update(boxes((100, 100)), np.array([0.9]))
    assert again != first


def test_a_weak_box_continues_a_track_but_never_starts_one():
    t = tracker()
    (weak_alone,) = t.update(boxes((200, 200)), np.array([0.3]))
    assert weak_alone == -1
    (first,) = t.update(boxes((100, 100)), np.array([0.9]))
    (weak,) = t.update(boxes((102, 100)), np.array([0.3]))
    assert weak == first


def test_trackers_sharing_a_counter_never_repeat_an_id():
    counter = itertools.count()
    people, things = ByteTracker(counter), ByteTracker(counter)
    a = people.update(boxes((50, 50)), np.array([0.9]))
    b = things.update(boxes((50, 50)), np.array([0.9]))
    assert a[0] != b[0]


# --- build_tracks / get_tracks with a fake detector on a tiny synthetic video -------------------------------


@pytest.fixture
def clip(tmp_path):
    """Two seconds, 8 fps, 64x48 frames: enough for the decoder to hand over 8 frames at 4 fps."""
    path = tmp_path / "Х-7.mov"  # Cyrillic Х, as in the distributed files
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (64, 48))
    for i in range(16):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


class FakeDetector:
    """A person walking right, a cup that stays, cash and a zero-area document on the frames OWLv2 looks at."""

    def __init__(self):
        self.frames = 0
        self.calls = 0

    def coco(self, frames):
        self.calls += 1
        out = []
        for _ in frames:
            x = 5 + 3 * self.frames
            self.frames += 1
            out.append(
                [Box("person", "person", x, 5, x + 10, 40, 0.9), Box("object", "cup", 30, 30, 36, 38, 0.8)]
            )
        return out

    def open_vocab(self, frames):
        return [
            [Box("cash", "cash", 20, 20, 30, 28, 0.3), Box("document", "document", 40, 5, 41, 5, 0.3)]
            for _ in frames
        ]


def test_build_tracks_gives_schema_valid_tracks_of_every_kind(clip):
    table = tracks.build_tracks(clip, FakeDetector())
    TRACKS_SCHEMA.validate(table)
    assert set(table["kind"]) == {"person", "object", "cash"}  # the document box has no area and is dropped
    assert table.groupby("kind")["track_id"].nunique().to_dict() == {"person": 1, "object": 1, "cash": 1}
    assert table["track_id"].nunique() == 3  # ids are unique across kinds


def test_open_vocabulary_kinds_are_seen_on_every_other_frame_only(clip):
    table = tracks.build_tracks(clip, FakeDetector())
    seen = table[table["kind"] == "person"]["t"].nunique()
    assert table[table["kind"] == "cash"]["t"].nunique() == -(
        -seen // tracks.OWL_EVERY
    )  # frames 0, 2, 4, ...


def test_the_document_prompt_is_always_on_and_extra_queries_are_objects_named_by_their_text(monkeypatch):
    assert ("document", "document") in detectors.owl_groups()
    assert [k for k in detectors.owl_groups() if k[0] == "object"] == []  # nothing passed #70 yet
    monkeypatch.setattr(detectors, "EXTRA_QUERIES", ("open door",))
    assert detectors.owl_groups()[("object", "open door")] == ("open door",)
    assert ("document", "document") in detectors.owl_groups()


def test_get_tracks_caches_by_video_and_reads_it_back(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(tracks, "TRACKS_DIR", tmp_path / "cache")
    fake = FakeDetector()
    built = tracks.get_tracks(clip, fake)
    calls = fake.calls
    again = tracks.get_tracks(clip, fake)  # a hit must not run the detector
    assert fake.calls == calls
    assert again.equals(built)
    assert (tmp_path / "cache" / tracks.setting_id() / "X-7.parquet").is_file()  # the id is the Latin X


def test_changing_the_queries_never_reads_an_old_cache(monkeypatch):
    before = tracks.setting_id()
    monkeypatch.setattr(detectors, "EXTRA_QUERIES", ("open door",))
    assert tracks.setting_id() != before
