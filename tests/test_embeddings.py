"""Padding, window pooling and the embedding cache; no model is loaded."""

import numpy as np
import pandas as pd
import pytest

from src.features.embeddings import (
    FINGERPRINT,
    PATCH,
    load_or_embed,
    pad_to_square,
    preprocess,
    window_features,
)


def test_pad_to_square_keeps_content_centred():
    frame = np.full((4, 10, 3), 7, dtype=np.uint8)
    out = pad_to_square(frame)
    assert out.shape == (10, 10, 3)
    assert (out[3:7] == 7).all() and (out[:3] == 0).all() and (out[7:] == 0).all()


def test_pad_to_patch_multiple_keeps_native_pixels():
    frame = np.full((20, 30, 3), 9, dtype=np.uint8)
    out = pad_to_square(frame, 14)
    assert out.shape == (42, 42, 3)
    assert (out[11:31, 6:36] == 9).all()


def test_preprocess_is_native_resolution():
    side = 848 // PATCH * PATCH + PATCH
    assert preprocess(np.zeros((848, 464, 3), dtype=np.uint8)).shape == (3, side, side)


def test_window_features_mean_and_empty_window():
    seconds = np.array([0.0, 0.5, 1.0, 1.5])
    emb = np.array([[0.0], [2.0], [4.0], [6.0]])
    windows = pd.DataFrame({"start": [0.0, 10.0], "end": [2.0, 12.0]})
    feats, kept = window_features(seconds, emb, windows)
    assert kept.tolist() == [True, False]
    assert feats[0, 0] == pytest.approx(3.0)


def test_cache_hit_does_not_build_the_model(tmp_path):
    np.savez(
        tmp_path / "v.npz",
        seconds=np.array([0.0]),
        embeddings=np.ones((1, 3)),
        elapsed=1.5,
        fingerprint=FINGERPRINT,
    )

    def boom():
        raise AssertionError("model built on cache hit")

    s, e, t = load_or_embed("v", tmp_path / "missing.mov", embedder_factory=boom, cache_dir=tmp_path)
    assert e.shape == (1, 3) and t == 1.5


def test_stale_cache_is_recomputed(tmp_path):
    np.savez(
        tmp_path / "v.npz",
        seconds=np.array([0.0]),
        embeddings=np.ones((1, 3)),
        elapsed=1.5,
        fingerprint="old",
    )

    def factory():
        raise RuntimeError("rebuilt")

    with pytest.raises(RuntimeError, match="rebuilt"):
        load_or_embed("v", tmp_path / "missing.mov", embedder_factory=factory, cache_dir=tmp_path)


def test_patch_mask_excludes_padding():
    from src.features.embeddings import frame_patch_mask

    mask = frame_patch_mask(28, 56, 14)  # 56x56 square, 28 px tall frame centred -> rows 1 and 2 of 4
    assert mask.reshape(4, 4).sum(axis=1).tolist() == [0, 4, 4, 0]


def test_cache_without_fingerprint_is_recomputed(tmp_path):
    np.savez(tmp_path / "v.npz", seconds=np.array([0.0]), embeddings=np.ones((1, 3)), elapsed=1.5)

    def factory():
        raise RuntimeError("rebuilt")

    with pytest.raises(RuntimeError, match="rebuilt"):
        load_or_embed("v", tmp_path / "missing.mov", embedder_factory=factory, cache_dir=tmp_path)


def _write_clip(path, w, h, n=4, fps=4):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        writer.write(np.full((h, w, 3), 40 * i, dtype=np.uint8))
    writer.release()
    return path


def _shapes_seen(path):
    """embed_video with a fake embedder that records the frame shapes and checks the patch mask matches them."""
    from src.features.embeddings import embed_video, frame_patch_mask

    seen = []

    def embedder(frames):
        assert len({f.shape for f in frames}) == 1
        x = np.stack([preprocess(f) for f in frames])
        assert frame_patch_mask(*frames[0].shape[:2]).size == (x.shape[-1] // PATCH) ** 2
        seen.extend(f.shape[:2] for f in frames)
        return np.zeros((len(frames), 768), dtype=np.float32)

    embed_video(path, embedder)
    return seen


def test_1080p_upload_is_capped_to_the_training_long_side(tmp_path):
    from src.features.embeddings import MAX_SIDE

    shapes = _shapes_seen(_write_clip(tmp_path / "landscape.mp4", 1920, 1080))
    assert shapes and all(max(s) == MAX_SIDE for s in shapes)
    assert shapes[0] == (477, 848)  # aspect kept
    # padded square has the training frame's token count, not ~5x
    assert preprocess(np.zeros((*shapes[0], 3), np.uint8)).shape[-1] == 854


def test_training_size_frames_are_not_resized(tmp_path):
    from src.features.embeddings import cap_long_side

    frame = np.random.default_rng(0).integers(0, 255, (848, 464, 3), dtype=np.uint8)
    assert cap_long_side(frame) is frame
    assert _shapes_seen(_write_clip(tmp_path / "portrait.mp4", 464, 848)) == [(848, 464)] * 4


def test_fingerprint_is_unchanged_by_the_cap():
    # the saved classifier is refused on a fingerprint mismatch, so the cap must not touch it
    assert FINGERPRINT == "dinov2-small|cls+meanpatch-on-frame|fps=4.0|pad-to-multiple-of-14"
