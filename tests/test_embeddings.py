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
