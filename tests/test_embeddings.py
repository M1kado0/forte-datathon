"""Padding, window pooling and the embedding cache; no model is loaded."""

import numpy as np
import pandas as pd
import pytest

from src.features.embeddings import INPUT_SIZE, load_or_embed, pad_to_square, preprocess, window_features


def test_pad_to_square_keeps_content_centred():
    frame = np.full((4, 10, 3), 7, dtype=np.uint8)
    out = pad_to_square(frame)
    assert out.shape == (10, 10, 3)
    assert (out[3:7] == 7).all() and (out[:3] == 0).all() and (out[7:] == 0).all()


def test_preprocess_shape():
    assert preprocess(np.zeros((30, 50, 3), dtype=np.uint8)).shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_window_features_mean_and_empty_window():
    seconds = np.array([0.0, 0.5, 1.0, 1.5])
    emb = np.array([[0.0], [2.0], [4.0], [6.0]])
    windows = pd.DataFrame({"start": [0.0, 10.0], "end": [2.0, 12.0]})
    feats, kept = window_features(seconds, emb, windows)
    assert kept.tolist() == [True, False]
    assert feats[0, 0] == pytest.approx(3.0)


def test_cache_hit_does_not_build_the_model(tmp_path):
    np.savez(tmp_path / "v.npz", seconds=np.array([0.0]), embeddings=np.ones((1, 3)), elapsed=1.5)

    def boom():
        raise AssertionError("model built on cache hit")

    s, e, t = load_or_embed("v", tmp_path / "missing.mov", embedder_factory=boom, cache_dir=tmp_path)
    assert e.shape == (1, 3) and t == 1.5
