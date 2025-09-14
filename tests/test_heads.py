"""Tests — shallow regression head."""
import numpy as np

from uda.config import HeadConfig
from uda.models.heads import build_head


def test_output_units_match_n_outputs():
    assert build_head(512, HeadConfig(), 1).output_shape[-1] == 1
    assert build_head(512, HeadConfig(), 2).output_shape[-1] == 2


def test_forward_pass_shape_and_finite():
    head = build_head(64, HeadConfig(hidden_units=[32], dropout=0.0), 1)
    x = np.random.rand(4, 64).astype("float32")
    y = np.asarray(head(x, training=False))
    assert y.shape == (4, 1)
    assert np.isfinite(y).all()


def test_depth_follows_hidden_units():
    shallow = build_head(16, HeadConfig(hidden_units=[8]), 1)
    deep = build_head(16, HeadConfig(hidden_units=[8, 8, 8]), 1)
    assert len(deep.layers) > len(shallow.layers)
