"""Tests for EfficientNet B0-B3 + ``cnn_scratch`` backbones.

Covers the binding contract for the extra backbones — the registry entries and
the ``NATIVE_INPUT_SIZES`` mirror. Tests assert
*behavior*: build success, output rank/shape, frozen-vs-trainable weight
contracts, the cross-module native-size consistency guard, preprocessing
domains, and the end-to-end forward pass.

All ImageNet backbones are built with ``weights=None`` so unit tests never
download weights (kept fast and offline so CI stays quick).
"""
from __future__ import annotations

import numpy as np
import pytest

from uda.models.backbones import (
    BACKBONES,
    build_backbone,
    native_input_size,
    preprocess_for,
)
from uda.config import BackboneConfig, ExperimentConfig
from uda.data import dataset as _dataset
from uda.models.model import build_model

#: The five backbones this feature adds (the four EfficientNets + scratch CNN).
NEW_NAMES = [
    "efficientnetb0",
    "efficientnetb1",
    "efficientnetb2",
    "efficientnetb3",
    "cnn_scratch",
]

#: Documented native square input sizes (height == width) for the new backbones.
EXPECTED_NATIVE: dict[str, tuple[int, int]] = {
    "efficientnetb0": (224, 224),
    "efficientnetb1": (240, 240),
    "efficientnetb2": (260, 260),
    "efficientnetb3": (300, 300),
    "cnn_scratch": (128, 128),
}

#: All nine backbones once the feature lands (legacy five + new five minus the
#: one already-counted scratch). Derived from the registry so the build test
#: also exercises the legacy entries staying intact.
ALL_NAMES = list(EXPECTED_NATIVE) + ["vgg19", "resnet50", "densenet201"]


# --------------------------------------------------------------------------- #
# Test 1 — every new backbone builds (and the registry exposes them).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NEW_NAMES, ids=NEW_NAMES)
def test_new_backbone_registered(name):
    """Each new name is present in the ``BACKBONES`` registry."""
    assert name in BACKBONES


@pytest.mark.parametrize("name", NEW_NAMES, ids=NEW_NAMES)
def test_new_backbone_builds_with_no_weights(name):
    """``build_backbone`` succeeds for every new backbone (offline, weights=None)."""
    cfg = BackboneConfig(name=name, weights=None, trainable=False, pooling="avg")
    model = build_backbone(cfg)
    # A frozen avg-pooled feature extractor -> rank-2 (None, feat_dim) output.
    assert len(model.output_shape) == 2
    assert model.output_shape[-1] and model.output_shape[-1] > 0


# --------------------------------------------------------------------------- #
# Test 2 — feature-vector rank with avg pooling; documented B0 width.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NEW_NAMES, ids=NEW_NAMES)
def test_avg_pool_gives_rank2_feature_vector(name):
    """``pooling="avg"`` yields a rank-2 ``(None, feat_dim)`` output, feat_dim > 0."""
    cfg = BackboneConfig(name=name, weights=None, pooling="avg")
    model = build_backbone(cfg)
    assert len(model.output_shape) == 2
    assert model.output_shape[-1] > 0


def test_efficientnetb0_avg_feature_dim_is_1280():
    """EfficientNetB0 global-avg-pooled feature width is the documented 1280."""
    cfg = BackboneConfig(name="efficientnetb0", weights=None, pooling="avg")
    model = build_backbone(cfg)
    assert model.output_shape[-1] == 1280


# --------------------------------------------------------------------------- #
# Test 3 — frozen => no trainable weights (holds for the scratch CNN too).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NEW_NAMES, ids=NEW_NAMES)
def test_frozen_has_no_trainable_weights(name):
    """``trainable=False`` => ``model.trainable_weights == []`` for all new backbones."""
    cfg = BackboneConfig(name=name, weights=None, trainable=False, pooling="avg")
    model = build_backbone(cfg)
    assert model.trainable is False
    assert model.trainable_weights == []


# --------------------------------------------------------------------------- #
# Test 4 — cnn_scratch shapes and trainability.
# --------------------------------------------------------------------------- #
def test_cnn_scratch_none_pooling_shape():
    """``pooling="none"`` exposes the raw ``(None, 8, 8, 256)`` feature maps."""
    cfg = BackboneConfig(name="cnn_scratch", weights=None, pooling="none")
    model = build_backbone(cfg)
    assert model.output_shape == (None, 8, 8, 256)


def test_cnn_scratch_grid2_flattens_to_2x2x256():
    """``pooling="grid2"`` avg-pools the 8x8x256 maps to 2x2 and flattens -> 1024."""
    cfg = BackboneConfig(name="cnn_scratch", weights=None, pooling="grid2")
    model = build_backbone(cfg)
    assert len(model.output_shape) == 2
    assert model.output_shape[-1] == 2 * 2 * 256


def test_cnn_scratch_trainable_has_weights():
    """``trainable=True`` => the random-init scratch CNN exposes trainable weights."""
    cfg = BackboneConfig(name="cnn_scratch", weights=None, trainable=True, pooling="avg")
    model = build_backbone(cfg)
    assert model.trainable is True
    assert len(model.trainable_weights) > 0


# --------------------------------------------------------------------------- #
# Test 5 — cnn_scratch rejects imagenet weights at build time.
# --------------------------------------------------------------------------- #
def test_cnn_scratch_rejects_imagenet_weights():
    """A ``cnn_scratch`` config asking for imagenet weights fails loudly."""
    cfg = BackboneConfig(name="cnn_scratch", weights="imagenet")
    with pytest.raises(ValueError):
        build_backbone(cfg)


# --------------------------------------------------------------------------- #
# Test 6 — the cross-module consistency guard (keeps the two tables honest).
# --------------------------------------------------------------------------- #
def test_native_sizes_match_expected_in_backbones():
    """``backbones.native_input_size`` returns the documented sizes."""
    for name, size in EXPECTED_NATIVE.items():
        assert native_input_size(name) == size


def test_native_sizes_match_expected_in_dataset():
    """``dataset.native_input_size`` mirrors the documented sizes."""
    for name, size in EXPECTED_NATIVE.items():
        assert _dataset.native_input_size(name) == size


def test_backbones_and_dataset_tables_agree():
    """For every backbone, both modules report the *same* native size, same key set.

    This is the single guard against ``BACKBONES`` / ``NATIVE_INPUT_SIZES`` drift:
    it must fail if either implementer's table diverges.
    """
    assert set(BACKBONES) == set(_dataset.NATIVE_INPUT_SIZES)
    for name in BACKBONES:
        assert native_input_size(name) == _dataset.native_input_size(name)


# --------------------------------------------------------------------------- #
# Test 7 — preprocessing domains: EfficientNet stays ~[0,255]; scratch -> [0,1].
# --------------------------------------------------------------------------- #
def test_preprocess_efficientnet_is_passthrough_0_255():
    """EfficientNet preprocess is a placeholder pass-through of the ``×255`` rescale.

    The Rescaling/Normalization is baked into the model graph and consumes
    ``[0, 255]``; ``preprocess_for`` must NOT re-normalize to ``[-1, 1]``.
    """
    x = np.linspace(0.0, 1.0, 2 * 8 * 8 * 3, dtype="float32").reshape(2, 8, 8, 3)
    out = preprocess_for("efficientnetb0", x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Pass-through of x*255: spans roughly [0, 255], not re-centered to negatives.
    assert out.min() >= -1e-3
    assert out.max() > 1.0
    assert out.max() <= 255.0 + 1e-3


def test_preprocess_cnn_scratch_returns_unit_range():
    """``cnn_scratch`` passthrough divides the ``×255`` back out -> input stays [0, 1]."""
    x = np.linspace(0.0, 1.0, 2 * 8 * 8 * 3, dtype="float32").reshape(2, 8, 8, 3)
    out = preprocess_for("cnn_scratch", x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # The /255 cancels the ×255 in preprocess_for: model sees its native [0, 1].
    np.testing.assert_allclose(out, x, atol=1e-5)


# --------------------------------------------------------------------------- #
# Test 8 — end-to-end build + forward pass for the scratch CNN.
# --------------------------------------------------------------------------- #
def test_cnn_scratch_end_to_end_forward_pass():
    """A full ``build_model`` on ``cnn_scratch`` runs and returns finite (2, 1) output."""
    cfg = ExperimentConfig(
        name="t",
        backbone=BackboneConfig(name="cnn_scratch", weights=None, trainable=True),
    )
    model = build_model(cfg)
    h, w = native_input_size("cnn_scratch")
    x = np.random.default_rng(0).random((2, h, w, 3)).astype("float32")
    y = np.asarray(model(x))
    assert y.shape == (2, 1)  # raw-degrees target -> n_outputs == 1
    assert np.isfinite(y).all()
