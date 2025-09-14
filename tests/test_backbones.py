"""Tests — frozen backbone registry (weights=None for speed/offline)."""
import numpy as np
import pytest

from uda.models.backbones import build_backbone, native_input_size, preprocess_for
from uda.config import BackboneConfig

NAMES = ["vgg19", "resnet50", "densenet201", "xception", "inceptionv3"]


@pytest.mark.parametrize("name", NAMES)
def test_native_sizes(name):
    assert native_input_size(name) in [(224, 224), (299, 299)]


@pytest.mark.parametrize("name", NAMES)
def test_build_frozen_feature_vector(name):
    cfg = BackboneConfig(name=name, weights=None, trainable=False, pooling="avg")
    model = build_backbone(cfg)
    assert model.trainable is False
    assert len(model.trainable_weights) == 0
    # global-avg-pooled -> a feature vector (rank-2 output)
    assert len(model.output_shape) == 2
    assert model.output_shape[-1] and model.output_shape[-1] > 0


def test_unknown_backbone_raises():
    with pytest.raises(KeyError):
        native_input_size("alexnet")


@pytest.mark.parametrize(
    "pooling,rank", [("avg", 2), ("max", 2), ("avgmax", 2), ("grid2", 2), ("grid3", 2), ("none", 4)]
)
def test_pooling_modes_build_frozen(pooling, rank):
    cfg = BackboneConfig(name="vgg19", weights=None, trainable=False, pooling=pooling)
    model = build_backbone(cfg)
    assert len(model.trainable_weights) == 0
    assert len(model.output_shape) == rank
    if pooling == "avgmax":
        assert model.output_shape[-1] == 1024  # concat of GAP+GMP over 512 channels
    if pooling.startswith("grid"):
        # grid pooling keeps a small spatial grid then flattens (orientation kept)
        assert model.output_shape[-1] > 512


@pytest.mark.parametrize("name", ["convnext_tiny", "convnext_base", "efficientnetv2b0"])
def test_modern_backbone_builds_frozen(name):
    """ConvNeXt / EfficientNetV2 are drop-in frozen feature extractors."""
    cfg = BackboneConfig(name=name, weights=None, trainable=False, pooling="grid2")
    model = build_backbone(cfg)
    assert len(model.trainable_weights) == 0
    assert len(model.output_shape) == 2 and model.output_shape[-1] > 0


def test_preprocess_scales_unit_range_to_imagenet():
    x = np.zeros((2, 224, 224, 3), dtype="float32")  # [0,1]
    out = preprocess_for("vgg19", x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    # VGG preprocessing subtracts the ImageNet mean from [0,255]; 0 -> negative.
    assert out.min() < 0.0
