"""Tests — assembled estimator (weights=None for speed)."""
import numpy as np

from uda.models.backbones import build_backbone
from uda.config import ExperimentConfig
from uda.models.heads import build_head
from uda.models.model import build_model, feature_dim


def _cfg(name="vgg19", target="raw"):
    return ExperimentConfig(
        name="t",
        backbone={"name": name, "weights": None},
        target={"kind": target},
    )


def _count(weights):
    return int(sum(int(np.prod(w.shape)) for w in weights))


def test_trainable_params_equal_head_only():
    cfg = _cfg("vgg19", "raw")
    model = build_model(cfg)
    bb = build_backbone(cfg.backbone)
    head = build_head(feature_dim(bb), cfg.head, 1)
    # backbone frozen => the model's trainable params are exactly the head's
    assert _count(model.trainable_weights) == _count(head.trainable_weights)


def test_sincos_model_has_two_outputs():
    model = build_model(_cfg("vgg19", "sincos2theta"))
    assert model.output_shape[-1] == 2


def test_raw_model_has_one_output():
    model = build_model(_cfg("resnet50", "raw"))
    assert model.output_shape[-1] == 1
