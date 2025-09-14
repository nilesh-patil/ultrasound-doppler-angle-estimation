"""Assembled estimator — frozen backbone -> shallow head.

The assembled model maps a preprocessed image batch to the
encoded angle target and is used for inference/export. Training (train.py) caches
backbone features first (the backbone is frozen) and fits the head on them, which
is equivalent and far faster — the paper's transfer-learning recipe.
"""
from __future__ import annotations

import keras
import numpy as np

from uda.models.backbones import build_backbone, native_input_size
from uda.config import ExperimentConfig
from uda.models.heads import build_head
from uda.models.targets import build_target

__all__ = ["build_model", "feature_dim"]


def feature_dim(backbone: keras.Model) -> int:
    """Flattened feature width (preserves spatial/orientation information)."""
    return int(np.prod(backbone.output_shape[1:]))


def build_model(cfg: ExperimentConfig) -> keras.Model:
    """Assemble ``backbone(frozen) -> head`` into one model.

    Parameters
    ----------
    cfg : ExperimentConfig
        Drives the backbone, head and target widths.

    Returns
    -------
    keras.Model
        Input ``(H, W, 3)`` (preprocessed) -> encoded target ``(n_outputs,)``.
    """
    backbone = build_backbone(cfg.backbone)
    target = build_target(cfg.target)
    head = build_head(feature_dim(backbone), cfg.head, target.n_outputs)

    h, w = native_input_size(cfg.backbone.name)
    inputs = keras.Input(shape=(h, w, 3), name="image")
    features = keras.layers.Flatten()(backbone(inputs))
    outputs = head(features)
    return keras.Model(inputs, outputs, name=f"uda_{cfg.backbone.name}")
