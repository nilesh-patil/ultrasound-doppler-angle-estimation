"""Frozen ImageNet backbone registry — the five paper feature extractors.

Each backbone is a ``keras.applications`` model used with
``include_top=False`` and ImageNet weights, frozen, with global average pooling so
it maps a preprocessed image batch ``(N, H, W, 3)`` to a feature vector
``(N, feat_dim)``. Adding EfficientNet/ConvNeXt later is one registry entry.

Preprocessing fidelity: the corpus delivers ``float32`` images in ``[0, 1]``; each
backbone's matching ``preprocess_input`` expects ``[0, 255]``, so
:func:`preprocess_for` rescales by 255 before applying it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import keras
import numpy as np
from keras import applications as kapp
from keras.applications import convnext as _convnext
from keras.applications import efficientnet as _effnet
from keras.applications import efficientnet_v2 as _effnet_v2

from uda.config import BackboneConfig

__all__ = [
    "BackboneSpec",
    "BACKBONES",
    "build_backbone",
    "preprocess_for",
    "native_input_size",
]


@dataclass(frozen=True)
class BackboneSpec:
    """A backbone factory plus its matching preprocessing and native input size."""

    factory: Callable[..., keras.Model]
    preprocess: Callable[[np.ndarray], np.ndarray]
    native: tuple[int, int]


def _passthrough_preprocess(x: np.ndarray) -> np.ndarray:
    """Undo the ``×255`` rescale so the scratch CNN sees its native ``[0, 1]``.

    A from-scratch network has no ImageNet statistics, so it consumes ``[0, 1]``
    images directly. :func:`preprocess_for` multiplies by 255 *before* calling
    this, so dividing by 255 here cancels that rescale and leaves the model on
    its ``[0, 1]`` domain (matching how the corpus is produced).
    """
    return np.asarray(x, dtype=np.float32) / 255.0


def _build_cnn_scratch(
    *,
    include_top: bool = False,
    weights: str | None = None,
    input_shape: tuple[int, int, int] = (128, 128, 3),
    pooling: str | None = None,
) -> keras.Model:
    """Build a small from-scratch Conv-BN-ReLU-MaxPool feature extractor.

    Mirrors the ``keras.applications`` factory signature so it drops into
    :func:`build_backbone` exactly like a pretrained model built with
    ``pooling=None``: it returns the raw ``(8, 8, 256)`` feature maps and lets
    :func:`_apply_pooling` control aggregation.

    Parameters
    ----------
    include_top : bool, optional
        Accepted for signature compatibility and ignored — there is no
        classifier head (always effectively ``False``).
    weights : str or None, optional
        Must be ``None`` (random init). ``"imagenet"`` raises ``ValueError`` —
        this network has no pretrained weights, so a misconfigured YAML fails
        loudly at build time.
    input_shape : tuple[int, int, int], optional
        Input ``(height, width, channels)``; defaults to ``(128, 128, 3)``.
    pooling : str or None, optional
        Accepted for signature compatibility and ignored — :func:`build_backbone`
        always passes ``None`` and pools itself via :func:`_apply_pooling`.

    Returns
    -------
    keras.Model
        ``inputs -> (8, 8, 256)`` feature maps, named ``"cnn_scratch"``, with no
        global pooling inside.

    Raises
    ------
    ValueError
        If ``weights`` is anything other than ``None``.
    """
    if weights is not None:
        raise ValueError(
            f"cnn_scratch has no pretrained weights; got weights={weights!r}, "
            "expected None"
        )
    from keras import layers

    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    return keras.Model(inputs, x, name="cnn_scratch")


#: name -> spec. Preprocess functions come from each model's own submodule so the
#: exact ImageNet normalization (mean-subtraction vs [-1,1] scaling) matches.
#:
#: EfficientNet note: ``keras.applications.efficientnet.preprocess_input`` is a
#: no-op placeholder — the Rescaling+Normalization is
#: baked into the model graph and consumes ``[0, 255]``. The ``×255`` rescale in
#: :func:`preprocess_for` already produces exactly that, so the placeholder is
#: registered only for API symmetry; do NOT add a second normalization.
BACKBONES: dict[str, BackboneSpec] = {
    "vgg19": BackboneSpec(kapp.VGG19, kapp.vgg19.preprocess_input, (224, 224)),
    "resnet50": BackboneSpec(kapp.ResNet50, kapp.resnet50.preprocess_input, (224, 224)),
    "densenet201": BackboneSpec(
        kapp.DenseNet201, kapp.densenet.preprocess_input, (224, 224)
    ),
    "xception": BackboneSpec(kapp.Xception, kapp.xception.preprocess_input, (299, 299)),
    "inceptionv3": BackboneSpec(
        kapp.InceptionV3, kapp.inception_v3.preprocess_input, (299, 299)
    ),
    # EfficientNet B0-B3 (frozen ImageNet feature extractors).
    "efficientnetb0": BackboneSpec(
        kapp.EfficientNetB0, _effnet.preprocess_input, (224, 224)
    ),
    "efficientnetb1": BackboneSpec(
        kapp.EfficientNetB1, _effnet.preprocess_input, (240, 240)
    ),
    "efficientnetb2": BackboneSpec(
        kapp.EfficientNetB2, _effnet.preprocess_input, (260, 260)
    ),
    "efficientnetb3": BackboneSpec(
        kapp.EfficientNetB3, _effnet.preprocess_input, (300, 300)
    ),
    # Modern backbones (ConvNeXt + EfficientNetV2), drop-in
    # frozen feature extractors. Like EfficientNet, their preprocess_input is a
    # baked-in no-op expecting [0,255] (preprocess_for's ×255 already supplies that).
    "convnext_tiny": BackboneSpec(
        kapp.ConvNeXtTiny, _convnext.preprocess_input, (224, 224)
    ),
    "convnext_small": BackboneSpec(
        kapp.ConvNeXtSmall, _convnext.preprocess_input, (224, 224)
    ),
    "convnext_base": BackboneSpec(
        kapp.ConvNeXtBase, _convnext.preprocess_input, (224, 224)
    ),
    "efficientnetv2b0": BackboneSpec(
        kapp.EfficientNetV2B0, _effnet_v2.preprocess_input, (224, 224)
    ),
    "efficientnetv2b1": BackboneSpec(
        kapp.EfficientNetV2B1, _effnet_v2.preprocess_input, (240, 240)
    ),
    "efficientnetv2b2": BackboneSpec(
        kapp.EfficientNetV2B2, _effnet_v2.preprocess_input, (260, 260)
    ),
    "efficientnetv2b3": BackboneSpec(
        kapp.EfficientNetV2B3, _effnet_v2.preprocess_input, (300, 300)
    ),
    # From-scratch Conv-BN-ReLU-MaxPool extractor (random init).
    "cnn_scratch": BackboneSpec(
        _build_cnn_scratch, _passthrough_preprocess, (128, 128)
    ),
}


def native_input_size(name: str) -> tuple[int, int]:
    """Return the ``(height, width)`` native input size for a backbone."""
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone: {name!r}")
    return BACKBONES[name].native


def _apply_pooling(x, pooling: str):
    """Aggregate conv feature maps ``(N, H, W, C)`` per the pooling mode.

    ``avg``/``max`` are global (rotation-invariant); ``avgmax`` concatenates both;
    ``gridG`` average-pools to a ~``G×G`` grid then flattens, retaining the coarse
    spatial layout that encodes vessel orientation; ``none`` returns the raw maps.
    """
    from keras import layers

    if pooling == "none":
        return x
    if pooling == "avg":
        return layers.GlobalAveragePooling2D()(x)
    if pooling == "max":
        return layers.GlobalMaxPooling2D()(x)
    if pooling == "avgmax":
        return layers.Concatenate()(
            [layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)]
        )
    if pooling.startswith("grid"):
        g = int(pooling[4:])
        hh, ww = int(x.shape[1]), int(x.shape[2])
        ph, pw = max(1, hh // g), max(1, ww // g)
        pooled = layers.AveragePooling2D(pool_size=(ph, pw), strides=(ph, pw))(x)
        return layers.Flatten()(pooled)
    raise ValueError(f"unknown pooling: {pooling!r}")


def build_backbone(cfg: BackboneConfig) -> keras.Model:
    """Construct a feature-extractor backbone with the configured pooling.

    Builds the ``keras.applications`` base with ``include_top=False`` and no built-in
    pooling, freezes it when ``cfg.trainable`` is False, then applies
    :func:`_apply_pooling` so the output is a feature vector (or raw maps for
    ``"none"``). The pooling layers carry no weights, so a frozen base stays frozen.
    """
    spec = BACKBONES.get(cfg.name)
    if spec is None:
        raise KeyError(f"unknown backbone: {cfg.name!r}")
    h, w = spec.native
    base = spec.factory(
        include_top=False, weights=cfg.weights, input_shape=(h, w, 3), pooling=None
    )
    features = _apply_pooling(base.output, cfg.pooling)
    model = keras.Model(base.input, features, name=f"{cfg.name}_features")
    # Set trainable on the wrapper so it propagates to the base layers (and so
    # ``model.trainable`` reports correctly); frozen => no trainable weights.
    model.trainable = bool(cfg.trainable)
    return model


def preprocess_for(name: str, x: np.ndarray) -> np.ndarray:
    """Apply a backbone's ImageNet preprocessing to ``[0, 1]`` images.

    Parameters
    ----------
    name : str
        Backbone name.
    x : np.ndarray
        Image batch ``(N, H, W, 3)``, ``float`` in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Preprocessed ``float32`` batch ready for the backbone.
    """
    spec = BACKBONES.get(name)
    if spec is None:
        raise KeyError(f"unknown backbone: {name!r}")
    arr = np.asarray(x, dtype=np.float32) * 255.0
    return np.asarray(spec.preprocess(arr), dtype=np.float32)
