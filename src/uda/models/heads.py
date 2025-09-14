"""Shallow regression head — ``BatchNorm -> [Dense -> ReLU -> Dropout]* -> Dense``.

Small and configurable (depth/width/dropout); the final
layer width is ``n_outputs`` (1 for raw degrees, 2 for the sin/cos target).
"""
from __future__ import annotations

import keras
from keras import layers

from uda.config import HeadConfig

__all__ = ["build_head"]


def build_head(input_dim: int, cfg: HeadConfig, n_outputs: int) -> keras.Model:
    """Build the shallow head that maps backbone features to the angle target.

    Parameters
    ----------
    input_dim : int
        Backbone feature-vector width.
    cfg : HeadConfig
        Depth (``hidden_units``), ``dropout``, ``batchnorm``, ``activation`` and
        ``final_activation``.
    n_outputs : int
        Output width (``AngleTarget.n_outputs``).

    Returns
    -------
    keras.Model
        The head model, input shape ``(input_dim,)``.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    reg = keras.regularizers.l2(cfg.l2) if cfg.l2 else None
    if cfg.batchnorm:
        x = layers.BatchNormalization()(x)
    for i, units in enumerate(cfg.hidden_units):
        x = layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i}")(x)
        x = layers.Activation(cfg.activation, name=f"act_{i}")(x)
        if cfg.dropout and cfg.dropout > 0:
            x = layers.Dropout(cfg.dropout, name=f"drop_{i}")(x)
    final_activation = None if cfg.final_activation == "linear" else cfg.final_activation
    outputs = layers.Dense(
        n_outputs, activation=final_activation, kernel_regularizer=reg, name="theta"
    )(x)
    return keras.Model(inputs, outputs, name="head")
