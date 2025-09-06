"""Determinism helpers: seed python/numpy and the active Keras backend.

Residual GPU nondeterminism (Metal / cuDNN reductions) cannot
be fully eliminated and is documented in the reproducibility report.
"""
from __future__ import annotations

import os
import random

import numpy as np


def backend_name() -> str:
    """Return the active Keras backend ('jax' or 'tensorflow')."""
    import keras

    return keras.backend.backend()


def set_seed(seed: int = 42) -> None:
    """Seed python ``random``, numpy, and the active Keras backend.

    Uses ``keras.utils.set_random_seed`` (one call seeds python/numpy/backend) plus
    ``PYTHONHASHSEED`` and, on TensorFlow, op-level determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import keras

    keras.utils.set_random_seed(seed)

    if backend_name() == "tensorflow":
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - best effort across versions
            pass
