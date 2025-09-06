"""Tests — determinism."""
import random

import numpy as np

from uda.seed import backend_name, set_seed


def test_numpy_reproducible():
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_python_random_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_backend_name_matches_keras():
    import keras

    assert backend_name() == keras.backend.backend()
