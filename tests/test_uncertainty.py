"""Tests — MC-dropout predictive uncertainty.

Covers ``uda.evaluation.uncertainty.mc_dropout_predict`` / ``predictive_interval`` /
``coverage``. MC dropout is intentionally stochastic, so these assert *shapes*
and *statistical/structural* properties (not bit-equality across runs). The
backend RNG is pinned with ``uda.seed.set_seed`` where a stable result is
needed. Fixtures are tiny synthetic heads (no backbone, no ImageNet weights) so
the whole file runs in well under a second on the JAX-CPU env.
"""
from __future__ import annotations

import numpy as np
import pytest

from uda.config import HeadConfig
from uda.models.heads import build_head
from uda.seed import set_seed

from uda.evaluation.uncertainty import coverage, mc_dropout_predict, predictive_interval


# --------------------------------------------------------------------------- #
# Fixtures: small synthetic heads / inputs (fast, weights-free).
# --------------------------------------------------------------------------- #
INPUT_DIM = 8


def _dropout_head(dropout: float, n_outputs: int = 1):
    """A tiny head with a controllable Dropout rate (batchnorm off for speed)."""
    cfg = HeadConfig(hidden_units=[16], dropout=dropout, batchnorm=False)
    return build_head(INPUT_DIM, cfg, n_outputs)


def _features(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic feature batch ``(n, INPUT_DIM)``."""
    return np.random.RandomState(seed).rand(n, INPUT_DIM).astype("float32")


# --------------------------------------------------------------------------- #
# 1. shapes — (mean, std) each (N, n_outputs).
# --------------------------------------------------------------------------- #
def test_mc_dropout_predict_returns_mean_std_with_input_shape():
    head = _dropout_head(0.5, n_outputs=1)
    mean, std = mc_dropout_predict(head, _features(5), n=20)
    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_mc_dropout_predict_preserves_multi_output_width():
    """A 2-output head (e.g. sin/cos) yields (N, 2) summaries — raw output space."""
    head = _dropout_head(0.5, n_outputs=2)
    mean, std = mc_dropout_predict(head, _features(7), n=12)
    assert mean.shape == (7, 2)
    assert std.shape == (7, 2)


def test_mc_dropout_predict_returns_host_numpy_arrays():
    """Backend tensors are converted to host numpy regardless of backend."""
    head = _dropout_head(0.5, n_outputs=1)
    mean, std = mc_dropout_predict(head, _features(4), n=8)
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)


def test_mc_dropout_predict_batching_does_not_change_shape():
    """A batch_size smaller than N still summarizes the whole input once."""
    head = _dropout_head(0.5, n_outputs=1)
    x = _features(10)
    mean, std = mc_dropout_predict(head, x, n=8, batch_size=3)
    assert mean.shape == (10, 1)
    assert std.shape == (10, 1)


# --------------------------------------------------------------------------- #
# 2. dropout actually fires (std > 0) vs no dropout (std ~ 0).
# --------------------------------------------------------------------------- #
def test_std_positive_when_dropout_active():
    set_seed(0)
    head = _dropout_head(0.5, n_outputs=1)
    _, std = mc_dropout_predict(head, _features(20), n=30)
    # Stochastic dropout masks => most rows vary across passes.
    assert np.mean(std > 0) > 0.5


def test_std_zero_without_dropout():
    set_seed(0)
    head = _dropout_head(0.0, n_outputs=1)  # no active Dropout layers
    _, std = mc_dropout_predict(head, _features(20), n=30)
    # No stochasticity => every pass is the same forward graph => std ~ 0.
    # atol=1e-5 (not the np.allclose 1e-8 default): repeated float32 matmuls on
    # the JAX/XLA backend leave a ~1e-7 residual even when the masks are fixed;
    # the contract is "dropout off => no meaningful spread", not bit-equality.
    assert np.allclose(std, 0.0, atol=1e-5)


def test_n_equals_one_gives_zero_std():
    """A single pass has no spread: std == 0 everywhere (population, ddof=0)."""
    head = _dropout_head(0.5, n_outputs=1)
    _, std = mc_dropout_predict(head, _features(6), n=1)
    assert np.allclose(std, 0.0)


# --------------------------------------------------------------------------- #
# 3. n < 1 raises ValueError.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_n", [0, -1, -5], ids=["zero", "neg1", "neg5"])
def test_n_below_one_raises(bad_n):
    head = _dropout_head(0.5, n_outputs=1)
    with pytest.raises(ValueError):
        mc_dropout_predict(head, _features(4), n=bad_n)


# --------------------------------------------------------------------------- #
# 4. predictive_interval — width == 2*z*std, lower < upper where std > 0.
# --------------------------------------------------------------------------- #
def test_predictive_interval_width_is_two_z_std():
    mean = np.array([[1.0], [2.0], [3.0]])
    std = np.array([[0.5], [1.0], [2.0]])
    z = 1.96
    lower, upper = predictive_interval(mean, std, z=z)
    assert np.allclose(upper - lower, 2.0 * z * std)


def test_predictive_interval_is_centered_on_mean():
    mean = np.array([[10.0], [20.0]])
    std = np.array([[1.0], [3.0]])
    lower, upper = predictive_interval(mean, std, z=2.0)
    assert np.allclose((lower + upper) / 2.0, mean)
    assert np.allclose(lower, mean - 2.0 * std)
    assert np.allclose(upper, mean + 2.0 * std)


def test_predictive_interval_lower_below_upper_where_std_positive():
    mean = np.zeros((4, 1))
    std = np.array([[0.0], [0.1], [1.0], [5.0]])
    lower, upper = predictive_interval(mean, std)
    pos = std > 0
    assert np.all(lower[pos] < upper[pos])
    # Degenerate std == 0 collapses the interval to a point.
    assert np.allclose(lower[~pos], upper[~pos])


def test_predictive_interval_preserves_shape():
    mean = np.zeros((3, 2))
    std = np.ones((3, 2))
    lower, upper = predictive_interval(mean, std)
    assert lower.shape == (3, 2)
    assert upper.shape == (3, 2)


# --------------------------------------------------------------------------- #
# 5. coverage — in [0, 1], monotone in z, exact endpoints on constructed cases.
# --------------------------------------------------------------------------- #
def test_coverage_is_a_fraction_in_unit_interval():
    rng = np.random.RandomState(0)
    mean = rng.randn(50, 1)
    std = np.abs(rng.randn(50, 1)) + 0.1
    y_true = mean + rng.randn(50, 1)
    c = coverage(y_true, mean, std)
    assert 0.0 <= c <= 1.0


def test_coverage_monotone_in_z():
    """A wider interval (larger z) covers at least as many points."""
    rng = np.random.RandomState(1)
    mean = rng.randn(40, 1)
    std = np.abs(rng.randn(40, 1)) + 0.1
    y_true = mean + rng.randn(40, 1)
    assert coverage(y_true, mean, std, z=3.0) >= coverage(y_true, mean, std, z=1.0)


def test_coverage_one_when_all_inside():
    mean = np.zeros((10, 1))
    std = np.ones((10, 1))
    y_true = mean.copy()  # every truth sits exactly at the center
    assert coverage(y_true, mean, std, z=1.96) == 1.0


def test_coverage_zero_when_all_far_outside():
    mean = np.zeros((10, 1))
    std = np.ones((10, 1))
    y_true = np.full((10, 1), 1000.0)  # far beyond mean + z*std
    assert coverage(y_true, mean, std, z=1.96) == 0.0


# --------------------------------------------------------------------------- #
# 6. coverage with std == 0 — covered iff y_true == mean exactly.
# --------------------------------------------------------------------------- #
def test_coverage_with_zero_std_requires_exact_match():
    mean = np.array([[1.0], [2.0], [3.0], [4.0]])
    std = np.zeros((4, 1))
    y_true = np.array([[1.0], [2.0], [99.0], [4.0]])  # row 2 differs
    # 3 of 4 exactly match => coverage 0.75.
    assert coverage(y_true, mean, std) == pytest.approx(0.75)


def test_coverage_with_zero_std_all_match_is_one():
    mean = np.array([[5.0], [6.0]])
    std = np.zeros((2, 1))
    assert coverage(mean.copy(), mean, std) == 1.0
