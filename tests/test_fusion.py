"""Tests — circular fusion of angle estimates (``uda.interpret.fusion``).

These tests cover the ``uda.interpret.fusion``
module. Everything here is post-hoc on arrays of per-image angle estimates: no
model is built and every test runs in well under a second. The module under test
is **Keras-free** (asserted explicitly in a fresh subprocess, mirroring
``tests/test_conformal.py``).

Two public entry points are exercised:

* ``circular_fuse(estimates, *, weights=None) -> np.ndarray`` — the double-angle
  weighted circular mean of ``K`` per-image estimates, mapped into ``[0, 180)``
  (180-periodic; never a linear mean).
* ``evaluate_fusion(y_true, learned, geometric, *, weights=None) -> {metrics_learned,
  metrics_geometric, metrics_fused, weights, n}`` — scores the learned, geometric,
  and fused estimates against a common truth via ``uda.evaluation.evaluate.metrics`` (single
  source of truth).

Vessel orientation is **180-periodic**, so the blend is the weighted circular mean
in double-angle (``2θ``) space ``0.5*atan2(Σ w sin 2θ, Σ w cos 2θ)`` mapped into
``[0, 180)`` — a linear ``w·a + (1−w)·b`` would corrupt any pair straddling the
0/180 seam. ``fusion`` takes arrays only, so it does **not** import ``uda.interpret.geometric``.
The honesty contract: ``evaluate_fusion`` always surfaces
``metrics_learned`` so fusing a weak prior with a strong learned estimate is never
*assumed* to win.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from uda.evaluation import evaluate
from uda.interpret.fusion import circular_fuse, evaluate_fusion


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _signed_wrap(delta: np.ndarray) -> np.ndarray:
    """Reference signed-wrap into ``(-90, 90]`` (the project's residual scale)."""
    return ((np.asarray(delta, dtype=float) + 90.0) % 180.0) - 90.0


def _double_angle_fuse(estimates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference ``(K, N)`` double-angle weighted circular mean -> ``[0, 180)``."""
    est = np.asarray(estimates, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    t = np.deg2rad(2.0 * est)  # (K, N)
    s = np.tensordot(w, np.sin(t), axes=(0, 0))  # (N,)
    c = np.tensordot(w, np.cos(t), axes=(0, 0))  # (N,)
    out = 0.5 * np.rad2deg(np.arctan2(s, c))
    return out % 180.0


# --------------------------------------------------------------------------- #
# 1. double-angle blend formula + range  (test 1)
# --------------------------------------------------------------------------- #
def test_circular_fuse_matches_double_angle_formula_and_range():
    """For hand-built ``(K, N)`` estimates + weights, ``circular_fuse`` equals the
    reference ``0.5*atan2(Σ w sin 2θ, Σ w cos 2θ)`` mapped into ``[0, 180)``; the
    output is length ``N`` and every value is in ``[0, 180)``."""
    rng = np.random.default_rng(0)
    estimates = rng.uniform(0.0, 180.0, size=(3, 11))
    weights = np.array([0.5, 0.3, 0.2])

    fused = circular_fuse(estimates, weights=weights)
    expected = _double_angle_fuse(estimates, weights)

    assert np.asarray(fused).shape == (11,)
    assert np.allclose(fused, expected, atol=1e-9)
    assert np.all(np.asarray(fused) >= 0.0)
    assert np.all(np.asarray(fused) < 180.0)


def test_circular_fuse_of_identical_arrays_returns_that_array():
    """Degenerate: the blend of ``K`` identical estimate rows is that array itself
    (a thing fused with copies of itself is unchanged), wrapped into ``[0, 180)``."""
    theta = np.array([5.0, 45.0, 90.0, 135.0, 170.0, 1.0, 179.0])
    estimates = np.stack([theta, theta, theta])  # (3, N) identical rows

    fused = np.asarray(circular_fuse(estimates))
    assert np.allclose(_signed_wrap(fused - theta), 0.0, atol=1e-7)
    assert np.all(fused >= 0.0) and np.all(fused < 180.0)


# --------------------------------------------------------------------------- #
# 2. seam correctness vs linear  (test 2)
# --------------------------------------------------------------------------- #
def test_circular_mean_of_1_and_179_is_boundary_not_90():
    """Fusing ``179°`` and ``1°`` (uniform weights) returns the 0/180 boundary (the
    circular bisector), NOT the linear ``90°``: a naive ``np.average`` would give the
    visibly-wrong 90."""
    estimates = np.array([[1.0], [179.0]])
    fused = float(np.asarray(circular_fuse(estimates))[0])

    # near the seam: ~0 or ~180, so signed-wrap distance to 0 is tiny
    assert abs(_signed_wrap(np.array([fused - 0.0]))[0]) < 1e-6
    # explicitly NOT the linear mean
    assert not fused == pytest.approx(90.0, abs=1.0)
    assert fused != pytest.approx(float(np.average([1.0, 179.0])), abs=1.0)


def test_seam_straddling_case_differs_from_linear_average():
    """A seam-straddling vector is visibly different from ``np.average`` (linear), but
    matches the reference double-angle mean."""
    a = np.array([2.0, 10.0, 170.0])
    b = np.array([178.0, 20.0, 178.0])
    estimates = np.stack([a, b])

    fused = np.asarray(circular_fuse(estimates))
    linear = (a + b) / 2.0  # 90, 15, 174
    expected = _double_angle_fuse(estimates, np.array([0.5, 0.5]))

    assert np.allclose(fused, expected, atol=1e-9)
    # the seam-crossing entries (0th and 2nd) diverge from the linear average
    assert abs(_signed_wrap(np.array([fused[0] - linear[0]]))[0]) > 80.0


# --------------------------------------------------------------------------- #
# 3. list and array inputs agree; unequal rows raise  (test 3)
# --------------------------------------------------------------------------- #
def test_list_of_arrays_and_stacked_array_agree():
    """A length-``K`` list of length-``N`` arrays and the equivalent ``(K, N)`` array
    give identical fused results."""
    rng = np.random.default_rng(1)
    rows = [rng.uniform(0.0, 180.0, size=9) for _ in range(4)]
    as_list = circular_fuse(rows)
    as_array = circular_fuse(np.stack(rows))
    assert np.allclose(np.asarray(as_list), np.asarray(as_array), atol=1e-12)


def test_rows_of_unequal_length_raise():
    """Estimate rows of mismatched length ``N`` are rejected (cannot be aligned)."""
    bad = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    with pytest.raises(ValueError):
        circular_fuse(bad)


# --------------------------------------------------------------------------- #
# 4. weights: default uniform, normalization, concentration  (test 4)
# --------------------------------------------------------------------------- #
def test_default_weights_are_uniform():
    """``weights=None`` is the uniform ``1/K`` blend (equals explicit equal weights)."""
    rng = np.random.default_rng(2)
    estimates = rng.uniform(0.0, 180.0, size=(3, 8))
    default = np.asarray(circular_fuse(estimates))
    explicit = np.asarray(circular_fuse(estimates, weights=[1.0, 1.0, 1.0]))
    assert np.allclose(default, explicit, atol=1e-12)


def test_weights_are_normalized_scale_invariant():
    """Scaling all weights by a positive constant leaves the result unchanged (the
    blend normalizes weights to sum to 1)."""
    rng = np.random.default_rng(3)
    estimates = rng.uniform(0.0, 180.0, size=(2, 10))
    base = np.asarray(circular_fuse(estimates, weights=[0.7, 0.3]))
    scaled = np.asarray(circular_fuse(estimates, weights=[7.0, 3.0]))
    assert np.allclose(base, scaled, atol=1e-12)


def test_weight_concentrated_on_one_member_returns_that_member():
    """A weight vector concentrated on member ``k`` (``[1, 0]``) returns that member
    (wrapped into ``[0, 180)``) — the other estimate is ignored."""
    member0 = np.array([5.0, 88.0, 150.0, 179.0])
    member1 = np.array([95.0, 12.0, 60.0, 1.0])
    estimates = np.stack([member0, member1])

    fused0 = np.asarray(circular_fuse(estimates, weights=[1.0, 0.0]))
    assert np.allclose(_signed_wrap(fused0 - member0), 0.0, atol=1e-7)

    fused1 = np.asarray(circular_fuse(estimates, weights=[0.0, 1.0]))
    assert np.allclose(_signed_wrap(fused1 - member1), 0.0, atol=1e-7)


# --------------------------------------------------------------------------- #
# 5. evaluate_fusion keys + metrics reuse (single source of truth)  (test 6)
# --------------------------------------------------------------------------- #
def test_evaluate_fusion_keys_and_metrics_reuse():
    """``evaluate_fusion`` returns ``{metrics_learned, metrics_geometric,
    metrics_fused, weights, n}``; each ``metrics_*`` equals
    ``uda.evaluation.evaluate.metrics(y_true, estimate)`` recomputed here in the CANONICAL
    truth-first order (so ME/MAPE/R2 are right), and ``metrics_fused`` uses exactly
    ``circular_fuse(...)``."""
    rng = np.random.default_rng(4)
    y_true = rng.uniform(20.0, 160.0, size=20)
    learned = y_true + rng.normal(0.0, 3.0, size=20)
    geometric = rng.uniform(0.0, 180.0, size=20)

    out = evaluate_fusion(y_true, learned, geometric)

    assert set(out) == {
        "metrics_learned",
        "metrics_geometric",
        "metrics_fused",
        "weights",
        "n",
    }
    assert out["n"] == 20

    assert out["metrics_learned"] == evaluate.metrics(y_true, learned)
    assert out["metrics_geometric"] == evaluate.metrics(y_true, geometric)

    fused = circular_fuse(np.stack([learned, geometric]), weights=out["weights"])
    assert out["metrics_fused"] == evaluate.metrics(y_true, fused)
    for m in (out["metrics_learned"], out["metrics_geometric"], out["metrics_fused"]):
        assert set(m) == {"mae", "rmse", "me", "mape", "r2"}


def test_evaluate_fusion_weight_all_on_learned_makes_fused_equal_learned():
    """With weight all on the learned member, ``metrics_fused == metrics_learned``
    (the fused estimate is the learned one), proving weights flow into the blend."""
    rng = np.random.default_rng(5)
    y_true = rng.uniform(20.0, 160.0, size=15)
    learned = y_true + rng.normal(0.0, 2.0, size=15)
    geometric = rng.uniform(0.0, 180.0, size=15)

    out = evaluate_fusion(y_true, learned, geometric, weights=[1.0, 0.0])
    # fusing with all weight on the learned member -> fused == learned, key by key
    for key in ("mae", "rmse", "me", "mape", "r2"):
        assert out["metrics_fused"][key] == pytest.approx(out["metrics_learned"][key])
    assert np.allclose(out["weights"], [1.0, 0.0])


# --------------------------------------------------------------------------- #
# 6. honest three-way comparison — fusion does not magically win  (test 7)
# --------------------------------------------------------------------------- #
def test_fused_mae_between_good_learned_and_noisy_geometric():
    """When ``geometric`` is pure noise and ``learned`` is near-truth, the fused MAE is
    **between** the two single-source MAEs — fusion does not beat the good source. The
    test asserts the *relationship*, matching the honesty contract (fusion not assumed
    to help)."""
    rng = np.random.default_rng(6)
    y_true = rng.uniform(30.0, 150.0, size=200)
    learned = y_true + rng.normal(0.0, 2.0, size=200)  # near truth
    geometric = rng.uniform(0.0, 180.0, size=200)  # pure noise

    out = evaluate_fusion(y_true, learned, geometric)
    mae_learned = out["metrics_learned"]["mae"]
    mae_geometric = out["metrics_geometric"]["mae"]
    mae_fused = out["metrics_fused"]["mae"]

    lo, hi = sorted((mae_learned, mae_geometric))
    assert lo - 1e-6 <= mae_fused <= hi + 1e-6
    # the noisy geometric is the worse source; fusion must not beat the learned one
    assert mae_learned <= mae_geometric
    assert mae_fused >= mae_learned - 1e-6


# --------------------------------------------------------------------------- #
# 7. Keras-free fresh-subprocess guard  (test 8)
# --------------------------------------------------------------------------- #
def test_fusion_module_is_keras_free():
    """Importing ``uda.interpret.fusion`` must not pull a heavy backend (keras/jax/tensorflow).
    Checked in a FRESH interpreter so other tests' backend imports don't pollute
    ``sys.modules`` (mirror ``tests/test_conformal.py``). ``fusion`` takes arrays, so
    it must not import ``uda.interpret.geometric`` either."""
    import subprocess

    code = (
        "import uda.interpret.fusion, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.interpret.fusion pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )
