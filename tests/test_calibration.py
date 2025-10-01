"""Tests — regression interval-reliability curves (``uda.evaluation.calibration``).

These cover the two public entry points of the (yet-to-be-written) ``uda.evaluation.calibration``
module:

* :func:`coverage_curve` — Gaussian interval-reliability curve. For each nominal
  coverage ``p`` it scores empirical coverage of ``mean ± z·std`` (with
  ``z = norm.ppf((1+p)/2)``) by delegating to :func:`uda.evaluation.uncertainty.coverage`, the
  single source of truth.
* :func:`conformal_calibration` — split-conformal calibration curve. For each
  ``alpha`` it reuses the §(a) half-width ``q`` from
  :func:`uda.evaluation.conformal.conformal_intervals` and reports ``mean(|test_resid| <= q)``.

Every test runs in well under a second, builds **no model**, and the module under
test is asserted **Keras-free** in a fresh interpreter (mirroring ``tests/test_cv.py``)
— the lazy ``uncertainty`` import is what keeps the import backend-free.

Honesty note: this is *interval* reliability for a regressor — nominal vs empirical
coverage — not classification ECE/Brier. The residuals fed to ``conformal_calibration``
are **already signed-wrapped by the caller**; the module never re-wraps.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.stats import norm

from uda.evaluation import conformal, uncertainty
from uda.evaluation.calibration import conformal_calibration, coverage_curve

DEFAULT_LEVELS = np.linspace(0.05, 0.95, 19)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _gaussian_sample(n: int = 5000, mu: float = 90.0, sigma: float = 4.0, seed: int = 0):
    """Synthetic, perfectly-calibrated batch: ``y_true ~ N(mean, std)`` per element.

    ``mean``/``std`` are constant vectors of length ``n`` so the Gaussian interval
    ``mean ± z·std`` is a genuine ``p``-credible interval for each draw — empirical
    coverage should hug the nominal diagonal.
    """
    rng = np.random.default_rng(seed)
    mean = np.full(n, mu, dtype=float)
    std = np.full(n, sigma, dtype=float)
    y_true = rng.normal(mu, sigma, size=n)
    return y_true, mean, std


def _exchangeable_resid(n_cal: int = 800, n_test: int = 800, sigma: float = 5.0, seed: int = 1):
    """Calibration/test residuals drawn i.i.d. from one distribution (exchangeable).

    Already on the signed-wrap scale (small sigma keeps |r| < 90), so the caller-owns-
    wrap contract is respected — the module is handed wrapped residuals verbatim.
    """
    rng = np.random.default_rng(seed)
    cal_resid = rng.normal(0.0, sigma, size=n_cal)
    test_resid = rng.normal(0.0, sigma, size=n_test)
    return cal_resid, test_resid


# --------------------------------------------------------------------------- #
# 1. Gaussian-correct synthetic -> coverage_curve hugs identity  (test 3)
# --------------------------------------------------------------------------- #
def test_gaussian_correct_coverage_curve_hugs_identity():
    """With ``y_true ~ N(mean, std)`` the empirical coverage tracks the nominal
    diagonal to within sampling noise at ``n = 5000``."""
    y_true, mean, std = _gaussian_sample(n=5000)
    curve = coverage_curve(y_true, mean, std)

    nominal = np.asarray(curve["nominal"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)
    assert np.max(np.abs(empirical - nominal)) < 0.05
    assert curve["miscoverage_area"] < 0.03


def test_default_levels_and_shape():
    """``levels=None`` evaluates ``linspace(0.05, 0.95, 19)``; arrays length-19 and
    ``nominal == levels`` exactly; keys are exactly the three documented ones."""
    y_true, mean, std = _gaussian_sample(n=2000)
    curve = coverage_curve(y_true, mean, std)

    assert set(curve.keys()) == {"nominal", "empirical", "miscoverage_area"}
    nominal = np.asarray(curve["nominal"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)
    assert nominal.shape == (19,)
    assert empirical.shape == (19,)
    np.testing.assert_allclose(nominal, DEFAULT_LEVELS)


# --------------------------------------------------------------------------- #
# 2. under-dispersed std -> over-confidence detected  (test 3, second half)
# --------------------------------------------------------------------------- #
def test_under_dispersed_std_is_overconfident_everywhere():
    """Halving the claimed ``std`` makes the intervals too tight: empirical coverage
    sits **below** nominal at every level, and ``miscoverage_area`` inflates well
    above the well-calibrated case."""
    y_true, mean, std = _gaussian_sample(n=5000)

    good = coverage_curve(y_true, mean, std)
    tight = coverage_curve(y_true, mean, std * 0.5)  # deliberately over-confident

    nominal = np.asarray(tight["nominal"], dtype=float)
    empirical = np.asarray(tight["empirical"], dtype=float)
    # over-confident => empirical < nominal at every nominal level
    assert np.all(empirical < nominal)
    # and the calibration gap is clearly larger than the honest model's
    assert tight["miscoverage_area"] > good["miscoverage_area"]
    assert tight["miscoverage_area"] > 0.1


# --------------------------------------------------------------------------- #
# 3. empirical monotone non-decreasing in nominal  (test 4)
# --------------------------------------------------------------------------- #
def test_empirical_monotone_non_decreasing_in_nominal():
    """``coverage`` is monotone in ``z`` and ``z`` is monotone in ``p``, so empirical
    coverage is non-decreasing as nominal increases."""
    y_true, mean, std = _gaussian_sample(n=3000, seed=5)
    curve = coverage_curve(y_true, mean, std)

    empirical = np.asarray(curve["empirical"], dtype=float)
    diffs = np.diff(empirical)
    assert np.all(diffs >= -1e-12)


# --------------------------------------------------------------------------- #
# 4. miscoverage_area ~ 0 for identity  (test 5)
# --------------------------------------------------------------------------- #
def test_miscoverage_area_formula_and_zero_on_identity():
    """``miscoverage_area == mean(|empirical - nominal|)`` recomputed from the
    returned arrays, and it is ~0 for a (near-)diagonal Gaussian-correct curve."""
    y_true, mean, std = _gaussian_sample(n=5000, seed=7)
    curve = coverage_curve(y_true, mean, std)

    nominal = np.asarray(curve["nominal"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)
    expected_area = float(np.mean(np.abs(empirical - nominal)))
    assert curve["miscoverage_area"] == pytest.approx(expected_area)
    # Gaussian-correct => the diagonal gap is tiny.
    assert curve["miscoverage_area"] < 0.03


# --------------------------------------------------------------------------- #
# 5. coverage_curve uses uncertainty.coverage  (test 2, single source of truth)
# --------------------------------------------------------------------------- #
def test_coverage_curve_uses_uncertainty_coverage_bit_for_bit():
    """For a hand-built batch and each nominal ``p``, ``empirical[i]`` equals
    ``uncertainty.coverage(y_true, mean, std, norm.ppf((1+p)/2))`` recomputed here —
    proving the ``uncertainty.coverage`` helper is the *only* place coverage is derived."""
    rng = np.random.default_rng(11)
    y_true = rng.normal(90.0, 6.0, size=400)
    mean = np.full(400, 90.0)
    std = np.full(400, 6.0)

    curve = coverage_curve(y_true, mean, std)
    nominal = np.asarray(curve["nominal"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)

    for i, p in enumerate(nominal):
        z = norm.ppf((1.0 + p) / 2.0)
        expected = uncertainty.coverage(y_true, mean, std, z=z)
        assert empirical[i] == pytest.approx(expected)


def test_coverage_curve_honors_explicit_levels():
    """An explicit ``levels`` is honored verbatim (``nominal == levels``)."""
    y_true, mean, std = _gaussian_sample(n=1000)
    levels = np.array([0.1, 0.5, 0.9])
    curve = coverage_curve(y_true, mean, std, levels=levels)
    np.testing.assert_allclose(np.asarray(curve["nominal"], dtype=float), levels)
    assert np.asarray(curve["empirical"]).shape == (3,)


# --------------------------------------------------------------------------- #
# 6. conformal_calibration reuses conformal.q + validity floor  (tests 6-9)
# --------------------------------------------------------------------------- #
def test_conformal_calibration_default_alphas_and_keys():
    """``alphas=None`` sweeps a small grid; keys are exactly the three documented;
    ``nominal == 1 - alpha`` and all three arrays share one length."""
    cal_resid, test_resid = _exchangeable_resid()
    curve = conformal_calibration(cal_resid, test_resid)

    assert set(curve.keys()) == {"alpha", "nominal", "empirical"}
    alpha = np.asarray(curve["alpha"], dtype=float)
    nominal = np.asarray(curve["nominal"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)
    assert alpha.shape == nominal.shape == empirical.shape
    assert alpha.size >= 1
    np.testing.assert_allclose(nominal, 1.0 - alpha)


def test_conformal_calibration_reuses_conformal_q_bit_for_bit():
    """Each ``empirical[i]`` equals ``mean(|test_resid| <= q_i)`` where ``q_i`` is
    pulled straight from ``conformal.conformal_intervals(cal_resid, np.zeros(1),
    alpha)["q"]`` — no private quantile re-derivation."""
    cal_resid, test_resid = _exchangeable_resid()
    alphas = np.array([0.05, 0.1, 0.2, 0.3])
    curve = conformal_calibration(cal_resid, test_resid, alphas=alphas)

    alpha = np.asarray(curve["alpha"], dtype=float)
    empirical = np.asarray(curve["empirical"], dtype=float)
    np.testing.assert_allclose(alpha, alphas)

    abs_test = np.abs(test_resid)
    for i, a in enumerate(alpha):
        q = conformal.conformal_intervals(cal_resid, np.zeros(1), float(a))["q"]
        expected = float(np.mean(abs_test <= q))
        assert empirical[i] == pytest.approx(expected)


def test_conformal_calibration_validity_floor_on_exchangeable_residuals():
    """Split-conformal's ``(n+1)`` guarantee is about the EXPECTATION over draws:
    averaged across many exchangeable (cal, test) draws, empirical coverage is
    ``>= nominal``. A single finite draw can dip a hair below (sampling noise of
    order ``1/sqrt(n_test)``), so the marginal floor is asserted on the mean."""
    alphas = np.array([0.05, 0.1, 0.2, 0.3])
    n_seeds = 40
    acc = np.zeros(len(alphas), dtype=float)
    for seed in range(n_seeds):
        cal_resid, test_resid = _exchangeable_resid(n_cal=1500, n_test=1500, seed=seed)
        curve = conformal_calibration(cal_resid, test_resid, alphas=alphas)
        acc += np.asarray(curve["empirical"], dtype=float)
    empirical = acc / n_seeds
    nominal = 1.0 - alphas
    # mean empirical coverage meets the marginal validity floor (>= nominal)
    assert np.all(empirical >= nominal - 0.01)
    # alphas ascending => q descending => mean empirical coverage non-increasing
    assert np.all(np.diff(empirical) <= 1e-9)


def test_conformal_calibration_caller_owns_the_wrap():
    """The module scores residuals verbatim and never re-wraps: a residual of
    magnitude ``2`` (already wrapped from a 179-vs-1 seam crossing) is scored as
    ``2``, never ``178``. With every |residual| == 2 and a generous ``q``, coverage
    is a perfect 1.0; if the module re-wrapped to 178 it would be 0.0."""
    cal_resid = np.full(50, 2.0)
    test_resid = np.full(50, -2.0)  # |.| == 2, the small-seam residual
    curve = conformal_calibration(cal_resid, test_resid, alphas=np.array([0.1]))

    empirical = np.asarray(curve["empirical"], dtype=float)
    # q is the (n+1)-inflated quantile of |cal_resid| == 2, so |test_resid|=2 <= q.
    assert empirical[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 7. Keras-free guard (fresh subprocess; mirror tests/test_cv.py)  (test 10)
# --------------------------------------------------------------------------- #
def test_calibration_module_is_keras_free():
    """Importing ``uda.evaluation.calibration`` must not pull a heavy backend
    (keras/jax/tensorflow) — the ``uncertainty`` import (which pulls ``keras``) is
    deferred lazily into the function body. Checked in a FRESH interpreter so other
    tests' backend imports don't pollute ``sys.modules``."""
    import subprocess

    code = (
        "import uda.evaluation.calibration, sys; "
        "bad = [b for b in ('keras', 'jax', 'tensorflow') if b in sys.modules]; "
        "print(','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"uda.evaluation.calibration pulled a backend: {r.stdout.strip()}\n{r.stderr[-300:]}"
    )
