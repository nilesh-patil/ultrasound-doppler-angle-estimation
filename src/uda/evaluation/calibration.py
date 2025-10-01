"""Regression interval-reliability curves (Keras-free).

This module answers a single honest question for a
*regression* angle estimate carrying an uncertainty: does its nominal-``p``
prediction interval actually contain the truth a fraction ``p`` of the time? This
is **interval reliability**, the regression analogue of a reliability curve — it is
**not** classification calibration. There is no class posterior here, so there is
deliberately no ECE, no Brier score, no per-class confidence: every quantity
returned is a nominal-vs-empirical *coverage* (and a coverage *gap*).

Two complementary curves, each reusing an existing single source of truth and
re-deriving nothing:

* :func:`coverage_curve` — **Gaussian** interval reliability. For each nominal
  coverage ``p`` the two-sided normal quantile is ``z = norm.ppf((1 + p) / 2)`` and
  the empirical coverage is :func:`uda.evaluation.uncertainty.coverage` (the
  single source of truth), i.e. the fraction of truths inside ``mean ± z*std``.
* :func:`conformal_calibration` — **split-conformal** calibration. For each
  ``alpha`` the half-width ``q`` comes straight from
  :func:`uda.evaluation.conformal.conformal_intervals` (so the ``(n+1)`` finite-sample
  inflation is inherited, never re-implemented) and the empirical coverage on the
  test residuals is ``mean(|test_resid| <= q)``.

Residual scale (critical): the residuals fed to :func:`conformal_calibration` are
**already signed-wrapped by the caller** into ``(-90, 90]`` — vessel orientation is
180-periodic and this module does not re-wrap. A residual of magnitude ``2`` (a
179-vs-1 seam crossing) is scored as ``2``, never ``178``: the caller owns angle
periodicity (this module does not re-wrap).

Keras-free: this module imports only ``numpy``, ``scipy.stats.norm``, and
``uda.evaluation.conformal`` at top level. ``uda.evaluation.uncertainty`` (which imports ``keras`` for its
MC-dropout helpers) is imported **lazily inside** :func:`coverage_curve`, mirroring
the deferral in ``conformal.py`` so ``import uda.evaluation.calibration`` stays backend-free.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from uda.evaluation import conformal

__all__ = ["coverage_curve", "conformal_calibration"]

# Default nominal-coverage grid for the Gaussian curve.
_DEFAULT_LEVELS = np.linspace(0.05, 0.95, 19)
# Default miscoverage grid for the conformal curve.
_DEFAULT_ALPHAS = np.array([0.05, 0.1, 0.2, 0.3])


def coverage_curve(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, *, levels=None
) -> dict:
    """Gaussian interval-reliability curve: nominal vs empirical coverage.

    For each nominal coverage ``p`` in ``levels`` the two-sided normal quantile is
    ``z = norm.ppf((1 + p) / 2)`` and the **empirical** coverage is
    ``uncertainty.coverage(y_true, mean, std, z)`` — the fraction of truths in the
    Gaussian interval ``mean ± z*std``. A perfectly calibrated model lies on the
    diagonal (empirical == nominal); because ``coverage`` is monotone in ``z`` and
    ``z`` is monotone in ``p``, ``empirical`` is non-decreasing in ``nominal``.

    No coverage is re-derived here: every point is a single call to
    :func:`uda.evaluation.uncertainty.coverage`, the only place an interval-membership fraction
    is computed.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth angles (degrees).
    mean, std : numpy.ndarray
        Per-element predictive mean and std defining each interval (e.g. from
        :func:`uda.evaluation.uncertainty.mc_dropout_predict`).
    levels : array-like or None, keyword-only
        Nominal coverages to evaluate; default ``np.linspace(0.05, 0.95, 19)``. Used
        verbatim as ``nominal`` when supplied.

    Returns
    -------
    dict
        ``{"nominal": numpy.ndarray, "empirical": numpy.ndarray,
        "miscoverage_area": float}`` where ``nominal == np.asarray(levels)``,
        ``empirical[i] == uncertainty.coverage(y_true, mean, std, z_i)`` with
        ``z_i = norm.ppf((1 + nominal[i]) / 2)``, and ``miscoverage_area ==
        mean(|empirical - nominal|)`` (the mean absolute gap from the diagonal; 0
        is perfectly calibrated).
    """
    nominal = _DEFAULT_LEVELS if levels is None else np.asarray(levels, dtype=float)

    # Lazy import: ``uda.evaluation.uncertainty`` pulls in ``keras`` at module top level, so
    # deferring it here keeps ``import uda.evaluation.calibration`` backend-free (the Keras-free
    # guard), exactly as ``conformal.py`` defers the same import.
    from uda.evaluation import uncertainty

    z = norm.ppf((1.0 + nominal) / 2.0)
    empirical = np.array(
        [uncertainty.coverage(y_true, mean, std, z=float(z_i)) for z_i in z],
        dtype=float,
    )

    miscoverage_area = float(np.mean(np.abs(empirical - nominal)))
    return {
        "nominal": nominal,
        "empirical": empirical,
        "miscoverage_area": miscoverage_area,
    }


def conformal_calibration(
    cal_resid: np.ndarray, test_resid: np.ndarray, *, alphas=None
) -> dict:
    """Split-conformal calibration curve: nominal ``1 - alpha`` vs empirical.

    For each ``alpha`` in ``alphas`` the half-width is
    ``q = conformal.conformal_intervals(cal_resid, np.zeros(1), alpha)["q"]`` — the
    finite-sample-valid quantile of ``|cal_resid|``; the throwaway
    ``np.zeros(1)`` test point only satisfies the signature, only ``q`` is used. The
    **empirical** coverage on the test residuals is ``mean(|test_resid| <= q)``.
    Valid conformal intervals sit on or above the diagonal (empirical ``>=``
    nominal), and a smaller ``alpha`` (larger ``q``) yields ``>=`` empirical
    coverage.

    No quantile is computed here: each ``q`` comes straight from
    :func:`uda.evaluation.conformal.conformal_intervals`, so the half-width matches
    bit-for-bit and the ``(n+1)`` inflation is inherited, not re-implemented.

    Parameters
    ----------
    cal_resid, test_resid : numpy.ndarray
        Calibration and test residuals, **already signed-wrapped by the caller**
        into ``(-90, 90]`` (this module does not re-wrap — the caller owns angle
        periodicity). The conformity score is the magnitude ``|·|``.
    alphas : array-like or None, keyword-only
        Miscoverage levels to sweep; default ``np.array([0.05, 0.1, 0.2, 0.3])``.

    Returns
    -------
    dict
        ``{"alpha": numpy.ndarray, "nominal": numpy.ndarray,
        "empirical": numpy.ndarray}`` where ``nominal == 1 - alpha`` and
        ``empirical[i] == mean(|test_resid| <= q_i)``.
    """
    alpha = _DEFAULT_ALPHAS if alphas is None else np.asarray(alphas, dtype=float)

    abs_test = np.abs(np.asarray(test_resid, dtype=float))
    empirical = np.array(
        [
            float(
                np.mean(
                    abs_test
                    <= conformal.conformal_intervals(
                        cal_resid, np.zeros(1), float(a)
                    )["q"]
                )
            )
            for a in alpha
        ],
        dtype=float,
    )

    return {
        "alpha": alpha,
        "nominal": 1.0 - alpha,
        "empirical": empirical,
    }
